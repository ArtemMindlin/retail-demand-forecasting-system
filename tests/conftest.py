from __future__ import annotations

import os
from pathlib import Path

import django
import numpy as np
import pandas as pd
import pytest

from retail_forecasting import tracking
from retail_forecasting.contracts.contracts_config import ModelConfig

# Configure Django before any test imports a view or the ORM-free settings.
# Values here are test-only: a throwaway password and non-secure cookies so the
# test client can hold a session over plain HTTP.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "retail_forecasting.api.settings")
os.environ.setdefault("DJANGO_DEBUG", "true")
os.environ.setdefault("DJANGO_SECRET_KEY", "test-only-secret-key")
os.environ.setdefault("AUTH_USERNAME", "test-operator")
os.environ.setdefault("AUTH_PASSWORD", "test-password")
os.environ.setdefault("COOKIE_SECURE", "false")
django.setup()

REPO_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture(autouse=True)
def _models_dir_never_points_at_the_repo(tmp_path_factory, monkeypatch) -> None:
    """Redirect the default models directory away from the repo's own ``models/``.

    ``ModelConfig.models_dir`` defaults to a RELATIVE ``Path("models")`` and
    ``run_experiment_from_frame`` persists every model it trains there unconditionally.
    Any test that builds Settings without naming a directory therefore overwrote the
    operational champions with models fitted on a 3-series synthetic panel -- silently,
    since the files are gitignored and the suite still passed. This backstop makes that
    impossible for tests that forget; tests that train should still pass models_dir
    explicitly so the intent is visible at the call site.
    """
    sandbox = tmp_path_factory.mktemp("models_dir_default")
    field = ModelConfig.model_fields["models_dir"]
    monkeypatch.setattr(field, "default", sandbox)
    ModelConfig.model_rebuild(force=True)
    yield
    ModelConfig.model_rebuild(force=True)


def make_synthetic_panel(num_series: int = 3, num_days: int = 70) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=num_days, freq="D")
    rows = []

    for store_id in range(1, num_series + 1):
        product_id = 100 + store_id
        for index, date in enumerate(dates):
            base = 10 + store_id
            seasonal = 2.0 * np.sin(2 * np.pi * index / 7)
            demand = max(base + seasonal + (index % 5) * 0.5, 0.1)
            rows.append(
                {
                    "city_id": 1,
                    "store_id": store_id,
                    "management_group_id": 1,
                    "first_category_id": 1,
                    "second_category_id": 1,
                    "third_category_id": store_id,
                    "product_id": product_id,
                    "date": date,
                    "observed_demand": demand,
                    "stockout_hours": float(index % 3),
                    "discount": 1.0 - 0.05 * (index % 2),
                    "holiday_flag": int(date.dayofweek == 6),
                    "activity_flag": int(index % 10 == 0),
                    "precpt": float(index % 4),
                    "avg_temperature": 15.0 + store_id,
                    "avg_humidity": 50.0 + index % 10,
                    "avg_wind_level": 3.0 + (index % 3),
                    "series_id": f"{store_id}_{product_id}",
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _mlflow_never_lands_in_the_repo(tmp_path_factory, monkeypatch) -> None:
    """Point the tracking store at a scratch database, artifacts included.

    `write_run_artifacts` records every run in MLflow, so without this the suite would write
    into the repo's own `mlflow.db` and `mlruns/`. Redirecting the store is enough BECAUSE
    `tracking._artifact_root` derives the artifact directory from it; MLflow's own default
    resolves `mlruns/` against the working directory instead, independently, which is the trap
    `tests/test_imputation_tuning.py` works around by moving the cwd.
    """
    store = tmp_path_factory.mktemp("mlflow_store") / "mlflow.db"
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", f"sqlite:///{store}")
