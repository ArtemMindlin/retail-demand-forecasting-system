from __future__ import annotations

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from retail_forecasting.api.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "Retail Forecasting API"}


def test_predict_orders_invalid_config() -> None:
    response = client.post("/predict_orders", json={"config_path": "non_existent.yaml"})
    assert response.status_code == 404


def test_predict_orders_returns_recommendations_on_success(tmp_path: pytest.TempPath) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(Path("configs/default.yaml").read_text())

    mock_artifacts = mock.MagicMock()
    mock_artifacts.run_directory = str(tmp_path / "output")
    mock_artifacts.reorder_recommendations = pd.DataFrame(
        {"series_id": ["1_101", "2_202"], "order_quantity": [10.0, 5.0]}
    )

    with mock.patch("retail_forecasting.api.main.run_experiment", return_value=mock_artifacts):
        response = client.post("/predict_orders", json={"config_path": str(config_file)})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["recommendations_generated"] == 2
    assert len(data["recommendations"]) == 2
