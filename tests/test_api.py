from __future__ import annotations

import secrets
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from retail_forecasting.api.main import _SESSIONS, app

# Inject a valid session token so protected endpoints don't return 401
_TEST_TOKEN = secrets.token_urlsafe(32)
_SESSIONS.add(_TEST_TOKEN)
client = TestClient(app, cookies={"rf_session": _TEST_TOKEN})


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "Retail Demand Forecasting API"
    assert body["version"] == "1.0.0"
    assert "timestamp" in body
    assert "uptime" in body
    assert "data_loaded" in body


def test_predict_orders_invalid_config() -> None:
    response = client.post("/predict_orders", json={"config_path": "non_existent.yaml"})
    assert response.status_code == 404


def test_predict_orders_returns_recommendations_on_success(tmp_path: pytest.TempPath) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(Path("configs/experiment.yaml").read_text())

    mock_artifacts = mock.MagicMock()
    mock_artifacts.run_directory = str(tmp_path / "output")
    mock_artifacts.reorder_recommendations = pd.DataFrame(
        {"series_id": ["1_101", "2_202"], "order_quantity": [10.0, 5.0]}
    )

    with (
        mock.patch("retail_forecasting.api.main.run_scoring", return_value=mock_artifacts),
        mock.patch("retail_forecasting.api.main.run_experiment", return_value=mock_artifacts),
    ):
        response = client.post("/predict_orders", json={"config_path": str(config_file)})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["recommendations_generated"] == 2
    assert len(data["recommendations"]) == 2


def test_get_dashboard() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Forecast.ai · Retail Demand Forecasting System" in response.text


def test_api_forecast_with_mock_df() -> None:
    dummy_df = pd.DataFrame(
        {
            "date": ["2026-05-01"] * 5 + ["2026-05-02"] * 5,
            "series_id": ["SKU-100", "SKU-200"] * 5,
            "y_true": [100, 150, 110, 140, 120, 130, 160, 125, 135, 145],
            "y_pred": [98, 148, 105, 138, 118, 128, 158, 120, 130, 140],
            "data_strategy": ["strategy_a"] * 10,
        }
    )

    with mock.patch("retail_forecasting.api.main.load_latest_predictions", return_value=dummy_df):
        payload = {
            "serviceLevel": 90.0,
            "shortageCost": 15.0,
            "holdingCost": 3.0,
            "capacity": 10000.0,
            "selectedSkuId": "SKU-100",
        }
        response = client.post("/api/forecast", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "forecast" in data
        assert "kpis" in data
        assert "recommendation" in data
        assert len(data["forecast"]) > 0
        assert data["recommendation"]["qStar"] >= 0


def test_api_skus_with_mock_df() -> None:
    dummy_df = pd.DataFrame(
        {
            "date": ["2026-05-01", "2026-05-02"],
            "series_id": ["SKU-100", "SKU-100"],
            "y_true": [100, 110],
            "y_pred": [98, 105],
            "data_strategy": ["strategy_a", "strategy_a"],
        }
    )

    with mock.patch("retail_forecasting.api.main.load_latest_predictions", return_value=dummy_df):
        response = client.get(
            "/api/skus",
            params={
                "service_level": 95.0,
                "shortage_cost": 18.0,
                "holding_cost": 4.0,
                "capacity": 12000.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == "SKU-100"
        assert "lastActual" in data[0]
        assert "q_star" in data[0]


def test_api_drift() -> None:
    response = client.get("/api/drift", params={"service_level": 95.0})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "psi" in data[0]
    assert "pre" in data[0]
    assert "post" in data[0]


def test_api_alerts() -> None:
    response = client.get("/api/alerts")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "title" in data[0]
    assert "sev" in data[0]


def test_api_retrain() -> None:
    response = client.post("/api/retrain")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "triggered" in data["message"].lower()


def test_load_latest_predictions_fallback() -> None:
    from retail_forecasting.api.main import _PREDICTIONS_CACHE, load_latest_predictions

    # Clear predictions cache first to force reload
    _PREDICTIONS_CACHE.clear()

    # Patch reports_dir Path.exists to return False to force raise FileNotFoundError
    with (
        mock.patch("pathlib.Path.exists", return_value=False),
        pytest.raises(FileNotFoundError),
    ):
        load_latest_predictions()


def test_api_run_success() -> None:
    from retail_forecasting.api.main import _RUN_LOCK, _RUN_STATE

    _RUN_STATE["status"] = "idle"
    _RUN_STATE["error"] = None
    if _RUN_LOCK.locked():
        _RUN_LOCK.release()

    with mock.patch("retail_forecasting.api.main._execute_pipeline_in_background"):
        response = client.post("/api/run")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "started" in data["message"]


def test_api_run_conflict() -> None:
    from retail_forecasting.api.main import _RUN_LOCK

    _RUN_LOCK.acquire(blocking=False)
    try:
        response = client.post("/api/run")
        assert response.status_code == 409
        data = response.json()
        assert "already in progress" in data["detail"]
    finally:
        _RUN_LOCK.release()


def test_api_run_status() -> None:
    from retail_forecasting.api.main import _RUN_STATE

    _RUN_STATE["status"] = "success"
    _RUN_STATE["error"] = None
    log_file_mock = mock.MagicMock()
    log_file_mock.exists.return_value = True
    open_mock = mock.mock_open(read_data="--- Pipeline Execution logs mock ---")

    with (
        mock.patch("retail_forecasting.api.main.Path", return_value=log_file_mock),
        mock.patch("builtins.open", open_mock),
    ):
        response = client.get("/api/run/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["error"] is None
        assert "Pipeline Execution logs mock" in data["logs"]


def test_api_download_predictions_success(tmp_path: pytest.TempPath) -> None:
    from retail_forecasting.api.main import _PREDICTIONS_CACHE

    _PREDICTIONS_CACHE.clear()

    pred_file = tmp_path / "predictions.csv"
    pred_file.write_text("date,series_id,observed_demand\n2026-05-01,SKU-100,10.0\n")

    with mock.patch("retail_forecasting.api.main._get_latest_run_path", return_value=tmp_path):
        response = client.get("/api/download/predictions")
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        disp = response.headers["content-disposition"]
        assert 'attachment; filename="predictions.csv"' in disp
        assert response.text == "date,series_id,observed_demand\n2026-05-01,SKU-100,10.0\n"


def test_api_download_costs_success(tmp_path: pytest.TempPath) -> None:
    from retail_forecasting.api.main import _PREDICTIONS_CACHE

    _PREDICTIONS_CACHE.clear()

    cost_file = tmp_path / "cost_summary.csv"
    cost_file.write_text("series_id,holding_cost,shortage_cost\nSKU-100,2.0,15.0\n")

    with mock.patch("retail_forecasting.api.main._get_latest_run_path", return_value=tmp_path):
        response = client.get("/api/download/costs")
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        disp = response.headers["content-disposition"]
        assert 'attachment; filename="costs.csv"' in disp
        assert response.text == "series_id,holding_cost,shortage_cost\nSKU-100,2.0,15.0\n"


def test_api_download_costs_fallback_success(tmp_path: pytest.TempPath) -> None:
    from retail_forecasting.api.main import _PREDICTIONS_CACHE

    _PREDICTIONS_CACHE.clear()

    cost_file = tmp_path / "costs.csv"
    cost_file.write_text("series_id,holding_cost,shortage_cost\nSKU-100,2.0,15.0\n")

    with mock.patch("retail_forecasting.api.main._get_latest_run_path", return_value=tmp_path):
        response = client.get("/api/download/costs")
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        disp = response.headers["content-disposition"]
        assert 'attachment; filename="costs.csv"' in disp
        assert response.text == "series_id,holding_cost,shortage_cost\nSKU-100,2.0,15.0\n"


def test_api_download_not_found() -> None:
    from retail_forecasting.api.main import _PREDICTIONS_CACHE

    _PREDICTIONS_CACHE.clear()

    with mock.patch(
        "retail_forecasting.api.main._get_latest_run_path",
        side_effect=FileNotFoundError("No runs found"),
    ):
        response = client.get("/api/download/predictions")
        assert response.status_code == 404
        assert "No runs found" in response.json()["detail"]


def test_api_download_file_missing(tmp_path: pytest.TempPath) -> None:
    from retail_forecasting.api.main import _PREDICTIONS_CACHE

    _PREDICTIONS_CACHE.clear()

    with mock.patch("retail_forecasting.api.main._get_latest_run_path", return_value=tmp_path):
        response = client.get("/api/download/predictions")
        assert response.status_code == 404
        assert "predictions.csv not found" in response.json()["detail"]
