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
    config_file.write_text(Path("configs/default.yaml").read_text())

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
    assert "Retail Forecasting · Decision Dashboard" in response.text


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

    # Patch reports_dir Path.exists to return False to force fallback
    with mock.patch("pathlib.Path.exists", return_value=False):
        df = load_latest_predictions()

    assert not df.empty
    assert "date" in df.columns
    assert "series_id" in df.columns
    assert "y_true" in df.columns
    assert "y_pred" in df.columns
    assert "data_strategy" in df.columns
    assert "SKU-1001" in df["series_id"].values
