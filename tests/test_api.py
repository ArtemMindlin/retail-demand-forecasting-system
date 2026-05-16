from __future__ import annotations

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
