"""Tests for the Django dashboard: auth, pages, and the documented JSON surface.

Most views are exercised against a real ``ArtifactStore`` pointed at ``tmp_path``
rather than mocks, so the tests cover artifact discovery and parsing too. Only
the pipeline entry points are mocked, since running them would train a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from django.test import Client, override_settings

from retail_forecasting.api import store as store_module

USERNAME = "test-operator"
PASSWORD = "test-password"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    """An empty reports directory wired into settings for one test."""
    reports = tmp_path / "reports"
    reports.mkdir()
    with override_settings(REPORTS_DIR=reports):
        store_module.reset_store()
        yield reports
    store_module.reset_store()


@pytest.fixture
def run_with_predictions(reports_dir: Path) -> Path:
    """A run directory holding a small but realistic predictions.csv."""
    run = reports_dir / "20260101_120000_test"
    run.mkdir()
    frame = pd.DataFrame(
        {
            "date": ["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04"] * 2,
            "series_id": ["SKU-100"] * 4 + ["SKU-200"] * 4,
            "y_true": [100, 110, 120, 130, 150, 140, 160, 145],
            "y_pred": [98, 105, 118, 128, 148, 138, 158, 140],
            "data_strategy": ["Observed"] * 8,
            "model_name": ["catboost"] * 8,
        }
    )
    frame.to_csv(run / "predictions.csv", index=False)
    return run


@pytest.fixture
def client() -> Client:
    """An unauthenticated test client."""
    return Client()


@pytest.fixture
def auth_client(client: Client) -> Client:
    """A client holding a valid operator session."""
    response = client.post("/login/", {"username": USERNAME, "password": PASSWORD})
    assert response.status_code == 302
    return client


# ── Authentication ────────────────────────────────────────────────────────────


def test_login_page_is_public(client: Client) -> None:
    response = client.get("/login/")
    assert response.status_code == 200
    assert "Iniciar sesión" in response.content.decode()


def test_wrong_credentials_are_rejected(client: Client) -> None:
    response = client.post("/login/", {"username": USERNAME, "password": "nope"})
    assert response.status_code == 401
    assert "Credenciales incorrectas" in response.content.decode()


@override_settings(AUTH_PASSWORD="")
def test_unset_password_disables_login(client: Client) -> None:
    """An unconfigured AUTH_PASSWORD must refuse everything, not accept ''."""
    response = client.post("/login/", {"username": USERNAME, "password": ""})
    assert response.status_code == 401


def test_pages_redirect_anonymous_users_to_login(client: Client) -> None:
    response = client.get("/drift/")
    assert response.status_code == 302
    assert response.headers["Location"] == "/login/?next=/drift/"


def test_json_endpoints_return_401_for_anonymous_users(client: Client) -> None:
    response = client.get("/api/skus")
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


def test_htmx_requests_get_a_redirect_header(client: Client) -> None:
    """htmx cannot follow a 302 into a fragment target, so it needs HX-Redirect."""
    response = client.get("/drift/", headers={"HX-Request": "true"})
    assert response.status_code == 401
    assert response.headers["HX-Redirect"] == "/login/"


def test_logout_clears_the_session(auth_client: Client) -> None:
    assert auth_client.post("/logout/").status_code == 302
    assert auth_client.get("/drift/").status_code == 302


# ── Health and API reference ──────────────────────────────────────────────────


def test_health_check_is_public(client: Client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "Retail Demand Forecasting API"
    assert "timestamp" in body
    assert "uptime" in body
    assert "data_loaded" in body


def test_api_reference_lists_the_documented_surface(auth_client: Client) -> None:
    response = auth_client.get("/api/")
    assert response.status_code == 200
    body = response.content.decode()
    for path in ("/api/forecast", "/api/skus", "/predict_orders", "/health"):
        assert path in body


# ── Pages ─────────────────────────────────────────────────────────────────────


def test_dashboard_renders_kpis_and_chart(auth_client: Client, run_with_predictions: Path) -> None:
    response = auth_client.get("/dashboard/")
    assert response.status_code == 200
    body = response.content.decode()
    assert "Coste de Inventario" in body
    assert "Cobertura Empírica" in body
    assert "forecast-svg" in body


def test_dashboard_shows_empty_state_without_predictions(
    auth_client: Client, reports_dir: Path
) -> None:
    response = auth_client.get("/dashboard/")
    assert response.status_code == 200
    assert "Todavía no hay predicciones" in response.content.decode()


def test_sku_table_filters_and_sorts_from_the_query_string(
    auth_client: Client, run_with_predictions: Path
) -> None:
    response = auth_client.get("/skus/", {"q": "SKU-100", "sort": "q_star", "dir": "asc"})
    assert response.status_code == 200
    body = response.content.decode()
    assert "SKU-100" in body
    assert "SKU-200" not in body


def test_drift_view_reads_the_real_report(auth_client: Client, run_with_predictions: Path) -> None:
    report = [
        {
            "name": "temperature_7d",
            "type": "numeric",
            "importance": 0.32,
            "psi": 0.27,
            "status": "critical",
            "pre": [0.2, 0.5, 0.3],
            "post": [0.1, 0.3, 0.6],
        }
    ]
    (run_with_predictions / "drift_report.json").write_text(json.dumps(report), encoding="utf-8")

    response = auth_client.get("/drift/")
    assert response.status_code == 200
    body = response.content.decode()
    assert "temperature_7d" in body
    assert "PSI 0.270" in body


def test_drift_view_says_so_when_the_report_is_missing(
    auth_client: Client, run_with_predictions: Path
) -> None:
    response = auth_client.get("/drift/")
    assert response.status_code == 200
    assert "Sin informe de drift" in response.content.decode()


def test_academic_modal_renders_live_parameters(
    auth_client: Client, run_with_predictions: Path
) -> None:
    response = auth_client.get("/dashboard/modulo/newsvendor/", {"shortage_cost": 20})
    assert response.status_code == 200
    body = response.content.decode()
    assert "Newsvendor" in body
    # CR* = 20 / (20 + 4) = 0.833 with the default holding cost.
    assert "0.833" in body


def test_unknown_academic_module_is_404(auth_client: Client, reports_dir: Path) -> None:
    assert auth_client.get("/dashboard/modulo/nope/").status_code == 404


def test_ops_view_reports_a_missing_simulation(auth_client: Client, reports_dir: Path) -> None:
    response = auth_client.get("/ops/")
    assert response.status_code == 200
    assert "simulación operacional no se ha ejecutado" in response.content.decode()


def test_eda_view_reports_a_missing_run(auth_client: Client, reports_dir: Path) -> None:
    response = auth_client.get("/eda/")
    assert response.status_code == 200
    assert "No hay ningún run de EDA" in response.content.decode()


# ── JSON surface ──────────────────────────────────────────────────────────────


def test_forecast_endpoint_returns_kpis_and_recommendation(
    auth_client: Client, run_with_predictions: Path
) -> None:
    response = auth_client.post(
        "/api/forecast",
        data=json.dumps({"serviceLevel": 90.0, "selectedSkuId": "SKU-100"}),
        content_type="application/json",
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sku"] == "SKU-100"
    assert set(body["kpis"]) == {"inventoryCost", "coverage", "mae", "driftPSI"}
    assert body["recommendation"]["qStar"] >= 0
    assert len(body["forecast"]) == 4


def test_forecast_endpoint_reports_missing_predictions(
    auth_client: Client, reports_dir: Path
) -> None:
    response = auth_client.post("/api/forecast", data="{}", content_type="application/json")
    assert response.status_code == 200
    assert response.json() == {"status": "no_predictions"}


def test_skus_endpoint_returns_one_row_per_series(
    auth_client: Client, run_with_predictions: Path
) -> None:
    response = auth_client.get("/api/skus", {"service_level": 90})
    assert response.status_code == 200
    rows = response.json()
    assert {row["id"] for row in rows} == {"SKU-100", "SKU-200"}
    assert all(row["coverageTarget"] == 90 for row in rows)


def test_skus_endpoint_is_empty_without_predictions(auth_client: Client, reports_dir: Path) -> None:
    assert auth_client.get("/api/skus").json() == []


def test_download_predictions(auth_client: Client, run_with_predictions: Path) -> None:
    response = auth_client.get("/api/download/predictions")
    assert response.status_code == 200
    assert response["Content-Type"] == "text/csv"
    assert b"series_id" in b"".join(response.streaming_content)


def test_download_costs_falls_back_to_the_legacy_filename(
    auth_client: Client, run_with_predictions: Path
) -> None:
    (run_with_predictions / "costs.csv").write_text("total_cost\n42\n", encoding="utf-8")
    response = auth_client.get("/api/download/costs")
    assert response.status_code == 200
    assert b"42" in b"".join(response.streaming_content)


def test_download_is_404_when_the_file_is_absent(
    auth_client: Client, run_with_predictions: Path
) -> None:
    assert auth_client.get("/api/download/costs").status_code == 404


def test_download_is_404_without_any_run(auth_client: Client, reports_dir: Path) -> None:
    assert auth_client.get("/api/download/predictions").status_code == 404


def test_predict_orders_rejects_a_missing_config(auth_client: Client, reports_dir: Path) -> None:
    response = auth_client.post(
        "/predict_orders",
        data=json.dumps({"config_path": "non_existent.yaml"}),
        content_type="application/json",
    )
    assert response.status_code == 404


def test_predict_orders_returns_recommendations(
    auth_client: Client, reports_dir: Path, tmp_path: Path
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(Path("configs/experiment.yaml").read_text(encoding="utf-8"))

    artifacts = mock.MagicMock()
    artifacts.run_directory = str(tmp_path / "output")
    artifacts.reorder_recommendations = pd.DataFrame(
        {"series_id": ["1_101", "2_202"], "order_quantity": [10.0, 5.0]}
    )

    with (
        mock.patch("retail_forecasting.api.views.json_api.run_scoring", return_value=artifacts),
        mock.patch("retail_forecasting.api.views.json_api.run_experiment", return_value=artifacts),
    ):
        response = auth_client.post(
            "/predict_orders",
            data=json.dumps({"config_path": str(config_file)}),
            content_type="application/json",
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["recommendations_generated"] == 2


# ── Overlay panels ────────────────────────────────────────────────────────────


def test_alerts_panel_lists_exceptions(auth_client: Client, run_with_predictions: Path) -> None:
    pd.DataFrame(
        {
            "series_id": ["SKU-100"],
            "risk_flag": ["extreme_uncertainty"],
            "notes": ["Revisar."],
            "order_quantity": [12],
        }
    ).to_csv(run_with_predictions / "exceptions.csv", index=False)

    response = auth_client.get("/alertas/")
    assert response.status_code == 200
    body = response.content.decode()
    assert "SKU-100" in body
    assert "Crítica" in body


def test_alerts_panel_is_empty_without_exceptions(
    auth_client: Client, run_with_predictions: Path
) -> None:
    response = auth_client.get("/alertas/")
    assert response.status_code == 200
    assert "Sin excepciones" in response.content.decode()


def test_config_editor_shows_the_current_file(auth_client: Client) -> None:
    response = auth_client.get("/configuracion/")
    assert response.status_code == 200
    assert "project:" in response.content.decode()


def test_config_editor_rejects_invalid_yaml(auth_client: Client, tmp_path: Path) -> None:
    target = tmp_path / "experiment.yaml"
    original = Path("configs/experiment.yaml").read_text(encoding="utf-8")
    target.write_text(original, encoding="utf-8")

    with override_settings(CONFIG_PATH=target):
        response = auth_client.post("/configuracion/", {"yaml": "project: [unclosed"})

    assert response.status_code == 400
    assert "YAML inválido" in response.content.decode()
    # The live file must be untouched by a rejected edit.
    assert target.read_text(encoding="utf-8") == original


def test_config_editor_saves_a_valid_file(auth_client: Client, tmp_path: Path) -> None:
    target = tmp_path / "experiment.yaml"
    original = Path("configs/experiment.yaml").read_text(encoding="utf-8")

    with override_settings(CONFIG_PATH=target):
        response = auth_client.post("/configuracion/", {"yaml": original})

    assert response.status_code == 200
    assert "guardada y validada" in response.content.decode()
    assert target.read_text(encoding="utf-8") == original


# ── Pipeline console ──────────────────────────────────────────────────────────


def test_pipeline_status_renders_the_console(auth_client: Client, reports_dir: Path) -> None:
    response = auth_client.get("/pipeline/status/")
    assert response.status_code == 200
    assert "retail_forecasting.run" in response.content.decode()


def test_pipeline_run_reports_a_missing_config(
    auth_client: Client, reports_dir: Path, tmp_path: Path
) -> None:
    with override_settings(CONFIG_PATH=tmp_path / "missing.yaml"):
        from retail_forecasting.api.views import pipeline as pipeline_views

        pipeline_views._runner = None
        response = auth_client.post("/pipeline/run/")
        pipeline_views._runner = None

    assert response.status_code == 200
    assert "Configuration file not found" in response.content.decode()
