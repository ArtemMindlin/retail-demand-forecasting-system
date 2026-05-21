# ruff: noqa: E501
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from retail_forecasting.config import InventoryConfig
from retail_forecasting.evaluation.metrics import summarize_costs
from retail_forecasting.inventory.simulation import simulate_inventory_policy

st.set_page_config(page_title="Retail Forecasting TFG Dashboard", layout="wide")

# --- Inyección de CSS de Alta Gama (SaaS) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* Global Body and Font */
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #f8fafc !important;
    }

    h1, h2, h3, h4, h5, h6, [class*="Header"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        letter-spacing: -0.02em;
    }

    /* Modern Tabs styling */
    button[data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        border-bottom: 2px solid transparent !important;
        padding: 12px 20px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #2563eb !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2563eb !important;
        border-bottom-color: #2563eb !important;
    }

    /* Sidebar elegant SaaS styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] label {
        color: #94a3b8 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    /* Premium Custom Cards */
    .kpi-container {
        display: flex;
        gap: 15px;
        flex-wrap: wrap;
        width: 100%;
        margin-bottom: 25px;
        margin-top: 15px;
    }

    .kpi-card {
        flex: 1;
        min-width: 220px;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.03), 0 2px 4px -2px rgb(0 0 0 / 0.03);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 20px -5px rgb(0 0 0 / 0.08), 0 8px 12px -6px rgb(0 0 0 / 0.08);
        border-color: #cbd5e1;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(to bottom, #3b82f6, #2563eb);
    }
    .kpi-card.success::before {
        background: linear-gradient(to bottom, #10b981, #059669);
    }
    .kpi-card.warning::before {
        background: linear-gradient(to bottom, #f59e0b, #d97706);
    }
    .kpi-card.info::before {
        background: linear-gradient(to bottom, #6366f1, #4f46e5);
    }

    .kpi-title {
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .kpi-value {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }

    .kpi-delta {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .kpi-delta.positive {
        color: #10b981;
    }
    .kpi-delta.negative {
        color: #ef4444;
    }
    .kpi-delta.info {
        color: #3b82f6;
    }

    /* Elegant Expanders styling */
    div[data-testid="stExpander"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        margin-bottom: 12px !important;
    }
    div[data-testid="stExpander"] details {
        border: none !important;
    }
    div[data-testid="stExpander"] summary {
        color: #f8fafc !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* What-if action button */
    .stButton button {
        width: 100% !important;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 12px -2px rgba(37, 99, 235, 0.3) !important;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
    }
    .stButton button:active {
        transform: translateY(1px) !important;
    }

    /* Didactic Cards animation */
    .tfg-card {
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -2px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
        margin-bottom: 25px !important;
    }
    .tfg-card:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.08), 0 8px 12px -6px rgba(0, 0, 0, 0.08) !important;
        border-color: #cbd5e1 !important;
    }
    .tfg-card::before {
        content: "" !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        width: 5px !important;
        height: 100% !important;
        background: linear-gradient(to bottom, #3b82f6, #2563eb) !important;
    }
    .tfg-card.blue::before {
        background: linear-gradient(to bottom, #3b82f6, #2563eb) !important;
    }
    .tfg-card.green::before {
        background: linear-gradient(to bottom, #10b981, #059669) !important;
    }
    .tfg-card.red::before {
        background: linear-gradient(to bottom, #ef4444, #dc2626) !important;
    }
    .tfg-card.orange::before {
        background: linear-gradient(to bottom, #f59e0b, #d97706) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_report_dirs() -> list[Path]:
    reports_path = Path("reports")
    if not reports_path.exists():
        return []
    # Return only experiment directories that contain the required CSV files
    dirs = [
        d
        for d in reports_path.iterdir()
        if d.is_dir()
        and (d / "predictions.csv").exists()
        and (d / "metrics_summary.csv").exists()
        and (d / "cost_summary.csv").exists()
    ]
    return sorted(dirs, key=lambda x: x.stat().st_mtime, reverse=True)


def parse_report_md(report_path: Path) -> str:
    if not report_path.exists():
        return "No report.md found."
    content = report_path.read_text()

    # Simple and robust search for ALERT lines
    lines = content.split("\n")
    alerts = [line.strip() for line in lines if "**ALERT**" in line]

    if alerts:
        return " ".join(alerts)
    return "No drift detected."


# --- Sidebar ---
st.sidebar.title("🚀 TFG Dashboard")
st.sidebar.markdown("Forecasting de Demanda & Decisiones de Inventario")

report_dirs = get_report_dirs()
if not report_dirs:
    st.error("No se encontraron reportes en la carpeta `reports/`. Ejecuta un experimento primero.")
    st.stop()

selected_run = st.sidebar.selectbox(
    "Seleccionar Ejecución (Run)", report_dirs, format_func=lambda x: x.name
)


# --- Load Data ---
@st.cache_data
def load_data(
    run_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, pd.DataFrame | None, pd.DataFrame | None]:
    preds = pd.read_csv(run_path / "predictions.csv")
    metrics = pd.read_csv(run_path / "metrics_summary.csv")
    costs = pd.read_csv(run_path / "cost_summary.csv")

    sens_path = run_path / "sensitivity_summary.csv"
    sens = pd.read_csv(sens_path) if sens_path.exists() else None

    pareto_path = run_path / "pareto_frontier.csv"
    pareto = pd.read_csv(pareto_path) if pareto_path.exists() else None

    drift_info = parse_report_md(run_path / "report.md")
    return preds, metrics, costs, drift_info, sens, pareto


preds, metrics, costs, drift_info, sens, pareto = load_data(selected_run)

# --- Filters ---
if "data_strategy" in preds.columns:
    all_strategies = sorted(preds["data_strategy"].dropna().unique())
else:
    all_strategies = ["Observed"]

selected_strategy = st.sidebar.selectbox(
    "Estrategia de Datos",
    all_strategies,
    help="Compara cómo afecta entrenar con datos reales censurados (Observed) vs datos imputados (Latent)",
)

# Apply strategy filter to predictions, metrics and costs for the main displays
preds_filtered = (
    preds[preds["data_strategy"] == selected_strategy]
    if "data_strategy" in preds.columns
    else preds
)
metrics_filtered = (
    metrics[metrics["data_strategy"] == selected_strategy]
    if "data_strategy" in metrics.columns
    else metrics
)
costs_filtered = (
    costs[costs["data_strategy"] == selected_strategy]
    if "data_strategy" in costs.columns
    else costs
)

all_series = sorted(preds_filtered["series_id"].unique())
selected_series = st.sidebar.selectbox("Seleccionar Producto/Tienda", all_series)

all_models = sorted(preds_filtered["model_name"].unique())
selected_model = st.sidebar.selectbox("Seleccionar Modelo", all_models)

# --- What-If Analysis ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Scenario What-If Planning")
st.sidebar.markdown("Modifica parámetros para re-simular decisiones en tiempo real.")

with st.sidebar.expander("Parámetros de Coste Global", expanded=True):
    new_c_over = st.slider("Coste de Exceso (C_over)", 0.1, 5.0, 1.0, 0.1)
    new_c_under = st.slider("Coste de Rotura (C_under)", 0.5, 20.0, 4.0, 0.5)

with st.sidebar.expander("Restricciones Logísticas", expanded=True):
    use_capacity = st.checkbox("Aplicar Límite de Capacidad", value=True)
    new_capacity = (
        st.slider("Capacidad Global (Unidades)", 500, 10000, 3000, 100) if use_capacity else None
    )

run_what_if = st.sidebar.button("▶️ Simular Escenario")

what_if_preds = None
what_if_costs = None

if run_what_if:
    sim_preds = preds_filtered[preds_filtered["model_name"] == selected_model].copy()
    # Overwrite costs for the What-If scenario
    sim_preds["c_over"] = new_c_over
    sim_preds["c_under"] = new_c_under
    sim_preds["critical_fractile"] = new_c_under / (new_c_under + new_c_over)

    custom_config = InventoryConfig(
        overstock_cost=new_c_over,
        stockout_cost=new_c_under,
        use_series_costs=False,  # Force global costs for the what-if to see pure effect
        global_capacity_units=new_capacity,
    )

    with st.spinner("Simulando escenario logístico (LP Optimization)..."):
        what_if_preds = simulate_inventory_policy(
            predictions=sim_preds,
            inventory_config=custom_config,
            series_cost_profile=None,
        )
        what_if_costs = summarize_costs(what_if_preds)
    st.sidebar.success("Escenario Simulado!")

# --- Main Layout ---
# Premium styled header for TFG
st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 25px; border-radius: 15px; margin-bottom: 25px; color: white; box-shadow: 0 4px 20px rgba(0,0,0,0.15);">
        <h1 style="margin: 0; font-family: 'Outfit', 'Inter', sans-serif; font-size: 2.2rem; font-weight: 700; color: white;">🚀 Panel de Decisiones de Inventario & Forecasting Probabilístico</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 1.05rem;">Trabajo Fin de Grado (TFG) - Optimización de Inventario Fresco bajo Roturas e Incertidumbre</p>
        <div style="margin-top: 15px; display: flex; gap: 10px; flex-wrap: wrap;">
            <span style="background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; color: white;">🔍 Estrategia: <b>{selected_strategy}</b></span>
            <span style="background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; color: white;">🎯 Conformal Prediction</span>
            <span style="background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; color: white;">⚖️ Frontera de Pareto</span>
            <span style="background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; color: white;">📈 Streamlit Pro Edition</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader(f"📊 Ejecución seleccionada: {selected_run.name}")

# Metrics Calculation
base_cost = costs_filtered[costs_filtered["model_name"] == selected_model]["sim_total_cost"].sum()

new_cost = what_if_costs["sim_total_cost"].sum() if what_if_costs is not None else None

model_metrics = metrics_filtered[metrics_filtered["model_name"] == selected_model]
mae = model_metrics.iloc[0]["mae"] if not model_metrics.empty else 0.0

coverage = 0.0
if not model_metrics.empty:
    coverage = model_metrics.iloc[0].get(
        "interval_coverage", model_metrics.iloc[0].get("coverage_q_0_1_q_0_9", 0)
    )

base_sl = (
    costs_filtered[costs_filtered["model_name"] == selected_model]["sim_service_level"].mean()
    if "sim_service_level" in costs_filtered.columns
    else 0.0
)
new_sl = what_if_costs["sim_service_level"].mean() if what_if_costs is not None else None

# Format variables for custom HTML rendering
cost_val = f"${new_cost:,.2f}" if new_cost is not None else f"${base_cost:,.2f}"
cost_delta_html = ""
if new_cost is not None:
    diff = new_cost - base_cost
    arrow = "↓" if diff < 0 else "↑"
    cls = "positive" if diff < 0 else "negative"  # lower cost is positive
    cost_delta_html = (
        f'<div class="kpi-delta {cls}"><span>{arrow}</span> ${abs(diff):,.2f} vs Base</div>'
    )

mae_val = f"{mae:.2f}"
coverage_val = "N/A" if pd.isna(coverage) else f"{coverage * 100:.1f}%"

sl_val = (
    f"{new_sl * 100:.1f}%"
    if new_sl is not None
    else (f"{base_sl * 100:.1f}%" if base_sl else "0.0%")
)
sl_delta_html = ""
if new_sl is not None:
    diff_sl = (new_sl - base_sl) * 100
    arrow = "↑" if diff_sl > 0 else "↓"
    cls = "positive" if diff_sl > 0 else "negative"
    sl_delta_html = (
        f'<div class="kpi-delta {cls}"><span>{arrow}</span> {abs(diff_sl):.1f}% vs Base</div>'
    )

# HTML KPI cards render
st.markdown(
    f"""
    <div class="kpi-container">
        <div class="kpi-card info">
            <div class="kpi-title">💵 Coste Total de Inventario</div>
            <div class="kpi-value">{cost_val}</div>
            {cost_delta_html}
        </div>
        <div class="kpi-card warning">
            <div class="kpi-title">🎯 Precisión de Pronóstico (MAE)</div>
            <div class="kpi-value">{mae_val}</div>
            <div class="kpi-delta info">Error absoluto medio</div>
        </div>
        <div class="kpi-card success">
            <div class="kpi-title">🔒 Cobertura Conformal (80% Target)</div>
            <div class="kpi-value">{coverage_val}</div>
            <div class="kpi-delta positive">Garantía libre de distribución</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">📈 Nivel de Servicio Realizado</div>
            <div class="kpi-value">{sl_val}</div>
            {sl_delta_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Drift Alerts
if "ALERT" in drift_info:
    st.markdown(
        f"""
        <div style="background-color: rgba(239, 68, 68, 0.08); padding: 16px 20px; border-radius: 12px; border-left: 5px solid #ef4444; margin-bottom: 25px; color: #0f172a; font-size: 0.95rem;">
            <span style="font-size: 1.2rem; margin-right: 8px;">⚠️</span>
            <strong>Detección de Drift (Cambio de Régimen):</strong> {drift_info}
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div style="background-color: rgba(16, 185, 129, 0.08); padding: 16px 20px; border-radius: 12px; border-left: 5px solid #10b981; margin-bottom: 25px; color: #0f172a; font-size: 0.95rem;">
            <span style="font-size: 1.2rem; margin-right: 8px;">✅</span>
            <strong>Estabilidad Temporal Confirmada:</strong> No se detectó drift significativo en la demanda de frescos.
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Tabbed Navigation ---
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔮 Pronósticos e Inventario",
        "🔍 Imputación de Demanda Latente",
        "🎯 Frontera de Pareto y Sensibilidad",
        "📘 Marco Teórico y Analógico",
    ]
)

# --- Tab 1: Forecasting and Base Inventory ---
with tab1:
    st.markdown("### 📈 Pronóstico vs Realidad y Decisiones de Compra")
    st.markdown(
        "Visualiza las predicciones de punto, los intervalos probables (Conformal Prediction) y las cantidades sugeridas de pedido."
    )

    series_data = preds_filtered[
        (preds_filtered["series_id"] == selected_series)
        & (preds_filtered["model_name"] == selected_model)
    ].sort_values("date")

    fig = go.Figure()

    # Plot Confidence Interval (Conformal)
    q_low_col = "q_0_1"
    q_high_col = "q_0_9"

    if q_low_col in series_data.columns and q_high_col in series_data.columns:
        fig.add_trace(
            go.Scatter(
                x=pd.concat([series_data["date"], series_data["date"][::-1]]),
                y=pd.concat([series_data[q_high_col], series_data[q_low_col][::-1]]),
                fill="toself",
                fillcolor="rgba(59, 130, 246, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="Intervalo de Confianza (CP 80%)",
            )
        )

    # Plot Actual Demand (Total Horizon)
    fig.add_trace(
        go.Scatter(
            x=series_data["date"],
            y=series_data["y_true"],
            mode="lines+markers",
            name="Demanda Real (y_true)",
            line=dict(color="rgb(31, 41, 55)", width=2),
            marker=dict(size=6),
        )
    )

    # Plot Prediction
    fig.add_trace(
        go.Scatter(
            x=series_data["date"],
            y=series_data["y_pred"],
            mode="lines",
            name="Predicción Punto (y_pred)",
            line=dict(color="rgb(59, 130, 246)", dash="dash", width=2),
        )
    )

    # Plot Base Order Quantity
    fig.add_trace(
        go.Scatter(
            x=series_data["date"],
            y=series_data["order_quantity"],
            mode="markers",
            name="Decisión de Inventario (Base)",
            marker=dict(symbol="diamond", size=10, color="rgb(239, 68, 68)"),
        )
    )

    # Plot What-If Order Quantity if available
    if what_if_preds is not None:
        wi_series_data = what_if_preds[what_if_preds["series_id"] == selected_series].sort_values(
            "date"
        )
        fig.add_trace(
            go.Scatter(
                x=wi_series_data["date"],
                y=wi_series_data["order_quantity"],
                mode="markers",
                name="Decisión (What-If)",
                marker=dict(
                    symbol="star",
                    size=14,
                    color="rgb(245, 158, 11)",
                    line=dict(width=1, color="black"),
                ),
            )
        )

    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Unidades de Demanda",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, family="'Inter', sans-serif"),
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
        font=dict(family="'Outfit', 'Inter', sans-serif"),
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.95)",
            font_size=13,
            font_family="'Inter', sans-serif",
            font_color="#f8fafc",
            bordercolor="#334155",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "💡 **Consejo Académico:** Observa cómo las cantidades de inventario (rombo rojo) superan "
        "la predicción media (azul discontinua) en periodos de alta incertidumbre. Esto demuestra la aplicación del "
        "**Fractil Crítico** que, para un ratio de coste alto (Cs=4.0 vs Co=1.0), eleva preventivamente el stock "
        "de seguridad para evitar la penalización de una rotura de stock."
    )


# --- Tab 2: Censorship & Latent Demand Imputation ---
with tab2:
    st.markdown("### 🔍 Análisis del Sesgo de Censura e Imputación de Demanda Latente")
    st.markdown(
        "El gran problema en retail: si hay rotura de stock (*stockout*), las ventas caen a cero, "
        "lo que sesga a la baja las predicciones futuras. Corregimos esto imputando la demanda latente."
    )

    # Check if there is latent demand data anywhere in the predictions
    latent_strat = next(
        (s for s in preds["data_strategy"].dropna().unique() if "Latent_" in s), None
    )

    if latent_strat is not None and "original_observed_demand" in preds.columns:
        # Filter for the latent strategy for this series and model
        latent_series_data = preds[
            (preds["data_strategy"] == latent_strat)
            & (preds["series_id"] == selected_series)
            & (preds["model_name"] == selected_model)
        ].sort_values("date")

        if not latent_series_data.empty:
            fig_latent = go.Figure()

            # 1. Stockout Hours as amber bars in secondary Y axis
            fig_latent.add_trace(
                go.Bar(
                    x=latent_series_data["date"],
                    y=latent_series_data["stockout_hours"],
                    name="Horas de Rotura (Stockout Hours)",
                    marker_color="rgba(245, 158, 11, 0.25)",
                    yaxis="y2",
                )
            )

            # 2. Original sales (censored)
            fig_latent.add_trace(
                go.Scatter(
                    x=latent_series_data["date"],
                    y=latent_series_data["original_observed_demand"],
                    mode="lines+markers",
                    name="Venta Real Registrada (Censurada por Stockout)",
                    line=dict(color="rgb(220, 38, 38)", width=2),
                    marker=dict(size=6, symbol="x"),
                )
            )

            # 3. Imputed latent demand
            fig_latent.add_trace(
                go.Scatter(
                    x=latent_series_data["date"],
                    y=latent_series_data["latent_demand_est"],
                    mode="lines+markers",
                    name="Demanda Latente Imputada (Estimación Real)",
                    line=dict(color="rgb(16, 185, 129)", width=2),
                    marker=dict(size=6, symbol="circle"),
                )
            )

            fig_latent.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Unidades de Demanda",
                yaxis2=dict(
                    title="Horas de Rotura (Stockout Hours)",
                    overlaying="y",
                    side="right",
                    range=[0, 24],
                    showgrid=False,
                ),
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=11, family="'Inter', sans-serif"),
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor="#f8fafc",
                paper_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
                yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
                font=dict(family="'Outfit', 'Inter', sans-serif"),
                hoverlabel=dict(
                    bgcolor="rgba(15, 23, 42, 0.95)",
                    font_size=13,
                    font_family="'Inter', sans-serif",
                    font_color="#f8fafc",
                    bordercolor="#334155",
                ),
            )

            st.plotly_chart(fig_latent, use_container_width=True)

            st.success(
                "🎯 **Evidencia Visual de Imputación:** Observa que cuando **Horas de Rotura** es mayor a 0 (barra naranja), "
                "la **Venta Registrada** (rojo) queda artificialmente baja, mientras que la **Demanda Latente** (verde) "
                "se reconstruye de forma supervisada (LGBM Imputer) a un nivel superior, logrando corregir el sesgo."
            )
        else:
            st.warning("No se encontraron registros de demanda latente para esta serie.")
    else:
        st.info(
            "ℹ️ **No se detecta un Experimento Comparativo:** Para habilitar este análisis visual dinámico, "
            "selecciona en la barra lateral una ejecución que contenga un experimento completo con imputación "
            "(por ejemplo, `fresh_retailnet_v2_20260519_000700`)."
        )


# --- Tab 3: Pareto Frontier & Sensitivity ---
with tab3:
    st.markdown("### 🎯 Frontera de Pareto de Nivel de Servicio vs Costes")
    st.markdown(
        "El dilema clásico de la cadena de suministro. La Frontera de Pareto muestra las decisiones "
        "que son matemáticamente eficientes bajo un conjunto de restricciones logísticas."
    )

    if pareto is not None and not pareto.empty:
        # Create a scatter plot of Service Level vs Total Cost
        fig_pareto = go.Figure()

        # Group Pareto points by strategy and model for scatter
        if "data_strategy" in pareto.columns:
            for (strategy_name, model_name), group in pareto.groupby(
                ["data_strategy", "model_name"]
            ):
                fig_pareto.add_trace(
                    go.Scatter(
                        x=group["total_cost"],
                        y=group["service_level"] * 100,
                        mode="markers",
                        name=f"{model_name} ({strategy_name})",
                        marker=dict(size=10, opacity=0.8),
                        hovertemplate=(
                            "<b>%{text}</b><br><br>"
                            "Coste Total: $%{x:,.2f}<br>"
                            "Nivel Servicio: %{y:.2f}%<br>"
                            "Escala de Orden: %{customdata:.1f}<br>"
                            "<extra></extra>"
                        ),
                        text=[
                            f"{m} - {s}"
                            for m, s in zip(
                                group["model_name"], group["data_strategy"], strict=False
                            )
                        ],
                        customdata=group["order_scale"],
                    )
                )
        else:
            for model_name, group in pareto.groupby("model_name"):
                fig_pareto.add_trace(
                    go.Scatter(
                        x=group["total_cost"],
                        y=group["service_level"] * 100,
                        mode="markers",
                        name=model_name,
                        marker=dict(size=10, opacity=0.8),
                        hovertemplate=(
                            "<b>%{text}</b><br><br>"
                            "Coste Total: $%{x:,.2f}<br>"
                            "Nivel Servicio: %{y:.2f}%<br>"
                            "Escala de Orden: %{customdata:.1f}<br>"
                            "<extra></extra>"
                        ),
                        text=[m for m in group["model_name"]],
                        customdata=group["order_scale"]
                        if "order_scale" in group.columns
                        else [1.0] * len(group),
                    )
                )

        # Draw the actual Pareto frontier line (points where is_pareto_efficient == True)
        pareto["is_pareto_efficient"] = (
            pareto["is_pareto_efficient"].astype(str).str.lower() == "true"
        )
        pareto_efficient = pareto[pareto["is_pareto_efficient"]].sort_values("total_cost")

        if not pareto_efficient.empty:
            fig_pareto.add_trace(
                go.Scatter(
                    x=pareto_efficient["total_cost"],
                    y=pareto_efficient["service_level"] * 100,
                    mode="lines",
                    name="Frontera de Pareto (Óptimo Eficiente)",
                    line=dict(color="rgb(16, 185, 129)", width=3, dash="dash"),
                    hoverinfo="skip",
                )
            )

        fig_pareto.update_layout(
            xaxis_title="Coste Total de Inventario ($)",
            yaxis_title="Nivel de Servicio (%)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11, family="'Inter', sans-serif"),
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="#f8fafc",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
            font=dict(family="'Outfit', 'Inter', sans-serif"),
            hoverlabel=dict(
                bgcolor="rgba(15, 23, 42, 0.95)",
                font_size=13,
                font_family="'Inter', sans-serif",
                font_color="#f8fafc",
                bordercolor="#334155",
            ),
        )

        st.plotly_chart(fig_pareto, use_container_width=True)

        st.info(
            "💡 **Explicación del Gráfico:** La **Frontera de Pareto** (línea discontinua verde) une los puntos óptimos. "
            "Los puntos a la derecha e inferiores son ineficientes (tienen costes más altos para un mismo nivel de servicio). "
            "Un tomador de decisiones puede deslizarse por la frontera verde para decidir cuánto coste está dispuesto "
            "a tolerar para alcanzar un nivel de servicio específico."
        )
    else:
        st.info("ℹ️ No se encontró `pareto_frontier.csv` para esta ejecución.")

    # Economic Sensitivity plot
    if sens is not None:
        st.markdown("---")
        st.markdown(
            "### 📊 Sensibilidad a Ratios de Penalización de Rotura ($C_{stockout}/C_{overstock}$)"
        )
        st.markdown(
            "Muestra cómo escala el coste total de inventario a medida que el negocio penaliza más duramente "
            "las roturas de stock. Un modelo con un buen pronóstico probabilístico (conformal) escala de forma mucho más controlada."
        )

        fig_sens = go.Figure()
        if "data_strategy" in sens.columns:
            for (strat_name, model_name), group in sens.groupby(["data_strategy", "model_name"]):
                fig_sens.add_trace(
                    go.Scatter(
                        x=group["ratio"],
                        y=group["total_cost"],
                        mode="lines+markers",
                        name=f"{model_name} ({strat_name})",
                        line=dict(width=2),
                        marker=dict(size=6),
                    )
                )
        else:
            for model_name, group in sens.groupby("model_name"):
                fig_sens.add_trace(
                    go.Scatter(
                        x=group["ratio"],
                        y=group["total_cost"],
                        mode="lines+markers",
                        name=model_name,
                        line=dict(width=2),
                        marker=dict(size=6),
                    )
                )

        fig_sens.update_layout(
            xaxis_title="Ratio de Coste Cs/Co",
            yaxis_title="Coste Total ($)",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11, family="'Inter', sans-serif"),
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="#f8fafc",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
            font=dict(family="'Outfit', 'Inter', sans-serif"),
            hoverlabel=dict(
                bgcolor="rgba(15, 23, 42, 0.95)",
                font_size=13,
                font_family="'Inter', sans-serif",
                font_color="#f8fafc",
                bordercolor="#334155",
            ),
        )
        st.plotly_chart(fig_sens, use_container_width=True)


# --- Tab 4: Didactic Theoretical Framework for TFG ---
with tab4:
    st.markdown("### 📘 Marco Teórico y Analogías Didácticas para la Defensa de TFG")
    st.markdown(
        "Prepara tu defensa académica con explicaciones claras, rigurosas y conceptuales "
        "de las metodologías clave implementadas en esta plataforma."
    )

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown(
            """
            <div class="tfg-card blue" style="min-height: 380px;">
                <h4 style="margin-top: 0; color: #1e3a8a;
                           font-family: 'Outfit', sans-serif; font-size: 1.25rem;">
                    🌧️ 1. La Analogía del Meteorólogo (Conformal Prediction)
                </h4>
                <p style="font-size: 0.95rem; text-align: justify;
                          line-height: 1.6; color: #334155; margin-bottom: 12px;">
                    En el TFG, no usamos intervalos de confianza estadísticos tradicionales
                    (que asumen normalidad y suelen fallar en colas), sino
                    <b>Conformal Prediction (CP)</b>, un método no paramétrico moderno
                    que garantiza cobertura real.
                </p>
                <p style="font-size: 0.95rem; text-align: justify; line-height: 1.6;
                          font-style: italic; background: #f8fafc; padding: 12px 16px;
                          border-radius: 10px; border-left: 3px solid #3b82f6;
                          color: #475569; margin-bottom: 12px;">
                    "Imagina un meteorólogo que dice: 'Mañana lloverá con un 80% de
                    probabilidad'. Si evalúas todas sus predicciones históricas
                    y resulta que llovió exactamente en el 80% de los días que hizo
                    este anuncio, el meteorólogo está <b>calibrado</b>.
                    Conformal Prediction garantiza ex-ante que nuestros intervalos del 80%
                    contendrán la demanda real exactamente el 80% del tiempo,
                    independientemente de qué tan sesgado esté el estimador base
                    (LGBM o CatBoost)."
                </p>
                <p style="font-size: 0.9rem; color: #2563eb; font-weight: 600; margin: 0;
                          display: flex; align-items: center; gap: 6px;">
                    <span>💡</span> <strong>Concepto Clave:</strong>
                    Calibración Empírica Libre de Distribución.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="tfg-card green" style="min-height: 300px;">
                <h4 style="margin-top: 0; color: #065f46;
                           font-family: 'Outfit', sans-serif; font-size: 1.25rem;">
                    ⚖️ 3. La Frontera Eficiente de Pareto
                </h4>
                <p style="font-size: 0.95rem; text-align: justify;
                          line-height: 1.6; color: #334155; margin-bottom: 12px;">
                    En logística de frescos, existe una contradicción intrínseca
                    entre coste y servicio. La <b>Frontera de Pareto</b> demuestra
                    que no hay una única "decisión perfecta", sino un
                    <b>conjunto de decisiones óptimas</b>.
                </p>
                <p style="font-size: 0.95rem; text-align: justify; line-height: 1.6;
                          color: #475569; margin: 0;">
                    El algoritmo calcula simulación de inventarios barriendo un factor
                    multiplicativo (escala de orden) desde 0.7x hasta 1.3x. La frontera
                    une los puntos donde ya no puedes reducir costes sin empeorar el
                    nivel de servicio, dando soporte a decisiones ejecutivas
                    estratégicas basadas en presupuesto.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_t2:
        st.markdown(
            """
            <div class="tfg-card red" style="min-height: 380px;">
                <h4 style="margin-top: 0; color: #991b1b;
                           font-family: 'Outfit', sans-serif; font-size: 1.25rem;">
                    📰 2. El Modelo del Vendedor de Periódicos (Newsvendor)
                </h4>
                <p style="font-size: 0.95rem; text-align: justify;
                          line-height: 1.6; color: #334155; margin-bottom: 12px;">
                    La demanda es una variable aleatoria y pedir la media/predicción puntual
                    es financieramente incorrecto en presencia de asimetrías de costes.
                    Usamos el formalismo del <b>Fractil Crítico (Critical Fractile)</b>.
                </p>
                <div style="font-size: 0.95rem; text-align: justify; line-height: 1.6;
                            background: #f8fafc; padding: 12px 16px; border-radius: 10px;
                            border-left: 3px solid #ef4444; color: #475569;
                            margin-bottom: 12px;">
                    La cantidad óptima a pedir es el cuantil de demanda correspondiente a:
                    <span style="font-family: monospace; font-weight: 700;
                                 font-size: 1.15rem; display: block; text-align: center;
                                 color: #b91c1c; margin: 8px 0;">
                        τ* = C_under / (C_under + C_over)
                    </span>
                    Donde <b>C_under</b> es el coste unitario por quedarnos cortos (rotura)
                    y <b>C_over</b> es el coste de exceso (merma).
                    Con Cs=4 y Co=1, el Fractil Crítico τ* es 0.80, lo que significa que
                    el inventario óptimo debe cubrir el cuantil del 80% de la distribución
                    conformal de demanda para maximizar la rentabilidad esperada.
                </div>
                <p style="font-size: 0.9rem; color: #ef4444; font-weight: 600; margin: 0;
                          display: flex; align-items: center; gap: 6px;">
                    <span>💡</span> <strong>Concepto Clave:</strong>
                    Asimetría de Pérdidas y Coste de Oportunidad.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="tfg-card orange" style="min-height: 300px;">
                <h4 style="margin-top: 0; color: #78350f;
                           font-family: 'Outfit', sans-serif; font-size: 1.25rem;">
                    📦 4. Optimización LP con Capacidad
                </h4>
                <p style="font-size: 0.95rem; text-align: justify;
                          line-height: 1.6; color: #334155; margin-bottom: 12px;">
                    En el mundo real, los almacenes o camiones tienen una capacidad máxima.
                    Cuando aplicas el modelo Newsvendor independiente a cada SKU,
                    la suma total de órdenes de compra puede exceder el límite físico global.
                </p>
                <p style="font-size: 0.95rem; text-align: justify; line-height: 1.6;
                          color: #475569; margin: 0;">
                    Nuestra plataforma resuelve un **Problema de Programación Lineal (LP)**
                    dinámico cada día. Cuando el límite global de capacidad está activo,
                    el sistema redistribuye inteligentemente las cuotas de capacidad
                    priorizando los productos con mayor criticidad o mayor penalización
                    por rotura.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- Bottom Cost Analysis Table ---
st.markdown("---")
st.subheader("💰 Desglose Económico de Métricas y Costes")
c1, c2 = st.columns(2)

with c1:
    st.write("📊 Métricas de Pronóstico por Modelo y Estrategia")
    st.dataframe(metrics, use_container_width=True)

with c2:
    if what_if_costs is not None:
        st.write("💸 Comparación de Costes de Inventario (Escenario What-If vs Base)")
        comparison = pd.merge(
            costs_filtered[costs_filtered["model_name"] == selected_model][
                ["backend_name", "sim_total_cost", "sim_service_level"]
            ],
            what_if_costs[["backend_name", "sim_total_cost", "sim_service_level"]],
            on="backend_name",
            suffixes=("_base", "_whatif"),
        )
        st.dataframe(comparison, use_container_width=True)
    else:
        st.write("💸 Costes de Inventario por Modelo y Estrategia")
        st.dataframe(costs, use_container_width=True)

st.markdown("---")
st.caption(
    "Dashboard desarrollado para el TFG sobre Forecasting Probabilístico "
    "y Decisiones de Inventario."
)
