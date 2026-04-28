import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import re

st.set_page_config(page_title="Retail Forecasting TFG Dashboard", layout="wide")

def get_report_dirs():
    reports_path = Path("reports")
    if not reports_path.exists():
        return []
    # Return directories sorted by modification time (newest first)
    dirs = [d for d in reports_path.iterdir() if d.is_dir()]
    return sorted(dirs, key=lambda x: x.stat().st_mtime, reverse=True)

def parse_report_md(report_path):
    if not report_path.exists():
        return "No report.md found."
    content = report_path.read_text()
    
    # Simple and robust search for ALERT lines
    lines = content.split("\n")
    alerts = [l.strip() for l in lines if "**ALERT**" in l]
    
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

selected_run = st.sidebar.selectbox("Seleccionar Ejecución (Run)", report_dirs, format_func=lambda x: x.name)

# --- Load Data ---
@st.cache_data
def load_data(run_path):
    preds = pd.read_csv(run_path / "predictions.csv")
    metrics = pd.read_csv(run_path / "metrics_summary.csv")
    costs = pd.read_csv(run_path / "cost_summary.csv")
    
    sens_path = run_path / "sensitivity_summary.csv"
    sens = pd.read_csv(sens_path) if sens_path.exists() else None
    
    drift_info = parse_report_md(run_path / "report.md")
    return preds, metrics, costs, drift_info, sens

preds, metrics, costs, drift_info, sens = load_data(selected_run)

# --- Filters ---
all_series = sorted(preds["series_id"].unique())
selected_series = st.sidebar.selectbox("Seleccionar Producto/Tienda", all_series)

all_models = sorted(preds["model_name"].unique())
selected_model = st.sidebar.selectbox("Seleccionar Modelo", all_models)

# --- Main Layout ---
st.title(f"📊 Análisis de Resultados: {selected_run.name}")

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Cost", f"${costs['total_cost'].sum():,.2f}")
with col2:
    # Filter metrics for the selected model
    model_metrics = metrics[metrics['model_name'] == selected_model]
    if not model_metrics.empty:
        mae = model_metrics.iloc[0]['mae']
        st.metric(f"MAE ({selected_model})", f"{mae:.2f}")
with col3:
    if not model_metrics.empty:
        coverage = model_metrics.iloc[0].get('coverage_q_0_1_q_0_9', 0)
        # Handle nan coverage for naive
        if pd.isna(coverage):
            st.metric("Interval Coverage", "N/A")
        else:
            st.metric("Interval Coverage (80% target)", f"{coverage*100:.1f}%")
with col4:
    st.info(f"Model: {selected_model}")

# Drift Alerts
if "ALERT" in drift_info:
    st.warning(f"⚠️ **Detección de Drift:** {drift_info}")
else:
    st.success("✅ Estabilidad confirmada: No se detectó Drift significativo.")

# --- Forecast Plot ---
st.subheader(f"📈 Forecast vs Realidad - {selected_series} ({selected_model})")

series_data = preds[(preds["series_id"] == selected_series) & (preds["model_name"] == selected_model)].sort_values("date")

fig = go.Figure()

# Plot Confidence Interval (Conformal)
q_low_col = "q_0_1"
q_high_col = "q_0_9"

if q_low_col in series_data.columns and q_high_col in series_data.columns:
    fig.add_trace(go.Scatter(
        x=pd.concat([series_data["date"], series_data["date"][::-1]]),
        y=pd.concat([series_data[q_high_col], series_data[q_low_col][::-1]]),
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="Intervalo de Confianza (CP)"
    ))

# Plot Actual Demand (Total Horizon)
fig.add_trace(go.Scatter(
    x=series_data["date"], y=series_data["y_true"],
    mode='lines+markers', name="Demanda Semanal (Target)",
    line=dict(color='black', width=2)
))

# Plot Daily Observed Demand (Context for Imputation)
if "original_observed_demand" in series_data.columns:
    fig.add_trace(go.Scatter(
        x=series_data["date"], y=series_data["original_observed_demand"],
        mode='lines', name="Venta Diaria (Observada)",
        line=dict(color='grey', width=1, dash='dot')
    ))

# Plot Prediction
fig.add_trace(go.Scatter(
    x=series_data["date"], y=series_data["y_pred"],
    mode='lines', name="Predicción (Punto)",
    line=dict(color='blue', dash='dash')
))

# Plot Latent Demand Recovery (Only where imputed)
if "is_imputed" in series_data.columns:
    # Fill NA with False to avoid masking error if column exists but has nulls
    imputed_mask = series_data["is_imputed"].fillna(False).astype(bool)
    if imputed_mask.any():
        imputed_points = series_data[imputed_mask]
        fig.add_trace(go.Scatter(
            x=imputed_points["date"], y=imputed_points["latent_demand_est"],
            mode='markers', name="Demanda Latente Recuperada",
            marker=dict(symbol='star', size=12, color='purple', line=dict(width=1, color='white'))
        ))

# Plot Order Quantity
fig.add_trace(go.Scatter(
    x=series_data["date"], y=series_data["order_quantity"],
    mode='markers', name="Decisión de Inventario",
    marker=dict(symbol='diamond', size=10, color='red')
))

fig.update_layout(
    xaxis_title="Fecha",
    yaxis_title="Unidades de Demanda",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# --- Economic Sensitivity Plot ---
if sens is not None:
    st.subheader("🎯 Sensibilidad Económica (Modelo vs Baseline)")
    st.markdown("¿Cómo cambia el coste total si el ratio $C_{stockout}/C_{overstock}$ aumenta?")
    
    fig_sens = go.Figure()
    for model_name in sens["model_name"].unique():
        model_sens = sens[sens["model_name"] == model_name]
        fig_sens.add_trace(go.Scatter(
            x=model_sens["ratio"], 
            y=model_sens["total_cost"],
            mode='lines+markers',
            name=model_name
        ))
    
    fig_sens.update_layout(
        xaxis_title="Ratio de Coste Cs/Co",
        yaxis_title="Coste Total ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_sens, use_container_width=True)

# --- Cost Analysis ---
st.subheader("💰 Desglose Económico")
c1, c2 = st.columns(2)

with c1:
    st.write("Métricas por Modelo")
    st.dataframe(metrics, use_container_width=True)

with c2:
    st.write("Costes de Inventario")
    st.dataframe(costs, use_container_width=True)

st.markdown("---")
st.caption("Dashboard desarrollado para el TFG sobre Forecasting Probabilístico y Decisiones de Inventario.")
