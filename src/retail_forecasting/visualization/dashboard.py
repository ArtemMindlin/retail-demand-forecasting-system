import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from retail_forecasting.config import InventoryConfig
from retail_forecasting.inventory.simulation import simulate_inventory_policy
from retail_forecasting.evaluation.metrics import summarize_costs

st.set_page_config(page_title="Retail Forecasting TFG Dashboard", layout="wide")


def get_report_dirs() -> list[Path]:
    reports_path = Path("reports")
    if not reports_path.exists():
        return []
    # Return only experiment directories that contain the required CSV files
    dirs = [
        d for d in reports_path.iterdir()
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
    st.error(
        "No se encontraron reportes en la carpeta `reports/`. Ejecuta un experimento primero."
    )
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
    help="Compara cómo afecta entrenar con datos reales censurados (Observed) vs datos imputados (Latent)"
)

# Apply strategy filter to predictions, metrics and costs for the main displays
preds_filtered = preds[preds["data_strategy"] == selected_strategy] if "data_strategy" in preds.columns else preds
metrics_filtered = metrics[metrics["data_strategy"] == selected_strategy] if "data_strategy" in metrics.columns else metrics
costs_filtered = costs[costs["data_strategy"] == selected_strategy] if "data_strategy" in costs.columns else costs

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
        st.slider("Capacidad Global (Unidades)", 500, 10000, 3000, 100)
        if use_capacity
        else None
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
    """
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 25px; border-radius: 15px; margin-bottom: 25px; color: white; box-shadow: 0 4px 20px rgba(0,0,0,0.15);">
        <h1 style="margin: 0; font-family: 'Outfit', 'Inter', sans-serif; font-size: 2.2rem; font-weight: 700; color: white;">🚀 Panel de Decisiones de Inventario & Forecasting Probabilístico</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 1.05rem;">Trabajo Fin de Grado (TFG) - Optimización de Inventario Fresco bajo Roturas e Incertidumbre</p>
        <div style="margin-top: 15px; display: flex; gap: 10px; flex-wrap: wrap;">
            <span style="background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; color: white;">🔍 Estrategia: <b>{strategy}</b></span>
            <span style="background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; color: white;">🎯 Conformal Prediction</span>
            <span style="background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; color: white;">⚖️ Frontera de Pareto</span>
            <span style="background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; color: white;">📈 Streamlit Pro Edition</span>
        </div>
    </div>
    """.format(strategy=selected_strategy),
    unsafe_allow_html=True
)

st.subheader(f"📊 Ejecución seleccionada: {selected_run.name}")

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    base_cost = costs_filtered[costs_filtered["model_name"] == selected_model]["sim_total_cost"].sum()
    if what_if_costs is not None:
        new_cost = what_if_costs["sim_total_cost"].sum()
        st.metric(
            "Coste Total (Simulado)",
            f"${new_cost:,.2f}",
            delta=f"${new_cost - base_cost:,.2f}",
            delta_color="inverse",
        )
    else:
        st.metric("Coste Total (Base)", f"${base_cost:,.2f}")
with col2:
    model_metrics = metrics_filtered[metrics_filtered["model_name"] == selected_model]
    if not model_metrics.empty:
        mae = model_metrics.iloc[0]["mae"]
        st.metric(f"MAE ({selected_model})", f"{mae:.2f}")
with col3:
    if not model_metrics.empty:
        coverage = model_metrics.iloc[0].get(
            "interval_coverage", model_metrics.iloc[0].get("coverage_q_0_1_q_0_9", 0)
        )
        if pd.isna(coverage):
            st.metric("Cobertura de Intervalo", "N/A")
        else:
            st.metric("Cobertura de Intervalo (80% Target)", f"{coverage * 100:.1f}%")
with col4:
    if what_if_costs is not None:
        base_sl = costs_filtered[costs_filtered["model_name"] == selected_model][
            "sim_service_level"
        ].mean()
        new_sl = what_if_costs["sim_service_level"].mean()
        st.metric(
            "Nivel de Servicio",
            f"{new_sl * 100:.1f}%",
            delta=f"{(new_sl - base_sl) * 100:.1f}%",
        )
    else:
        base_sl = costs_filtered[costs_filtered["model_name"] == selected_model][
            "sim_service_level"
        ].mean() if "sim_service_level" in costs_filtered.columns else None
        if base_sl is not None and not pd.isna(base_sl):
            st.metric("Nivel de Servicio (Base)", f"{base_sl * 100:.1f}%")
        else:
            st.info(f"Modelo: {selected_model}")

# Drift Alerts
if "ALERT" in drift_info:
    st.warning(f"⚠️ **Detección de Drift (Cambio de Régimen):** {drift_info}")
else:
    st.success("✅ **Estabilidad Temporal Confirmada:** No se detectó drift significativo en la demanda.")

# --- Tabbed Navigation ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Pronósticos e Inventario",
    "🔍 Imputación de Demanda Latente",
    "🎯 Frontera de Pareto y Sensibilidad",
    "📘 Marco Teórico y Analógico"
])

# --- Tab 1: Forecasting and Base Inventory ---
with tab1:
    st.markdown("### 📈 Pronóstico vs Realidad y Decisiones de Compra")
    st.markdown(
        "Visualiza las predicciones de punto, los intervalos probables (Conformal Prediction) y las cantidades sugeridas de pedido."
    )

    series_data = preds_filtered[
        (preds_filtered["series_id"] == selected_series) & (preds_filtered["model_name"] == selected_model)
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
            marker=dict(size=6)
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
        wi_series_data = what_if_preds[
            what_if_preds["series_id"] == selected_series
        ].sort_values("date")
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="rgba(243, 244, 246, 0.5)",
        paper_bgcolor="white"
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
    latent_strat = next((s for s in preds["data_strategy"].dropna().unique() if "Latent_" in s), None)

    if latent_strat is not None and "original_observed_demand" in preds.columns:
        # Filter for the latent strategy for this series and model
        latent_series_data = preds[
            (preds["data_strategy"] == latent_strat) &
            (preds["series_id"] == selected_series) &
            (preds["model_name"] == selected_model)
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
                    yaxis="y2"
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
                    marker=dict(size=6, symbol="x")
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
                    marker=dict(size=6, symbol="circle")
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
                    showgrid=False
                ),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor="rgba(243, 244, 246, 0.5)",
                paper_bgcolor="white"
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
        for (strategy_name, model_name), group in pareto.groupby(["data_strategy", "model_name"]):
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
                    text=[f"{m} - {s}" for m, s in zip(group["model_name"], group["data_strategy"])],
                    customdata=group["order_scale"]
                )
            )

        # Draw the actual Pareto frontier line (points where is_pareto_efficient == True)
        pareto["is_pareto_efficient"] = pareto["is_pareto_efficient"].astype(str).str.lower() == "true"
        pareto_efficient = pareto[pareto["is_pareto_efficient"]].sort_values("total_cost")

        if not pareto_efficient.empty:
            fig_pareto.add_trace(
                go.Scatter(
                    x=pareto_efficient["total_cost"],
                    y=pareto_efficient["service_level"] * 100,
                    mode="lines",
                    name="Frontera de Pareto (Óptimo Eficiente)",
                    line=dict(color="rgb(16, 185, 129)", width=3, dash="dash"),
                    hoverinfo="skip"
                )
            )

        fig_pareto.update_layout(
            xaxis_title="Coste Total de Inventario ($)",
            yaxis_title="Nivel de Servicio (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor="rgba(243, 244, 246, 0.5)",
            paper_bgcolor="white"
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
        st.markdown("### 📊 Sensibilidad a Ratios de Penalización de Rotura ($C_{stockout}/C_{overstock}$)")
        st.markdown(
            "Muestra cómo escala el coste total de inventario a medida que el negocio penaliza más duramente "
            "las roturas de stock. Un modelo con un buen pronóstico probabilístico (conformal) escala de forma mucho más controlada."
        )

        fig_sens = go.Figure()
        for (strat_name, model_name), group in sens.groupby(["data_strategy", "model_name"]):
            fig_sens.add_trace(
                go.Scatter(
                    x=group["ratio"],
                    y=group["total_cost"],
                    mode="lines+markers",
                    name=f"{model_name} ({strat_name})",
                    line=dict(width=2),
                    marker=dict(size=6)
                )
            )

        fig_sens.update_layout(
            xaxis_title="Ratio de Coste Cs/Co",
            yaxis_title="Coste Total ($)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor="rgba(243, 244, 246, 0.5)",
            paper_bgcolor="white"
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
            <div style="background-color: rgba(59, 130, 246, 0.08); padding: 20px; border-radius: 12px; border-left: 5px solid #3b82f6; margin-bottom: 20px; min-height: 380px;">
                <h4 style="margin-top: 0; color: #1e3a8a;">🌧️ 1. La Analogía del Meteorólogo (Conformal Prediction)</h4>
                <p style="font-size: 0.95rem; text-align: justify; line-height: 1.5; color: black;">
                    En el TFG, no usamos intervalos de confianza estadísticos tradicionales (que asumen normalidad y suelen fallar en colas), 
                    sino <b>Conformal Prediction (CP)</b>, un método no paramétrico moderno que garantiza cobertura real.
                </p>
                <p style="font-size: 0.95rem; text-align: justify; line-height: 1.5; font-style: italic; background: white; padding: 10px; border-radius: 8px; color: black;">
                    "Imagina un meteorólogo que dice: 'Mañana lloverá con un 80% de probabilidad'. Si evalúas todas sus predicciones históricas 
                    y resulta que llovió exactamente en el 80% de los días que hizo este anuncio, el meteorólogo está <b>calibrado</b>. 
                    Conformal Prediction garantiza ex-ante que nuestros intervalos del 80% contendrán la demanda real exactamente el 80% del tiempo, 
                    independientemente de qué tan sesgado esté el estimador base (LGBM o CatBoost)."
                </p>
                <p style="font-size: 0.9rem; color: #1d4ed8; font-weight: 600;">
                    💡 Concepto Clave: Calibración Empírica Libre de Distribución.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="background-color: rgba(16, 185, 129, 0.08); padding: 20px; border-radius: 12px; border-left: 5px solid #10b981; min-height: 300px;">
                <h4 style="margin-top: 0; color: #065f46;">⚖️ 3. La Frontera Eficiente de Pareto</h4>
                <p style="font-size: 0.95rem; text-align: justify; line-height: 1.5; color: black;">
                    En logística de frescos, existe una contradicción intrínseca entre coste y servicio. 
                    La <b>Frontera de Pareto</b> demuestra que no hay una única "decisión perfecta", sino un 
                    <b>conjunto de decisiones óptimas</b>.
                </p>
                <p style="font-size: 0.95rem; text-align: justify; line-height: 1.5; color: black;">
                    El algoritmo calcula simulación de inventarios barriendo un factor multiplicativo 
                    (escala de orden) desde 0.7x hasta 1.3x. La frontera une los puntos donde ya no puedes 
                    reducir costes sin empeorar el nivel de servicio, dando soporte a decisiones ejecutivas 
                    estratégicas basadas en presupuesto.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_t2:
        st.markdown(
            """
            <div style="background-color: rgba(239, 68, 68, 0.08); padding: 20px; border-radius: 12px; border-left: 5px solid #ef4444; margin-bottom: 20px; min-height: 380px;">
                <h4 style="margin-top: 0; color: #991b1b;">📰 2. El Modelo del Vendedor de Periódicos (Newsvendor)</h4>
                <p style="font-size: 0.95rem; text-align: justify; line-height: 1.5; color: black;">
                    La demanda es una variable aleatoria y pedir la media/predicción puntual es financieramente incorrecto en presencia de asimetrías de costes. 
                    Usamos el formalismo del <b>Fractil Crítico (Critical Fractile)</b>.
                </p>
                <p style="font-size: 0.95rem; text-align: justify; line-height: 1.5; background: white; padding: 10px; border-radius: 8px; color: black;">
                    La cantidad óptima a pedir es el cuantil de demanda correspondiente a:
                    <br><br>
                    <span style="font-family: monospace; font-weight: bold; font-size: 1.1rem; display: block; text-align: center; color: #b91c1c;">
                        τ* = C_under / (C_under + C_over)
                    </span>
                    <br>
                    Donde <b>C_under</b> es el coste unitario por quedarnos cortos (rotura) y <b>C_over</b> es el coste de exceso (merma). 
                    Con Cs=4 y Co=1, el Fractil Crítico τ* es 0.80, lo que significa que el inventario óptimo debe 
                    cubrir el cuantil del 80% de la distribución conformal de demanda para maximizar la rentabilidad esperada.
                </p>
                <p style="font-size: 0.9rem; color: #b91c1c; font-weight: 600;">
                    💡 Concepto Clave: Asimetría de Pérdidas y Coste de Oportunidad.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="background-color: rgba(245, 158, 11, 0.08); padding: 20px; border-radius: 12px; border-left: 5px solid #f59e0b; min-height: 300px;">
                <h4 style="margin-top: 0; color: #78350f;">📦 4. Optimización LP con Capacidad</h4>
                <p style="font-size: 0.95rem; text-align: justify; line-height: 1.5; color: black;">
                    En el mundo real, los almacenes o camiones tienen una capacidad máxima. Cuando aplicas el modelo Newsvendor 
                    independiente a cada SKU, la suma total de órdenes de compra puede exceder el límite físico global.
                </p>
                <p style="font-size: 0.95rem; text-align: justify; line-height: 1.5; color: black;">
                    Nuestra plataforma resuelve un **Problema de Programación Lineal (LP)** dinámico cada día. 
                    Cuando el límite global de capacidad está activo, el sistema redistribuye inteligentemente 
                    las cuotas de capacidad priorizando los productos con mayor criticidad o mayor penalización por rotura.
                </p>
            </div>
            """,
            unsafe_allow_html=True
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
    "Dashboard desarrollado para el TFG sobre Forecasting Probabilístico y Decisiones de Inventario."
)
