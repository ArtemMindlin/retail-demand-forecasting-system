from __future__ import annotations

from retail_forecasting.config import Settings

# We will import RunArtifacts inside functions if needed to avoid circular imports,
# or just type hint it with Any or 'RunArtifacts' if we use TYPE_CHECKING.
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def generate_post_mortem_report(artifacts: Any, settings: Settings) -> str:
    """
    Identifies the Top 5 most problematic SKUs based on their simulated total cost
    and attempts to diagnose the root cause (Drift, High Intermittency, etc.).
    """
    preds = artifacts.predictions
    if preds.empty or "sim_total_cost" not in preds.columns:
        return "_No post-mortem analysis available (requires dynamic simulation)._"

    # We focus on the worst performing SKUs across all models, or specifically the champion/selected model.
    # Let's aggregate by series_id across the whole run to find the worst offenders.
    sku_costs = (
        preds.groupby("series_id")["sim_total_cost"].sum().sort_values(ascending=False)
    )

    if sku_costs.empty:
        return "_No cost data to analyze._"

    top_5_skus = sku_costs.head(5).index.tolist()

    report_lines = []

    for sku in top_5_skus:
        sku_data = preds[preds["series_id"] == sku]
        total_cost = sku_costs[sku]

        # Diagnostics
        diagnostics = []

        # 1. Drift Check
        # Find folds where this SKU was evaluated
        sku_folds = set(sku_data["fold_id"].unique())
        drift_folds = {event.fold_id for event in artifacts.drifts}
        if sku_folds.intersection(drift_folds):
            diagnostics.append(
                "📉 **Concept Drift:** El modelo se vio afectado por un cambio brusco de tendencia detectado en uno de los folds (Page-Hinkley event)."
            )

        # 2. Intermittency Check
        zero_demand_pct = (sku_data["y_true"] == 0).mean()
        if zero_demand_pct > 0.6:
            diagnostics.append(
                f"🛑 **Alta Intermitencia:** El {zero_demand_pct*100:.0f}% de los periodos tuvieron demanda cero. El modelo sobre-pronostica sistemáticamente."
            )

        # 3. High Error (Low predictability)
        # Check if MAE is very high compared to mean demand
        mean_demand = sku_data["y_true"].mean()
        mae = (sku_data["y_true"] - sku_data["y_pred"]).abs().mean()
        if mean_demand > 0 and (mae / mean_demand) > 0.8:
            diagnostics.append(
                f"⚠️ **Baja Predictibilidad:** El MAE ({mae:.1f}) es más del 80% de la demanda media ({mean_demand:.1f}). La serie presenta un ruido intrínseco muy alto."
            )

        # 4. Stockout Risk
        if "sim_service_level_hit" in sku_data.columns:
            service_level = sku_data["sim_service_level_hit"].mean()
            if service_level < 0.8:
                diagnostics.append(
                    f"📦 **Riesgo Crítico de Rotura:** El nivel de servicio dinámico cayó al {service_level*100:.1f}%. El modelo es demasiado conservador para la varianza real."
                )

        if not diagnostics:
            diagnostics.append(
                "🔍 **Desviación Estándar:** El coste elevado se debe a la acumulación normal de errores menores, sin una anomalía estructural detectada."
            )

        report_lines.append(f"### SKU: `{sku}`")
        report_lines.append(
            f"- **Impacto Económico:** ${total_cost:,.2f} en sobrecostes logísticos."
        )
        for diag in diagnostics:
            report_lines.append(f"- {diag}")
        report_lines.append("")

    return "\n".join(report_lines)
