from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil

import pandas as pd

from retail_forecasting.utils.io import (
    dataframe_to_markdown,
    ensure_directory,
    make_run_directory,
)

MEMORIA_FIGURE_EXPORTS = [
    {
        "filename": "coverage_heatmap.png",
        "caption": "Cobertura temporal del panel por serie y fecha.",
        "label": "fig:eda_coverage_heatmap",
    },
    {
        "filename": "weekday_demand_profile.png",
        "caption": "Perfil semanal de demanda observada con media y mediana.",
        "label": "fig:eda_weekday_profile",
    },
    {
        "filename": "zero_demand_rate_by_series.png",
        "caption": "Series mas intermitentes segun su proporcion de demanda cero.",
        "label": "fig:eda_zero_demand_rate",
    },
    {
        "filename": "stockout_hours_distribution.png",
        "caption": "Distribucion de horas de stockout en el panel preparado.",
        "label": "fig:eda_stockout_distribution",
    },
    {
        "filename": "stockout_band_demand.png",
        "caption": "Demanda media y numero de observaciones por banda de stockout.",
        "label": "fig:eda_stockout_band_demand",
    },
    {
        "filename": "stockout_vs_demand_scatter.png",
        "caption": "Relacion entre horas de stockout y demanda observada.",
        "label": "fig:eda_stockout_vs_demand",
    },
    {
        "filename": "correlation_heatmap.png",
        "caption": "Mapa de correlaciones entre variables numericas del panel preparado.",
        "label": "fig:eda_correlation_heatmap",
    },
    {
        "filename": "covariate_vs_demand_grid.png",
        "caption": "Relaciones muestreadas entre covariables exogenas y demanda observada.",
        "label": "fig:eda_covariates_vs_demand",
    },
    {
        "filename": "representative_series_panels.png",
        "caption": "Series representativas con demanda observada y overlay de stockout.",
        "label": "fig:eda_representative_series",
    },
]


@dataclass
class EdaArtifacts:
    panel: pd.DataFrame
    dataset_summary: pd.DataFrame
    config_alignment_summary: pd.DataFrame
    missingness_summary: pd.DataFrame
    numeric_summary: pd.DataFrame
    series_summary: pd.DataFrame
    temporal_summary: pd.DataFrame
    weekday_summary: pd.DataFrame
    series_gap_summary: pd.DataFrame
    stockout_summary: pd.DataFrame
    stockout_by_series_summary: pd.DataFrame
    stockout_demand_bands: pd.DataFrame
    correlation_summary: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    run_directory: Path | None = None


def write_eda_artifacts(
    artifacts: EdaArtifacts,
    output_dir: str | Path,
    run_name: str,
    make_plots: bool,
    render_plots: callable | None = None,
    memoria_dir: str | Path | None = None,
) -> EdaArtifacts:
    """Persist EDA summaries, plots, and a Markdown report."""
    run_dir = make_run_directory(output_dir, run_name)
    ensure_directory(run_dir)

    outputs = {
        "dataset_summary.csv": artifacts.dataset_summary,
        "config_alignment_summary.csv": artifacts.config_alignment_summary,
        "missingness_summary.csv": artifacts.missingness_summary,
        "numeric_summary.csv": artifacts.numeric_summary,
        "series_summary.csv": artifacts.series_summary,
        "temporal_summary.csv": artifacts.temporal_summary,
        "weekday_summary.csv": artifacts.weekday_summary,
        "series_gap_summary.csv": artifacts.series_gap_summary,
        "stockout_summary.csv": artifacts.stockout_summary,
        "stockout_by_series_summary.csv": artifacts.stockout_by_series_summary,
        "stockout_demand_bands.csv": artifacts.stockout_demand_bands,
        "correlation_summary.csv": artifacts.correlation_summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(run_dir / filename, index=False)

    if make_plots and render_plots is not None:
        render_plots(
            panel=artifacts.panel,
            weekday_summary=artifacts.weekday_summary,
            series_summary=artifacts.series_summary,
            output_dir=run_dir,
        )

    report_text = build_eda_report(artifacts)
    (run_dir / "eda_report.md").write_text(report_text, encoding="utf-8")

    if memoria_dir is not None:
        export_figures_to_memoria(
            run_directory=run_dir,
            memoria_dir=memoria_dir,
        )

    artifacts.run_directory = run_dir
    return artifacts


def build_eda_report(artifacts: EdaArtifacts) -> str:
    """Render the Markdown report for an EDA run."""
    report = [
        "# Exploratory Data Analysis Report",
        "",
        "## Dataset Summary",
        "",
        dataframe_to_markdown(artifacts.dataset_summary),
        "",
        "## Configuration Alignment",
        "",
        dataframe_to_markdown(artifacts.config_alignment_summary),
        "",
        "## Alerts",
        "",
        *(
            [f"- **ALERT**: {warning}" for warning in artifacts.warnings]
            if artifacts.warnings
            else ["- No configuration-alignment issues detected."]
        ),
        "",
        "## Temporal Coverage",
        "",
        dataframe_to_markdown(artifacts.temporal_summary),
        "",
        "## Missingness Summary",
        "",
        dataframe_to_markdown(artifacts.missingness_summary.head(12)),
        "",
        "## Weekly Seasonality",
        "",
        dataframe_to_markdown(artifacts.weekday_summary),
        "",
        "## Stockout Summary",
        "",
        dataframe_to_markdown(artifacts.stockout_summary),
        "",
        "## Demand by Stockout Band",
        "",
        dataframe_to_markdown(artifacts.stockout_demand_bands),
        "",
        "## Top Series by Observed Demand",
        "",
        dataframe_to_markdown(
            artifacts.series_summary.head(10),
            columns=[
                "series_id",
                "history_days",
                "observed_demand_sum",
                "observed_demand_mean",
                "stockout_day_rate",
            ],
        ),
        "",
        "## Correlation Summary",
        "",
        dataframe_to_markdown(artifacts.correlation_summary.head(12)),
        "",
        "## Generated Figures",
        "",
        "- `coverage_heatmap.png`: continuity and date coverage by series.",
        "- `observed_demand_distribution.png`: full and positive-demand histograms.",
        "- `observed_demand_boxplot_top_series.png`: demand dispersion for the top-demand series.",
        "- `weekday_demand_profile.png`: mean and median demand by weekday.",
        "- `zero_demand_rate_by_series.png`: most intermittent series.",
        "- `stockout_hours_distribution.png`: full and positive stockout-hour distributions.",
        "- `stockout_band_demand.png`: demand and count by stockout intensity band.",
        "- `stockout_vs_demand_scatter.png`: raw relation and mean trend for stockout hours.",
        "- `correlation_heatmap.png`: pairwise numeric correlation structure.",
        "- `covariate_vs_demand_grid.png`: sampled relations for key exogenous variables.",
        "- `top_series_total_demand.png`: aggregate ranking of the highest-volume series.",
        "- `representative_series_panels.png`: small multiples of demand with stockout overlay.",
        "",
        "## Interpretation Notes",
        "",
        "- The EDA is computed on the canonical prepared panel, not on raw column names.",
        "- Demand and stockout summaries are descriptive only and do not alter target semantics.",
        "- Weekly and stockout diagnostics are intended to guide feature and experiment design.",
    ]
    return "\n".join(report)


def export_figures_to_memoria(
    run_directory: str | Path,
    memoria_dir: str | Path,
) -> None:
    """Copy selected EDA figures into the memoria tree and emit a LaTeX fragment."""
    run_dir = Path(run_directory)
    memoria_root = Path(memoria_dir)
    target_dir = ensure_directory(memoria_root / "figures" / "eda")

    for figure in MEMORIA_FIGURE_EXPORTS:
        source = run_dir / figure["filename"]
        if source.exists():
            shutil.copy2(source, target_dir / figure["filename"])

    latex_fragment = build_memoria_eda_figures_tex()
    (target_dir / "eda_figures.tex").write_text(latex_fragment, encoding="utf-8")


def build_memoria_eda_figures_tex() -> str:
    """Build the generated LaTeX fragment for the EDA chapter."""
    blocks: list[str] = []
    for figure in MEMORIA_FIGURE_EXPORTS:
        blocks.extend(
            [
                r"\begin{figure}[htbp]",
                r"    \centering",
                f"    \\includegraphics[width=0.95\\textwidth]{{figures/eda/{figure['filename']}}}",
                f"    \\caption{{{figure['caption']}}}",
                f"    \\label{{{figure['label']}}}",
                r"\end{figure}",
                "",
            ]
        )

    return "\n".join(blocks).strip() + "\n"
