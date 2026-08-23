"""Which of the generated EDA figures the thesis includes.

The list is what `export_figures_to_memoria` copies into `memoria/figures/eda/`, and it is
pinned against the figures chapter 3 actually includes by
`test_every_figure_the_thesis_includes_is_exported_by_the_eda_run`. A figure the run draws but
that is absent here stays in `reports/` and never reaches the thesis.

Captions and labels used to live here too, feeding a generated `eda_figures.tex`. Nothing ever
included that fragment -- chapter 3 writes its own figure environments by hand -- so eleven
captions and eleven interpretation paragraphs were maintained for a file no LaTeX run read.
"""

from __future__ import annotations

MEMORIA_FIGURE_EXPORTS: tuple[str, ...] = (
    "observed_demand_distribution.png",
    "observed_demand_boxplot_top_series.png",
    "coverage_heatmap.png",
    "weekday_demand_profile.png",
    "category_seasonality_high.png",
    "category_seasonality_medium.png",
    "category_seasonality_low.png",
    "acf_demand.png",
    "zero_demand_rate_by_series.png",
    "stockout_hours_distribution.png",
    "stockout_band_demand.png",
    "stockout_vs_demand_scatter.png",
    "correlation_heatmap.png",
    "covariate_vs_demand_grid.png",
    "representative_series_panels.png",
)
