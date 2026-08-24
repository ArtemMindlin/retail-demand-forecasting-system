from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from retail_forecasting.config import Settings, build_config_hash
from retail_forecasting.contracts.contracts_quality import EdaRunMetadata
from retail_forecasting.data.dataset import load_prepared_panel, panel_cache_filename
from retail_forecasting.eda.profiling import (
    build_correlation_summary,
    build_dataset_summary,
    build_missingness_summary,
    build_numeric_summary,
    render_profiling_figures,
)
from retail_forecasting.eda.series import build_series_summary, render_series_figures
from retail_forecasting.eda.stockout import (
    build_stockout_by_series_summary,
    build_stockout_demand_bands,
    build_stockout_summary,
    render_stockout_figures,
)
from retail_forecasting.eda.temporal import (
    build_series_gap_summary,
    build_temporal_summary,
    build_weekday_summary,
    render_temporal_figures,
)
from retail_forecasting.tracking import log_eda_run_to_mlflow
from retail_forecasting.utils.io import make_run_directory
from retail_forecasting.utils.logging import Table, fields, get_logger, rule, thousands
from retail_forecasting.utils.provenance import get_git_commit, utc_timestamp

logger = get_logger(__name__)


def _elapsed(seconds: float) -> str:
    """`45s` or `2m10s`, the precision a stage line needs."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def build_run_metadata(
    settings: Settings, panel: pd.DataFrame, split: str, config_path: Path | None
) -> EdaRunMetadata:
    """Describe the run: what it read, what the config asked for, and from which commit."""
    return EdaRunMetadata(
        split=split,
        panel_source=str(
            settings.dataset.processed_panel_dir / panel_cache_filename(settings.dataset, split)
        ),
        n_series=int(panel["series_id"].nunique()),
        rows=len(panel),
        date_min=str(panel["date"].min().date()),
        date_max=str(panel["date"].max().date()),
        configured_top_n_series=settings.dataset.top_n_series,
        configured_min_history_days=settings.dataset.min_history_days,
        configured_max_rows=settings.dataset.max_rows,
        imputation_strategy=settings.preprocessing.imputation_strategy,
        drop_negative_sales=settings.preprocessing.drop_negative_sales,
        fill_missing_values=settings.preprocessing.fill_missing_values,
        config_hash=build_config_hash(settings),
        config_path=None if config_path is None else str(config_path),
        created_at=utc_timestamp(),
        git_commit=get_git_commit(),
    )


def run_eda(settings: Settings, split: str = "train", config_path: Path | None = None) -> Path:
    """Read the panel, summarise it, draw it, and return the directory it all landed in."""
    if split not in settings.dataset.splits:
        raise ValueError(
            f"No hay un split llamado '{split}' en dataset.splits. "
            f"Disponibles: {', '.join(sorted(settings.dataset.splits))}."
        )

    rule(logger, "análisis exploratorio del panel")
    started = time.monotonic()
    panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split=split,
    )
    run_dir = make_run_directory(
        settings.reporting.output_dir, f"eda_{settings.reporting.run_name}"
    )
    fields(
        logger,
        {
            "panel": f"{thousands(panel['series_id'].nunique())} series, "
            f"{thousands(len(panel))} filas",
            "ventana": f"{panel['date'].min().date()} → {panel['date'].max().date()}",
            "split": split,
            "fuente": settings.dataset.processed_panel_dir
            / panel_cache_filename(settings.dataset, split),
            "carpeta": run_dir,
        },
    )

    stages = Table(logger, {"etapa": 18, "salidas": 7, "tiempo": 6})
    stages.header()

    def done(stage: str, count: int, since: float) -> float:
        stages.row({"etapa": stage, "salidas": count, "tiempo": _elapsed(time.monotonic() - since)})
        return time.monotonic()

    series_summary = build_series_summary(panel)
    weekday_summary = build_weekday_summary(panel)
    stockout_demand_bands = build_stockout_demand_bands(panel)

    summaries = {
        "dataset_summary.csv": build_dataset_summary(panel),
        "missingness_summary.csv": build_missingness_summary(panel),
        "numeric_summary.csv": build_numeric_summary(panel),
        "series_summary.csv": series_summary,
        "temporal_summary.csv": build_temporal_summary(panel),
        "weekday_summary.csv": weekday_summary,
        "series_gap_summary.csv": build_series_gap_summary(panel),
        "stockout_summary.csv": build_stockout_summary(panel),
        "stockout_by_series_summary.csv": build_stockout_by_series_summary(panel),
        "stockout_demand_bands.csv": stockout_demand_bands,
        "correlation_summary.csv": build_correlation_summary(panel),
    }
    for filename, frame in summaries.items():
        frame.to_csv(run_dir / filename, index=False)
    mark = done("resúmenes", len(summaries), started)

    drawn = 0
    for stage, render in (
        ("panel", lambda: render_profiling_figures(panel, run_dir)),
        ("series", lambda: render_series_figures(panel, series_summary, run_dir)),
        (
            "temporal",
            lambda: render_temporal_figures(panel, weekday_summary, series_summary, run_dir),
        ),
        ("stockout", lambda: render_stockout_figures(panel, stockout_demand_bands, run_dir)),
    ):
        render()
        total = len(list(run_dir.glob("*.png")))
        mark = done(f"figuras · {stage}", total - drawn, mark)
        drawn = total

    metadata = build_run_metadata(settings, panel, split, config_path)
    (run_dir / "eda_metadata.json").write_text(
        json.dumps(metadata.model_dump(), indent=2), encoding="utf-8"
    )

    try:
        log_eda_run_to_mlflow(metadata=metadata, run_dir=run_dir)
    except Exception as exc:  # noqa: BLE001 - see above
        logger.warning(
            "el registro en MLflow falló y la corrida no se ve afectada: %s: %s. "
            "Todo lo que produjo está en %s",
            type(exc).__name__,
            exc,
            run_dir,
        )

    fields(
        logger,
        {
            "total": f"{drawn} figuras y {len(summaries)} tablas "
            f"en {_elapsed(time.monotonic() - started)}"
        },
    )
    return run_dir
