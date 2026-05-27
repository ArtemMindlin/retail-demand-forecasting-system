from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from retail_forecasting.config import load_config
from retail_forecasting.data.dataset import load_prepared_panel


def generate_acf_demand(output_path: Path, max_lags: int = 28) -> None:
    settings = load_config("configs/default.yaml")
    eda_dataset_config = settings.dataset.model_copy(
        update={"top_n_series": None, "max_rows": None}
    )
    panel = load_prepared_panel(eda_dataset_config, settings.preprocessing, split="train")

    daily_mean = panel.groupby("date")["observed_demand"].mean().sort_index().to_numpy()
    n = len(daily_mean)

    mean = daily_mean.mean()
    centered = daily_mean - mean
    var = (centered**2).sum()

    acf_values = np.array(
        [(centered[: n - lag] * centered[lag:]).sum() / var for lag in range(max_lags + 1)]
    )
    confidence_bound = 1.96 / np.sqrt(n)
    lags = np.arange(max_lags + 1)
    seasonal_lags = {7, 14, 21, 28}

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhspan(-confidence_bound, confidence_bound, color="#1f77b4", alpha=0.12, label="95% CI")

    for lag, val in zip(lags, acf_values, strict=False):
        color = "#d62728" if lag in seasonal_lags else "#1f77b4"
        ax.vlines(lag, 0, val, colors=color, linewidth=1.8)
        ax.plot(lag, val, "o", color=color, markersize=4)

    for s_lag in seasonal_lags:
        if s_lag <= max_lags:
            ax.axvline(float(s_lag), color="#d62728", linewidth=0.7, linestyle="--", alpha=0.45)

    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("ACF of daily aggregate demand (lags 0–28)")
    ax.set_xticks(lags)
    ax.legend(loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"ACF plot saved to {output_path}")


if __name__ == "__main__":
    out = Path("memoria/figures/eda/acf_demand.png")
    generate_acf_demand(out)
