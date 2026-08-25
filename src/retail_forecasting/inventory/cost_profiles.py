from __future__ import annotations

import pandas as pd

from retail_forecasting.config import InventoryConfig


def attach_series_costs(
    predictions: pd.DataFrame,
    inventory_config: InventoryConfig,
) -> pd.DataFrame:
    """Attach the inventory cost coefficients and the critical fractile to a prediction frame.

    One `c_over`/`c_under` pair for the whole catalogue, straight from the config.

    A synthetic PER-SERIES profile used to live here, deriving both coefficients from proxies
    of each series' sales shape. It is gone: it produced a `total_cost` in an invented
    currency, and that total was the champion-selection criterion. Measured before removing
    it, the differentiation cut cost 2.05% on reconstructed demand and RAISED it 3.66% on
    censored demand -- under either accounting -- so it was not reliably worth its nine
    uncalibrated constants. See invariant 43.
    """
    enriched = predictions.copy()
    enriched["c_over"] = inventory_config.overstock_cost
    enriched["c_under"] = inventory_config.stockout_cost
    enriched["critical_fractile"] = inventory_config.stockout_cost / (
        inventory_config.stockout_cost + inventory_config.overstock_cost
    )
    return enriched
