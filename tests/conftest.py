from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_panel(num_series: int = 3, num_days: int = 70) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=num_days, freq="D")
    rows = []

    for store_id in range(1, num_series + 1):
        product_id = 100 + store_id
        for index, date in enumerate(dates):
            base = 10 + store_id
            seasonal = 2.0 * np.sin(2 * np.pi * index / 7)
            demand = max(base + seasonal + (index % 5) * 0.5, 0.1)
            rows.append(
                {
                    "city_id": 1,
                    "store_id": store_id,
                    "management_group_id": 1,
                    "first_category_id": 1,
                    "second_category_id": 1,
                    "third_category_id": store_id,
                    "product_id": product_id,
                    "date": date,
                    "observed_demand": demand,
                    "stockout_hours": float(index % 3),
                    "discount": 1.0 - 0.05 * (index % 2),
                    "holiday_flag": int(date.dayofweek == 6),
                    "activity_flag": int(index % 10 == 0),
                    "precpt": float(index % 4),
                    "avg_temperature": 15.0 + store_id,
                    "avg_humidity": 50.0 + index % 10,
                    "avg_wind_level": 3.0 + (index % 3),
                    "series_id": f"{store_id}_{product_id}",
                }
            )

    return pd.DataFrame(rows)
