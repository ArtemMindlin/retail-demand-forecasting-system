from __future__ import annotations

from retail_forecasting.config import FeatureConfig
from retail_forecasting.features.engineering import build_supervised_frame
from tests import make_synthetic_panel


def test_build_supervised_frame_uses_only_past_features() -> None:
    panel = make_synthetic_panel(num_series=1, num_days=20)
    feature_config = FeatureConfig(lags=[1, 2], rolling_windows=[3])

    supervised, feature_columns = build_supervised_frame(panel, feature_config, horizon=3)
    assert "demand_lag_1" in feature_columns
    assert "target_lead_time_demand" in supervised.columns

    row = supervised.iloc[0]
    source = panel.loc[panel["series_id"] == row["series_id"]].sort_values("date").reset_index(drop=True)
    source_index = source.index[source["date"] == row["date"]][0]

    expected_lag_1 = source.loc[source_index - 1, "observed_demand"]
    expected_target = source.loc[source_index : source_index + 2, "observed_demand"].sum()

    assert row["demand_lag_1"] == expected_lag_1
    assert row["target_lead_time_demand"] == expected_target
