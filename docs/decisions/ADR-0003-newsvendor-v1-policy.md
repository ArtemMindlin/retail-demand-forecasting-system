# ADR-0003: Use Single-Period Newsvendor as the v1 Inventory Policy

## Status

Accepted.

## Context

The project is not only forecasting demand; it evaluates whether forecasts induce better inventory decisions. A full multi-period inventory simulation would require lead-time dynamics, replenishment state, service constraints, and stronger assumptions than the v1 currently models.

The v1 needs a simple policy that turns point or quantile forecasts into an order quantity and exposes the economic trade-off between overstock and stockout.

## Decision

The v1 inventory layer uses a single-period newsvendor policy.

When quantile forecasts are available, the order quantity is selected around the critical fractile:

```text
critical_fractile = stockout_cost / (stockout_cost + overstock_cost)
```

When only a point forecast is available, the point forecast is used as the order quantity.

## Consequences

- Forecast quality can be evaluated by operational cost, not just MAE/RMSE.
- Cost assumptions are explicit and configurable.
- The policy does not model inventory carryover, lead-time variability, or multi-period replenishment.

## Harness Rules

- Inventory decision logic lives in `src/retail_forecasting/inventory/`.
- Models must not compute `order_quantity` or cost columns.
- Evaluation summarizes costs but must not change order quantities.
