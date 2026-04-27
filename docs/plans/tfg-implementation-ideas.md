# TFG Implementation Ideas

Date: 2026-04-21

## Purpose

This document is the living backlog for implementation ideas that could raise the technical and academic level of the TFG.

It is intentionally not a generic wish list. Each idea should be scoped enough to decide whether it belongs in the core TFG, an extension, or future work.

## Positioning

The current project is a forecast-to-decision pipeline for retail demand forecasting. It already covers data loading, temporal feature engineering, walk-forward validation, baseline and boosting models, quantile outputs, a single-period newsvendor decision layer, economic metrics, and reporting.

Recent related work, especially the VN2 winner report "One Global Model, Many Behaviors", overlaps strongly with the broad direction of global retail forecasting plus cost-aware inventory decisions. Therefore, new implementation work should avoid simply reproducing that setting. The strongest differentiation is to focus on:

- probabilistic and calibrated uncertainty;
- economic sensitivity analysis;
- stockouts as censored demand;
- robust temporal evaluation;
- clear comparison between predictive and decision-aware rankings.

## Prioritization Rules

- Prefer additions that make the TFG more defensible, not just larger.
- Prefer experiments that answer a clear research question.
- Avoid adding models unless they test a specific hypothesis.
- Keep the v1 pipeline stable before adding operational complexity.
- Treat multi-agent or reinforcement learning approaches as future work unless the whole TFG is deliberately re-scoped.

## High Priority Ideas

### 1. Economic Sensitivity Analysis

Status: proposed

Goal: evaluate whether model rankings change when the business cost ratio changes.

Motivation: the current newsvendor decision depends directly on the ratio between stockout cost and overstock cost. A model that is best when stockouts are expensive may not be best when overstock is expensive.

Suggested scope:

- Run the same backtest under several cost ratios, for example `1:1`, `2:1`, `4:1`, `8:1`, and `10:1`.
- Report total cost, overstock units, stockout units, and mean order quantity per model and scenario.
- Add a compact plot or table showing ranking changes by scenario.

Expected contribution:

- Strong experimental rigor.
- Clear decision-oriented narrative.
- Low implementation risk.

Evidence needed:

- Cost tables by scenario.
- Ranking comparison against MAE/RMSE ranking.
- Short interpretation of when economic and predictive rankings disagree.

Risks:

- If all rankings are identical, the result is still useful but less striking. In that case, focus on stability and explain why.

### 2. Probabilistic Calibration With Conformal Prediction

Status: proposed

Goal: improve the reliability of prediction intervals or quantiles before using them for inventory decisions.

Motivation: the current project already produces quantile forecasts and reports pinball loss and coverage. Existing results suggest interval coverage may be weak, which makes calibration a natural and defensible extension.

Suggested scope:

- Start with a split-conformal or rolling conformal method compatible with temporal validation.
- Calibrate prediction intervals on past validation-like data only.
- Compare raw quantiles against calibrated intervals.
- Measure coverage, interval width, pinball loss where applicable, and downstream inventory cost.

Expected contribution:

- Strong methodological differentiation from winner-report style systems focused mainly on point forecasts and heuristic buffers.
- Better connection between uncertainty estimation and decision quality.

Evidence needed:

- Raw vs calibrated coverage.
- Raw vs calibrated cost under multiple cost ratios.
- Confirmation that calibration does not use future information.

Risks:

- Temporal conformal methods can introduce leakage if implemented carelessly.
- Better coverage may increase overstock cost if intervals become too conservative.

### 3. Stockout Treatment as Censored Demand

Status: proposed

Goal: compare observed-demand modeling against simple stockout-aware corrections.

Motivation: observed sales are not always true demand when stockouts occur. The current v1 explicitly models observed demand and uses stockout lags as features, but it does not recover latent demand.

Suggested scope:

- Baseline treatment: use observed demand as currently implemented.
- Masking treatment: exclude or downweight observations with high stockout intensity when training.
- Simple imputation treatment: replace high-stockout demand observations using a conservative historical estimate, such as rolling median or same-series recent non-stockout demand.
- Evaluate predictive metrics and inventory cost under the same backtesting protocol.

Expected contribution:

- Directly addresses one of the core methodological weaknesses.
- Makes the stockout discussion empirical rather than only conceptual.

Evidence needed:

- Performance by stockout regime.
- Cost impact of each treatment.
- Clear statement that imputed demand is an estimate, not ground truth.

Risks:

- Imputation may inject bias.
- Aggressive masking may reduce training data too much.

### 4. Predictive Ranking vs Economic Ranking

Status: proposed

Goal: formalize whether the best predictive model is also the best decision model.

Motivation: current reports already show a possible disagreement: the seasonal naive baseline can have better MAE/RMSE while the boosting model can be close or slightly better in total cost depending on backend.

Suggested scope:

- Build a report section that compares rankings by MAE, RMSE, pinball loss, coverage, total cost, stockout cost, and overstock cost.
- Add a simple rank-correlation or rank-disagreement summary.
- Highlight cases where lower error does not imply lower cost.

Expected contribution:

- Very aligned with the core TFG thesis.
- Low implementation risk.

Evidence needed:

- Ranking table by metric.
- Interpretation of at least one disagreement case.

Risks:

- Requires enough model variants or scenarios to make ranking analysis meaningful.

## Medium Priority Ideas

### 5. Multi-Period Inventory Simulation With Fixed Lead Time

Status: proposed

Goal: move beyond the current single-period newsvendor evaluation toward a weekly operational inventory simulation.

Motivation: the current project predicts demand over a horizon and evaluates each decision independently. A more realistic inventory setting would account for weekly orders, integer quantities, inventory carried over time, and fixed delivery lead time.

Operational setting:

- Decision period: weekly.
- Order: integer quantity `Q_t^(i) >= 0`.
- Timing: order placed at the end of week `t`, before week `t+1`.
- Lead time: fixed two weeks.
- Arrival: an order placed at the end of week `t` becomes available at the start of week `t+3`.

Suggested scope:

- Implement as a separate simulator module, not as a replacement for the v1 newsvendor policy.
- Track beginning inventory, arrivals, demand, fulfilled sales, lost sales, ending inventory, and outstanding orders.
- Start with synthetic or derived initial inventory assumptions if real inventory is unavailable.
- Compare the existing newsvendor evaluation against the multi-period simulator.

Expected contribution:

- Stronger operational realism.
- Clear distinction between one-step decision evaluation and dynamic inventory control.

Evidence needed:

- Simulator invariants and tests.
- Cost comparison under the same forecast inputs.
- Sensitivity to initial inventory assumptions.

Risks:

- FreshRetailNet may not expose all inventory-state variables needed for a realistic simulation.
- If initial inventory must be assumed, the analysis must state that clearly.
- This can grow quickly in scope.

### 6. Adaptive Retraining and Drift Diagnostics

Status: proposed

Goal: test whether temporal adaptation improves forecast and decision performance.

Motivation: retail demand can change due to promotions, seasonality, product behavior, and stockout regimes. The current project uses walk-forward retraining, but does not yet compare retraining policies.

Suggested scope:

- Compare expanding window, sliding window, and fixed retraining schedules.
- Add a simple drift diagnostic based on fold-level degradation, distribution shift, or stockout-regime changes.
- Optionally test recency weighting as a lightweight alternative.

Expected contribution:

- Better robustness story.
- Natural extension of existing walk-forward validation.

Evidence needed:

- Fold-level performance under each policy.
- Cost stability over time.

Risks:

- Short dataset windows may limit strong drift conclusions.

### 7. Stronger Baselines With a Clear Hypothesis

Status: proposed

Goal: improve the experimental comparison without turning the TFG into a generic model benchmark.

Motivation: the seasonal naive baseline is competitive. Adding one or two carefully chosen baselines can make conclusions more credible.

Suggested scope:

- Add moving average or exponential smoothing as a simple local baseline.
- Add CatBoost or improved LightGBM categorical handling as a stronger global tabular baseline.
- Only add models if they test a stated hypothesis, such as global sharing vs local heuristics.

Expected contribution:

- Stronger experimental validity.

Evidence needed:

- Same backtest and cost evaluation for every model.
- Clear explanation of why each baseline exists.

Risks:

- Too many models can dilute the thesis.

## Optional Ideas

### 8. Decision Support Dashboard

Status: optional

Goal: present forecasts, costs, and sensitivity results interactively.

Motivation: useful for professional impact, but not the main academic contribution.

Suggested scope:

- Streamlit or simple static dashboard.
- Show model ranking, cost scenarios, and stockout-regime diagnostics.

Expected contribution:

- Better communication and portfolio value.

Risks:

- Can consume time without improving the research contribution.

### 9. Multi-Agent or Reinforcement Learning Inventory Control

Status: future work

Goal: explore decentralized or sequential inventory policies.

Motivation: potentially interesting, but it would require a different problem formulation, simulator, state/action design, and reward structure.

Suggested scope:

- Keep as literature review or future work unless the TFG is re-scoped.

Expected contribution:

- High novelty if done properly.

Risks:

- High implementation risk.
- Easy to produce a fragile toy result.

## Candidate Research Questions

1. Does the ranking of forecasting models change when evaluated by inventory cost rather than predictive error?
2. How sensitive are inventory decisions to the assumed stockout and overstock cost ratio?
3. Do calibrated probabilistic forecasts reduce operational cost compared with raw quantile forecasts?
4. How much does stockout-aware demand treatment affect forecasting and ordering decisions?
5. Is a single-period newsvendor approximation sufficient for evaluating retail inventory decisions, or does a multi-period lead-time simulation change the conclusions?
6. Which retraining strategy provides the most stable cost performance across temporal folds?

## Current Recommendation

The best next implementation sequence is:

1. economic sensitivity analysis;
2. predictive vs economic ranking report;
3. conformal calibration;
4. stockout treatment experiments;
5. multi-period lead-time simulator only after the first four are stable.

This order gives the TFG a stronger contribution with controlled implementation risk.

## Parking Lot

Use this section to add new ideas before deciding whether they deserve a full entry.

- Add an external holdout using the official eval split if its temporal semantics are verified.
- Add categorical handling improvements for global boosting models.
- Add configuration fingerprints to processed data caches to avoid stale-cache experiments.
