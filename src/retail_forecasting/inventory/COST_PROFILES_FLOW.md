# Per-Series Cost Generation Flow: `build_series_cost_profile`

This diagram illustrates the heuristic logic used to generate synthetic inventory costs (overstock/stockout) for each unique series (Store + Product).

```mermaid
%%{init: { 'flowchart': { 'htmlLabels': false } } }%%
graph TD
    Start(["Start: build_series_cost_profile"]) --> Validate["Validate required columns: observed_demand, stockout_hours, etc."]

    Validate --> SeriesStats["Group by series_id: Calculate Mean Demand, Intermittency, Stockout Rate"]
    SeriesStats --> CatStats["Group by category_id: Calculate average instability benchmarks"]

    CatStats --> Scoring["Heuristic Scoring: perishability, criticality, and slow-moving scores"]
    Scoring --> Adjust["Adjust Base Costs: Multiply settings.inventory costs by scores"]

    Adjust --> Fractile["Calculate Critical Fractile: c_under / (c_under + c_over)"]
    Fractile --> End(["Return Cost Profile DataFrame"])

    style Start fill:#e1f5fe,stroke:#01579b
    style End fill:#e1f5fe,stroke:#01579b
