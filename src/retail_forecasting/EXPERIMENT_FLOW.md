# Experiment Flow: `run_experiment` Pipeline

This diagram illustrates the end-to-end experiment pipeline, comparing Observed demand vs. Latent (imputed) demand strategies.

```mermaid
%%{init: { 
  'theme': 'neutral', 
  'flowchart': { 'htmlLabels': false }
} }%%
graph TD
    Start(["Start: run_experiment"]) --> LoadData["load_prepared_panel: Load Train Split"]
    
    LoadData --> StrategyA["Strategy A: Observed Demand (Baseline)"]
    
    subgraph Strategy_Execution [Strategy Execution: run_experiment_from_frame]
        direction TB
        LabelRegime["label_stockout_regime: Identify Censorship"]
        LabelRegime --> FeatureEng["build_supervised_frame: Feature Engineering"]
        FeatureEng --> BuildFolds["build_walk_forward_folds: Time-Series CV"]
        
        subgraph Backtesting_Loop [Cross-Validation Loop (per fold)]
            direction TB
            Calibrate["Conformal Calibration (Latest 21 days)"]
            Calibrate --> FitPredict["Fit & Predict Models: Naive, Ridge, LGBM, CatBoost, ARIMA"]
            FitPredict --> Inventory["choose_order_quantity: Newsvendor Decision"]
            Inventory --> Drift["PageHinkleyDetector: Update & Monitor Drift"]
        end
        
        BuildFolds --> Backtesting_Loop
        Backtesting_Loop --> Summarize["Summarize: Metrics, Costs, Pareto, Sensitivity"]
    end

    StrategyA --> Strategy_Execution
    
    Strategy_Execution --> Impute["LatentDemandImputer: Impute Censored Demand"]
    
    Impute --> StrategyB["Strategy B: Latent Demand (Imputed)"]
    StrategyB --> Strategy_Execution_Latent["Strategy Execution: run_experiment_from_frame"]
    
    Strategy_Execution_Latent --> Merge["Merge Artifacts: Strategy A + Strategy B"]
    Merge --> Write["write_run_artifacts: Export CSVs & Markdown Report"]
    Write --> End(["End"])

    style Start fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style End fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style Strategy_Execution fill:#f8f9fa,stroke:#333
    style Strategy_Execution_Latent fill:#f8f9fa,stroke:#333
    style Backtesting_Loop fill:#ffffff,stroke:#546e7a,stroke-dasharray: 5 5
```
