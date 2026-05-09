# Data Preparation Flow: `prepare_daily_panel`

This diagram illustrates the sequence of cleaning, filtering, and feature preparation steps required to transform raw data into a modeling-ready daily panel.

```mermaid
%%{init: { 'flowchart': { 'htmlLabels': false } } }%%
graph TD
    Start(["Start: prepare_daily_panel"]) --> Rename["Rename Columns: dt->date, sale_amount->observed_demand, etc."]
    Rename --> DateConv["Convert 'date' to datetime"]
    
    DateConv --> NegSales{"drop_negative_sales=True?"}
    NegSales -- "Yes" --> DropNeg["Filter out negative sales"]
    NegSales -- "No" --> Dedup["drop_duplicates: Store + Product + Date"]
    DropNeg --> Dedup
    
    Dedup --> SeriesID["Create series_id: store_id + product_id"]
    SeriesID --> Sort["Sort by series_id & date"]
    
    Sort --> MinHist["Filter series by min_history_days"]
    
    MinHist --> TopN{"top_n_series set?"}
    TopN -- "Yes" --> FilterTop["Keep N largest series by demand"]
    TopN -- "No" --> MissingVals["preprocessing.fill_missing_values?"]
    FilterTop --> MissingVals
    
    MissingVals -- "Yes" --> Impute["Fill Flags with 0.0 & Weather with Median"]
    MissingVals -- "No" --> End(["Return Cleaned Panel"])
    Impute --> End

    style Start fill:#e1f5fe,stroke:#01579b
    style End fill:#e1f5fe,stroke:#01579b
```
