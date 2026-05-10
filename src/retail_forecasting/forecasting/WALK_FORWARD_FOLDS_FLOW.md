# Walk-Forward Fold Construction

This diagram documents the control flow of `build_walk_forward_folds()`.

```mermaid
flowchart TD
    A["Input: panel, validation_config, horizon"] --> B["Extract global unique dates<br/>unique_dates = sorted(date.drop_duplicates())"]

    B --> C["Compute minimum required dates<br/>D_min = initial_train_days + n_folds * fold_size_days + horizon - 1"]

    C --> D{"len(unique_dates) < D_min?"}
    D -- "Yes" --> E["Raise ValueError<br/>not enough dates"]
    D -- "No" --> F["Initialize folds = []"]

    F --> G["Compute last valid validation origin<br/>last_valid_index = len(unique_dates) - horizon"]

    G --> H["Loop fold_id<br/>0 ... n_folds - 1"]

    H --> I["validation_start_index = initial_train_days + fold_id * fold_size_days"]
    I --> J["validation_end_index = validation_start_index + fold_size_days - 1"]

    J --> K{"validation_end_index > last_valid_index?"}
    K -- "Yes" --> L["Break loop<br/>target horizon would be incomplete"]
    K -- "No" --> M["Map validation indexes to dates"]

    M --> N["validation_start_date = unique_dates[validation_start_index]"]
    N --> O["validation_end_date = unique_dates[validation_end_index]"]
    O --> P["train_end_date = validation_start_date - horizon days"]

    P --> Q["Create FoldSpec"]

    Q --> R{"FoldSpec Pydantic validation"}
    R --> R1["fold_id >= 0"]
    R --> R2["horizon > 0"]
    R --> R3["validation_end_date >= validation_start_date"]
    R --> R4["train_end_date = validation_start_date - horizon"]

    R1 --> S["Append FoldSpec to folds"]
    R2 --> S
    R3 --> S
    R4 --> S

    S --> H

    L --> T{"folds is empty?"}
    H --> U["Loop finished"]
    U --> T

    T -- "Yes" --> V["Raise ValueError<br/>no valid fold could be created"]
    T -- "No" --> W["Return list[FoldSpec]"]
```

Key points:

- Folds are built over the panel's global calendar of unique dates, not per-series calendars.
- Validation windows move forward by `fold_size_days` for each `fold_id`.
- `last_valid_index` prevents validation origins whose target horizon would be incomplete.
- `FoldSpec` validates the temporal gap and basic fold invariants at runtime.
