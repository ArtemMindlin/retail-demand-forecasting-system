# Project Conventions

These conventions keep the repo predictable for humans and agents.

## Naming

Use Python naming conventions consistently:

- files and modules: `snake_case.py`
- functions: `snake_case`
- variables: `snake_case`
- classes: `PascalCase`
- constants: `UPPER_SNAKE_CASE`
- tests: `test_<behavior_or_contract>.py`

Prefer domain names already used by the project:

- `panel` for prepared daily demand data
- `supervised_frame` for model-ready rows with features and target
- `predictions` for model outputs plus inventory decisions
- `metrics_summary`, `fold_metrics`, and `cost_summary` for evaluation outputs
- `target_lead_time_demand` for the current lead-time demand target
- `order_quantity` for the inventory decision

Do not introduce alternate names for established columns unless the schema intentionally changes and `docs/contracts/dataframes.md` is updated.

## Documentation Files

Use these file naming conventions:

- contracts: `docs/contracts/<domain>.md`
- general docs: `snake_case.md`

Keep `CLAUDE.md` short. Put detailed rules in the main `docs/` files and link to them from `CLAUDE.md`.

## Web Layer

Conventions for `src/retail_forecasting/api/`. See `docs/web_layer.md` for the
architecture.

- templates: `templates/views/<tab>.html` for whole pages, `templates/partials/<component>.html` for fragments swapped by htmx
- static assets: `static/css/`, `static/js/`, `static/img/`, and `static/vendor/` for third-party libraries served from our own domain
- URL names match the tab ids in `api/context.py`; the navigation context processor resolves the active tab from `request.resolver_match.url_name`
- one view module per plane: `views/pages.py`, `views/ops.py`, `views/eda.py`, `views/experiments.py`, `views/panels.py`, `views/pipeline.py`, `views/json_api.py`

Two template rules that are easy to get wrong and fail silently:

- `{# … #}` is single-line only. Use `{% comment %}…{% endcomment %}` for anything longer, or it renders as visible text.
- Any number that lands in JavaScript, CSS, or an SVG attribute must bypass es-ES localization: use `|floatformat:"3u"` or `|unlocalize`. Applying `|unlocalize` after `floatformat` does nothing.

## Tests

Use test names that describe the protected contract, not implementation details.

Examples:

- `test_temporal_leakage_contract.py`
- `test_dataframe_contracts.py`
- `test_raw_column_boundaries.py`
- `test_config_contract.py`

For harness tests, prefer deterministic synthetic data over remote data or cached datasets.

## Commit Messages

Use short Conventional Commit style subjects:

- `docs: add harness conventions`
- `test: add dataframe contract checks`
- `fix: remove hardcoded quantile coverage bounds`
- `chore: ignore generated artifacts`
- `refactor: simplify architecture import contract`

Keep the subject in imperative mood and under roughly 72 characters when practical.

Recommended commit body:

```text
Context:
- Why this change is needed.

Changes:
- What changed.

Validation:
- Commands run, for example `uv run pytest`.
```

## Commit Splitting

When the worktree contains multiple unrelated changes, split commits by intent.

Good split for this repo:

1. Docs/system-of-record changes.
2. Test harness additions.
3. Production code changes.
4. Generated artifact cleanup or ignore rules.

Before staging or committing a large change, inspect `git status --short` and propose the split. Ask for confirmation after each large project change before staging or committing.

Do not mix generated notebooks, reports, PDFs, or local cache cleanup into a code or docs commit unless the user explicitly asks.
