# CLAUDE.md

Project rules for Claude Code working in this repository.

## What this project is

A subnational GDP estimator built from VIIRS nighttime lights. Scope for v1: one country, ADM2 districts, non-commercial research use, log-log panel regression plus XGBoost. No CNN. Read PLAN.md for the full phase plan and SETUP.md for environment setup.

## Non-negotiable rules

### Earth Engine

1. Never call `.getInfo()` on anything larger than a scalar or a small list. Use `Export.table.toDrive` style batch exports via `ee.batch.Export`. Interactive calls on hundreds of polygons across hundreds of months will burn the monthly EECU quota and then throttle the account for the rest of the month.
2. Always pass `project=` explicitly to `ee.Initialize()`. Read it from `GEE_PROJECT_ID` in `.env`. Never rely on a default project.
3. Batch extraction by year, not all at once. Each year is a separate export task with a resumable checkpoint.
4. Before any new extraction run, check whether the output parquet already exists and skip. Recomputation is the main way quota gets wasted.
5. Record the dataset ID, the list of image IDs used, and the extraction timestamp as columns in every output. GEE reprocesses datasets and results change over time.

### Joins and matching

6. Every join between light data and GDP labels must be followed by an assertion that the row count is what you expect. Silent row loss from name mismatches is the single most likely source of a wrong answer in this project.
7. Never fuzzy-match region names at runtime. Fuzzy matching runs once as a script that writes `data/crosswalk_review.csv` for a human to check. The reviewed and committed `config/crosswalk_<country>.csv` is what the pipeline uses.
8. `region_id` is a stable synthetic key. Never join on name strings.

### Data handling

9. All GDP values are deflated to constant prices before modelling. Nominal values are a bug.
10. Zero radiance is not zero economic activity. A polygon-month with `cf_cvg` at or below 1 is missing data, not a dark district. Flag it, do not impute it as zero.
11. Raw downloads live in `data/raw/` and are gitignored. Only `data/processed/*.parquet` and the crosswalk are committed.

### Modelling

12. Never use random K-fold cross validation on this panel. Use grouped CV by region for spatial holdout and forward-chaining splits for temporal holdout. Random folds leak and produce fake R2 above 0.95.
13. Every published estimate carries an uncertainty band. Point estimates of district GDP without intervals are not shippable.
14. The within-estimator elasticity of log GDP on log SOL should land near 0.3, and cross-sectional near 0.6 to 1.0. If it is far outside that, assume the extraction is wrong before assuming the model is interesting.

## Conventions

- Python 3.11, `src/` layout, package name `gdp_proxy`.
- Type hints on all public functions. `ruff` for lint and format, line length 100.
- Config lives in `config/countries.yaml`. No hardcoded country names, years, CRS values, or thresholds in source files.
- Secrets and project IDs live in `.env`, loaded with `python-dotenv`. `.env` is gitignored, `.env.example` is committed.
- Tabular interchange is parquet, not CSV, except for human-reviewed files like the crosswalk.
- Logging with the `logging` module, not `print`, outside of CLI entrypoints.

## Definition of done for a phase

Each phase in PLAN.md has an exit test. That exit test must exist as a real pytest in `tests/` and pass. Do not report a phase complete based on a manual check or a notebook cell.

## Things to ask me about rather than decide

- Changing the target country or admin level
- Adding a data source that requires a licence or payment
- Any change that makes the project commercial, which would break the GADM and GEE non-commercial terms
- Dropping regions from the training set for any reason

## Useful commands

```bash
python -m gdp_proxy.doctor              # Phase 0 exit test
pytest -q                               # all tests
ruff check . && ruff format .           # lint and format
python -m gdp_proxy.pipeline --check    # is there a new VIIRS month
streamlit run app/streamlit_app.py      # dashboard
```
