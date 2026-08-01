# Phase 5: model and evaluation

Two modules. `model.py` fits, `evaluate.py` judges. Keep them apart so the
evaluation code cannot be quietly tuned to flatter the model.

## The one thing that ruins this phase

Random K-fold cross validation on a panel leaks. Adjacent years of the same
district are nearly identical, so a random fold puts 2019 Pune in train and 2020
Pune in test, and the model scores 0.97 by remembering the district rather than
learning anything about light. Rule 12.

Use two honest splits instead, and report both:

- **Spatial holdout**: `GroupKFold` grouped by `region_id`. Answers "can this
  predict a district it has never seen", which is the actual product question,
  because you are predicting districts with no published GDP.
- **Temporal holdout**: forward chaining. Train on years up to t, test on t+1.
  Answers "can this predict next year", which is the monitoring question.

Spatial R2 will be much lower than random-fold R2. That is the point. If your
spatial R2 comes back above 0.9, look for the leak before believing it.

## model.py contract

```python
def fit_panel(df, cfg) -> PanelResult: ...  # log-log, region + year FE
def fit_xgboost(df, cfg, splits) -> ModelResult: ...
def spatial_splits(df, n_splits=5) -> Iterator[tuple]: ...
def temporal_splits(df, min_train_years=4) -> Iterator[tuple]: ...
def predict_with_intervals(model, X, alpha=0.1) -> pd.DataFrame: ...
def save_model(model, path) -> None: ...
```

### The baseline comes first

Fit `log(gdp_constant) ~ log(sol) + C(region_id) + C(year)` with `linearmodels`
`PanelOLS` before touching XGBoost. It takes ten lines, it is interpretable, and
its coefficient is a diagnostic for the entire upstream pipeline.

The within-estimator elasticity should land near 0.3. The cross-sectional
elasticity, from a specification without region fixed effects, should land
between 0.6 and 1.0. These ranges come from a large published literature. Rule
14: if you are far outside them, the extraction is wrong, not the finding
interesting. An elasticity of 0.05 usually means the join lost most of the
signal or the labels are nominal. An elasticity above 2 usually means units.

Report the elasticity with its standard error, always. It is the single number
that tells the user whether to trust anything downstream.

### XGBoost

Features from Phase 4, target `log(gdp_constant)`. Tune shallowly: this panel
has a few hundred training units, and a deep boosted forest on 300 rows will
memorise. Cap `max_depth` around 4, use early stopping on the grouped folds,
and expect the gain over the log-log baseline to be modest. If XGBoost beats the
panel model by a huge margin, suspect leakage through a feature that encodes
region identity, such as `area_km2` or an unnormalised `sol`.

### Uncertainty is not optional

Rule 13. A district GDP point estimate with no interval is not shippable,
because the whole use case is someone deciding where to put money. Two workable
routes:

- quantile regression with XGBoost, fitting the 5th and 95th percentiles
- split conformal prediction on the spatial holdout residuals, which gives
  coverage guarantees without a distributional assumption

Conformal is the better default here: it is simple, it makes no assumption about
residual shape, and its coverage claim is honest.

## The ADM1 to ADM2 extrapolation

Be explicit about this in code and in output, because it is the project's
central weakness. The model trains on state-level labels, roughly 30 to 36 units
per year, and predicts 640 districts. That is a twentyfold resolution jump.

Three things reduce the damage:

1. Pool years so the training panel is several hundred region-years rather than
   30 rows.
2. Prefer a hierarchical specification that lets district predictions shrink
   toward their state's fitted value, rather than treating districts as
   independent draws.
3. Validate against the districts that *do* have published GDDP. That subset is
   small and it is the only real out-of-sample evidence you have. Never train on
   it. Hold it out entirely and report performance on it separately.

Widen the intervals for districts far from the training distribution, and carry
a per-district `extrapolation_flag` through to the dashboard.

## evaluate.py contract

```python
def spatial_holdout_metrics(...) -> dict: ...
def temporal_holdout_metrics(...) -> dict: ...
def elasticity_check(panel_result, cfg) -> list[tuple[str, bool, str]]: ...
def district_validation(preds, gddp_labels) -> dict: ...
def interval_coverage(preds, actuals, nominal=0.90) -> float: ...
def report(...) -> str: ...
```

Metrics on the log scale (R2, RMSE, MAE) and on the level scale after
exponentiating, since a good log R2 can still mean large rupee errors in big
districts. Report both.

`interval_coverage` should come out near its nominal level. If your 90 percent
intervals contain the truth 60 percent of the time, they are decoration.

## Exit test

`tests/test_model.py::test_spatial_holdout_and_elasticity`

Assert: the within elasticity is in a configurable plausible band, spatial
holdout R2 is reported and is *below* 0.95 (a leak detector, not a quality bar),
and interval coverage is within 10 points of nominal.

Offline tests with synthetic data:

- `spatial_splits` never puts the same `region_id` in train and test
- `temporal_splits` never puts a future year in train
- a deliberately leaky random split scores higher than the grouped split on
  synthetic autocorrelated data, proving the splitter is doing its job
- `predict_with_intervals` returns lower bound below point estimate below upper
