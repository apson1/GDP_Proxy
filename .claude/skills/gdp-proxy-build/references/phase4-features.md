# Phase 4: features

Short phase, but the feature set decides what the model can see. The core
insight is that Sum of Lights alone saturates: once a city core maxes out the
sensor, extra activity produces no extra radiance. Features that measure the
*spread* and *edge* of light keep responding after brightness stops.

## features.py contract

```python
def load_panel(cfg: dict) -> pd.DataFrame: ...  # all yearly parquets, concatenated
def annualise(monthly: pd.DataFrame, cfg: dict) -> pd.DataFrame: ...
def add_level_features(df, boundaries, cfg) -> pd.DataFrame: ...
def add_growth_features(df, cfg) -> pd.DataFrame: ...
def build_features(cfg: dict) -> pd.DataFrame: ...
def validate_features(df, cfg) -> FeatureReport: ...
```

## Monthly to annual

Labels are annual, so the panel has to be annualised, and how you do it matters.
Take the mean of *valid* months, not the sum, and record `n_valid_months`. A
district with four usable months and one with twelve should not be compared as
if both were complete. Drop region-years below a configurable
`min_valid_months`, default 6, and record how many you dropped rather than
quietly shrinking the panel.

Do not `fillna(0)`. Rule 10. A masked month is missing.

## Feature list

Levels, per region-year:

- `sol` — sum of masked radiance
- `sol_per_area` — `sol / area_km2`, removes the trivial "big districts are bright" effect
- `sol_per_capita` — where population exists
- `mean_rad`, `median_rad`, `p90_rad`
- `lit_pixels`, `lit_share` — `lit_pixels / valid_pixels`
- `gini_light` — concentration of radiance within the district
- `n_valid_months` — a data-quality feature the model is allowed to use

Dynamics:

- `sol_yoy`, `lit_pixels_yoy` — year-on-year log difference, not percent change, so it is symmetric and matches the log-log model
- `newly_lit` — pixels lit this year that were dark in the configured baseline year
- 3-year rolling means of `sol` and `lit_share`

Logs: `log_sol = np.log(sol)` for strictly positive levels, `np.log1p` for
counts. Guard against `log(0)` explicitly rather than adding a magic epsilon;
if `sol` is exactly zero for a well-observed district, that is worth surfacing,
not smoothing.

## Why lit_share and gini_light earn their place

`sol` is one number and it conflates two different things: a district getting
brighter, and a district lighting up more area. Those mean different things
economically. Electrification of villages shows up in `lit_share` and barely at
all in `sol`, because a newly connected village is dim. Industrial expansion in
an already-bright corridor shows up in `sol` and `p90_rad`.

`gini_light` separates a district whose light is one dense city from one with
the same total spread evenly. Those have very different GDP for identical SOL.

## Validation

`validate_features` should assert:

- one row per region-year, no duplicates
- no infinite values after the log transforms
- `lit_share` within [0, 1]
- `sol_per_area` positive where `sol` is positive
- the panel is balanced or the imbalance is reported, with counts per year
- `n_valid_months` distribution printed, so a masking regression is visible

## Exit test

`tests/test_features.py::test_national_sol_tracks_national_gdp`

Aggregate `sol` to the national level by year and correlate with national GDP
from WDI. Expect above 0.9. This is a cheap, strong check on the entire upstream
chain: if extraction, masking or annualisation is broken, this correlation
collapses, and it needs no subnational labels to run.

Also test offline with synthetic panels:

- annualisation ignores masked months and sets `n_valid_months` correctly
- a region-year below `min_valid_months` is dropped and counted
- `gini_light` returns 0 for uniform light and approaches 1 for a single hot pixel
- year-on-year features are NaN in the first year rather than 0
