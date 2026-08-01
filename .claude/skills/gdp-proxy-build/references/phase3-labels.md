# Phase 3: labels and region matching

This is the phase that decides whether the project is credible. It is also the
one that takes twice as long as planned, because region names are a mess and the
mess is invisible until you count rows.

## What you are building

Two modules.

`labels.py` turns published GDP figures into a tidy, deflated panel keyed by
region name and year. `match.py` maps those names onto the `region_id` values
that `boundaries.py` produced, via a human-reviewed crosswalk.

They are separate on purpose. Ingest is mechanical and rerunnable. Matching
involves human judgement that must be captured in a file and committed, not
redone at runtime.

## labels.py contract

```python
def load_dose(cfg: dict, path: Path | None = None) -> pd.DataFrame: ...
def load_national_deflator(iso3: str, base_year: int) -> pd.DataFrame: ...
def deflate(df: pd.DataFrame, deflator: pd.DataFrame, base_year: int) -> pd.DataFrame: ...
def load_labels(cfg: dict) -> pd.DataFrame: ...  # orchestrates the above
def validate_labels(df: pd.DataFrame, cfg: dict) -> LabelReport: ...
```

Output schema, one row per region-year:

| column | notes |
|---|---|
| `source_region_name` | as published, untouched |
| `parent_name` | state or province, needed to disambiguate duplicate district names |
| `year` | int |
| `gdp_constant` | deflated to `base_year` local currency |
| `gdp_nominal` | kept for audit only, never modelled on |
| `population` | if available |
| `admin_level` | 1 or 2 |
| `label_source` | `dose`, `mospi_gsdp`, `gddp_<state>` |
| `ingested_at` | ISO timestamp |

### Sources

DOSE lives on Zenodo (record 20035157), covers 1661 regions across 83 countries
to 2020, and is mostly ADM1. It already carries deflators and constant-price
columns; prefer its own constant-price series over redoing the deflation, and
document which you used.

World Bank WDI needs no API key: `https://api.worldbank.org/v2/country/{iso3}/indicator/NY.GDP.DEFL.ZS?format=json&per_page=200`.

District-level GDDP exists for only some Indian states and arrives as PDF or XLS
with merged cells and footnote rows. Write a small per-state adapter rather than
one clever universal parser. Put the raw files in `data/raw/labels/` and commit
the adapter, not the file.

### Deflation

Rule 9 in CLAUDE.md: never model on nominal values. A district growing 8 percent
nominal in a year with 7 percent inflation grew 1 percent, and nightlights will
show roughly 1 percent. Train on nominal and the model learns the inflation
series.

`validate_labels` should fail if `gdp_constant` is missing, if the base year is
not in the deflator series, or if year-on-year real growth exceeds 50 percent
for more than a handful of region-years, which usually means a units break
(lakh versus crore, or a currency redenomination mid-series).

## match.py contract

```python
def propose_crosswalk(
    boundaries: gpd.GeoDataFrame, labels: pd.DataFrame, cfg: dict
) -> pd.DataFrame: ...  # writes crosswalk_review.csv
def load_crosswalk(cfg: dict) -> pd.DataFrame: ...  # reads the reviewed file
def apply_crosswalk(labels: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame: ...
def validate_crosswalk(crosswalk, boundaries, labels) -> MatchReport: ...
```

Rule 7 is the shape of this module. Fuzzy matching runs **once**, as a proposal
step, and writes `data/crosswalk_review.csv` with a confidence score for a human
to check. The reviewed file is committed to `config/crosswalk_<country>.csv` and
that committed file is the only thing the pipeline reads. Fuzzy matching at
runtime means the mapping silently changes when a library version changes.

`crosswalk_review.csv` columns: `source_region_name`, `parent_name`,
`proposed_region_id`, `proposed_name`, `score`, `method`, `needs_review`,
`decision`. Leave `decision` blank for the human. Sort worst score first so the
reviewer sees the problems in the first screen rather than the last.

Normalise before scoring: casefold, strip diacritics, drop punctuation, drop
administrative suffixes (`district`, `zilla`, `taluk`), and collapse whitespace.
Score with `rapidfuzz.fuzz.token_sort_ratio`, then block by `parent_name` so
"Aurangabad" in Maharashtra cannot match "Aurangabad" in Bihar. Duplicate
district names across states are common and matching across the state boundary
is a real, quiet error. Mark anything under 90 as `needs_review`.

Use the `name_variants` column that `boundaries.py` already carries from GADM's
VARNAME field. It resolves a lot of transliteration cases for free.

## Exit test

`tests/test_match.py::test_every_label_maps_to_exactly_one_polygon`

The test asserts a bijection where one should exist: every labelled region maps
to exactly one `region_id`, no `region_id` receives two labels for the same
year, and the unmatched set is either empty or explicitly listed in a committed
`config/unmatched_<country>.csv` with a reason per row.

Unmatched regions are allowed. Unmatched regions that nobody looked at are not.
A district created in 2019 will not appear in 2014 boundaries and that is a fact
about the world, not a bug. Write the reason down.

Also test offline, with small synthetic frames:

- normalisation collapses "Bangalore Urban District" and "bengaluru urban"
- blocking prevents a cross-state match on an identical district name
- `apply_crosswalk` raises rather than dropping rows when a name is missing
- a duplicate `proposed_region_id` in the reviewed file is caught

## Reporting

Report matched count, unmatched count, and the score distribution. If you
matched 612 of 640, say which 28 and why. That is the number the user needs to
judge whether the panel is trustworthy, and it is the number a summary sentence
would hide.
