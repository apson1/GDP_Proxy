# Phase 6: pipeline and dashboard

## pipeline.py

One idempotent entrypoint that can be run on a schedule and does nothing when
there is nothing new. NOAA posts a monthly VIIRS composite roughly three to six
weeks after the month ends, so a weekly run that no-ops most of the time is the
right cadence.

```python
def latest_available_month(cfg) -> date: ...  # cheap: aggregate_max on time_start
def latest_processed_month(cfg) -> date | None: ...
def run(cfg, check_only: bool = False) -> PipelineResult: ...
```

Sequence: check for a new composite, extract it, rebuild features for the
affected year, run inference, append to `data/processed/estimates.parquet`,
write a run record. Every step checks whether its output already exists and
skips. Rule 4, and here it also means a crashed run can be re-run safely.

`--check` must be genuinely cheap. It is one `aggregate_max` on
`system:time_start`, a scalar, which is the one `getInfo` call this project
allows. Do not filter and count images to answer it.

Schedule with GitHub Actions using a service account, not your personal
credentials. `gee-community/ee-initialize-github-actions` documents the pattern:
store the service account JSON as a repository secret and initialise from it. Do
not commit credentials, and remember the service account also needs to be
registered against the Earth Engine project.

The estimates table needs a `vintage` column. When you re-run a month because a
composite was reprocessed, you want both rows, not a silent overwrite. Someone
will eventually ask why a number changed.

## app/streamlit_app.py

The dashboard's job is to make the uncertainty as visible as the estimate. A
choropleth of point estimates alone invites people to read a 15 percent
difference between two districts as real when the intervals overlap completely.

Pages:

1. **Map** — choropleth of estimated GDP or growth, year slider, a toggle for
   estimate versus interval width so users can see where the model is unsure.
2. **District detail** — the light time series, the GDP estimate with its band,
   the `n_valid_months` history, and whether this district had a published label
   or was extrapolated.
3. **Diagnostics** — the elasticity with its standard error, both holdout
   scores, interval coverage, and the count of unmatched regions. Publishing
   these is what separates a research tool from a plausible-looking dashboard.
4. **Caveats** — plain text, not a footnote. Nightlights underrepresent
   subsistence agriculture, informal services and remittance-driven consumption.
   Dense cores saturate. LED conversion depresses measured radiance while
   activity rises. A user making a decision on this needs to know all of it.

### Performance

Naive rendering of 640 detailed polygons across 150 months will hang the browser.

- Precompute everything into parquet. The app reads files, never calls Earth
  Engine on page load, and never fits a model at request time.
- Simplify geometries for display with `shapely.simplify` at a tolerance
  appropriate to the zoom, and keep the full-resolution geometry only for
  extraction. Store the simplified version separately.
- `st.cache_data` on the loaders.
- `pydeck` rather than per-feature folium markers.

## Phase 7: verification

The point of this phase is to check the project against the outside world, not
against itself.

- **Synthetic raster test.** Build a small raster with a known radiance pattern
  and a known polygon, run it through the masking and zonal stats, and assert
  the exact expected SOL. This is the only test that proves the extraction
  arithmetic rather than its plumbing.
- **Reproduce a published elasticity.** Pick a paper covering your country and
  see whether your within-estimator lands in the same neighbourhood. If it does
  not, you have a finding to explain or a bug to fix, and either is worth
  knowing before you publish.
- **Face validity.** Known boom districts should rank high, known conflict or
  decline districts low. Ask the user to name a few they know; they will spot a
  wrong answer faster than any metric.
- **Vintage stability.** Re-extract one already-processed year and diff it
  against the stored parquet. GEE reprocesses datasets; a large diff is
  information, and the provenance columns already in the output are what let you
  investigate it.
