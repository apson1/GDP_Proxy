# Nightlights GDP Proxy: Build Plan

Scope decisions locked for v1:

- Geography: one country, ADM2 (district) inference. Reference build below uses India (about 640 districts). Kenya (47 counties) and Indonesia (514 kabupaten) are drop-in alternatives.
- Use: research and portfolio, non-commercial. This makes Google Earth Engine and GADM both free and legal.
- Model: log-log panel regression plus XGBoost. No CNN in v1.

---

## 1. Data source audit

Every source below was checked for cost and access friction as of July 2026.

| Source | What you get | Cost | Access friction |
|---|---|---|---|
| GEE `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` | Monthly avg radiance, 463m, stray-light corrected, 2014-01 to present | Free (non-commercial) | Google account, verified Cloud project, ~1 day approval |
| GEE `NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG` | Same but no stray-light correction, 2012-04 to present | Free | Same |
| GEE `NOAA/VIIRS/DNB/ANNUAL_V22` | Annual cleaned composites, outlier-removed, best for model training | Free | Same |
| NASA Black Marble VNP46A3/A4 (`NASA/VIIRS/002/VNP46A2` in GEE) | Lunar-BRDF and atmosphere corrected monthly/annual | Free | Earthdata Login for direct download, or use GEE |
| GADM v4.1 / v5 | ADM0 to ADM3 boundaries, every country | Free | Direct download, no login. Non-commercial only |
| geoBoundaries | Same coverage, CC-BY license | Free | Direct download or API |
| DOSE (Zenodo) | Subnational GDP, 1661 regions, 83 countries, 1960-2020 | Free | Direct download, CC license |
| MOSPI / data.gov.in | India GSDP by state, some GDDP by district | Free | Direct download, format is messy XLS |
| EOG VIIRS Nightfire (flare locations) | Gas flare point locations and volumes | Free | Requires application and license approval, not instant |
| World Bank WDI API | National GDP, deflators, population | Free | No key required |

Verdict: everything core is free. Two caveats. EOG Nightfire needs a license application that is not instant, so build the flare mask from the published static flare-site CSV or a buffer approach first. GEE non-commercial is free but now metered (see roadblock 1).

---

## 2. Repository layout

```
GDP_Proxy/
  CLAUDE.md                 project rules for Claude Code
  PLAN.md                   this file
  pyproject.toml
  .env.example              GEE project id, Earthdata creds
  config/
    countries.yaml          country, admin level, CRS, year range
  data/
    raw/                    downloaded, gitignored
    interim/
    processed/              zonal stats parquet
  src/gdp_proxy/
    auth.py                 GEE init, Earthdata session
    boundaries.py           GADM/geoBoundaries fetch, dissolve, validate
    ee_extract.py           reduceRegions zonal stats, export to Drive/GCS
    masks.py                flare mask, water mask, background noise floor
    features.py             SOL, mean radiance, lit pixel count, growth rates
    labels.py               DOSE + MOSPI ingest, deflate, harmonise names
    match.py                fuzzy region-name matching to boundary ids
    model.py                log-log panel, XGBoost, CV splits
    evaluate.py             holdout metrics, elasticity checks
    pipeline.py             monthly orchestration entrypoint
  app/
    streamlit_app.py
  notebooks/
  tests/
```

---

## 3. Phased plan

### Phase 0: Environment and access (half a day)

1. Create a Google Cloud project, enable the Earth Engine API, register it as non-commercial. Attach a billing account to get the Contributor tier (1000 EECU-hours/month, still free) rather than Community tier (150).
2. `earthengine authenticate` and store the project id in `.env`.
3. Create a NASA Earthdata Login account. Generate a LAADS token even if you plan to stay in GEE, as a fallback path.
4. Submit the EOG data licence application now so approval lands before you need it.
5. Set up the Python env: `earthengine-api`, `geemap`, `geopandas`, `rasterio`, `pyarrow`, `xgboost`, `statsmodels`, `linearmodels`, `streamlit`, `pydeck`.

Exit test: a script prints the number of images in `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` and the date of the newest one.

### Phase 1: Boundaries (half a day)

1. Download GADM level 2 for the target country. Reproject to an equal-area CRS for area calculations, keep EPSG:4326 for GEE.
2. Fix invalid geometries, dissolve slivers, assign a stable `region_id` that does not depend on the name string.
3. Snapshot the boundary version. Districts split and rename constantly, and this is the single largest source of silent error in the project.

Exit test: polygon count matches the official district count for the census year, and total area matches the country's official land area within 1 percent.

### Phase 2: Zonal extraction (1-2 days)

1. Write `ee_extract.py` to loop months, apply masks, and run `reduceRegions` with a combined reducer: sum, mean, count, and a lit-pixel count above threshold.
2. Masks to apply, in order:
   - Cloud-free coverage: drop pixels where `cf_cvg` is 0 or 1 for that month.
   - Background noise floor: zero out radiance below about 0.3 to 0.5 nW/cm2/sr. Calibrate the threshold on a known dark region such as desert or forest interior.
   - Water: mask with JRC Global Surface Water occurrence above 50 percent to kill boat and fishing lights.
   - Gas flares: mask a 5 to 10 km buffer around known flare sites.
3. Export to CSV in Google Drive or GCS, then load to parquet. Do not try to pull hundreds of months through `getInfo()`.
4. Batch by year to stay inside memory limits and to make retries cheap.

Exit test: SOL for the whole country by year correlates above 0.9 with national GDP, and the December-to-June seasonal pattern looks sane for a country with monsoon cloud cover.

### Phase 3: Labels and matching (1-2 days, the real work)

1. Pull DOSE from Zenodo. Filter to your country. This gives ADM1 for most countries.
2. Pull district-level GDDP where the state statistics department publishes it. In India, roughly a dozen states do. Expect PDFs and XLS with merged cells.
3. Deflate to constant prices using the country deflator from World Bank WDI. Never train on nominal values.
4. Match region names to `region_id`. Use a deterministic crosswalk CSV that you hand-check, not fuzzy matching at runtime. Fuzzy match once, output a review file, commit the reviewed crosswalk.

Exit test: every labelled region maps to exactly one polygon, and the unmatched list is empty or explicitly documented.

### Phase 4: Features (half a day)

- `sol`: sum of masked radiance in the polygon
- `sol_per_area`, `sol_per_capita`
- `mean_rad`, `median_rad`, `p90_rad`
- `lit_pixels`: count above threshold; `lit_share`: lit pixels / total pixels
- `newly_lit`: pixels lit this year that were dark in baseline year
- `gini_light`: concentration of light within the polygon, a proxy for urban versus dispersed activity
- `sol_yoy`, `lit_pixels_yoy`, 3-year rolling means
- Log transforms of all level variables, with `log1p` for counts

### Phase 5: Model (1-2 days)

1. Baseline: `log(GDP) ~ log(SOL) + region FE + year FE` on the ADM1 panel. Report the elasticity. Published values land around 0.3 for the within estimator and 0.6 to 1.0 for the cross-sectional estimator. If you are far outside that range, your extraction is wrong, not your model.
2. XGBoost on the full feature set with grouped cross validation. Group by region for spatial holdout, and run a separate forward-chaining time split. Random K-fold will give you a fake high R2 because of panel autocorrelation.
3. Predict ADM2 from a model trained on ADM1. State this limitation loudly. Validate against the districts where GDDP does exist.
4. Report prediction intervals via quantile regression or conformal prediction. A point estimate of district GDP with no uncertainty band is not usable by anyone.

Exit test: spatial-holdout R2 on log GDP, plus a plot of predicted versus actual for the held-out districts with real GDDP.

### Phase 6: Pipeline and dashboard (1-2 days)

1. `pipeline.py` checks for a new VIIRS month, extracts, features, infers, and appends to the processed store. Idempotent, safe to re-run.
2. Schedule with GitHub Actions on a cron. NOAA typically posts a monthly composite 3 to 6 weeks after month end, so run weekly and no-op when there is nothing new.
3. Streamlit app: choropleth of estimated GDP and growth, time slider, district drill-down with the light time series, and a page that shows model diagnostics honestly.
4. Cache the parquet in the repo or a small object store. Do not have the dashboard call GEE on page load.

### Phase 7: Verification

- Unit tests on masking and zonal stats using a synthetic raster with a known answer.
- Reproduce one published elasticity from the literature for your country as an external check.
- Sanity checks: known boom districts should rank high, known conflict or decline districts should rank low.

---

## 4. Roadblocks, ranked by how likely they are to hurt you

1. GEE compute quota. Since April 2026, non-commercial projects are metered. Community tier is 150 EECU-hours per month, Contributor tier is 1000 and only needs a billing account attached. Exceeding the quota puts you in Restricted mode, which throttles rather than bills you. Mitigation: attach billing to get Contributor, use `reduceRegions` with the coarsest acceptable `scale`, export as batch tasks rather than interactive calls, and cache aggressively so you never recompute a month.

2. Label scarcity at ADM2. This is the project's core weakness, not a detail. DOSE mostly stops at ADM1. If you train on 36 states and predict 640 districts, you are extrapolating across a 20x resolution gap with about 30 training units per year. Mitigation: pool years for a panel of a few hundred region-years, pull the district GDDP that does exist for validation, use hierarchical or multilevel models rather than pretending districts are independent, and publish uncertainty bands.

3. Region name and boundary churn. District splits, renames, and transliteration variants will silently drop or duplicate rows. India created dozens of new districts in the last decade. Mitigation: a committed, hand-reviewed crosswalk file; a hard assertion that row counts match after every join; and freezing to one boundary vintage with a documented mapping to newer vintages.

4. Top-coding and saturation. Dense urban cores saturate the DNB sensor, so SOL underestimates growth in already-bright places and the log-log fit flattens at the top. Mitigation: add lit-area and light-concentration features that keep responding after radiance saturates, and check residuals against baseline brightness.

5. Cloud and seasonality. Monsoon and high-latitude summer months produce months with almost no valid observations for some polygons. `cf_cvg` will be 0. Mitigation: use annual composites for training, require a minimum cloud-free observation count per polygon-month, and interpolate or flag rather than treating 0 radiance as 0 activity.

6. Sensor and product discontinuities. VCMCFG versus VCMSLCFG differ in stray-light handling, and NOAA VIIRS versus NASA Black Marble differ in atmospheric correction. Mixing them mid-series creates fake growth. Mitigation: pick one product family for the entire series, and if you must switch, estimate the offset on an overlap period.

7. The LED transition. Cities replacing sodium lamps with LEDs emit less in the DNB band even as activity rises. This produces a downward bias in exactly the middle-income places you care about. Mitigation: acknowledge it, check whether affected regions show a step change, and treat DNB estimates in rapidly-LED-ifying regions as a lower bound.

8. Structural blind spots. Nightlights track energy-intensive and urban activity. They badly underrepresent subsistence agriculture, informal services, and remittance-driven consumption. In an agrarian district, light growth may be near zero while household income doubles. State this in the dashboard, not just in a footnote.

9. Flare mask data access. EOG requires a licence application, and interim access only gives you the reduced ezCSV format. Mitigation: apply early, and in the meantime mask the small number of known petroleum basins by hand from the published global flare survey.

10. Streamlit performance. Rendering 640 polygons across 150 months in a naive folium map will time out. Mitigation: simplify geometries with `topojson` or `shapely.simplify`, precompute everything into parquet, and use `pydeck` or `st.map` rather than per-feature folium markers.

11. Reproducibility of GEE results. GEE datasets get reprocessed and versions get deprecated. A number you extracted in March may differ in July. Mitigation: record the dataset id, the image ids used, and the extraction date in every output parquet.

12. GADM licence. Free for academic and non-commercial only. This is fine for your stated use, but if the project ever becomes commercial you must swap to geoBoundaries. Writing `boundaries.py` with a pluggable source now costs an hour and saves a rewrite later.

---

## 5. Suggested Claude Code workflow

1. Start with `CLAUDE.md` capturing: the coding conventions, the fact that all GEE calls must be batch exports rather than `getInfo`, the assertion-on-join rule, and the config file location.
2. Work one phase per branch. Each phase above has an exit test. Make that exit test an actual pytest so Claude Code can verify it rather than claim success.
3. Give Claude Code the crosswalk review file as a human checkpoint. Do not let name matching happen silently.
4. Keep `data/raw` gitignored and commit only the processed parquet plus the crosswalk.

## 6. Realistic timeline

About 7 to 10 working days to a defensible v1 with a live dashboard. Phase 3 is the one that always takes twice as long as planned.
