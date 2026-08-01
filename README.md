# GDP_Proxy

**A nowcast of Indian state GDP from satellite nighttime lights, disaggregated to 676
districts by light share.**

This is what the numbers are, in the order that determines whether they are fit for your
purpose:

1. **A state GDP nowcast.** The model is estimated at state level, where the labels are.
   The cross-sectional elasticity of log GDP on log lights is **0.809 (se 0.015)**, inside
   the 0.6–1.0 range the published literature reports. For years after the label series ends
   (DOSE stops at 2019) the state total is predicted rather than observed, and every row
   says which it is.
2. **Disaggregated to districts by light share**, not measured at district level. No
   district in India publishes GDP that this model trains on. Each state's total is split
   across its districts in proportion to their share of the state's light, so district
   figures sum exactly to the state total and the national total is right by construction.
3. **The split is validated** against published district GDDP from Tamil Nadu, Maharashtra
   and Karnataka — **321 district-years across 95 districts**, held out of training
   entirely. Light share correlates **0.806** (Spearman 0.879) with true GDP share, mean
   absolute error **0.99 share points**. The split carries real information.
4. **The split is also biased, measurably and in a known direction.** Light under-allocates
   service economies (`corr(services share, error) = −0.59`) and over-allocates agricultural
   ones (`+0.54`); large districts are under-allocated as the sensor saturates (top decile
   receives 0.80× its true share, bottom half 1.22×). A one-parameter correction was fitted
   and **rejected** by leave-one-state-out. The shipped figures are uncorrected.
5. **There is no validated growth estimator.** The within-district relationship is not
   identified on a six-year label overlap: after removing state and year effects, 0.2% of
   GDP variance survives, and the measured within elasticity is **−0.073 (se 0.065)** against
   a literature value near 0.3. **Do not read year-on-year changes as growth, and do not rank
   districts by growth rate.**

**Fit for**: comparing how large or how prosperous districts are, in a given year, with an
uncertainty band, accepting a known sectoral tilt. **Not fit for**: growth rates, year-on-year
change, or ranking service-led against agriculture-led districts without discounting for the
bias in point 4.

Both **total GDP** and **GDP per capita** are published, with per capita as the headline.
They rank districts very differently, and that is the point: a populous poor district is
high on total output and low on prosperity.

Official GDP in developing nations suffers from long publication lags, sparse subnational
coverage, and occasional political manipulation. Nighttime lights, captured monthly by the
VIIRS Day/Night Band at 463 m, are observed everywhere on the same schedule and cannot be
revised for political convenience. That is the case for doing this at all.

> **Nightlights are a proxy, not a measurement.** They track energy-intensive and urban
> activity, and badly underrepresent subsistence agriculture, informal services and
> remittance-driven consumption. Read [What this cannot tell you](#what-this-cannot-tell-you)
> before using any number from this project. It is not optional context.

---

## Scope (locked for v1)

| Decision | Value | Why |
|---|---|---|
| Geography | India, ADM2 (676 districts) | Largest subnational label gap worth closing |
| Labels | DOSE V2.14, ADM1 (33 states) | Only free, harmonised subnational GDP panel |
| Period | Labels 1980–2019; lights 2014– | VIIRS monthly starts 2014 |
| Model | Log-log panel FE + XGBoost | Interpretable baseline first, no CNN in v1 |
| Use | Research, non-commercial | Keeps GADM and Earth Engine free and legal |

Kenya (47 counties) ships configured in `config/countries.yaml` as a second target;
Indonesia (514 kabupaten) is a documented candidate that would need a config block adding.
Nothing country-specific is hardcoded in source — country, admin level, CRS, year range and
every threshold live in that one file.

---

## How it works

```
VIIRS monthly composites (Earth Engine)
  │  mask: cloud coverage → water → gas flares → noise floor
  ▼
Zonal statistics per district-month  ──────────►  data/processed/sol_*.parquet
  │  batch Export.table.toDrive, never getInfo
  ▼
Annualise on VALID months only (+ n_valid_months)
  │  sol, lit_share, gini_light, p90_rad, YoY log-diffs
  ▼
Features  ◄──── join ────  Deflated GDP labels (DOSE, constant 2015 LCU)
  │                            via committed, human-reviewed crosswalk
  ▼
Model: log(GDP) ~ log(SOL) + region FE + year FE, then XGBoost
  │  spatial (GroupKFold) + temporal (forward-chaining) holdouts
  ▼
Estimates + conformal prediction intervals  ────►  estimates.parquet → Streamlit
```

The masking order matters and is not obvious. **Cloud-masked means "we could not see";
noise-floor-zeroed means "we looked and it was dark."** Collapsing that distinction turns a
monsoon month into a poor district, which is the single most consequential bug available
in this project.

---

## Repository layout

```
src/gdp_proxy/
  config.py       countries.yaml + .env loading
  auth.py         Earth Engine init (explicit project id)
  doctor.py       Phase 0 exit test
  boundaries.py   GADM / geoBoundaries, stable synthetic region_id
  masks.py        coverage, water, flare, noise floor
  extract.py      batch reduceRegions, CSV → validated parquet
  labels.py       DOSE ingest, deflation to constant prices
  match.py        GID + fuzzy crosswalk, one row per district
  features.py     annualisation, SOL and derived features
  population.py   GHSL district population, epoch interpolation
  gddp.py         district GDDP adapters, VALIDATION ONLY, never trained on
  model.py        panel FE, XGBoost, honest CV splits, light-share allocation
  evaluate.py     holdout metrics, elasticity gate, allocation validation
  pipeline.py     idempotent monthly orchestration
app/streamlit_app.py   map, district detail, diagnostics, caveats
config/
  countries.yaml          all tunables; no constants in source
  crosswalk_india.csv     reviewed by a human, committed
  unmatched_india.csv     regions with no label, each with a reason
  face_validity_india.csv 12 districts named by a human, boom and lagging
.github/workflows/
  check-new-month.yml     weekly cron: python -m gdp_proxy.pipeline --check
tests/                    186 tests
```

Docs: **[SETUP.md](SETUP.md)** for environment and Earth Engine access (start here) ·
**[PLAN.md](PLAN.md)** for the phase plan and ranked roadblocks ·
**[CLAUDE.md](CLAUDE.md)** for the non-negotiable engineering rules ·
**[BUILD.md](BUILD.md)** for the one-shot build brief.

---

## Quickstart

Environment setup, including the three separate things Earth Engine access requires, is in
**[SETUP.md](SETUP.md)**. Once `python -m gdp_proxy.doctor` passes:

```bash
python -m gdp_proxy.boundaries
```

```bash
python -m gdp_proxy.extract --pilot --submit
```

```bash
python -m gdp_proxy.extract --ingest
```

```bash
python -m gdp_proxy.labels
```

```bash
python -m gdp_proxy.match --propose
```

Review `data/crosswalk_review.csv` by hand, commit it as `config/crosswalk_india.csv`, then:

```bash
python -m gdp_proxy.pipeline --check
```

```bash
streamlit run app/streamlit_app.py
```

Development:

```bash
ruff check . && ruff format . && pytest -q
```

**Start with the pilot, not the full series.** Earth Engine noncommercial projects are
metered monthly (150 EECU-hours Community, 1000 Contributor). Exceeding the quota throttles
the account for the rest of the calendar month. Every extraction step checks for an existing
parquet and skips; recomputation is the main way quota disappears.

---

## Current status

Phases 0–7 are built and tested: **186 tests — 185 passing, 1 xfailed**. The suite is green.

The one `xfail` is `test_known_boom_and_decline_districts_rank_correctly`, marked
`strict=True` with the measured allocation bias as its reason: light under-allocates service
economies (Bangalore, Surat) and over-allocates districts with lit extractive industry
(Dantewada). It is a documented property, not a regression — and because it is strict, the
suite will fail loudly if it ever starts passing rather than going quietly green.

| Phase | Status |
|---|---|
| 0 Environment | Done — `doctor.py` passes, Earth Engine live |
| 1 Boundaries | Done — 676 GADM 4.1 ADM2 districts snapshotted |
| 2 Extraction | Done for 2014–2019 + 2024–2025; 2020–2023 not yet extracted |
| 3 Labels + matching | Done — crosswalk reviewed and committed, 672/676 matched |
| 4 Features | Done — 3,988 district-years in the modelling frame |
| 5 Model | Done — cross-sectional elasticity asserted; within reported only (see below) |
| 6 Pipeline + dashboard | Done — app run end to end against real estimates; weekly `--check` on GitHub Actions |
| 7 Verification | Done — synthetic-raster arithmetic verified on real Earth Engine |

### Region matching (real numbers)

DOSE V2.14 carries `GID_1`, GADM's own state code, so state matching is an **exact code
join**, not a fuzzy name match:

| | count |
|---|---|
| ADM2 districts (GADM 4.1) | 676 |
| ADM1 label regions (DOSE V2.14) | 33 |
| **Matched** | **672** |
| — exact GID (`IND.x_1`) | 630 |
| — disputed-territory GID (`Z0n.x_1` → `IND.x_1`) | 42 |
| — fuzzy name fallback | 0 |
| **Unmatched** | **4** |

The 4 unmatched are Lakshadweep, Dadra and Nagar Haveli, and Daman and Diu (2 districts) —
small union territories absent from DOSE's 33 regions. Each is written into
`config/unmatched_india.csv` with a reason. Unmatched regions are allowed; unmatched
regions nobody looked at are not.

GADM files territory under international dispute using `Z`-prefixed codes (`Z01.14_1` for
Jammu and Kashmir) while DOSE uses the national code (`IND.14_1`). Those 42 districts are
matched deterministically but flagged for human confirmation, never folded in silently.

One state maps to many districts, so the crosswalk is **one-to-many by design**. The
invariant is the other direction: every district maps to exactly one state.

### Elasticity

Fitted at ADM1, where the label varies (33 states × 6 years, n=191, annual composites):

| Estimator | Value | Expected | Verdict |
|---|---|---|---|
| Cross-sectional (pooled) | **+0.809** (se 0.015) | 0.6–1.0 | In band, asserted |
| Within (region + year FE) | **−0.073** (se 0.065) | ~0.3 | Not identified, reported only |
| Within (region FE only) | +0.358 (se 0.065) | — | **Contaminated, ignore** |

Only the cross-sectional estimate is asserted by the test suite. The band was **not**
widened to accommodate the within estimate.

**The entity-FE-only figure is a trap and is labelled as such in code.** It lands at
+0.358, temptingly close to the literature's ~0.3, but it retains the 2016/2017 VIIRS
calibration step and reproduces that value by coincidence. Split at the break it flips
sign: **−0.201 before 2017, +0.354 after.** A structural elasticity cannot flip sign at a
calendar boundary. The national year-on-year ratios confirm it — one year carries the whole
result (2017: dlog SOL +0.226 against dlog GDP +0.076) while every other year is nonsense
(−3.86, +1.73, +3.56). `python -m gdp_proxy.model` prints the sign flip on every run and a
test fails if any code path promotes this number to headline.

This also rules out rural electrification as the explanation: Saubhagya launched September
2017 and peaked through 2018, yet 2018 and 2019 show dlog SOL of only +0.040 and +0.020.

### The 2016/2017 calibration discontinuity

National SOL steps and never returns. Switching from monthly `VCMSLCFG` to the
inter-calibrated annual composites **reduces but does not remove it**:

| Series | 2016→2017 dlog SOL |
|---|---|
| Monthly VCMSLCFG | +0.328 |
| Annual V21 | +0.226 |

A 25% one-year national jump is still not plausible growth. Training defaults to the annual
series (`training_series: annual`); the two families are never concatenated.

Note the annual product is split across two GEE asset ids by processing version:
**ANNUAL_V21 covers 2013–2021, ANNUAL_V22 only 2022–2025.** The extractor picks by year and
records which id produced each row.

### Allocation validation (the within-state split)

The estimates' *coherence* is guaranteed by construction; their *correctness within a state*
is not, and this is the only external test of it. District GDDP from three states is ingested
purely as validation and can never enter training — `assert_no_validation_labels` raises in
`build_training_frame` if it does, because training on it would make the check circular.

| State | District-years | Years | Pearson r | MAE (share points) |
|---|---|---|---|---|
| Tamil Nadu | 192 | 2014–2019 | 0.799 | 0.84 |
| Maharashtra | 99 | 2022–2024 | 0.754 | 1.10 |
| Karnataka | 30 | 2022 | 0.941 | 1.61 |
| **Pooled** | **321** | | **0.806** | **0.99** |

**The bias is systematic and has not been corrected.** Two gradients, both measured:

- **Sectoral.** In Karnataka, where sector shares are published:
  `corr(services share, allocation error) = −0.59`, `corr(agriculture share, error) = +0.54`,
  industry ≈ 0. The more service-led a district, the more light **under**-allocates its GDP.
- **Size / saturation.** `corr(log GDP share, error) = −0.38`. Top-decile districts by GDP
  share receive on average **0.80×** their true share; bottom-half districts receive
  **1.22×**. This is the DNB saturation effect predicted in PLAN.md roadblock 4, showing up
  in real data.

Worst-mispredicted districts are dense service and port economies — Thane (ratio 0.31),
Dakshina Kannada (0.25), Chennai (0.35), Udupi (0.21) — against over-allocated peri-urban
and agricultural ones: Ahmadnagar (2.01), Bangalore Rural (1.95), Kolar (1.89).

### A correction was tested and rejected

The obvious fix is a one-parameter reweighting, `corrected_share ∝ sol_share^α` renormalised
within each state-year, with α > 1 shifting mass toward large districts. One parameter over
321 observations is hard to overfit. It was fitted and tested **leave-one-state-out**: fit α
on two states, score the held-out third.

| Held out | n | α (fitted on other two) | MAE uncorr. | MAE corr. | Δ | r uncorr. | r corr. |
|---|---|---|---|---|---|---|---|
| Karnataka | 30 | 0.954 | 1.61 | 1.68 | **+0.07** | 0.941 | 0.934 |
| Maharashtra | 99 | 0.914 | 1.10 | 1.15 | **+0.04** | 0.754 | 0.756 |
| Tamil Nadu | 192 | 1.103 | 0.84 | 0.92 | **+0.09** | 0.799 | 0.796 |

**Rejected on both criteria.** Held-out MAE got *worse* in all three folds, and α straddles
1.0 (0.914–1.103), so the folds disagree about which direction to correct.

The mechanism is visible in the per-state fits. Each state's own optimum differs sharply —
Karnataka 1.376, Maharashtra 0.987, Tamil Nadu 0.904 — tracking per-state size gradients that
range from −0.28 to −0.77. No single global exponent can fit them. This is a state effect,
not a time effect: within Tamil Nadu, α is stable across all six years (0.867–1.043).

A size-weighted loss, which lets the largest economies dominate the fit, is worse still —
degraded in 3 of 3 folds.

**The most tempting α is the one that fails hardest.** Setting α = 1.376 (Karnataka's
in-sample optimum) raises face validity from 9/12 to 11/12, rescuing Bangalore (0.60 → 0.90)
and Surat (0.45 → 0.69). It is also the value LOSO rejects most strongly. Fitting to 30
Karnataka districts and applying it to all 676 would be exactly the overfitting the test
exists to catch, so it has not been applied.

No two-parameter version was tried: adding a second term to a model whose first term does not
generalise would add flexibility in the wrong place.

### The bias is state-specific and learnable, not noise

This is the substantive finding, and it is worth stating separately because it is easy to
misread the rejection above as "the correction did not work because the signal is weak".
It is the opposite. **α is stable within a state and unstable between states.**

| | α |
|---|---|
| Tamil Nadu, per year across six years | 0.867, 0.904, 0.895, 0.924, 0.955, 1.043 |
| Maharashtra, per year across three years | 0.987, 0.978, 0.954 |
| **Between states** (each state's own optimum) | **0.904 (TN), 0.987 (MH), 1.376 (KA)** |

Within a state the exponent barely moves across years. Between states it moves by half its
own magnitude, and the per-state size gradients differ threefold (−0.28, −0.42, −0.77).

So the bias is not noise that more data would average away. It is a real, stable, per-state
property — plausibly reflecting each state's industrial mix and settlement pattern — and it
is *learnable* given district GDDP for that state. A global correction fails precisely
because it averages three genuinely different quantities into one.

**Per-state correction was considered and declined.** It would be fitted for 3 states out of
33 and undefined for the other 30, producing a map where a tenth of the country is corrected
and the rest is not. Two adjacent districts either side of a state line would then be
adjusted on different bases, and the discontinuity would be an artefact of which state
publishes GDDP rather than anything economic. A partially-corrected map is harder to reason
about than a uniformly-biased one whose bias is documented.

The path that would actually work is more states: with district GDDP for, say, 15 of 33, α
could be modelled from state characteristics (sectoral composition, urbanisation) rather than
fitted per state, and applied everywhere on the same basis.

**Conclusion: light share is biased, and the bias could not be corrected in a way that
generalises across states.** The uncorrected share is what ships, and
`allocation_share_uncorrected` is published on every row so the raw allocation is always
reconstructible. Full fold results are in `diagnostics.json` under
`allocation_validation.share_correction`; per-district detail in
`data/processed/allocation_validation.parquet`.

### Coverage: no region is excluded

**All 33 label regions are in the panel** (n=191 state-years). Nothing is dropped.

The coverage-gap hazard is real but it lives on the **light** side: if a district is missing
in one year and present the next, the state's summed SOL jumps and that jump reads as
growth. That is now asserted directly — `assert_light_panel_balanced` raises unless every
state's district count is constant across the years it covers. It currently passes: 676
districts × 6 years, zero gaps.

An earlier version guarded this by requiring a balanced *label* panel, which excluded three
states for gaps in the **GDP** series. Gaps in GDP cannot manufacture fake light growth, so
that exclusion bought nothing and cost Telangana — one of the faster-growing states. The
estimator uses explicit fixed-effect dummies, which handle an unbalanced panel correctly.

Three regions have a short label series, all retained, each with a recorded reason in
`diagnostics.json`:

| Region | Label-years | Missing | Status |
|---|---|---|---|
| Telangana | 5 | 2014 | **Correct by history** — created 2014-06-02 |
| Kerala | 5 | 2014 | Gap in DOSE V2.14; MOSPI publishes it |
| Puducherry | 1 | 2015–2019 | DOSE series truncated after 2014 |

Telangana's gap is **not a defect and must not be backfilled**: the state did not exist for
most of 2014, so inventing a 2014 GSDP would be fabricated data. It is logged at INFO;
genuine gaps are logged at WARNING. Any region with a short series and no entry in
`label_coverage_notes` is reported as UNEXPLAINED rather than assumed benign.

---

## What this cannot tell you

Read this before quoting any number.

- **Subsistence agriculture, informal services and remittance-driven consumption** barely
  emit light. In an agrarian district, household income can double while measured radiance
  is flat.
- **Dense urban cores saturate** the sensor. Growth is understated exactly where it is
  largest. `lit_share` and `p90_rad` are in the feature set because they keep responding
  after brightness flattens.
- **LED conversion** lowers emitted light in the DNB band while activity rises. Estimates
  for rapidly-LED-ifying cities are a **lower bound**, not an estimate.
- **The ADM1 → ADM2 jump is the central weakness.** The model trains on 33 state labels and
  predicts 676 districts — a twentyfold resolution gap. District estimates are
  extrapolations, flagged as such, and carry wide intervals. Districts within a state share
  one label and are **not independent observations**.
- **Point estimates without intervals are not shippable.** Every published estimate carries
  an uncertainty band. If two districts' bands overlap, the difference between them is not
  a finding.
- **The light-share allocation breaks in two predictable directions.** Service economies are
  understated, because an office district produces far more output per lumen than a factory
  or a highway (Bangalore and Surat both rank lower on per capita than local knowledge
  says). Extractive and sparsely-populated districts are overstated per capita, because a
  lit mine in a district of 300,000 people yields a high figure that reflects the mine, not
  household prosperity (Dantewada).
- **The population denominator is not fully independent of the numerator.** GDP per capita
  uses GHSL population, and GHSL distributes census counts across a grid using built-up-area
  layers derived partly from satellite imagery. That imagery is optical and radar, **not**
  nighttime lights, so the coupling is indirect and weak — but not zero. Built-up area and
  lit area correlate, so a district whose built-up extent GHSL overestimates may receive
  both more people and more light. The practical effect is mild: it can slightly compress
  per-capita differences between districts.

---

## Design rules worth knowing

The full list is in [CLAUDE.md](CLAUDE.md). The four that shape everything:

1. **Missing is not zero.** `cf_cvg ≤ 1` is missing data, not a dark district. Never
   `fillna(0)` on radiance.
2. **Assert row counts after every join.** A merge that loses a third of the districts looks
   exactly like a merge that worked.
3. **No fuzzy matching at runtime.** It runs once, writes a review file, a human checks it,
   and the committed file is what the pipeline reads.
4. **No random K-fold.** Panel autocorrelation leaks and reports R² above 0.95. Spatial
   (grouped) and temporal (forward-chaining) splits only. A spatial R² above 0.9 means look
   for the leak before believing it.

Published elasticity of log GDP on log SOL lands near **0.3 within** and **0.6–1.0
cross-sectional**. Far outside that range means the extraction is wrong, not the finding
interesting.

---

## Data sources and licensing

| Source | Use | Licence |
|---|---|---|
| NOAA VIIRS DNB (`VCMSLCFG`) via Earth Engine | Monthly radiance | Free, **noncommercial** registration |
| GADM 4.1 | District boundaries | Free, **academic / noncommercial only**, no redistribution |
| DOSE V2.14 (Zenodo 20035157) | Subnational GDP labels | Free, CC |
| JRC Global Surface Water | Water mask | Free |
| JRC GHSL P2023A `GHS_POP` | District population (per-capita denominator) | Free |
| World Bank WDI | Deflators, national GDP | Free, no key |

**This project is non-commercial by design.** Making it commercial would breach both the
GADM and Earth Engine terms. `boundaries.py` supports geoBoundaries (CC-BY) as a pluggable
alternative, but switching changes every `region_id` and requires re-keying the panel.

### The boundary snapshot is not in this repository

GADM permits academic and non-commercial *use*, but **not redistribution without prior
permission**, and publishing their data in a public repo is redistribution. So the district
boundary snapshot is gitignored, along with the two other files that carry GADM's district
names or GID codes at scale. Rebuild them locally in this order:

```bash
python -m gdp_proxy.boundaries
```

That writes `data/processed/boundaries_india_adm2_gadm.parquet` (~11 MB) and is a
prerequisite for extraction, matching and the dashboard map. `python -m gdp_proxy.match
--propose` and `python -m gdp_proxy.evaluate` regenerate the other two.

The processed parquets that *are* committed — light, population, features, labels,
estimates — are keyed only by the opaque synthetic `region_id` and contain no GADM
geometry, names or codes.

`data/processed/model.pkl` is also gitignored: it is a regenerable build artefact, and
pickle is a code-execution format that should never be handed to a stranger from a public
repository. Rebuild it with `python -m gdp_proxy.model`.

Raw downloads in `data/raw/` are gitignored. Only processed parquet and the reviewed
crosswalk are committed.
