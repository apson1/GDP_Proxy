---
name: gdp-proxy-build
description: Build, extend or debug the VIIRS nighttime-lights subnational GDP estimator in this repository. Use this skill whenever the work touches nightlights, VIIRS, DNB, sum of lights, SOL, Earth Engine extraction, GADM or geoBoundaries district boundaries, DOSE or GSDP labels, the region crosswalk, the district GDP model, the monthly pipeline, or the Streamlit dashboard. Also use it for any request phrased as "build the next phase", "finish the project", "run the whole build", or "why is my elasticity wrong" in this repo, even when the user does not name a phase or a file. It carries the phase contracts, exit tests and the domain traps that make this project silently produce confident wrong numbers.
---

# Building the GDP proxy

This repo estimates district-level GDP from satellite nighttime lights. The
danger in this project is not that it crashes. It is that it runs, produces a
clean-looking map, and the numbers are wrong. Almost every failure mode here is
silent: a join that drops 200 districts, a mask that turns cloud into darkness,
a cross-validation split that leaks and reports R2 of 0.97.

So the working style is: build a phase, prove it with a test that would fail if
the phase were broken, and only then move on. `CLAUDE.md` in the repo root holds
the non-negotiable rules. Read it first; it is short and every rule in it exists
because that specific mistake produces a plausible wrong answer.

## Current state

Phases 0 to 2 are built and tested. Do not rewrite them.

| Module | Does | Status |
|---|---|---|
| `config.py` | loads `config/countries.yaml` plus `.env` | done |
| `auth.py` | `init_ee()`, explicit project id | done |
| `doctor.py` | Phase 0 exit test | done |
| `boundaries.py` | GADM and geoBoundaries loaders, `region_id`, validation | done |
| `masks.py` | coverage, water, flare, noise floor | done |
| `extract.py` | batch `reduceRegions`, CSV ingest, validation | done |
| `labels.py` | DOSE and GSDP ingest, deflation | Phase 3 |
| `match.py` | name to `region_id` crosswalk | Phase 3 |
| `features.py` | SOL and derived features | Phase 4 |
| `model.py` | panel regression and XGBoost | Phase 5 |
| `evaluate.py` | holdout metrics, elasticity checks | Phase 5 |
| `pipeline.py` | monthly orchestration | Phase 6 |
| `app/streamlit_app.py` | dashboard | Phase 6 |

Read the existing modules before writing new ones. Match their shape: lazy `ee`
imports, a `validate_*` function returning a report object with named checks, a
`main()` CLI, and pure logic separated from network calls so tests run offline.

## Build order and how to know a phase is finished

Work the phases in order. Each has a reference file with the full contract.

1. Phase 3, labels and matching — `references/phase3-labels.md`
2. Phase 4, features — `references/phase4-features.md`
3. Phase 5, model — `references/phase5-model.md`
4. Phase 6, pipeline and dashboard — `references/phase6-pipeline.md`

Before starting any phase, read its reference file. Before finishing any phase,
check `references/pitfalls.md` for the traps specific to that stage.

A phase is finished when its exit test exists as a real pytest and passes. Not
when a notebook cell looked right. If the exit test needs data the user has not
produced yet, write the test, mark it `@pytest.mark.network` or
`@pytest.mark.needs_data`, and say clearly which artefact is missing rather than
weakening the assertion until it passes.

Run after every phase:

```bash
ruff check . && ruff format .
pytest -q                 # offline tests, must be green
```

## The three habits that keep this project honest

**Assert the row count after every join.** This is the highest-value habit in
the codebase. Region names arrive from four sources with four spellings, and a
merge that loses a third of the districts looks exactly like a merge that
worked. Every merge gets an explicit expected count and a failure that names the
dropped regions.

```python
before = len(light)
merged = light.merge(labels, on="region_id", how="inner", validate="m:1")
if len(merged) != expected:
    lost = set(light.region_id) - set(merged.region_id)
    raise ValueError(f"Join dropped {len(lost)} regions: {sorted(lost)[:10]}")
```

**Missing is not zero.** A district under monsoon cloud has no light reading.
That is not the same as a district with no economic activity. `is_missing` is
already computed in `extract.py`; carry it forward, never `fillna(0)` on
radiance, and let the model see a gap rather than a fabricated zero.

**Distrust a good result.** If spatial-holdout R2 comes back above 0.9, or the
within-region elasticity of log GDP on log SOL is not somewhere near 0.3, the
first hypothesis is a leak or a bad extraction, not a breakthrough. The
literature range is roughly 0.3 within and 0.6 to 1.0 cross-sectional. Check the
split before celebrating.

## Working with the user's quota

Earth Engine noncommercial projects are metered monthly: 150 EECU-hours on the
Community tier, 1000 on Contributor. Burning it stalls the project for the rest
of the calendar month, so treat compute as a budget, not a free resource.

- Never call `.getInfo()` on anything bigger than a scalar or a short list.
- Check for an existing output parquet and skip before submitting any export.
- Default to a two-year pilot before a full-series run.
- If an export fails with "computed value is too large", raise `tile_scale` in
  the config rather than reducing the region set.

## When something is genuinely ambiguous

Some choices change what the project means, and guessing wastes the user's time
in a way that is hard to undo later. Stop and ask rather than deciding:

- changing the target country or admin level
- dropping regions from the training set for any reason
- adding a data source that needs a licence or payment
- anything that makes the project commercial, which breaks the GADM and Earth
  Engine noncommercial terms

Everything else, decide and note the reasoning in the commit message.

## Reporting back

When a phase completes, report the numbers that would reveal a problem, not a
summary of the work. For Phase 3 that is matched and unmatched region counts.
For Phase 5 it is the elasticity and both holdout scores. The user cannot check
"labels ingested successfully". They can check "612 of 640 districts matched, 28
unmatched listed in crosswalk_review.csv, all in states created after 2014".
