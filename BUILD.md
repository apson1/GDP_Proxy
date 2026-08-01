# One-shot build brief

Paste the block below into Claude Code from the repo root. It is written to be
handed over without further explanation.

---

```
Build out phases 3 through 7 of this project in one pass, using the
gdp-proxy-build skill.

Read first, in this order:
  1. CLAUDE.md               non-negotiable rules
  2. .claude/skills/gdp-proxy-build/SKILL.md
  3. PLAN.md                 phase plan and known roadblocks
  4. The existing modules in src/gdp_proxy/ so the new code matches their shape

Phases 0 to 2 are done and tested. Do not rewrite them.

Work phase by phase. For each phase: read its reference file under
.claude/skills/gdp-proxy-build/references/, write the module, write the tests,
run `ruff check . && ruff format .` and `pytest -q`, and only then start the
next phase. Do not batch all the code and test at the end.

Build:
  Phase 3  src/gdp_proxy/labels.py, src/gdp_proxy/match.py
  Phase 4  src/gdp_proxy/features.py
  Phase 5  src/gdp_proxy/model.py, src/gdp_proxy/evaluate.py
  Phase 6  src/gdp_proxy/pipeline.py, app/streamlit_app.py
  Phase 7  tests/test_verification.py

Tests that need real data or Earth Engine credentials go behind
@pytest.mark.network or @pytest.mark.needs_data. Never weaken an assertion to
make it pass. If a phase cannot be completed without an artefact I have not
produced yet, build everything around it, mark the gap clearly, and keep going.

Stop and ask me before: changing the target country or admin level, dropping
regions from the training set, adding a paid or licensed data source, or making
any change that would break the GADM and Earth Engine non-commercial terms.

When you finish, give me one short report containing: matched and unmatched
region counts from Phase 3, the within and cross-sectional elasticities with
standard errors from Phase 5, spatial and temporal holdout scores, interval
coverage, and the list of tests that are skipped and why. Do not summarise the
work; give me the numbers that would reveal a problem.
```

---

## What I need to supply for the build to run end to end

The code can be written without these, but the exit tests cannot pass.

| Needed | For | How |
|---|---|---|
| Boundary snapshot | Phase 3 onward | `python -m gdp_proxy.boundaries` |
| GEE boundary asset | Phase 2 extraction | SETUP.md Part E |
| Extracted SOL parquets | Phase 4 onward | `python -m gdp_proxy.extract --pilot --submit`, then `--ingest` |
| DOSE download | Phase 3 | Zenodo record 20035157, into `data/raw/labels/` |
| State GDDP files | Phase 5 district validation | state statistics departments, into `data/raw/labels/` |
| Reviewed crosswalk | Phase 3 exit test | review `data/crosswalk_review.csv`, commit to `config/` |

The crosswalk review is mine to do, by design. It is the one point where a human
has to look at the region name matches, and automating it away is how districts
get mapped to the wrong state.
