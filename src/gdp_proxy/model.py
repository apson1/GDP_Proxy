"""Phase 5a: models.

Two things ruin this phase, and both are guarded here rather than trusted to
discipline.

Random K-fold on a panel leaks (rule 12). Adjacent years of one district are
nearly identical, so a random fold trains on 2019 Pune and tests on 2020 Pune and
scores 0.97 by memorising the district. This module only exposes honest splits:
``spatial_splits`` (GroupKFold by ``region_id``) and ``temporal_splits`` (forward
chaining). There is no code path that shuffles rows into folds.

The panel baseline comes before the tree model, because its coefficient is a
diagnostic for the whole upstream pipeline. The within-estimator elasticity of
log GDP on log SOL should land near 0.3 and the cross-sectional near 0.6-1.0
(rule 14). Far outside that means the extraction is wrong, not the finding
interesting.

Every prediction carries an interval (rule 13). Point estimates of district GDP
with no band are not shippable, because the use case is deciding where money
goes. Intervals come from split-conformal residuals on the spatial holdout, which
make no distributional assumption.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "log_gdp"

DEFAULT_FEATURES = [
    "log_sol",
    "log_sol_per_area",
    "log_lit_pixels",
    "lit_share",
    "gini_light",
    "mean_rad",
    "median_rad",
    "p90_rad",
    "n_valid_months",
    "sol_yoy",
    "lit_pixels_yoy",
    "newly_lit",
    "sol_roll3",
    "lit_share_roll3",
]


def model_features(cfg: dict[str, Any]) -> list[str]:
    return list((cfg.get("model") or {}).get("features") or DEFAULT_FEATURES)


def ensure_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add log_gdp from gdp_constant if absent. Never model on nominal (rule 9)."""
    out = df.copy()
    if TARGET not in out.columns:
        if "gdp_constant" not in out.columns:
            raise ValueError("Need gdp_constant (or log_gdp) to build the modelling target.")
        out[TARGET] = np.log(out["gdp_constant"].where(out["gdp_constant"] > 0))
    return out


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


def spatial_splits(df: pd.DataFrame, n_splits: int = 5) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold by region_id: no district appears in both train and test.

    This answers the real product question, whether the model can predict a
    district it has never seen, which is precisely what predicting unlabelled
    districts requires.
    """
    from sklearn.model_selection import GroupKFold

    groups = df["region_id"].to_numpy()
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        raise ValueError(f"Need at least 2 distinct regions for spatial CV, got {n_groups}.")
    gkf = GroupKFold(n_splits=n_splits)
    dummy_x = np.zeros((len(df), 1))
    yield from gkf.split(dummy_x, groups=groups)


def temporal_splits(
    df: pd.DataFrame, min_train_years: int = 4
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Forward chaining: train on years <= t, test on t+1. Never a future year in train.

    Answers the monitoring question, whether the model can predict next year.
    """
    years = np.sort(df["year"].unique())
    if len(years) <= min_train_years:
        raise ValueError(
            f"Need more than {min_train_years} years for temporal CV, got {len(years)}."
        )
    positions = np.arange(len(df))
    year_arr = df["year"].to_numpy()
    for i in range(min_train_years, len(years)):
        train_years = years[:i]
        test_year = years[i]
        train_idx = positions[np.isin(year_arr, train_years)]
        test_idx = positions[year_arr == test_year]
        if len(test_idx) == 0:
            continue
        yield train_idx, test_idx


# --------------------------------------------------------------------------
# panel baseline
# --------------------------------------------------------------------------


LABEL_KEY = ["source_region_name", "parent_name"]

# How each level feature rolls up from district to label region. SOL and pixel
# counts are extensive (sum); radiance percentiles and month counts are intensive
# (mean). Getting this wrong silently rescales the regressor.
LABEL_AGG = {
    "sol": "sum",
    "lit_pixels": "sum",
    "valid_pixels": "sum",
    "area_km2": "sum",
    "mean_rad": "mean",
    "median_rad": "mean",
    "p90_rad": "mean",
    "n_valid_months": "mean",
}


def assert_light_panel_balanced(df: pd.DataFrame) -> None:
    """Assert the LIGHT panel has no coverage gaps. This is the real guard.

    The hazard is an unbalanced *district* sum: if a district is missing in 2016
    and present in 2017, the state total jumps by that district's light and the
    jump is read as growth. That hazard lives entirely on the satellite side.

    Guarding it by dropping *label* regions was the wrong instrument. It excluded
    three states for gaps in the GDP series, which cannot manufacture fake light
    growth, while doing nothing the extraction checks were not already doing.

    Raises rather than warning: if this ever fails, every within-estimate
    downstream is contaminated by coverage-driven variation.
    """
    if "region_id" not in df.columns or "year" not in df.columns:
        return

    if "source_region_name" not in df.columns:
        # No label grouping available: fall back to global completeness.
        n_years = df["year"].nunique()
        per_district = df.groupby("region_id")["year"].nunique()
        incomplete = per_district[per_district < n_years]
        if len(incomplete):
            raise ValueError(
                f"Light panel is unbalanced: {len(incomplete)} district(s) are not "
                f"observed in all {n_years} years, e.g. {sorted(incomplete.index)[:5]}."
            )
        return

    # State-relative on purpose. A state whose GDP series is short (Telangana has
    # no 2014 because it did not exist) legitimately contributes fewer years, and
    # that is not a light coverage gap. What must never happen is a state's
    # district count changing across the years it *does* cover, because then the
    # state total moves for reasons unrelated to economic activity.
    counts = df.groupby(["source_region_name", "year"])["region_id"].nunique()
    spread = counts.groupby(level=0).agg(["min", "max", "size"])
    offenders = spread[spread["min"] != spread["max"]]
    if len(offenders):
        detail = ", ".join(
            f"{region} ({row['min']}-{row['max']} districts across {row['size']} years)"
            for region, row in offenders.head(5).iterrows()
        )
        raise ValueError(
            f"Light panel is unbalanced within {len(offenders)} label region(s): "
            f"{detail}. The state's summed SOL would change because of coverage, "
            "and that change would be read as growth. Re-extract the missing "
            "district-years rather than dropping regions."
        )


def aggregate_to_label_level(df: pd.DataFrame, require_balanced: bool = False) -> pd.DataFrame:
    """Roll a district panel up to the resolution the GDP label actually varies at.

    ``apply_crosswalk`` fans one ADM1 label out to its ~21 districts, so every
    district in a state carries the *same* ``gdp_constant`` and within-state
    variance of ``log_gdp`` is exactly zero. Regressing district ``log_sol`` on
    that is not an imprecise version of the right regression, it is a different
    and wrong one: a district in Uttar Pradesh is assigned the whole of UP's GDP.

    SOL is extensive, so the state aggregate is the sum of its districts.

    ``require_balanced`` defaults to **False**. The coverage-gap hazard it was
    introduced to guard lives on the light side, and is now asserted directly by
    ``assert_light_panel_balanced``. Applying it here dropped label regions for
    gaps in the *GDP* series, which cannot manufacture fake light growth, and
    excluded three states (Kerala, Puducherry, Telangana) for no benefit.

    The estimator uses explicit fixed-effect dummies, which handle an unbalanced
    panel correctly, so keeping every state costs nothing in the maths.
    """
    key = [c for c in LABEL_KEY if c in df.columns]
    if not key:
        raise ValueError(
            "Frame has no label-region columns (source_region_name/parent_name); "
            "cannot aggregate to label level. Rebuild it with build_training_frame."
        )

    d = df.copy()
    n_dropped_unbalanced = 0
    dropped_label_regions: list[str] = []
    if require_balanced and "region_id" in d.columns:
        n_years = d["year"].nunique()
        per_district = d.groupby("region_id")["year"].nunique()
        complete = set(per_district[per_district == n_years].index)
        n_dropped_unbalanced = int(d["region_id"].nunique() - len(complete))
        kept = d[d["region_id"].isin(complete)]

        # CLAUDE.md lists dropping training regions as an ask-first decision, so
        # this can never be silent. Name every label region lost entirely, on
        # every run, at WARNING.
        before_regions = set(d["source_region_name"].unique()) if key else set()
        after_regions = set(kept["source_region_name"].unique()) if key else set()
        dropped_label_regions = sorted(before_regions - after_regions)

        if n_dropped_unbalanced:
            logger.warning(
                "require_balanced=True drops %d district(s) not observed in all %d "
                "years. The light-side hazard is now asserted directly by "
                "assert_light_panel_balanced; this option is rarely needed.",
                n_dropped_unbalanced,
                n_years,
            )
        if dropped_label_regions:
            logger.warning(
                "DROPPED FROM TRAINING ENTIRELY: %d label region(s) -> %s. "
                "Dropping training regions is an ask-first decision (CLAUDE.md).",
                len(dropped_label_regions),
                ", ".join(dropped_label_regions),
            )
        d = kept

    agg = {c: how for c, how in LABEL_AGG.items() if c in d.columns}
    agg["gdp_constant"] = "first"
    panel = d.groupby(key + ["year"], as_index=False).agg(agg)
    panel["n_districts"] = (
        d.groupby(key + ["year"])["region_id"].nunique().reset_index(drop=True)
        if "region_id" in d.columns
        else 1
    )

    panel["region_id"] = panel[key].astype(str).agg("|".join, axis=1)
    panel["log_sol"] = np.log(panel["sol"].where(panel["sol"] > 0))
    panel["log_gdp"] = np.log(panel["gdp_constant"].where(panel["gdp_constant"] > 0))
    if {"lit_pixels", "valid_pixels"}.issubset(panel.columns):
        panel["lit_share"] = (
            panel["lit_pixels"] / panel["valid_pixels"].where(panel["valid_pixels"] > 0)
        ).clip(upper=1.0)
    if "area_km2" in panel.columns:
        panel["log_sol_per_area"] = np.log(
            (panel["sol"] / panel["area_km2"]).where(panel["sol"] > 0)
        )

    panel.attrs["n_dropped_unbalanced"] = n_dropped_unbalanced
    panel.attrs["dropped_label_regions"] = dropped_label_regions
    logger.info(
        "Label-level panel: %d region-years (%d regions x %d years)",
        len(panel),
        panel["region_id"].nunique(),
        panel["year"].nunique(),
    )
    return panel


def allocate_to_districts(
    districts: pd.DataFrame, state_gdp: pd.DataFrame, share_col: str = "sol"
) -> pd.DataFrame:
    """Split each state's GDP across its districts in proportion to light share.

    This exists because the model cannot predict district GDP directly. It trains
    on ADM1 labels fanned out to ADM2, so its prediction for a district is an
    estimate of that district's *state* total. Publishing those as district GDP
    overstates the national total by the number of districts per state (26x on
    this panel) and inverts per-capita rankings, because a state total divided by
    a large district population looks poor.

    Downscaling instead makes the estimates coherent by construction::

        district_gdp = state_gdp * (district_SOL / sum of district_SOL in state)

    The shares sum to 1 within every state-year, so district estimates sum
    exactly to the state total and the national total is right.

    **What this assumes**: that economic output distributes within a state the
    way light does. That assumption is not testable here, because no district in
    this panel has a published GDP to check it against. It is the standard
    approach in the downscaling literature, and it is an assumption, not a
    result. Districts whose economy is unlike their light footprint (heavy
    subsistence agriculture, or a single large lit industrial site) will be
    misallocated, and nothing in this pipeline will detect it.
    """
    key = [c for c in LABEL_KEY if c in districts.columns]
    if not key:
        raise ValueError("districts frame needs source_region_name/parent_name to allocate.")
    if share_col not in districts.columns:
        raise ValueError(f"districts frame has no '{share_col}' column to allocate on.")

    d = districts.copy()
    if (d[share_col] < 0).any():
        raise ValueError(f"Negative {share_col} cannot be an allocation share.")

    totals = d.groupby(key + ["year"])[share_col].transform("sum")
    zero_total = totals <= 0
    if zero_total.any():
        bad = d.loc[zero_total, key + ["year"]].drop_duplicates()
        raise ValueError(
            f"{len(bad)} state-year(s) have zero total {share_col}, so GDP cannot be "
            f"allocated: {bad.head(3).to_dict(orient='records')}"
        )
    d["allocation_share"] = d[share_col] / totals

    before = len(d)
    merged = d.merge(state_gdp, on=key + ["year"], how="inner", validate="m:1")
    if merged.empty:
        raise ValueError("No overlap between district frame and state GDP frame.")
    logger.info(
        "Allocation: %d of %d district-years matched a state GDP figure", len(merged), before
    )

    for col in ("state_gdp", "state_gdp_lower", "state_gdp_upper"):
        if col in merged.columns:
            merged[col.replace("state_gdp", "district_gdp")] = (
                merged[col] * merged["allocation_share"]
            )

    # Shares must sum to 1 within each state-year, else the national total is wrong.
    sums = merged.groupby(key + ["year"])["allocation_share"].sum()
    worst = float((sums - 1.0).abs().max())
    if worst > 1e-6:
        raise ValueError(
            f"Allocation shares do not sum to 1 within state-years (worst deviation "
            f"{worst:.2e}). District estimates would not sum to the state total."
        )
    return merged


def label_coverage_report(df: pd.DataFrame, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Explain every label region with fewer years than the extraction window.

    A short series is not automatically a defect. Telangana was created on
    2014-06-02, so DOSE correctly carries no 2014 figure; reporting that as
    missing data invites someone to "fix" it by inventing a number. Reasons come
    from ``label_coverage_notes`` in config, and anything unlisted is flagged as
    unexplained so it gets looked at rather than assumed benign.
    """
    if "source_region_name" not in df.columns:
        return []

    notes = cfg.get("label_coverage_notes") or {}
    all_years = {int(y) for y in df["year"].unique()}
    out: list[dict[str, Any]] = []

    for region, grp in df.groupby("source_region_name"):
        observed = {int(y) for y in grp["year"].unique()}
        missing = sorted(int(y) for y in all_years - observed)
        if not missing:
            continue
        note = notes.get(region, {})
        out.append(
            {
                "region": str(region),
                "observed_years": sorted(observed),
                "missing_years": missing,
                "n_label_years": len(observed),
                "correct_by_history": bool(note.get("correct_by_history", False)),
                "reason": str(note.get("reason", "")).strip()
                or "UNEXPLAINED: no entry in label_coverage_notes for this region.",
                "explained": bool(note),
            }
        )
    return sorted(out, key=lambda r: r["region"])


@dataclass
class PanelResult:
    within_elasticity: float
    within_se: float
    cross_elasticity: float
    cross_se: float
    n_obs: int
    n_regions: int
    n_years: int
    within_rsquared: float
    level: str = "label"
    series: str = "unknown"
    # CONTAMINATED DIAGNOSTIC, NOT A VALIDATION SIGNAL.
    #
    # Entity-FE-only keeps the common national trend, which on this panel means
    # it also keeps the 2016/2017 VIIRS calibration step. It lands near 0.35,
    # close to the literature's ~0.3, purely by coincidence: the artefact happens
    # to have about the right size. The proof is the sub-period split below,
    # where it flips sign (-0.20 before the break, +0.35 after). A structural
    # elasticity does not flip sign at a calendar boundary.
    #
    # Never quote this as agreement with the literature and never assert it
    # against the plausible band. See test_entity_only_is_never_the_headline.
    within_entity_only: float = float("nan")
    within_entity_only_se: float = float("nan")
    # Sub-period entity-only estimates either side of the calibration break,
    # carried permanently so the sign flip is visible to anyone reading this.
    entity_only_pre_break: float = float("nan")
    entity_only_post_break: float = float("nan")
    break_year: int | None = None
    # Label regions lost to the balanced-panel requirement (CLAUDE.md: dropping
    # training regions is an ask-first decision, so it is never silent).
    dropped_label_regions: list[str] = field(default_factory=list)
    n_dropped_districts: int = 0
    # Label regions with a short series, each with a reason and whether the gap
    # is an administrative fact rather than missing data.
    label_coverage: list[dict[str, Any]] = field(default_factory=list)

    def render(self) -> str:
        flip = ""
        if self.break_year and np.isfinite(self.entity_only_pre_break):
            flip = (
                f"\n      sub-period: {self.entity_only_pre_break:+.3f} before "
                f"{self.break_year}, {self.entity_only_post_break:+.3f} after"
            )
            if self.entity_only_pre_break * self.entity_only_post_break < 0:
                flip += "  <- SIGN FLIP: artefact, not an elasticity"
        dropped = ""
        if self.dropped_label_regions:
            dropped = (
                f"\n  DROPPED from training: {len(self.dropped_label_regions)} label region(s) "
                f"({', '.join(self.dropped_label_regions)}) and "
                f"{self.n_dropped_districts} district(s), for incomplete year coverage"
            )
        if self.label_coverage:
            dropped += "\n  Short label series (all retained):"
            for row in self.label_coverage:
                tag = "by history" if row["correct_by_history"] else "missing data"
                dropped += (
                    f"\n    {row['region']}: {row['n_label_years']} year(s), "
                    f"missing {row['missing_years']} [{tag}]"
                )
        return (
            f"Panel log-log elasticity of log GDP on log SOL  "
            f"[fitted at {self.level} level, {self.series}]\n"
            f"  within (region+year FE)  {self.within_elasticity:+.3f} "
            f"(se {self.within_se:.3f})   [expect ~0.3]\n"
            f"  cross-sectional (pooled) {self.cross_elasticity:+.3f} "
            f"(se {self.cross_se:.3f})   [expect 0.6-1.0]\n"
            f"  [CONTAMINATED, not a validation signal]\n"
            f"    within (region FE only) {self.within_entity_only:+.3f} "
            f"(se {self.within_entity_only_se:.3f}) "
            f"- retains the {self.break_year or 'calibration'} product step{flip}\n"
            f"  n={self.n_obs}, regions={self.n_regions}, years={self.n_years}, "
            f"within R2={self.within_rsquared:.3f}{dropped}"
        )


def fit_panel(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    level: str = "label",
    series: str | None = None,
) -> PanelResult:
    """Fit log(gdp) ~ log(sol) with region+year FE (within) and pooled (cross).

    ``level="label"`` (the default, and the one whose elasticity is reported)
    aggregates to the label's own admin level first. Anything else regresses a
    varying regressor on an outcome that is constant within group.

    ``level="district"`` is available as a clearly-labelled *separate*
    specification. It fits on the fanned-out rows with standard errors clustered
    by label region, because the rows are not independent. Its point estimate is
    still attenuated by construction; it is diagnostic, not the headline.
    """
    import statsmodels.api as sm
    from linearmodels.panel import PanelOLS

    if level not in {"label", "district"}:
        raise ValueError(f"level must be 'label' or 'district', got {level!r}")

    # The coverage-gap hazard is on the light side. Assert it there, hard, before
    # anything is aggregated or dropped.
    assert_light_panel_balanced(df)

    coverage = label_coverage_report(df, cfg)
    for row in coverage:
        if row["correct_by_history"]:
            logger.info(
                "%s has %d of %d label-years (missing %s). This is correct: %s",
                row["region"],
                row["n_label_years"],
                row["n_label_years"] + len(row["missing_years"]),
                row["missing_years"],
                row["reason"].split(".")[0],
            )
        else:
            logger.warning(
                "%s has an incomplete label series: %d year(s), missing %s. %s",
                row["region"],
                row["n_label_years"],
                row["missing_years"],
                row["reason"],
            )

    if level == "label":
        df = aggregate_to_label_level(df)
    dropped_label_regions = list(df.attrs.get("dropped_label_regions", []))
    n_dropped_districts = int(df.attrs.get("n_dropped_unbalanced", 0))

    d = ensure_target(df).dropna(subset=[TARGET, "log_sol"]).copy()
    if d["region_id"].nunique() < 2 or d["year"].nunique() < 2:
        raise ValueError("Panel needs at least 2 regions and 2 years for two-way FE.")

    # Guard against a fan-out bug silently reappearing: at label level the
    # regression must use exactly one row per label region-year.
    if level == "label":
        expected = d.groupby(["region_id", "year"]).ngroups
        if len(d) != expected:
            raise ValueError(
                f"Label-level panel has {len(d)} rows but only {expected} distinct "
                "region-years. The frame is still fanned out; fit_panel would "
                "regress state GDP on district SOL."
            )

    panel = d.set_index(["region_id", "year"])
    within = PanelOLS(
        panel[TARGET], panel[["log_sol"]], entity_effects=True, time_effects=True
    ).fit(cov_type="clustered", cluster_entity=True)
    # Entity FE only. On a 6-year panel the year effects absorb the common
    # national trend, which is most of the identifying variation, so this is
    # reported alongside. It is NOT protected from a product-level calibration
    # step the way the two-way estimator is.
    entity_only = PanelOLS(panel[TARGET], panel[["log_sol"]], entity_effects=True).fit(
        cov_type="clustered", cluster_entity=True
    )

    # Sub-period entity-only, either side of the known calibration break. This is
    # the diagnostic that exposes the contamination: a structural elasticity is
    # stable across a calendar boundary, an artefact-driven one flips sign.
    break_year = cfg.get("monthly_break_year")
    pre = post = float("nan")
    if break_year:
        break_year = int(break_year)
        for lo_hi, slot in (("pre", "pre"), ("post", "post")):
            sub = d[d["year"] < break_year] if lo_hi == "pre" else d[d["year"] >= break_year]
            if sub["year"].nunique() >= 2 and sub["region_id"].nunique() >= 2:
                s = sub.set_index(["region_id", "year"])
                try:
                    r = PanelOLS(s[TARGET], s[["log_sol"]], entity_effects=True).fit(
                        cov_type="clustered", cluster_entity=True
                    )
                    value = float(r.params["log_sol"])
                except Exception as exc:  # noqa: BLE001 - diagnostic must not crash the fit
                    logger.warning("Sub-period %s fit failed: %s", slot, exc)
                    value = float("nan")
                if slot == "pre":
                    pre = value
                else:
                    post = value

    if np.isfinite(pre) and np.isfinite(post) and pre * post < 0:
        logger.warning(
            "Entity-only elasticity flips sign across %s (%.3f -> %.3f). This is a "
            "product calibration artefact, not a structural elasticity. Do not "
            "report within_entity_only as agreement with the literature.",
            break_year,
            pre,
            post,
        )

    x = sm.add_constant(d[["log_sol"]])
    if level == "district" and "source_region_name" in d.columns:
        cross = sm.OLS(d[TARGET], x).fit(
            cov_type="cluster", cov_kwds={"groups": d["source_region_name"]}
        )
    else:
        cross = sm.OLS(d[TARGET], x).fit(cov_type="HC1")

    return PanelResult(
        within_elasticity=float(within.params["log_sol"]),
        within_se=float(within.std_errors["log_sol"]),
        cross_elasticity=float(cross.params["log_sol"]),
        cross_se=float(cross.bse["log_sol"]),
        n_obs=int(d.shape[0]),
        n_regions=int(d["region_id"].nunique()),
        n_years=int(d["year"].nunique()),
        within_rsquared=float(within.rsquared_within),
        level=level,
        series=series or str(df.attrs.get("series", "unknown")),
        within_entity_only=float(entity_only.params["log_sol"]),
        within_entity_only_se=float(entity_only.std_errors["log_sol"]),
        entity_only_pre_break=pre,
        entity_only_post_break=post,
        break_year=break_year if break_year else None,
        dropped_label_regions=dropped_label_regions,
        n_dropped_districts=n_dropped_districts,
        label_coverage=coverage,
    )


# --------------------------------------------------------------------------
# XGBoost
# --------------------------------------------------------------------------


@dataclass
class ModelResult:
    model: Any
    features: list[str]
    calib_residuals: np.ndarray
    n_train: int
    oof_r2: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.features])


def _xgb_params(cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "max_depth": 4,
        "n_estimators": 400,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
    }
    defaults.update((cfg.get("model") or {}).get("xgb") or {})
    return defaults


def fit_xgboost(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> ModelResult:
    """Fit a shallow XGBoost regressor with conformal calibration on spatial OOF.

    Calibration residuals come from out-of-fold spatial predictions, so each is a
    residual from a model that never saw that region. That is what makes the
    interval a genuine out-of-sample band rather than an in-sample illusion.
    """
    from xgboost import XGBRegressor

    features = model_features(cfg)
    d = ensure_target(df).dropna(subset=[TARGET]).reset_index(drop=True)
    missing = [f for f in features if f not in d.columns]
    if missing:
        raise ValueError(f"Feature panel is missing columns {missing}.")

    X, y = d[features], d[TARGET].to_numpy()
    params = _xgb_params(cfg)

    if splits is None:
        n_splits = int((cfg.get("model") or {}).get("n_spatial_splits", 5))
        splits = list(spatial_splits(d, n_splits=n_splits))

    # Out-of-fold predictions for conformal calibration and an honest OOF R2.
    oof = np.full(len(d), np.nan)
    for train_idx, test_idx in splits:
        fold = XGBRegressor(**params)
        fold.fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = fold.predict(X.iloc[test_idx])

    mask = ~np.isnan(oof)
    calib_residuals = np.abs(y[mask] - oof[mask])
    oof_r2 = _r2(y[mask], oof[mask]) if mask.sum() > 2 else None

    final = XGBRegressor(**params)
    final.fit(X, y)

    return ModelResult(
        model=final,
        features=features,
        calib_residuals=calib_residuals,
        n_train=len(d),
        oof_r2=oof_r2,
        params=params,
    )


def predict_with_intervals(model: ModelResult, X: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    """Point estimate plus a split-conformal interval at level ``1 - alpha``.

    The half-width is the ``1 - alpha`` quantile of the absolute OOF residuals, so
    a 90% interval (alpha=0.1) is calibrated to cover the truth 90% of the time on
    data drawn like the spatial holdout.
    """
    point = model.predict(X)
    if len(model.calib_residuals) == 0:
        half = np.full(len(X), np.nan)
    else:
        q = float(np.quantile(model.calib_residuals, 1 - alpha, method="higher"))
        half = np.full(len(X), q)
    return pd.DataFrame(
        {
            "prediction": point,
            "lower": point - half,
            "upper": point + half,
        },
        index=X.index,
    )


# --------------------------------------------------------------------------
# helpers and persistence
# --------------------------------------------------------------------------


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def save_model(model: ModelResult, path: str | Path) -> None:
    """Persist a fitted ModelResult (booster plus calibration) to disk."""
    import pickle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(model, fh)
    logger.info("Saved model to %s", path)


def load_model(path: str | Path) -> ModelResult:
    import pickle

    with Path(path).open("rb") as fh:
        return pickle.load(fh)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    """Fit the panel baseline and XGBoost, save the model, print the diagnostics.

    Deliberately prints the elasticity first. It is the number that says whether
    anything downstream is worth reading.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Fit the GDP proxy models")
    parser.add_argument("--country", default=None)
    parser.add_argument(
        "--series", default=None, choices=["annual", "monthly"], help="override training_series"
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--district-spec",
        action="store_true",
        help="also report the district-level fit (clustered SEs) as a separate specification",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from .config import country_config
    from .evaluate import (
        conformal_coverage,
        elasticity_check,
        spatial_holdout_metrics,
        temporal_holdout_metrics,
    )
    from .pipeline import MODEL_PATH, build_training_frame

    cfg = country_config(args.country)
    if args.series:
        cfg = {**cfg, "training_series": args.series}
    series = str(cfg.get("training_series", "monthly"))

    frame = build_training_frame(cfg)

    # 1. Panel baseline, fitted at the label's own admin level.
    panel = fit_panel(frame, cfg, level="label", series=series)
    print()
    print(panel.render())
    print()
    for name, passed, detail in elasticity_check(panel, cfg):
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    if args.district_spec:
        district = fit_panel(frame, cfg, level="district", series=series)
        print()
        print("Separate specification, NOT the headline estimate:")
        print(district.render())
        print("  Attenuated by construction: districts share one state label.")

    # 2. Honest holdouts on the district frame.
    print()
    spatial = spatial_holdout_metrics(frame, cfg)
    temporal = temporal_holdout_metrics(frame, cfg)
    print(
        f"Spatial holdout:  log R2 {spatial['log_r2']:+.3f}, "
        f"level R2 {spatial['level_r2']:+.3f}, n {spatial['log_n']}"
    )
    print(
        f"Temporal holdout: log R2 {temporal['log_r2']:+.3f}, "
        f"level R2 {temporal['level_r2']:+.3f}, n {temporal['log_n']}"
    )
    if spatial["log_r2"] > 0.95:
        print("  WARNING: spatial R2 above 0.95 on a panel this small usually means a leak.")

    # 3. Intervals.
    emp, nom = conformal_coverage(frame, cfg)
    print(f"Interval coverage: {emp:.2f} empirical vs {nom:.2f} nominal")

    # 4. Fit and persist the shipping model.
    #
    # Go through the properly-imported module, not the local names. Under
    # ``python -m gdp_proxy.model`` this file is loaded as ``__main__``, so a
    # ModelResult built from the local class pickles as ``__main__.ModelResult``
    # and cannot be unpickled by anything else, including the pipeline.
    import gdp_proxy.model as _model

    model = _model.fit_xgboost(frame, cfg)
    print(f"XGBoost OOF (spatial) R2: {model.oof_r2:+.3f}" if model.oof_r2 is not None else "")
    if not args.no_save:
        _model.save_model(model, MODEL_PATH)
        # Round-trip immediately: fail here rather than in the pipeline later.
        reloaded = _model.load_model(MODEL_PATH)
        if type(reloaded).__module__ != "gdp_proxy.model":
            raise RuntimeError(
                f"Model pickled under {type(reloaded).__module__!r}; it would not "
                "load outside this entrypoint."
            )
        print(f"Saved model to {MODEL_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
