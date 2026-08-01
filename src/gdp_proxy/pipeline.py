"""Phase 6a: monthly orchestration.

One idempotent entrypoint that can run on a schedule and does nothing when there
is nothing new. NOAA posts a monthly VIIRS composite roughly three to six weeks
after the month ends, so a weekly run that no-ops most of the time is the right
cadence.

The ``--check`` path is deliberately cheap: a single ``aggregate_max`` on
``system:time_start``, a scalar, which is the one ``getInfo`` this project allows
(rule 1). It never filters and counts images to answer "is there a new month".

Every stage checks whether its output already exists and skips (rule 4), so a
crashed run is safe to re-run. The estimates table carries a ``vintage`` column:
when a month is re-run because a composite was reprocessed, both rows are kept,
because someone will eventually ask why a number changed.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from .config import DATA_DIR, ConfigError, country_config

logger = logging.getLogger(__name__)

PROCESSED_DIR = DATA_DIR / "processed"
ESTIMATES_PATH = PROCESSED_DIR / "estimates.parquet"
MODEL_PATH = PROCESSED_DIR / "model.pkl"


# --------------------------------------------------------------------------
# pure decision logic (offline-testable)
# --------------------------------------------------------------------------


def needs_run(available: date, processed: date | None) -> bool:
    """True when a composite newer than the latest processed month exists."""
    return processed is None or available > processed


def affected_years(available: date, processed: date | None, start_year: int) -> list[int]:
    """Years whose features must be rebuilt given a newly available month.

    A new month only changes its own year's annual aggregate, so the affected set
    is just that year, unless nothing has ever been processed, in which case the
    whole series from start_year is in scope.
    """
    if processed is None:
        return list(range(start_year, available.year + 1))
    if available <= processed:
        return []
    return list(range(processed.year, available.year + 1))


# --------------------------------------------------------------------------
# state of the world
# --------------------------------------------------------------------------


def latest_available_month(cfg: dict[str, Any]) -> date:
    """The newest VIIRS composite date. One scalar getInfo, the cheap check."""
    import ee

    from .auth import init_ee

    init_ee()
    coll = ee.ImageCollection(cfg["viirs_monthly"])
    ms = coll.aggregate_max("system:time_start").getInfo()
    if ms is None:
        raise ConfigError(f"{cfg['viirs_monthly']} returned no images.")
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return date(dt.year, dt.month, 1)


def latest_processed_month(cfg: dict[str, Any]) -> date | None:
    """The newest month present in the extracted monthly parquets, or None."""
    stem = f"sol_{cfg['country']}_adm{cfg['admin_level']}_"
    paths = sorted(PROCESSED_DIR.glob(f"{stem}*.parquet"))
    if not paths:
        return None
    latest: date | None = None
    for path in paths:
        try:
            df = pd.read_parquet(path, columns=["year", "month"])
        except Exception as exc:  # noqa: BLE001 - a corrupt file should not crash --check
            logger.warning("Could not read %s: %s", path.name, exc)
            continue
        if df.empty:
            continue
        y = int(df["year"].max())
        m = int(df.loc[df["year"] == y, "month"].max())
        cand = date(y, m, 1)
        if latest is None or cand > latest:
            latest = cand
    return latest


# --------------------------------------------------------------------------
# training frame (features + labels via the reviewed crosswalk)
# --------------------------------------------------------------------------


def build_training_frame(cfg: dict[str, Any]) -> pd.DataFrame:
    """Join the feature panel to deflated labels through the committed crosswalk.

    ``apply_crosswalk`` fans each ADM1 label out to the ADM2 districts it covers,
    so the label frame arriving here is already keyed by district-year and the
    join below is 1:1 on ``region_id`` + ``year``.

    Note what this means statistically: districts in the same state share one GDP
    value, so they are not independent observations. ``n_districts_in_label`` is
    carried through so the modelling step can weight or aggregate rather than
    treating N copies of a state total as N measurements. Every join asserts its
    row count (rule 6), and a silent empty join is refused outright.
    """
    from .features import build_features
    from .labels import load_labels
    from .match import apply_crosswalk, load_crosswalk
    from .model import assert_light_panel_balanced

    features = build_features(cfg)
    # Assert on the raw light panel, before the label join. After the join a
    # district legitimately loses years wherever its state's GDP series is short
    # (Telangana has no 2014 because it did not exist), which would mask a real
    # coverage gap behind a label gap.
    assert_light_panel_balanced(features)

    labels = load_labels(cfg)
    crosswalk = load_crosswalk(cfg)
    labels = apply_crosswalk(labels, crosswalk)  # attaches region_id, raises on gaps

    labels = labels.dropna(subset=["gdp_constant"])
    key = ["region_id", "year"]
    labels["year"] = labels["year"].astype(int)

    # Carry the label region through so the panel baseline can be fitted at the
    # level the label actually varies at. See aggregate_to_label_panel.
    carry = [
        c
        for c in ("gdp_constant", "n_districts_in_label", "source_region_name", "parent_name")
        if c in labels.columns
    ]
    n_labels = len(labels)
    frame = features.merge(labels[key + carry], on=key, how="inner", validate="1:1")
    if frame.empty:
        raise ConfigError(
            f"Feature-label join produced zero rows. The region_id spaces do not "
            f"overlap: features are ADM{cfg['admin_level']} districts, labels map to a "
            "different level. Check that the crosswalk assigns label regions to the "
            "same boundary vintage the features were extracted from."
        )
    logger.info(
        "Training frame: %d district-years (from %d label rows, %d feature rows, "
        "%d distinct districts)",
        len(frame),
        n_labels,
        len(features),
        frame["region_id"].nunique(),
    )
    if "n_districts_in_label" in frame.columns:
        shared = int((frame["n_districts_in_label"] > 1).sum())
        if shared:
            logger.warning(
                "%d of %d training rows share a label with other districts "
                "(ADM1 labels fanned out to ADM2). These are not independent "
                "observations; group by label region when splitting.",
                shared,
                len(frame),
            )

    import numpy as np

    frame["log_gdp"] = np.log(frame["gdp_constant"].where(frame["gdp_constant"] > 0))

    # District GDDP is validation-only. If it ever reaches training, the
    # allocation check becomes circular: the split would be fitted to the data
    # used to judge it. Enforced here, not left to a comment.
    from .gddp import assert_no_validation_labels

    assert_no_validation_labels(frame, where="training frame")
    return frame


def aggregate_to_label_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Deprecated alias. The canonical implementation lives in model.py.

    Kept so existing callers keep working; ``fit_panel`` now aggregates itself.
    """
    from .model import aggregate_to_label_level

    return aggregate_to_label_level(frame)


# --------------------------------------------------------------------------
# estimates store
# --------------------------------------------------------------------------


def make_vintage(cfg: dict[str, Any]) -> str:
    """A vintage tag combining the run time and the source series actually used.

    Must reflect the configured ``training_series``, not always the monthly id:
    a vintage naming the wrong product is worse than no vintage, because it is
    the field someone consults to explain why a number moved.
    """
    from .extract import series_of

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    series = series_of(cfg)
    if series == "annual":
        ids = sorted({str(v["id"]) for v in (cfg.get("viirs_annual_versions") or [])}) or [
            str(cfg.get("viirs_annual", "unknown"))
        ]
        return f"{stamp}|annual|{'+'.join(i.split('/')[-1] for i in ids)}"
    return f"{stamp}|monthly|{str(cfg['viirs_monthly']).split('/')[-1]}"


def append_estimates(estimates: pd.DataFrame, cfg: dict[str, Any], vintage: str) -> pd.DataFrame:
    """Append a new batch of estimates under a vintage, never overwriting old rows.

    If an identical (vintage, region_id, year) already exists the append is a
    no-op, so a re-run does not duplicate. A *new* vintage for the same region-year
    is kept alongside the old one on purpose (rule 11 provenance).
    """
    estimates = estimates.copy()
    estimates["vintage"] = vintage
    estimates["run_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if ESTIMATES_PATH.exists():
        prior = pd.read_parquet(ESTIMATES_PATH)
        combined = pd.concat([prior, estimates], ignore_index=True)
        combined = combined.drop_duplicates(subset=["vintage", "region_id", "year"], keep="first")
    else:
        combined = estimates

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(ESTIMATES_PATH, index=False)
    logger.info(
        "Estimates store now holds %d rows across %d vintage(s)",
        len(combined),
        combined["vintage"].nunique(),
    )
    return combined


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


@dataclass
class PipelineResult:
    checked_at: str
    latest_available: date | None
    latest_processed: date | None
    new_data: bool
    action: str
    affected_years: list[int] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"Pipeline run at {self.checked_at}",
            f"  latest available   {self.latest_available}",
            f"  latest processed   {self.latest_processed}",
            f"  new data           {self.new_data}",
            f"  action             {self.action}",
            f"  affected years     {self.affected_years}",
        ]
        for m in self.messages:
            lines.append(f"  - {m}")
        return "\n".join(lines)


def run(cfg: dict[str, Any], check_only: bool = False) -> PipelineResult:
    """Check for a new composite; if there is one and not check-only, process it.

    Processing is staged and each stage skips when its output exists: submit the
    export for the affected year (Earth Engine), then ingest the downloaded CSV,
    rebuild features, run inference, and append to the estimates store under a new
    vintage. Stages that need artefacts not present yet are recorded in messages
    rather than crashing the run.
    """
    result = PipelineResult(
        checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        latest_available=None,
        latest_processed=None,
        new_data=False,
        action="noop",
    )

    result.latest_processed = latest_processed_month(cfg)
    result.latest_available = latest_available_month(cfg)
    result.new_data = needs_run(result.latest_available, result.latest_processed)
    result.affected_years = affected_years(
        result.latest_available, result.latest_processed, int(cfg["start_year"])
    )

    if not result.new_data:
        result.action = "noop"
        result.messages.append("Up to date. Nothing to extract.")
        return result

    if check_only:
        result.action = "new-month-available"
        result.messages.append(
            f"New composite {result.latest_available} available; run without --check to process."
        )
        return result

    result.action = "processed"
    _process(cfg, result)
    return result


def _process(cfg: dict[str, Any], result: PipelineResult) -> None:
    """Drive the affected years through extract -> features -> infer -> append."""
    from .extract import ingest, output_path, submit_year

    for year in result.affected_years:
        if output_path(cfg, year).exists():
            result.messages.append(f"{year}: extraction parquet exists, skipping submit.")
            continue
        try:
            submit_year(cfg, year)
            result.messages.append(f"{year}: export task submitted; awaiting Drive CSV.")
        except Exception as exc:  # noqa: BLE001 - report, do not crash the schedule
            result.messages.append(f"{year}: could not submit export ({exc}).")

    # Ingest any downloaded CSVs that are not yet parquet.
    try:
        from .boundaries import load_snapshot

        expected = len(load_snapshot(cfg["country"]))
        for year in result.affected_years:
            if not output_path(cfg, year).exists():
                try:
                    ingest(cfg, year, expected)
                    result.messages.append(f"{year}: ingested CSV to parquet.")
                except ConfigError:
                    pass  # CSV not downloaded yet; a later run will pick it up
    except ConfigError as exc:
        result.messages.append(f"No boundary snapshot for ingest step: {exc}")

    _infer(cfg, result)


def _infer(cfg: dict[str, Any], result: PipelineResult) -> None:
    """Build features, predict with intervals, and append to the estimates store."""
    try:
        from .evaluate import predict_panel
        from .features import build_features
        from .model import load_model
    except ImportError as exc:  # pragma: no cover
        result.messages.append(f"Inference deps unavailable: {exc}")
        return

    if not MODEL_PATH.exists():
        result.messages.append(
            f"No trained model at {MODEL_PATH}; train one before inference "
            "(python -m gdp_proxy.model_train or the notebook)."
        )
        return

    # A model trained on V21 must not predict from V22 without checking that the
    # two versions are on the same scale. There is no overlapping year to
    # calibrate against, so the continuous monthly series is used as a bridge.
    # A silent offset would shift every district estimate by a constant factor
    # and look entirely plausible.
    try:
        from .features import assert_version_boundary_consistent

        boundary = assert_version_boundary_consistent(cfg)
        result.messages.append(
            f"Version boundary OK: annual/monthly ratio within "
            f"[{boundary['lower_bound']:.3f}, {boundary['upper_bound']:.3f}] "
            f"for {boundary['checked_years']}"
        )
    except ConfigError as exc:
        result.messages.append(f"Version-boundary check could not run or failed: {exc}")
        raise

    try:
        features = build_features(cfg)
    except ConfigError as exc:
        result.messages.append(f"Cannot build features yet: {exc}")
        return

    # District GDP is an ALLOCATION, not a direct prediction. The model predicts
    # a district's state total (its label is the state's GDP), so publishing the
    # raw prediction per district overstates the national total by the number of
    # districts per state. Downscale the state total by light share instead.
    from .model import LABEL_KEY, allocate_to_districts

    model = load_model(MODEL_PATH)
    conformal = predict_panel(model, features, cfg)

    states = state_gdp_by_year(cfg, features)
    # Carry the interval as a relative band around the state total, using the
    # conformal half-width the model produced on the log scale.
    half = (conformal["upper"] - conformal["prediction"]).median()
    states["state_gdp_lower"] = states["state_gdp"] * float(_safe_exp(-half))
    states["state_gdp_upper"] = states["state_gdp"] * float(_safe_exp(half))

    region_to_state = _region_to_state(cfg)
    dist = features.merge(region_to_state, on="region_id", how="inner")
    allocated = allocate_to_districts(dist, states, share_col="sol")

    preds = allocated[
        ["region_id", "year", "allocation_share", "state_gdp_source"]
        + [c for c in LABEL_KEY if c in allocated.columns]
    ].copy()
    # allocation_share is the RAW light share. No bias correction is applied:
    # a one-parameter power correction was fitted and rejected by
    # leave-one-state-out (see diagnostics share_correction). Keeping the raw
    # share means the allocation can always be reconstructed from the state
    # total, whatever is decided later.
    preds["allocation_share_uncorrected"] = allocated["allocation_share"]
    preds["share_correction"] = "none"
    preds["gdp_estimate"] = allocated["district_gdp"]
    preds["gdp_lower"] = allocated["district_gdp_lower"]
    preds["gdp_upper"] = allocated["district_gdp_upper"]
    preds["series"] = str(cfg.get("training_series", "monthly"))
    preds = _add_per_capita(preds, features)
    preds = _add_extrapolation_flags(preds, cfg)

    append_estimates(preds, cfg, make_vintage(cfg))
    n_extrap = int(preds["extrapolation_flag"].sum())
    n_pred_state = int((preds["state_gdp_source"] == "predicted").sum())
    result.messages.append(
        f"Inference appended {len(preds)} district estimates by light-share "
        f"allocation ({n_extrap} flagged extrapolated, {n_pred_state} from a "
        "predicted rather than published state total)."
    )


def _region_to_state(cfg: dict[str, Any]) -> pd.DataFrame:
    """region_id -> label region, from the committed crosswalk."""
    from .match import load_crosswalk

    cw = load_crosswalk(cfg)
    cw = cw[cw["source_region_name"] != ""]
    return cw[["region_id", "source_region_name", "parent_name"]].drop_duplicates("region_id")


def state_gdp_by_year(cfg: dict[str, Any], features: pd.DataFrame) -> pd.DataFrame:
    """State GDP per year: the published label where it exists, else predicted.

    Downscaling needs a state total to split. For years the label series covers
    it uses the published figure directly, which is ground truth and needs no
    model. For later years (DOSE ends 2019) it predicts the state total from the
    cross-sectional log-log relationship fitted at state level, which is the
    elasticity this panel actually identifies (~0.81).

    ``state_gdp_source`` records which, per row, because a downscaled published
    figure and a downscaled prediction are different products and should never
    be read as the same thing.
    """
    import numpy as np

    from .labels import load_labels
    from .match import apply_crosswalk, load_crosswalk
    from .model import LABEL_KEY, aggregate_to_label_level

    key = list(LABEL_KEY)

    # Observed state GDP, keyed by label region and year.
    labels = apply_crosswalk(load_labels(cfg), load_crosswalk(cfg))
    labels = labels.dropna(subset=["gdp_constant"])
    labels["year"] = labels["year"].astype(int)
    observed = (
        labels.groupby(key + ["year"], as_index=False)["gdp_constant"]
        .first()
        .rename(columns={"gdp_constant": "state_gdp"})
    )
    observed["state_gdp_source"] = "published"

    # State-level light panel for every year we have features for, including the
    # years with no label at all.
    frame = features.merge(
        labels[key + ["region_id", "year"]].drop_duplicates(),
        on=["region_id", "year"],
        how="left",
    )
    # Districts keep their label region across all years, including unlabelled ones.
    region_to_state = (
        labels[["region_id"] + key].drop_duplicates("region_id").set_index("region_id")
    )
    for col in key:
        frame[col] = frame["region_id"].map(region_to_state[col])
    frame = frame.dropna(subset=key)
    frame["gdp_constant"] = np.nan
    state_panel = aggregate_to_label_level(frame, require_balanced=False)

    fitted = state_panel.merge(observed, on=key + ["year"], how="left")
    have = fitted["state_gdp"].notna()
    if have.sum() < 10:
        raise ConfigError(
            f"Only {int(have.sum())} state-years have a published GDP; too few to "
            "calibrate the level for unlabelled years."
        )

    # Cross-sectional fit on the state panel: log(gdp) = a + b*log(sol).
    import statsmodels.api as sm

    train = fitted[have & fitted["log_sol"].notna()]
    ols = sm.OLS(np.log(train["state_gdp"]), sm.add_constant(train[["log_sol"]])).fit(
        cov_type="HC1"
    )
    logger.info(
        "State-level level model: log_gdp = %.3f + %.3f*log_sol (n=%d)",
        float(ols.params["const"]),
        float(ols.params["log_sol"]),
        int(ols.nobs),
    )

    predicted = np.exp(ols.predict(sm.add_constant(fitted[["log_sol"]], has_constant="add")))
    fitted["state_gdp"] = fitted["state_gdp"].where(have, predicted)
    fitted["state_gdp_source"] = fitted["state_gdp_source"].fillna("predicted")

    n_pred = int((fitted["state_gdp_source"] == "predicted").sum())
    if n_pred:
        logger.warning(
            "%d state-year(s) have no published GDP and use a predicted state total; "
            "those district estimates are model output, not downscaled measurement.",
            n_pred,
        )
    return fitted[key + ["year", "state_gdp", "state_gdp_source"]]


def _add_per_capita(preds: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Divide the estimate and both interval bounds by district population.

    Published alongside the total, never instead of it, because the two rank
    districts very differently and each answers a different question. Total says
    how much output a district produces; per capita says how well off the people
    in it are. A populous poor district is high on one and low on the other, and
    conflating them is exactly the error the face-validity check surfaced.

    The interval is scaled by the same denominator, so its coverage carries over
    unchanged: dividing a bound by a positive constant preserves containment.
    """
    pop_cols = [
        c
        for c in (
            "region_id",
            "year",
            "population",
            "population_interpolated",
            "population_is_projection",
        )
        if c in features.columns
    ]
    if "population" not in pop_cols:
        logger.warning("No population column; per-capita estimates will be NaN.")
        for col in ("gdp_per_capita_estimate", "gdp_per_capita_lower", "gdp_per_capita_upper"):
            preds[col] = float("nan")
        return preds

    before = len(preds)
    out = preds.merge(features[pop_cols], on=["region_id", "year"], how="left")
    if len(out) != before:
        raise ValueError(f"Population join changed row count {before} -> {len(out)}")

    denom = out["population"].where(out["population"] > 0)
    out["gdp_per_capita_estimate"] = out["gdp_estimate"] / denom
    out["gdp_per_capita_lower"] = out["gdp_lower"] / denom
    out["gdp_per_capita_upper"] = out["gdp_upper"] / denom

    n_missing = int(denom.isna().sum())
    if n_missing:
        logger.warning("%d region-years have no usable population denominator", n_missing)
    return out


def _add_extrapolation_flags(preds: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Mark how far each estimate sits from anything the model actually saw.

    Two distinct kinds of extrapolation, kept separate because they have
    different severity:

    ``extrapolated_admin`` is True for every row, and says so honestly: the model
    trains on ADM1 state labels and predicts ADM2 districts, a roughly twentyfold
    resolution jump. No district in this project has its own published GDP.

    ``extrapolated_year`` is True where the year lies outside the label window
    entirely (2024 and 2025 have no GDP labels at all, since DOSE ends 2019), so
    the estimate rests on the light-to-GDP relationship holding five years beyond
    anything observed.

    ``extrapolation_flag`` is their union, which is what the dashboard reads.
    """
    out = preds.copy()
    try:
        from .labels import load_labels

        label_years = load_labels(cfg)["year"].dropna().astype(int)
        lo, hi = int(label_years.min()), int(label_years.max())
    except Exception as exc:  # noqa: BLE001 - flagging must not break inference
        logger.warning("Could not read label year range for extrapolation flags: %s", exc)
        lo = hi = None

    out["extrapolated_admin"] = True
    if lo is not None:
        out["extrapolated_year"] = (out["year"] < lo) | (out["year"] > hi)
        out["label_year_range"] = f"{lo}-{hi}"
    else:
        out["extrapolated_year"] = False
        out["label_year_range"] = "unknown"

    out["extrapolation_flag"] = out["extrapolated_admin"] | out["extrapolated_year"]
    return out


def _safe_exp(x: float) -> float:
    import numpy as np

    return float(np.exp(x)) if pd.notna(x) else float("nan")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Monthly VIIRS -> GDP estimate pipeline")
    parser.add_argument("--country", default=None)
    parser.add_argument("--check", action="store_true", help="cheap check for a new month; no work")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = country_config(args.country)

    result = run(cfg, check_only=args.check)
    print()
    print(result.render())
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
