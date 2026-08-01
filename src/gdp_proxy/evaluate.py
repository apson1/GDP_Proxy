"""Phase 5b: evaluation.

Kept separate from ``model.py`` so the scoring code cannot be quietly tuned to
flatter the fit. Every metric is reported on two scales: the log scale the model
optimises, and the level (rupee) scale after exponentiating, because a good log
R2 can still hide large absolute errors in the biggest districts.

The spatial-holdout R2 here is a leak detector as much as a quality bar. If it
comes back above 0.95 on a panel this small, the first hypothesis is a leak, not
a breakthrough (pitfalls.md).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .model import (
    ModelResult,
    ensure_target,
    fit_xgboost,
    model_features,
    predict_with_intervals,
    spatial_splits,
    temporal_splits,
)

logger = logging.getLogger(__name__)

TARGET = "log_gdp"


# --------------------------------------------------------------------------
# metric primitives
# --------------------------------------------------------------------------


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        f"{prefix}_r2": float(r2),
        f"{prefix}_rmse": float(np.sqrt(np.mean(err**2))),
        f"{prefix}_mae": float(np.mean(np.abs(err))),
        f"{prefix}_n": int(len(y_true)),
    }


def _holdout(
    df: pd.DataFrame, cfg: dict[str, Any], splits: list[tuple[np.ndarray, np.ndarray]]
) -> dict[str, float]:
    """Train per fold, collect out-of-fold predictions, score on log and level scales."""
    from xgboost import XGBRegressor

    from .model import _xgb_params

    features = model_features(cfg)
    d = ensure_target(df).dropna(subset=[TARGET]).reset_index(drop=True)
    X, y = d[features], d[TARGET].to_numpy()
    params = _xgb_params(cfg)

    oof = np.full(len(d), np.nan)
    for train_idx, test_idx in splits:
        m = XGBRegressor(**params)
        m.fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = m.predict(X.iloc[test_idx])

    mask = ~np.isnan(oof)
    out = _metrics(y[mask], oof[mask], "log")
    # Level scale: a strong log R2 can still mean big rupee errors in large districts.
    out.update(_metrics(np.exp(y[mask]), np.exp(oof[mask]), "level"))
    return out


def spatial_holdout_metrics(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, float]:
    """Score by GroupKFold on region_id: performance on districts never seen."""
    d = ensure_target(df).dropna(subset=[TARGET]).reset_index(drop=True)
    n_splits = int((cfg.get("model") or {}).get("n_spatial_splits", 5))
    return _holdout(d, cfg, list(spatial_splits(d, n_splits=n_splits)))


def temporal_holdout_metrics(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, float]:
    """Score by forward chaining: performance on the next unseen year."""
    d = ensure_target(df).dropna(subset=[TARGET]).reset_index(drop=True)
    min_train = int((cfg.get("model") or {}).get("min_train_years", 4))
    return _holdout(d, cfg, list(temporal_splits(d, min_train_years=min_train)))


# --------------------------------------------------------------------------
# elasticity gate
# --------------------------------------------------------------------------


def elasticity_check(panel_result, cfg: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Check both elasticities against their plausible bands (rule 14)."""
    mcfg = cfg.get("model") or {}
    lo_w, hi_w = mcfg.get("within_elasticity_band", [0.15, 0.6])
    lo_c, hi_c = mcfg.get("cross_elasticity_band", [0.6, 1.0])

    w = panel_result.within_elasticity
    c = panel_result.cross_elasticity
    return [
        (
            "within elasticity in band",
            lo_w <= w <= hi_w,
            f"{w:.3f} (se {panel_result.within_se:.3f}); expect [{lo_w}, {hi_w}]. "
            "Below suggests the join lost signal or labels are nominal; above ~2 means units.",
        ),
        (
            "cross-sectional elasticity in band",
            lo_c <= c <= hi_c,
            f"{c:.3f} (se {panel_result.cross_se:.3f}); expect [{lo_c}, {hi_c}].",
        ),
    ]


# --------------------------------------------------------------------------
# district (ADM2) out-of-sample validation
# --------------------------------------------------------------------------


def district_validation(preds: pd.DataFrame, gddp_labels: pd.DataFrame) -> dict[str, Any]:
    """Score predictions against the districts that DO have published GDDP.

    This subset is the only real out-of-sample evidence for the ADM1->ADM2
    extrapolation and must never be in the training set. ``preds`` needs
    region_id, year, prediction (log scale); ``gddp_labels`` needs region_id,
    year, gdp_constant.
    """
    key = ["region_id", "year"]
    before = len(gddp_labels)
    merged = gddp_labels.merge(preds, on=key, how="inner", validate="1:1")
    if merged.empty:
        return {"n": 0, "note": "no overlap between predictions and district GDDP"}
    lost = before - len(merged)

    y_true = np.log(merged["gdp_constant"].where(merged["gdp_constant"] > 0))
    valid = ~y_true.isna()
    out = _metrics(y_true[valid].to_numpy(), merged.loc[valid, "prediction"].to_numpy(), "district")
    out["n_matched"] = int(valid.sum())
    out["n_gddp_unmatched"] = int(lost)
    return out


# --------------------------------------------------------------------------
# interval coverage
# --------------------------------------------------------------------------


def interval_coverage(preds: pd.DataFrame, actuals: np.ndarray, nominal: float = 0.90) -> float:
    """Fraction of actuals falling inside [lower, upper].

    A 90% interval that covers 60% of the time is decoration. This returns the
    empirical coverage so it can be compared to ``nominal`` directly.
    """
    actuals = np.asarray(actuals, dtype=float)
    inside = (actuals >= preds["lower"].to_numpy()) & (actuals <= preds["upper"].to_numpy())
    valid = ~np.isnan(actuals) & ~np.isnan(preds["lower"].to_numpy())
    if valid.sum() == 0:
        return float("nan")
    return float(inside[valid].mean())


def conformal_coverage(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[float, float]:
    """Measured coverage of conformal intervals on the spatial holdout.

    Fits a model on OOF folds, builds intervals for the held-out rows, and reports
    (empirical_coverage, nominal). Uses the same spatial splits so the coverage
    claim is on genuinely unseen districts.
    """
    from xgboost import XGBRegressor

    from .model import _xgb_params

    alpha = float((cfg.get("model") or {}).get("conformal_alpha", 0.1))
    features = model_features(cfg)
    d = ensure_target(df).dropna(subset=[TARGET]).reset_index(drop=True)
    X, y = d[features], d[TARGET].to_numpy()
    params = _xgb_params(cfg)
    n_splits = int((cfg.get("model") or {}).get("n_spatial_splits", 5))
    splits = list(spatial_splits(d, n_splits=n_splits))

    covered: list[bool] = []
    for held_i, (train_idx, test_idx) in enumerate(splits):
        # Calibrate conformal width on the OTHER folds' OOF residuals, then apply
        # to this fold, so calibration and test never share a region.
        inner = [s for j, s in enumerate(splits) if j != held_i]
        calib_resid = []
        for tr, te in inner:
            m = XGBRegressor(**params)
            m.fit(X.iloc[tr], y[tr])
            calib_resid.extend(np.abs(y[te] - m.predict(X.iloc[te])))
        q = float(np.quantile(np.abs(calib_resid), 1 - alpha, method="higher"))

        m = XGBRegressor(**params)
        m.fit(X.iloc[train_idx], y[train_idx])
        pred = m.predict(X.iloc[test_idx])
        lower, upper = pred - q, pred + q
        covered.extend((y[test_idx] >= lower) & (y[test_idx] <= upper))

    return float(np.mean(covered)), 1 - alpha


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def report(
    panel_result,
    spatial: dict[str, float],
    temporal: dict[str, float],
    coverage: tuple[float, float] | None,
    cfg: dict[str, Any],
    district: dict[str, Any] | None = None,
) -> str:
    lines = [panel_result.render(), ""]
    for name, passed, detail in elasticity_check(panel_result, cfg):
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    lines.append("")
    lines.append(
        f"Spatial holdout:  log R2 {spatial['log_r2']:.3f}, "
        f"level R2 {spatial['level_r2']:.3f}, n {spatial['log_n']}"
    )
    lines.append(
        f"Temporal holdout: log R2 {temporal['log_r2']:.3f}, "
        f"level R2 {temporal['level_r2']:.3f}, n {temporal['log_n']}"
    )
    if spatial["log_r2"] > 0.95:
        lines.append("  WARNING: spatial R2 above 0.95 on a small panel usually means a leak.")
    if coverage is not None:
        emp, nom = coverage
        lines.append(f"Interval coverage: {emp:.2f} empirical vs {nom:.2f} nominal")
    if district is not None and district.get("n_matched", 0):
        lines.append(
            f"District (ADM2) validation: log R2 {district['district_r2']:.3f} "
            f"on {district['n_matched']} published-GDDP districts"
        )
    return "\n".join(lines)


def build_model(df: pd.DataFrame, cfg: dict[str, Any]) -> ModelResult:
    """Convenience: fit the final shipping model with conformal calibration."""
    return fit_xgboost(df, cfg)


def predict_panel(model: ModelResult, df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Predict every region-year with intervals and an extrapolation flag."""
    alpha = float((cfg.get("model") or {}).get("conformal_alpha", 0.1))
    intervals = predict_with_intervals(model, df, alpha=alpha)
    out = pd.concat([df[["region_id", "year"]].reset_index(drop=True), intervals], axis=1)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    """Score an existing model and write data/processed/diagnostics.json.

    Separate from ``model`` on purpose: evaluation that lives in the same command
    as fitting tends to acquire knobs that flatter the fit.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate the GDP proxy model")
    parser.add_argument("--country", default=None)
    parser.add_argument(
        "--series", default=None, choices=["annual", "monthly"], help="override training_series"
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from .config import country_config
    from .model import fit_panel
    from .pipeline import PROCESSED_DIR, build_training_frame

    cfg = country_config(args.country)
    if args.series:
        cfg = {**cfg, "training_series": args.series}
    series = str(cfg.get("training_series", "monthly"))

    frame = build_training_frame(cfg)
    panel = fit_panel(frame, cfg, level="label", series=series)
    spatial = spatial_holdout_metrics(frame, cfg)
    temporal = temporal_holdout_metrics(frame, cfg)
    coverage = conformal_coverage(frame, cfg)

    print()
    print(report(panel, spatial, temporal, coverage, cfg))
    print()

    diagnostics = {
        "series": series,
        "panel_level": panel.level,
        "within_elasticity": panel.within_elasticity,
        "within_se": panel.within_se,
        # Deliberately nested under a name that cannot be mistaken for a
        # validation signal. It retains the 2016/2017 calibration step and
        # reproduces the literature's ~0.3 by coincidence; the sub-period sign
        # flip is the proof. Never quote as agreement with published estimates.
        "contaminated_diagnostics": {
            "_warning": (
                "within_entity_only retains the VIIRS calibration discontinuity and "
                "is NOT a validation signal. It flips sign across the break, which a "
                "structural elasticity cannot do. Do not compare it to the literature."
            ),
            "within_entity_only": panel.within_entity_only,
            "within_entity_only_se": panel.within_entity_only_se,
            "entity_only_pre_break": panel.entity_only_pre_break,
            "entity_only_post_break": panel.entity_only_post_break,
            "break_year": panel.break_year,
            "sign_flips_across_break": bool(
                panel.entity_only_pre_break == panel.entity_only_pre_break
                and panel.entity_only_post_break == panel.entity_only_post_break
                and panel.entity_only_pre_break * panel.entity_only_post_break < 0
            ),
        },
        # Regions excluded from training by the balanced-panel requirement.
        # CLAUDE.md treats dropping training regions as ask-first, so this is
        # recorded on every run rather than left to the logs.
        "dropped_label_regions": panel.dropped_label_regions,
        "n_dropped_districts": panel.n_dropped_districts,
        # Every label region with a short series, retained in training, with the
        # reason. correct_by_history=true means the gap is an administrative fact
        # (the region did not exist yet) and must NOT be backfilled: inventing a
        # figure for a state that did not exist would be fabrication.
        "label_coverage": panel.label_coverage,
        "n_unexplained_label_gaps": sum(1 for r in panel.label_coverage if not r["explained"]),
        "cross_elasticity": panel.cross_elasticity,
        "cross_se": panel.cross_se,
        "n_obs": panel.n_obs,
        "n_regions": panel.n_regions,
        "n_years": panel.n_years,
        "spatial_log_r2": spatial["log_r2"],
        "spatial_level_r2": spatial["level_r2"],
        "temporal_log_r2": temporal["log_r2"],
        "temporal_level_r2": temporal["level_r2"],
        "interval_coverage": coverage[0],
        "nominal_coverage": coverage[1],
        "elasticity_checks": [
            {"name": n, "passed": p, "detail": d} for n, p, d in elasticity_check(panel, cfg)
        ],
    }

    # Recorded every run: a version offset between annual products would scale
    # every district estimate by a constant and look entirely plausible.
    try:
        from .features import check_version_boundary

        diagnostics["version_boundary"] = check_version_boundary(cfg)
    except Exception as exc:  # noqa: BLE001 - diagnostics are best effort
        logger.warning("Version-boundary check unavailable: %s", exc)
        diagnostics["version_boundary"] = {"ok": None, "error": str(exc)}

    # The only external test of the within-state split. Without it the district
    # figures are coherent but unvalidated.
    try:
        from .gddp import validate_allocation

        diagnostics["allocation_validation"] = validate_allocation(cfg)
    except Exception as exc:  # noqa: BLE001 - diagnostics are best effort
        logger.warning("Allocation validation unavailable: %s", exc)
        diagnostics["allocation_validation"] = {"available": False, "reason": str(exc)}

    try:
        from .match import load_crosswalk, load_unmatched

        cw = load_crosswalk(cfg)
        diagnostics["n_matched"] = int((cw["source_region_name"] != "").sum())
        diagnostics["n_unmatched"] = int((cw["source_region_name"] == "").sum())
        diagnostics["unmatched_adm1"] = sorted(load_unmatched(cfg)["name"].tolist())
    except Exception as exc:  # noqa: BLE001 - diagnostics are best effort
        logger.warning("Could not read crosswalk counts: %s", exc)

    if not args.no_write:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out = PROCESSED_DIR / "diagnostics.json"
        out.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
