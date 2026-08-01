"""District GDDP: validation-only labels.

The project's district estimates are an *allocation*, not a prediction: a state's
GDP is split across its districts in proportion to light share. Nothing in the
training pipeline can test whether that split is right, because the labels it
trains on are state-level. This module supplies the only external evidence
available — the district GDP that a handful of Indian states actually publish.

**These labels must never enter training.** Every row carries
``validation_only=True`` and ``assert_no_validation_labels`` raises if any of
them reach a training frame. Training on them would make the allocation check
circular, and the check is the only thing standing between "coherent" and
"correct".

One adapter per state, because each publishes a different shape. Raw files live
in ``data/raw/labels/`` and are gitignored; the adapters are committed.

Sources (all open, no licence barrier):
  Tamil Nadu   constant-price GDDP 2011-12 to 2019-20, 32 districts
  Maharashtra  real GDDP 2021-22 to 2024-25
  Karnataka    GDDP 2022-23 plus agriculture/industry/services shares

Fiscal years are converted to the calendar year they mostly cover: Indian FY
2018-19 runs April 2018 to March 2019, so it maps to 2018.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DATA_DIR, ConfigError

logger = logging.getLogger(__name__)

LABELS_RAW_DIR = DATA_DIR / "raw" / "labels"

GDDP_COLUMNS = [
    "source_region_name",
    "district_name",
    "year",
    "gddp",
    "price_basis",
    "label_source",
    "validation_only",
]


def _fiscal_to_calendar(label: str) -> int | None:
    """'2018-19' -> 2018. Indian FY starts in April, so the first year dominates."""
    m = re.search(r"(19|20)\d{2}", str(label))
    return int(m.group(0)) if m else None


def _to_number(value: Any) -> float:
    """Indian tables use thousands separators and footnote marks."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    text = re.sub(r"[^0-9.\-]", "", str(value))
    if text in ("", "-", "."):
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _clean_district(name: Any) -> str:
    """Strip footnote markers state tables attach to district names."""
    return re.sub(r"[#$*+†‡]+", "", str(name)).strip()


# --------------------------------------------------------------------------
# per-state adapters
# --------------------------------------------------------------------------


def load_gddp_tamil_nadu(path: Path | None = None) -> pd.DataFrame:
    """Tamil Nadu DES: wide table, one column per fiscal year, constant prices.

    Values are in lakh rupees at 2011-12 constant prices. Only the relative
    share within the state matters for the allocation check, so the unit and the
    base year cancel out; they are recorded anyway.
    """
    path = Path(path or LABELS_RAW_DIR / "gddp_tamilnadu_constant_2011_2019.csv")
    if not path.exists():
        raise ConfigError(f"No Tamil Nadu GDDP file at {path}.")

    raw = pd.read_csv(path)
    # Row 2 of the published sheet is a column-number legend, not data.
    raw = raw[raw["District"].astype(str).str.strip().str.lower() != "2"]
    raw = raw[~raw["District"].isna()]

    year_cols = {c: _fiscal_to_calendar(c) for c in raw.columns if _fiscal_to_calendar(c)}
    year_cols = {c: y for c, y in year_cols.items() if c not in ("Sl No.", "District")}

    rows = []
    for rec in raw.itertuples(index=False):
        district = _clean_district(rec.District)
        if not district or district.lower() in ("total", "tamil nadu", "state"):
            continue
        for col, year in year_cols.items():
            value = _to_number(getattr(rec, col.replace(" ", "_").replace("-", "_"), None))
            if np.isnan(value):
                value = _to_number(raw.loc[raw["District"] == rec.District, col].iloc[0])
            rows.append(
                {
                    "source_region_name": "Tamil Nadu",
                    "district_name": district,
                    "year": year,
                    "gddp": value,
                    "price_basis": "constant_2011_12_lakh_inr",
                    "label_source": "gddp_tamil_nadu",
                }
            )
    out = pd.DataFrame(rows).dropna(subset=["gddp"])
    logger.info("Tamil Nadu GDDP: %d district-years", len(out))
    return out


def load_gddp_maharashtra(path: Path | None = None) -> pd.DataFrame:
    """Maharashtra Economic Survey: real GDDP columns per fiscal year."""
    path = Path(path or LABELS_RAW_DIR / "gddp_maharashtra.csv")
    if not path.exists():
        raise ConfigError(f"No Maharashtra GDDP file at {path}.")

    raw = pd.read_csv(path)
    real_cols = [c for c in raw.columns if c.lower().startswith("real gddp")]
    if not real_cols:
        raise ConfigError(f"{path.name} has no 'Real GDDP' columns; got {list(raw.columns)}")

    rows = []
    for _, rec in raw.iterrows():
        district = _clean_district(rec.get("District"))
        if not district or district.lower() in ("total", "maharashtra", "state"):
            continue
        for col in real_cols:
            year = _fiscal_to_calendar(col)
            if year is None:
                continue
            rows.append(
                {
                    "source_region_name": "Maharashtra",
                    "district_name": district,
                    "year": year,
                    "gddp": _to_number(rec[col]),
                    "price_basis": "real_crore_inr",
                    "label_source": "gddp_maharashtra",
                }
            )
    out = pd.DataFrame(rows).dropna(subset=["gddp"])
    logger.info("Maharashtra GDDP: %d district-years", len(out))
    return out


def load_gddp_karnataka(path: Path | None = None) -> pd.DataFrame:
    """Karnataka Economic Survey: single year, with sectoral composition.

    The sector shares are why this file earns its place: the face-validity
    failures pointed at a sectoral bias in light-share allocation, and this is
    the only source here that measures sector directly.
    """
    path = Path(path or LABELS_RAW_DIR / "gddp_karnataka_2022_23.csv")
    if not path.exists():
        raise ConfigError(f"No Karnataka GDDP file at {path}.")

    raw = pd.read_csv(path)
    gddp_col = next((c for c in raw.columns if "Gross District Domestic Product" in c), None)
    if gddp_col is None:
        raise ConfigError(f"{path.name} has no GDDP column; got {list(raw.columns)}")

    rows = []
    for _, rec in raw.iterrows():
        district = _clean_district(rec.get("District"))
        if not district or district.lower() in ("total", "karnataka", "state"):
            continue
        rows.append(
            {
                "source_region_name": "Karnataka",
                "district_name": district,
                "year": 2022,  # FY 2022-23
                "gddp": _to_number(rec[gddp_col]),
                "price_basis": "current_lakh_inr",
                "label_source": "gddp_karnataka",
                "share_agriculture": _to_number(rec.get("Agriculture (%)")),
                "share_industry": _to_number(rec.get("Industry (%)")),
                "share_services": _to_number(rec.get("Services (%)")),
            }
        )
    out = pd.DataFrame(rows).dropna(subset=["gddp"])
    logger.info("Karnataka GDDP: %d district-years", len(out))
    return out


ADAPTERS = {
    "tamil_nadu": load_gddp_tamil_nadu,
    "maharashtra": load_gddp_maharashtra,
    "karnataka": load_gddp_karnataka,
}


def load_all_gddp(states: list[str] | None = None) -> pd.DataFrame:
    """Every available district GDDP source, stacked and marked validation-only."""
    frames = []
    for name, loader in ADAPTERS.items():
        if states and name not in states:
            continue
        try:
            frames.append(loader())
        except ConfigError as exc:
            logger.warning("Skipping %s GDDP: %s", name, exc)
    if not frames:
        raise ConfigError(
            "No district GDDP available. Without it the within-state allocation "
            "cannot be validated; see the README statement on this."
        )
    out = pd.concat(frames, ignore_index=True)
    # Rule enforced in code, not comment: these never train.
    out["validation_only"] = True
    return out


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------


def match_gddp_to_regions(
    gddp: pd.DataFrame,
    boundaries: pd.DataFrame,
    min_score: float = 88.0,
    aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Attach ``region_id`` to GDDP rows by district name, blocked on state.

    Blocking on the state is what keeps this honest: India has several districts
    sharing a name across states, and a cross-state match would silently compare
    one district's GDP to another's light. Unmatched rows are dropped and counted
    rather than force-matched, because a wrong match corrupts the very number
    this module exists to measure.
    """
    from rapidfuzz import fuzz

    from .match import _variant_list, normalise

    bnd = boundaries.copy()
    bnd["norm_name"] = bnd["name"].map(normalise)
    bnd["norm_parent"] = bnd["parent_name"].map(normalise)
    # GADM 4.1 carries pre-rename district names (Bellary, Belgaum, Gulbarga)
    # while the state surveys use the current ones (Ballari, Belagavi,
    # Kalaburagi). VARNAME resolves most of these without a hand-written alias
    # table, which would rot as more districts are renamed.
    if "name_variants" in bnd.columns:
        bnd["variants"] = bnd["name_variants"].map(_variant_list)
    else:
        bnd["variants"] = [[] for _ in range(len(bnd))]

    # Published tables interleave division/region subtotals with districts.
    # Summing them alongside districts would double-count the state.
    aggregate = (
        gddp["district_name"]
        .astype(str)
        .str.contains(r"\b(?:DIV|DIVISION|REGION|TOTAL|ALL)\b", case=False, regex=True)
    )
    if aggregate.any():
        logger.info(
            "Excluding %d aggregate row(s) (divisions/subtotals), e.g. %s",
            int(aggregate.sum()),
            sorted(gddp.loc[aggregate, "district_name"].unique())[:4],
        )
        gddp = gddp.loc[~aggregate]

    out_rows = []
    unmatched: list[str] = []
    for rec in gddp.itertuples(index=False):
        state_norm = normalise(rec.source_region_name)
        block = bnd[bnd["norm_parent"] == state_norm]
        if block.empty:
            unmatched.append(f"{rec.district_name} ({rec.source_region_name}: no such state)")
            continue
        # Apply the committed rename map before scoring. These are different
        # words, not spelling variants, so no amount of fuzz bridges them.
        published = str(rec.district_name).strip()
        target = normalise((aliases or {}).get(published, published))

        def _best_score(row, t=target):
            cands = [row["norm_name"], *row["variants"]]
            return max(fuzz.token_sort_ratio(t, c) for c in cands if c)

        scores = block.apply(_best_score, axis=1)
        best = scores.idxmax()
        if scores[best] < min_score:
            unmatched.append(
                f"{rec.district_name} ({rec.source_region_name}, best {scores[best]:.0f})"
            )
            continue
        row = rec._asdict()
        row["region_id"] = block.loc[best, "region_id"]
        row["matched_name"] = block.loc[best, "name"]
        row["match_score"] = float(scores[best])
        out_rows.append(row)

    matched = pd.DataFrame(out_rows)
    if unmatched:
        logger.warning(
            "%d GDDP district-year(s) unmatched and excluded from validation: %s",
            len(unmatched),
            sorted(set(unmatched))[:12],
        )
    # A district must not receive two different GDDP rows in one year.
    if not matched.empty:
        dupes = matched.duplicated(subset=["region_id", "year"]).sum()
        if dupes:
            raise ValueError(
                f"{dupes} region-year(s) matched by more than one GDDP district; "
                "the boundary vintage and the published district list disagree."
            )
    logger.info("Matched %d of %d GDDP district-years", len(matched), len(gddp))
    return matched


def allocation_error_report(matched: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Compare each district's true GDP share against its light share.

    The allocation assigns ``state_gdp * (district_SOL / state_SOL)``. This
    computes the same share from published GDDP and puts the two side by side, so
    the question "is the split right" gets a number instead of an assumption.

    Shares are computed within state-year over the **matched** districts only, so
    both sides are normalised over exactly the same set and the comparison is not
    contaminated by districts one source covers and the other does not.
    """
    key = ["source_region_name", "year"]
    sol = features[["region_id", "year", "sol"]].copy()

    df = matched.merge(sol, on=["region_id", "year"], how="inner")
    if df.empty:
        raise ConfigError(
            "No overlap between GDDP years and extracted light years. Extract the "
            "years the GDDP sources cover before validating the allocation."
        )

    df["gddp_share"] = df["gddp"] / df.groupby(key)["gddp"].transform("sum")
    df["sol_share"] = df["sol"] / df.groupby(key)["sol"].transform("sum")
    df["share_error"] = df["sol_share"] - df["gddp_share"]
    df["abs_share_error"] = df["share_error"].abs()
    # Ratio form: >1 means light over-allocates GDP to this district.
    df["allocation_ratio"] = df["sol_share"] / df["gddp_share"].where(df["gddp_share"] > 0)
    return df


GROUP_KEY = ["source_region_name", "year"]


def apply_power_correction(df: pd.DataFrame, alpha: float) -> pd.Series:
    """Reweight light shares by ``share ** alpha``, renormalised within state-year.

    One parameter. ``alpha > 1`` concentrates mass on the largest districts,
    which is the direction the pooled size gradient (-0.38) points; ``alpha < 1``
    spreads it out. ``alpha == 1`` is the uncorrected allocation.
    """
    powered = df["sol_share"] ** float(alpha)
    totals = powered.groupby([df[k] for k in GROUP_KEY]).transform("sum")
    return powered / totals


def _share_mae(df: pd.DataFrame, alpha: float) -> float:
    return float((apply_power_correction(df, alpha) - df["gddp_share"]).abs().mean())


def fit_alpha(df: pd.DataFrame, bounds: tuple[float, float] = (0.3, 3.0)) -> float:
    """Alpha minimising mean absolute share error. Fitted on whatever is passed."""
    from scipy.optimize import minimize_scalar

    result = minimize_scalar(lambda a: _share_mae(df, a), bounds=bounds, method="bounded")
    return float(result.x)


def leave_one_state_out_correction(errors: pd.DataFrame) -> dict[str, Any]:
    """Does a fitted share correction generalise to a state it was not fitted on?

    Fits alpha on all-but-one state and scores the held-out state. This is the
    only honest test available: the bias is obvious in-sample, and an in-sample
    fit would "improve" the numbers while encoding one state's industrial mix as
    a universal law.

    The verdict is deliberately strict. A correction earns its place only if
    alpha is stable across folds *and* held-out MAE improves in **every** fold.
    Anything less means we would be shipping a state-specific adjustment applied
    to 33 states, most of which we cannot check.
    """
    states = sorted(errors["source_region_name"].unique())
    if len(states) < 3:
        raise ConfigError(
            f"Leave-one-state-out needs at least 3 states with GDDP; have {len(states)}."
        )

    folds = []
    for held_out in states:
        train = errors[errors["source_region_name"] != held_out]
        test = errors[errors["source_region_name"] == held_out]
        alpha = fit_alpha(train)

        base_mae = _share_mae(test, 1.0)
        corr_mae = _share_mae(test, alpha)
        base_r = float(np.corrcoef(test["sol_share"], test["gddp_share"])[0, 1])
        corr_r = float(np.corrcoef(apply_power_correction(test, alpha), test["gddp_share"])[0, 1])
        folds.append(
            {
                "held_out_state": held_out,
                "n_test": int(len(test)),
                "alpha_fitted_on_others": alpha,
                "mae_uncorrected_pts": 100 * base_mae,
                "mae_corrected_pts": 100 * corr_mae,
                "delta_mae_pts": 100 * (corr_mae - base_mae),
                "pearson_uncorrected": base_r,
                "pearson_corrected": corr_r,
                "improved": bool(corr_mae < base_mae),
            }
        )

    alphas = [f["alpha_fitted_on_others"] for f in folds]
    per_state_alpha = {s: fit_alpha(errors[errors["source_region_name"] == s]) for s in states}

    # The substantive finding: alpha is stable WITHIN a state across years and
    # unstable BETWEEN states. That makes the bias a real per-state property
    # rather than noise, and is exactly why one global exponent cannot work.
    per_state_year_alpha: dict[str, dict[int, float]] = {}
    for state in states:
        grp = errors[errors["source_region_name"] == state]
        if grp["year"].nunique() < 2:
            continue
        per_state_year_alpha[state] = {
            int(year): fit_alpha(sub) for year, sub in grp.groupby("year")
        }
    within_spread = {
        s: float(max(v.values()) - min(v.values())) for s, v in per_state_year_alpha.items()
    }
    between_spread = float(max(per_state_alpha.values()) - min(per_state_alpha.values()))

    improved_everywhere = all(f["improved"] for f in folds)
    # Straddling 1.0 means the folds disagree about which way to correct.
    stable = (min(alphas) > 1.0) or (max(alphas) < 1.0)

    return {
        "form": "corrected_share proportional to sol_share ** alpha, "
        "renormalised within state-year",
        "n_parameters": 1,
        "folds": folds,
        "alpha_range": [min(alphas), max(alphas)],
        "alpha_pooled_in_sample": fit_alpha(errors),
        "alpha_per_state_in_sample": per_state_alpha,
        "alpha_per_state_per_year": per_state_year_alpha,
        "alpha_within_state_spread": within_spread,
        "alpha_between_state_spread": between_spread,
        "heterogeneity": (
            "alpha is stable within a state across years and unstable between "
            "states, so the bias is a real per-state property rather than noise. "
            "Per-state correction was declined: it would cover 3 of 33 states and "
            "produce an inconsistently corrected map."
        ),
        "improved_in_every_fold": improved_everywhere,
        "alpha_sign_consistent": bool(stable),
        "accepted": bool(improved_everywhere and stable),
        "applied_to_estimates": False,
        "verdict": (
            "ACCEPTED"
            if (improved_everywhere and stable)
            else "REJECTED: does not generalise across states; uncorrected share retained"
        ),
    }


def validate_allocation(cfg: dict[str, Any]) -> dict[str, Any]:
    """End-to-end allocation check. Returns a summary suitable for diagnostics.json.

    Reports the correlation and mean absolute error between light share and
    published GDP share, plus the sectoral and size gradients that say *how* the
    split is wrong rather than just how much.
    """
    import geopandas as gpd

    from .boundaries import load_snapshot
    from .features import build_features

    boundaries = load_snapshot(cfg["country"])
    if not isinstance(boundaries, gpd.GeoDataFrame):  # pragma: no cover - defensive
        raise ConfigError("Boundary snapshot did not load as a GeoDataFrame.")

    matched = match_gddp_to_regions(
        load_all_gddp(), boundaries, aliases=cfg.get("district_aliases")
    )
    errors = allocation_error_report(matched, build_features(cfg))

    by_state = {}
    for state, grp in errors.groupby("source_region_name"):
        by_state[str(state)] = {
            "n": int(len(grp)),
            "years": sorted(int(y) for y in grp["year"].unique()),
            "pearson": float(np.corrcoef(grp["sol_share"], grp["gddp_share"])[0, 1]),
            "mae_share_points": float(100 * grp["abs_share_error"].mean()),
        }

    worst = (
        errors.sort_values("abs_share_error", ascending=False)
        .drop_duplicates("region_id")
        .head(10)[
            [
                "matched_name",
                "source_region_name",
                "year",
                "sol_share",
                "gddp_share",
                "share_error",
                "allocation_ratio",
            ]
        ]
    )

    summary: dict[str, Any] = {
        "n_district_years": int(len(errors)),
        "n_districts": int(errors["region_id"].nunique()),
        "states": sorted(errors["source_region_name"].unique().tolist()),
        "pearson": float(np.corrcoef(errors["sol_share"], errors["gddp_share"])[0, 1]),
        "spearman": float(errors[["sol_share", "gddp_share"]].corr(method="spearman").iloc[0, 1]),
        "mae_share_points": float(100 * errors["abs_share_error"].mean()),
        "by_state": by_state,
        "worst_districts": worst.to_dict(orient="records"),
        # Size gradient: saturation predicts big districts are under-allocated.
        "size_gradient_corr": float(
            np.corrcoef(
                np.log(errors["gddp_share"].where(errors["gddp_share"] > 0)), errors["share_error"]
            )[0, 1]
        ),
    }

    sect = (
        errors.dropna(subset=["share_services"]) if "share_services" in errors else pd.DataFrame()
    )
    if len(sect) > 3:
        summary["sectoral_gradient"] = {
            sector: float(np.corrcoef(sect[f"share_{sector}"], sect["share_error"])[0, 1])
            for sector in ("services", "industry", "agriculture")
            if f"share_{sector}" in sect
        }

    # Per-state size gradients. If these differ a lot, no single global exponent
    # can fit them all, which is the mechanism behind the correction's failure.
    summary["size_gradient_by_state"] = {
        str(state): float(
            np.corrcoef(np.log(grp["gddp_share"].where(grp["gddp_share"] > 0)), grp["share_error"])[
                0, 1
            ]
        )
        for state, grp in errors.groupby("source_region_name")
    }

    try:
        summary["share_correction"] = leave_one_state_out_correction(errors)
    except ConfigError as exc:
        summary["share_correction"] = {"available": False, "reason": str(exc)}
    return summary


def assert_no_validation_labels(df: pd.DataFrame, where: str = "training frame") -> None:
    """Raise if validation-only labels have leaked into a training frame.

    Training on district GDDP would make the allocation check circular: the split
    would be fitted to the very data used to judge it, and a bad split would then
    look good. This is the assertion that keeps the check independent.
    """
    if "validation_only" not in df.columns:
        return
    leaked = df["validation_only"].fillna(False).astype(bool)
    if leaked.any():
        sources = sorted(df.loc[leaked, "label_source"].astype(str).unique())
        raise ValueError(
            f"{int(leaked.sum())} validation-only district GDDP row(s) reached the "
            f"{where} (sources: {sources}). These labels exist solely to test the "
            "within-state allocation and must never be trained on, or the test "
            "becomes circular."
        )
