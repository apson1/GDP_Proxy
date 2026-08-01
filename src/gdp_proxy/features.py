"""Phase 4: features.

Sum of Lights alone saturates: once a city core maxes out the sensor, extra
activity produces no extra radiance. The features here that measure the *spread*
and *edge* of light (``lit_share``, ``gini_light``, ``p90_rad``) keep responding
after brightness flattens, which is exactly where SOL stops working.

Two rules dominate the annualisation:

  Rule 10  A masked month is missing, not dark. We average over *valid* months
           and never ``fillna(0)``. A district with four usable months is not the
           same as one with twelve, and ``n_valid_months`` carries that forward.
  Rule 6   Every join asserts its row count. The boundary merge that adds area
           and population is the one that silently drops districts.

All logic here is pure pandas/numpy and runs offline. The only network function,
``load_national_gdp``, is isolated for the exit-test correlation.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import DATA_DIR, ConfigError, country_config

logger = logging.getLogger(__name__)

OUT_DIR = DATA_DIR / "processed"

WDI_URL = (
    "https://api.worldbank.org/v2/country/{iso3}/indicator/{indicator}?format=json&per_page=500"
)

# Columns averaged across valid months when going monthly -> annual.
MONTHLY_MEAN_COLS = ["sol", "mean_rad", "median_rad", "p90_rad", "lit_pixels", "valid_pixels"]


# --------------------------------------------------------------------------
# loading the monthly panel
# --------------------------------------------------------------------------


def load_panel(cfg: dict[str, Any], series: str | None = None) -> pd.DataFrame:
    """Concatenate the extracted yearly parquets for the configured series.

    Annual and monthly files are kept apart on disk and never concatenated:
    VCMSLCFG monthly carries a calibration step at 2017 that ANNUAL_V22 does not,
    and splicing the two families produces growth that is not there.
    """
    series = series or str(cfg.get("training_series", "monthly")).lower()
    suffix = "_annual" if series == "annual" else ""
    stem = f"sol_{cfg['country']}_adm{cfg['admin_level']}{suffix}_"

    paths = sorted(OUT_DIR.glob(f"{stem}*.parquet"))
    if series == "monthly":
        # The monthly glob would otherwise also match the _annual files.
        paths = [p for p in paths if "_annual_" not in p.name]

    if not paths:
        raise ConfigError(
            f"No extracted parquets matching {stem}*.parquet in {OUT_DIR} "
            f"(training_series={series}). Run:\n"
            f"  python -m gdp_proxy.extract --series {series} --submit\n"
            "then download the CSVs from Drive and run --ingest."
        )
    frames = [pd.read_parquet(p) for p in paths]
    panel = pd.concat(frames, ignore_index=True)

    if "series" in panel.columns and panel["series"].nunique() > 1:
        raise ConfigError(
            f"Panel mixes product families {sorted(panel['series'].unique())}. "
            "Splicing VIIRS products mid-series creates fake growth."
        )
    panel.attrs["series"] = series
    logger.info("Loaded %d %s rows from %d yearly files", len(panel), series, len(paths))
    return panel


# --------------------------------------------------------------------------
# monthly -> annual
# --------------------------------------------------------------------------


def annualise(monthly: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Collapse months to years using the mean of *valid* months only.

    Records ``n_valid_months`` and drops region-years with fewer than
    ``min_valid_months`` valid observations, reporting how many were dropped
    rather than quietly shrinking the panel. Never imputes zero for a masked
    month (rule 10).
    """
    min_valid = int(cfg.get("min_valid_months", 6))

    if "is_missing" not in monthly.columns:
        raise ConfigError("Monthly panel has no is_missing column; re-run Phase 2 ingest.")

    # The annual composite is already a year: one row per region, and the
    # min_valid_months gate is meaningless against a single period.
    if str(monthly.attrs.get("series", "")) == "annual" or (
        "series" in monthly.columns and (monthly["series"] == "annual").all()
    ):
        min_valid = 1

    valid = monthly.loc[~monthly["is_missing"].astype(bool)].copy()

    present = [c for c in MONTHLY_MEAN_COLS if c in valid.columns]
    grouped = valid.groupby(["region_id", "year"], sort=True)
    annual = grouped[present].mean()
    annual["n_valid_months"] = grouped.size()
    annual = annual.reset_index()

    before = len(annual)
    kept = annual.loc[annual["n_valid_months"] >= min_valid].reset_index(drop=True)
    n_dropped = before - len(kept)
    kept.attrs["n_dropped_low_months"] = n_dropped
    kept.attrs["min_valid_months"] = min_valid
    if n_dropped:
        logger.info("Dropped %d region-years below %d valid months", n_dropped, min_valid)
    return kept


# --------------------------------------------------------------------------
# concentration
# --------------------------------------------------------------------------


def gini(values: np.ndarray | list[float]) -> float:
    """Gini coefficient of a non-negative array. 0 = uniform, ->1 = all in one cell.

    Standard mean-absolute-difference form. Returns 0 for an empty or all-zero
    input, since there is no concentration to measure.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0.0
    if np.any(arr < 0):
        arr = arr - arr.min()  # shift so the coefficient stays well defined
    total = arr.sum()
    if total == 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * arr)) / (n * total))


def _approx_gini_light(lit_share: float, median_rad: float, p90_rad: float) -> float:
    """Coarse within-district light concentration from summary stats.

    A true within-district Gini needs the per-pixel radiance array, which the
    Phase 2 extractor does not export (it emits sum/mean/percentiles, not a
    histogram). Until a histogram reducer is added there, this reconstructs a
    100-point representative distribution: the dark share at 0, and the lit share
    split between the median and the 90th percentile. It is monotonic in the two
    things that matter, how much area is dark and how peaked the lit area is, and
    is flagged as an approximation in the feature report.
    """
    if not np.isfinite(lit_share):
        return float("nan")
    lit_share = min(max(lit_share, 0.0), 1.0)
    n = 100
    n_lit = int(round(lit_share * n))
    n_dark = n - n_lit
    if n_lit == 0:
        return 0.0
    half = n_lit // 2
    lit_vals = [median_rad] * (n_lit - half) + [p90_rad] * half
    sample = np.array([0.0] * n_dark + lit_vals, dtype=float)
    return gini(sample)


# --------------------------------------------------------------------------
# level features
# --------------------------------------------------------------------------


def add_level_features(
    df: pd.DataFrame,
    boundaries: pd.DataFrame,
    cfg: dict[str, Any],
    population: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add area/population-normalised levels, lit_share, gini_light and logs.

    ``population`` is a region-**year** frame (GHSL epochs interpolated), not a
    static column, because district populations move enough over a decade that a
    single figure would distort per-capita trends.
    """
    bnd_cols = ["region_id", "area_km2"]
    if "population" in boundaries.columns and population is None:
        bnd_cols.append("population")
    bnd = boundaries[bnd_cols].drop_duplicates("region_id")

    before = len(df)
    merged = df.merge(bnd, on="region_id", how="left", validate="m:1")
    if len(merged) != before:
        raise ValueError(f"Boundary join changed row count {before} -> {len(merged)}")

    if population is not None:
        pop_cols = [
            c
            for c in (
                "region_id",
                "year",
                "population",
                "population_interpolated",
                "population_is_projection",
            )
            if c in population.columns
        ]
        merged = merged.drop(columns=[c for c in ("population",) if c in merged.columns])
        merged = merged.merge(population[pop_cols], on=["region_id", "year"], how="left")
        if len(merged) != before:
            raise ValueError(f"Population join changed row count {before} -> {len(merged)}")
        missing_pop = merged["population"].isna()
        if missing_pop.any():
            lost = sorted(merged.loc[missing_pop, "region_id"].unique())
            raise ValueError(
                f"{int(missing_pop.sum())} region-years have no population; "
                f"region_id/year mismatch with the GHSL panel: {lost[:10]}"
            )
    lost = merged["area_km2"].isna()
    if lost.any():
        missing = sorted(merged.loc[lost, "region_id"].unique())
        raise ValueError(
            f"{int(lost.sum())} region-years have no boundary area; region_id "
            f"mismatch between panel and boundaries: {missing[:10]}"
        )

    merged["sol_per_area"] = merged["sol"] / merged["area_km2"]

    if "population" in merged.columns:
        pop = merged["population"].where(merged["population"] > 0)
        merged["sol_per_capita"] = merged["sol"] / pop
        merged["log_population"] = _safe_log(merged["population"])
        merged["log_sol_per_capita"] = _safe_log(merged["sol_per_capita"])
    else:
        merged["sol_per_capita"] = np.nan
        merged["log_population"] = np.nan
        merged["log_sol_per_capita"] = np.nan

    denom = merged["valid_pixels"].where(merged["valid_pixels"] > 0)
    merged["lit_share"] = (merged["lit_pixels"] / denom).clip(upper=1.0)

    merged["gini_light"] = [
        _approx_gini_light(ls, mr, p9)
        for ls, mr, p9 in zip(
            merged["lit_share"],
            merged.get("median_rad", pd.Series(np.nan, index=merged.index)),
            merged.get("p90_rad", pd.Series(np.nan, index=merged.index)),
            strict=False,
        )
    ]

    merged["log_sol"] = _safe_log(merged["sol"])
    merged["log_sol_per_area"] = _safe_log(merged["sol_per_area"])
    merged["log_lit_pixels"] = np.log1p(merged["lit_pixels"])

    return merged


def _safe_log(series: pd.Series) -> pd.Series:
    """log of strictly positive values; exact zeros become NaN, not -inf.

    A zero SOL in a well-observed district is worth surfacing (rule 10), not
    smoothing with a magic epsilon. NaN propagates and shows up in validation.
    """
    out = np.log(series.where(series > 0))
    return out


# --------------------------------------------------------------------------
# dynamics
# --------------------------------------------------------------------------


def add_growth_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add year-on-year log differences, newly_lit, and 3-year rolling means."""
    baseline_year = int(cfg.get("baseline_year", cfg.get("start_year", df["year"].min())))
    out = df.sort_values(["region_id", "year"]).copy()
    grp = out.groupby("region_id", sort=False)

    # Log differences, not percent change: symmetric and matches the log-log model.
    out["sol_yoy"] = grp["sol"].transform(lambda s: _safe_log(s).diff())
    out["lit_pixels_yoy"] = grp["lit_pixels"].transform(lambda s: np.log1p(s).diff())

    # newly_lit: lit pixels relative to the baseline year. We only have counts,
    # not pixel identities, so this is (lit_now - lit_baseline) clipped at zero,
    # documented as an approximation of true newly-electrified area.
    baseline = (
        out.loc[out["year"] == baseline_year, ["region_id", "lit_pixels"]]
        .rename(columns={"lit_pixels": "_lit_baseline"})
        .drop_duplicates("region_id")
    )
    out = out.merge(baseline, on="region_id", how="left", validate="m:1")
    out["newly_lit"] = (out["lit_pixels"] - out["_lit_baseline"]).clip(lower=0)
    out = out.drop(columns=["_lit_baseline"])

    out["sol_roll3"] = grp["sol"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    # lit_share exists after add_level_features; guard for standalone use.
    if "lit_share" in out.columns:
        out["lit_share_roll3"] = out.groupby("region_id", sort=False)["lit_share"].transform(
            lambda s: s.rolling(3, min_periods=1).mean()
        )

    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def build_features(cfg: dict[str, Any], write: bool = False) -> pd.DataFrame:
    """Monthly parquets -> annual, normalised, dynamic feature panel."""
    from .boundaries import load_snapshot

    monthly = load_panel(cfg)
    annual = annualise(monthly, cfg)
    boundaries = load_snapshot(cfg["country"])

    population = None
    try:
        from .population import load_population

        population = load_population(cfg, sorted(int(y) for y in annual["year"].unique()))
    except ConfigError as exc:
        # Per-capita is the headline metric, so its absence is worth saying out
        # loud rather than silently emitting NaN columns.
        logger.warning(
            "No district population available, so per-capita features will be NaN "
            "and only total GDP can be reported: %s",
            exc,
        )

    leveled = add_level_features(annual, boundaries, cfg, population=population)
    features = add_growth_features(leveled, cfg)

    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"features_{cfg['country']}_adm{cfg['admin_level']}.parquet"
        features.to_parquet(out, index=False)
        logger.info("Wrote %s", out)
    return features


# --------------------------------------------------------------------------
# national GDP (network, for the exit correlation only)
# --------------------------------------------------------------------------


def load_national_gdp(iso3: str, indicator: str) -> pd.DataFrame:
    """National GDP series from World Bank WDI. Network call."""
    import requests

    url = WDI_URL.format(iso3=iso3, indicator=indicator)
    payload = requests.get(url, timeout=60).json()
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise ConfigError(f"WDI returned no GDP data for {iso3}")
    rows = [
        {"year": int(r["date"]), "national_gdp": float(r["value"])}
        for r in payload[1]
        if r.get("value") is not None
    ]
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def annual_monthly_ratio(cfg: dict[str, Any]) -> pd.DataFrame:
    """National annual-SOL / monthly-SOL ratio per year, the version-boundary bridge.

    The annual family is split across processing versions with **no overlapping
    year**: V21 ends 2021, V22 starts 2022. There is therefore no direct way to
    estimate an offset across that boundary. The monthly VCMSLCFG series does run
    continuously across both eras, so it can act as a bridge: the ratio of annual
    to monthly national SOL should be stable from year to year if the two annual
    versions are on the same scale.

    A silent version offset is the dangerous case. It would scale every district
    estimate by a constant, which is invisible on a map and entirely plausible on
    inspection. We already found a calibration break *inside* one product family,
    so the boundary between families is not assumed seamless.
    """
    base = dict(cfg)
    annual = build_features({**base, "training_series": "annual"})
    monthly = build_features({**base, "training_series": "monthly"})

    na = annual.groupby("year")["sol"].sum()
    nm = monthly.groupby("year")["sol"].sum()
    common = sorted(set(na.index) & set(nm.index))
    if not common:
        raise ConfigError(
            "No year has both an annual and a monthly extraction, so the version "
            "boundary cannot be checked. Extract the same year in both series."
        )

    rows = []
    for year in common:
        rows.append(
            {
                "year": int(year),
                "annual_sol": float(na[year]),
                "monthly_sol": float(nm[year]),
                "ratio": float(na[year] / nm[year]) if nm[year] else float("nan"),
                "annual_dataset": _annual_dataset_for_year(cfg, int(year)),
            }
        )
    return pd.DataFrame(rows)


def _annual_dataset_for_year(cfg: dict[str, Any], year: int) -> str:
    for v in cfg.get("viirs_annual_versions") or []:
        if int(v["start_year"]) <= year <= int(v["end_year"]):
            return str(v["id"])
    return str(cfg.get("viirs_annual", "unknown"))


def check_version_boundary(
    cfg: dict[str, Any], baseline_end_year: int = 2021, n_sd: float = 3.0
) -> dict[str, Any]:
    """Compare each post-boundary ratio to the pre-boundary baseline.

    Returns the ratio series plus the baseline statistics. ``ok`` is False when a
    post-boundary year sits outside ``mean +/- n_sd * sd`` of the baseline, which
    means the two annual versions are not on the same scale and predictions made
    with one from a model trained on the other would be uniformly shifted.
    """
    ratios = annual_monthly_ratio(cfg)
    baseline = ratios[ratios["year"] <= baseline_end_year]
    later = ratios[ratios["year"] > baseline_end_year]

    if len(baseline) < 2:
        raise ConfigError(
            f"Need at least 2 baseline years (<= {baseline_end_year}) to estimate the "
            f"ratio spread; got {len(baseline)}."
        )

    mean = float(baseline["ratio"].mean())
    sd = float(baseline["ratio"].std(ddof=1))
    lo, hi = mean - n_sd * sd, mean + n_sd * sd

    offenders = later[(later["ratio"] < lo) | (later["ratio"] > hi)]
    return {
        "ok": bool(len(offenders) == 0),
        "baseline_mean": mean,
        "baseline_sd": sd,
        "n_sd": float(n_sd),
        "lower_bound": lo,
        "upper_bound": hi,
        "baseline_years": [int(y) for y in baseline["year"]],
        "checked_years": [int(y) for y in later["year"]],
        "offending_years": [int(y) for y in offenders["year"]],
        "ratios": ratios.to_dict(orient="records"),
    }


def assert_version_boundary_consistent(
    cfg: dict[str, Any], baseline_end_year: int = 2021, n_sd: float = 3.0
) -> dict[str, Any]:
    """Raise if an annual version boundary has shifted the national scale."""
    result = check_version_boundary(cfg, baseline_end_year, n_sd)
    if not result["ok"]:
        detail = ", ".join(
            f"{r['year']}: {r['ratio']:.4f} ({r['annual_dataset'].split('/')[-1]})"
            for r in result["ratios"]
            if r["year"] in result["offending_years"]
        )
        raise ConfigError(
            f"Annual/monthly SOL ratio outside the V21 baseline for {detail}. "
            f"Baseline {result['baseline_mean']:.4f} +/- {n_sd}*{result['baseline_sd']:.4f} "
            f"= [{result['lower_bound']:.3f}, {result['upper_bound']:.3f}]. "
            "The annual product versions are not on the same scale, so a model "
            "trained on one and predicting with the other would shift every "
            "district estimate by a constant factor. Estimate and apply the "
            "offset before producing estimates."
        )
    return result


def national_sol_by_year(features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate district SOL to the national total per year."""
    return (
        features.groupby("year", as_index=False)["sol"]
        .sum()
        .rename(columns={"sol": "national_sol"})
    )


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@dataclass
class FeatureReport:
    country: str
    n_rows: int
    n_regions: int
    n_dropped_low_months: int
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    per_year_counts: dict[int, int] = field(default_factory=dict)
    valid_month_hist: dict[int, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.checks)

    def render(self) -> str:
        lines = [
            f"Features: {self.country}",
            f"  rows                {self.n_rows}",
            f"  regions             {self.n_regions}",
            f"  dropped (<months)   {self.n_dropped_low_months}",
            f"  regions per year    {self.per_year_counts}",
            f"  valid-month hist    {self.valid_month_hist}",
            "",
        ]
        for name, passed, detail in self.checks:
            lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        return "\n".join(lines)


def validate_features(df: pd.DataFrame, cfg: dict[str, Any]) -> FeatureReport:
    """Assert the feature panel is clean, finite, and its imbalance is visible."""
    report = FeatureReport(
        country=cfg.get("country", "unknown"),
        n_rows=len(df),
        n_regions=int(df["region_id"].nunique()),
        n_dropped_low_months=int(df.attrs.get("n_dropped_low_months", 0)),
        per_year_counts={int(y): int(c) for y, c in df["year"].value_counts().sort_index().items()},
        valid_month_hist={
            int(m): int(c) for m, c in df["n_valid_months"].value_counts().sort_index().items()
        },
    )

    dupes = int(df.duplicated(subset=["region_id", "year"]).sum())
    report.checks.append(("one row per region-year", dupes == 0, f"{dupes} duplicates"))

    log_cols = [c for c in ("log_sol", "log_sol_per_area", "log_lit_pixels") if c in df.columns]
    n_inf = int(np.isinf(df[log_cols].to_numpy(dtype=float)).sum()) if log_cols else 0
    report.checks.append(
        ("no infinite log values", n_inf == 0, f"{n_inf} infinities in {log_cols}")
    )

    if "lit_share" in df.columns:
        bad = df["lit_share"].dropna()
        out_of_range = int(((bad < 0) | (bad > 1)).sum())
        report.checks.append(
            ("lit_share within [0,1]", out_of_range == 0, f"{out_of_range} out of range")
        )

    if {"sol", "sol_per_area"}.issubset(df.columns):
        pos_sol = df["sol"] > 0
        bad = int((pos_sol & ~(df["sol_per_area"] > 0)).sum())
        report.checks.append(
            ("sol_per_area positive where sol positive", bad == 0, f"{bad} violations")
        )

    counts = df["region_id"].value_counts()
    balanced = counts.nunique() == 1
    detail = "balanced" if balanced else f"{counts.min()}-{counts.max()} years per region"
    report.checks.append(("panel balanced (reported, not required)", True, detail))

    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the annual feature panel")
    parser.add_argument("--country", default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = country_config(args.country)

    features = build_features(cfg, write=not args.no_write)
    report = validate_features(features, cfg)
    print()
    print(report.render())
    print()
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
