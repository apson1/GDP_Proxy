"""Phase 3a: GDP labels.

Turns published GDP figures into a tidy, deflated panel keyed by region name and
year. This is deliberately mechanical and rerunnable. The judgement calls, which
name maps to which polygon, live in ``match.py`` and are captured in a committed
crosswalk, never redone here.

Two rules from CLAUDE.md dominate this module:

  Rule 9   All modelling is on constant prices. A district growing 8% nominal in
           a year of 7% inflation grew 1%, and nightlights will show roughly 1%.
           Train on nominal values and the model learns the inflation series.
  Rule 6   Assert row counts after every join. Region names arrive from several
           sources with several spellings; a merge that loses a third of the
           rows looks exactly like a merge that worked.

Network calls (World Bank WDI) are isolated in one function so the parsing,
deflation and validation logic runs offline in tests.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR, ConfigError, country_config

logger = logging.getLogger(__name__)

LABELS_RAW_DIR = DATA_DIR / "raw" / "labels"
OUT_DIR = DATA_DIR / "processed"

WDI_URL = (
    "https://api.worldbank.org/v2/country/{iso3}/indicator/{indicator}?format=json&per_page=500"
)

OUTPUT_COLUMNS = [
    "source_region_name",
    "parent_name",
    # GADM's own ADM1 code (DOSE's GID_1). Carried through so match.py can do an
    # exact code join instead of guessing at transliterated state names.
    "source_gid",
    "year",
    "gdp_constant",
    "gdp_nominal",
    "population",
    "admin_level",
    "label_source",
    "ingested_at",
]

# DOSE column names vary slightly between releases. Resolve the first present
# candidate rather than hard-coding one spelling, and fail loudly listing what
# was actually in the file if a required field is absent.
DOSE_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "iso3": ["GID_0", "iso3", "country_iso"],
    "region": ["region", "region_name", "GID_1_name", "name"],
    "parent": ["country", "country_name", "GID_0"],
    "gid1": ["GID_1", "gid_1"],
    "year": ["year"],
    "gdp_nominal": ["grp_lcu", "grp_lcu_current", "grp"],
    "gdp_constant": ["grp_lcu_2015", "grp_lcu_const", "grp_lcu_constant"],
    # DOSE V2.14 ships a per-capita constant series; multiply by pop for levels.
    "gdp_constant_pc": ["grp_pc_lcu_2015", "grp_pc_lcu_const"],
    "population": ["pop", "population"],
    "deflator": ["deflator_2015", "deflator"],
}


def _first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _float_series(df: pd.DataFrame, col: str | None) -> pd.Series:
    """Coerce a column to float64, or return an all-NaN float column if absent.

    Returning a typed NaN column rather than a scalar ``pd.NA`` matters: assigning
    a scalar pd.NA to a DataFrame column silently produces **object** dtype, and
    an object-dtype numeric column blows up much later in ``np.log`` with an
    opaque "ufunc does not support argument 0 of type float" error, nowhere near
    the line that set it.
    """
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").astype("float64")


# --------------------------------------------------------------------------
# DOSE ingest
# --------------------------------------------------------------------------


def load_dose(cfg: dict[str, Any], path: Path | None = None) -> pd.DataFrame:
    """Load DOSE, filter to the target country, normalise to the output schema.

    DOSE (Zenodo record 20035157) is ADM1 for most countries and already ships a
    constant-price series in 2015 local currency. We prefer that series over
    redoing the deflation and record the choice in ``label_source``. The raw file
    is not committed; drop it in ``data/raw/labels/`` and point ``dose_file`` at
    it in config.
    """
    if path is None:
        path = LABELS_RAW_DIR / str(cfg.get("dose_file", "dose_v2.csv"))
    path = Path(path)
    if not path.exists():
        raise ConfigError(
            f"No DOSE file at {path}. Download record {cfg.get('dose_record', '20035157')} "
            f"from Zenodo and place it in {LABELS_RAW_DIR}/ (see config dose_file)."
        )

    raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return normalise_dose(raw, cfg)


def normalise_dose(raw: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Pure reshape of a raw DOSE frame to the output schema. Testable offline."""
    overrides = dict(cfg.get("dose_columns") or {})
    resolved: dict[str, str | None] = {}
    for key, candidates in DOSE_COLUMN_CANDIDATES.items():
        chosen = overrides.get(key) or _first_col(raw, candidates)
        resolved[key] = chosen

    for required in ("region", "year", "gdp_nominal"):
        if resolved[required] is None:
            raise ConfigError(
                f"DOSE file has no column for '{required}'. Tried "
                f"{DOSE_COLUMN_CANDIDATES[required]}; available columns are {list(raw.columns)}."
            )

    df = raw.copy()

    iso3 = cfg["iso3"]
    if resolved["iso3"] is not None:
        df = df[df[resolved["iso3"]].astype(str).str.upper() == iso3.upper()]
    if df.empty:
        raise ConfigError(f"DOSE file contains no rows for {iso3}.")

    population = _float_series(df, resolved["population"])
    gdp_constant, method = _dose_constant_series(df, resolved, population, cfg)

    out = pd.DataFrame(
        {
            "source_region_name": df[resolved["region"]].astype(str),
            "parent_name": (
                df[resolved["parent"]].astype(str)
                if resolved["parent"] is not None
                else cfg["iso3"]
            ),
            "source_gid": (
                df[resolved["gid1"]].astype(str)
                if resolved["gid1"] is not None
                else pd.Series("", index=df.index, dtype="object")
            ),
            "year": pd.to_numeric(df[resolved["year"]], errors="coerce").astype("Int64"),
            "gdp_nominal": _float_series(df, resolved["gdp_nominal"]),
            "gdp_constant": gdp_constant,
            "population": population,
            "admin_level": 1,
            "label_source": "dose",
            "deflation_method": method,
        }
    )
    out = out.dropna(subset=["year"]).reset_index(drop=True)
    logger.info("DOSE: %d region-years for %s (constant price via %s)", len(out), iso3, method)
    return out


def _dose_constant_series(
    df: pd.DataFrame,
    resolved: dict[str, str | None],
    population: pd.Series,
    cfg: dict[str, Any],
) -> tuple[pd.Series, str]:
    """Pick the constant-price series, preferring DOSE's own when configured.

    DOSE V2.14 ships ``grp_pc_lcu_2015`` (per capita) and ``deflator_2015``,
    already based on 2015, which is what ``base_year`` defaults to. Using the
    authors' own validated series removes the WDI fetch and a whole class of
    dtype and coverage failures. ``use_dose_constant: false`` falls back to
    deflating nominal values with WDI, which is what sources lacking a constant
    series need.

    Always returns a float64 Series, never object.
    """
    if not bool(cfg.get("use_dose_constant", True)):
        return pd.Series(np.nan, index=df.index, dtype="float64"), "pending"

    level_col = resolved.get("gdp_constant")
    if level_col is not None:
        return _float_series(df, level_col), "dose_constant_level_2015"

    pc_col = resolved.get("gdp_constant_pc")
    if pc_col is not None:
        if resolved.get("population") is None:
            raise ConfigError(
                f"DOSE file has the per-capita constant column '{pc_col}' but no "
                "population column to convert it to levels. Set use_dose_constant: "
                "false to deflate nominal values via WDI instead."
            )
        levels = (_float_series(df, pc_col) * population).astype("float64")
        return levels, "dose_constant_pc_x_pop_2015"

    # No DOSE constant series at all; the WDI deflation path fills this in.
    return pd.Series(np.nan, index=df.index, dtype="float64"), "pending"


# --------------------------------------------------------------------------
# deflation
# --------------------------------------------------------------------------


def load_national_deflator(iso3: str, base_year: int, indicator: str) -> pd.DataFrame:
    """Fetch the national GDP deflator series from World Bank WDI. Network call.

    Returns one row per year with the raw index value. WDI's base year is
    country-specific; ``deflate`` rebases to ``base_year`` itself, so the raw
    index is all that is needed here.
    """
    url = WDI_URL.format(iso3=iso3, indicator=indicator)
    logger.info("Fetching WDI deflator %s for %s", indicator, iso3)
    payload = requests.get(url, timeout=60).json()
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise ConfigError(f"WDI returned no deflator data for {iso3}: {str(payload)[:200]}")

    defl = pd.DataFrame(
        [{"year": rec.get("date"), "deflator": rec.get("value")} for rec in payload[1]]
    )
    # WDI interleaves nulls with floats for years it has no estimate for. Coerce
    # explicitly and drop them: a None surviving into the arithmetic turns the
    # whole column object-dtype, which fails much later inside np.log.
    defl["year"] = pd.to_numeric(defl["year"], errors="coerce").astype("Int64")
    defl["deflator"] = pd.to_numeric(defl["deflator"], errors="coerce")
    n_before = len(defl)
    defl = defl.dropna(subset=["year", "deflator"]).copy()
    if len(defl) < n_before:
        logger.info("WDI: dropped %d year(s) with no deflator value", n_before - len(defl))

    defl["year"] = defl["year"].astype(int)
    defl["deflator"] = defl["deflator"].astype("float64")
    defl = defl.sort_values("year").reset_index(drop=True)

    if defl.empty:
        raise ConfigError(f"WDI deflator series for {iso3} is empty after dropping nulls.")
    if base_year not in set(defl["year"]):
        raise ConfigError(
            f"Base year {base_year} is not in the WDI deflator series for {iso3} "
            f"(range {int(defl['year'].min())}-{int(defl['year'].max())})."
        )
    return defl


def deflate(df: pd.DataFrame, deflator: pd.DataFrame, base_year: int) -> pd.DataFrame:
    """Fill ``gdp_constant`` from ``gdp_nominal`` using a national deflator.

    Only rows whose ``gdp_constant`` is still missing are deflated; rows that
    already carry DOSE's own constant series are left untouched. The deflator is
    rebased so that ``base_year`` has factor 1.0.
    """
    if base_year not in set(deflator["year"]):
        raise ConfigError(f"Base year {base_year} not in deflator series.")

    defl = deflator.copy()
    defl["deflator"] = pd.to_numeric(defl["deflator"], errors="coerce")
    defl = defl.dropna(subset=["deflator"])
    if base_year not in set(defl["year"]):
        raise ConfigError(f"Base year {base_year} has a null deflator; cannot rebase.")
    base_value = float(defl.loc[defl["year"] == base_year, "deflator"].iloc[0])

    factors = defl.assign(defl_factor=lambda d: base_value / d["deflator"])[["year", "defl_factor"]]

    before = len(df)
    merged = df.merge(factors, on="year", how="left", validate="m:1")
    if len(merged) != before:
        raise ValueError(f"Deflator join changed row count {before} -> {len(merged)}")

    # Coerce up front so an object-dtype column arriving from a caller cannot
    # survive into the output and fail later inside np.log.
    for col in ("gdp_constant", "gdp_nominal"):
        if col in merged.columns:
            coerced = pd.to_numeric(merged[col], errors="coerce")
            bad = coerced.isna() & merged[col].notna()
            if bad.any():
                offenders = merged.loc[bad, col].unique()[:5]
                raise ValueError(
                    f"Column '{col}' holds non-numeric values that cannot be deflated: "
                    f"{list(offenders)}. Fix the ingest adapter rather than coercing here."
                )
            merged[col] = coerced.astype("float64")

    needs = merged["gdp_constant"].isna()
    merged.loc[needs, "gdp_constant"] = (
        merged.loc[needs, "gdp_nominal"] * merged.loc[needs, "defl_factor"]
    )
    merged.loc[needs, "deflation_method"] = f"wdi_deflator_{base_year}"

    unmatched = int((needs & merged["defl_factor"].isna()).sum())
    if unmatched:
        missing_years = sorted(merged.loc[needs & merged["defl_factor"].isna(), "year"].unique())
        logger.warning(
            "%d region-years had no deflator (years %s); gdp_constant left NaN",
            unmatched,
            missing_years[:10],
        )

    out = merged.drop(columns=["defl_factor"])

    # Post-condition. The assignment above can re-widen the column to object if
    # anything upstream slipped through, and the failure would otherwise surface
    # several functions away inside np.log.
    out["gdp_constant"] = out["gdp_constant"].astype("float64")
    if out["gdp_constant"].dtype != np.float64:
        raise ValueError(
            f"deflate() produced gdp_constant with dtype {out['gdp_constant'].dtype}, "
            "expected float64. An object-dtype numeric column crashes np.log downstream."
        )
    return out


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def load_labels(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load every configured label source, deflate, and stack into one panel."""
    base_year = int(cfg.get("base_year", 2015))
    sources = cfg.get("label_sources") or ["dose"]

    frames: list[pd.DataFrame] = []
    if "dose" in sources:
        frames.append(load_dose(cfg))

    # District-level GDDP adapters (gddp_*) would append here. Each is a small
    # per-state parser writing the same schema; none are committed yet, so the
    # panel is ADM1-only until a raw GDDP file is dropped in and an adapter added.

    if not frames:
        raise ConfigError(f"No usable label sources among {sources}.")

    panel = pd.concat(frames, ignore_index=True)

    if panel["gdp_constant"].isna().any():
        deflator = load_national_deflator(
            cfg["iso3"], base_year, str(cfg.get("wdi_deflator_indicator", "NY.GDP.DEFL.ZS"))
        )
        panel = deflate(panel, deflator, base_year)

    panel["ingested_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cols = OUTPUT_COLUMNS + [c for c in ("deflation_method",) if c in panel.columns]
    return panel[[c for c in cols if c in panel.columns]].reset_index(drop=True)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@dataclass
class LabelReport:
    country: str
    n_rows: int
    n_regions: int
    year_min: int | None
    year_max: int | None
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.checks)

    def render(self) -> str:
        lines = [
            f"Labels: {self.country}",
            f"  rows        {self.n_rows}",
            f"  regions     {self.n_regions}",
            f"  years       {self.year_min}-{self.year_max}",
            "",
        ]
        for name, passed, detail in self.checks:
            lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        return "\n".join(lines)


def validate_labels(df: pd.DataFrame, cfg: dict[str, Any]) -> LabelReport:
    """Assert the label panel is deflated, keyed cleanly, and free of unit breaks."""
    base_year = int(cfg.get("base_year", 2015))
    max_growth = float(cfg.get("max_real_yoy_growth", 0.5))
    years = df["year"].dropna()

    report = LabelReport(
        country=cfg.get("country", cfg.get("iso3", "unknown")),
        n_rows=len(df),
        n_regions=int(df["source_region_name"].nunique()),
        year_min=int(years.min()) if len(years) else None,
        year_max=int(years.max()) if len(years) else None,
    )

    missing_constant = int(df["gdp_constant"].isna().sum())
    report.checks.append(
        (
            "gdp_constant present",
            missing_constant == 0,
            f"{missing_constant} region-years have no constant-price value",
        )
    )

    # Rule 9: nominal must never be the modelling target. Surface it if the two
    # series are identical, which means deflation silently did nothing.
    if not df["gdp_nominal"].isna().all():
        paired = df.dropna(subset=["gdp_constant", "gdp_nominal"])
        identical = len(paired) > 0 and bool(
            (paired["gdp_constant"] == paired["gdp_nominal"]).all()
        )
        report.checks.append(
            (
                "constant differs from nominal",
                not identical,
                "constant series equals nominal; deflation did not run" if identical else "ok",
            )
        )

    in_range = report.year_min is not None and report.year_min <= base_year <= report.year_max
    report.checks.append(
        (
            "base year within panel",
            bool(in_range),
            f"base_year {base_year} vs panel {report.year_min}-{report.year_max}",
        )
    )

    no_dupes = not df.duplicated(subset=["source_region_name", "parent_name", "year"]).any()
    n_dupes = int(df.duplicated(subset=["source_region_name", "parent_name", "year"]).sum())
    report.checks.append(("no duplicate region-years", no_dupes, f"{n_dupes} duplicates"))

    # A genuine lakh/crore break hits many region-years at once, so tolerate a
    # thin tail (~1% of the panel) but no more. On a small panel this is zero,
    # which is correct: a lone 100x jump there is almost certainly a units error.
    n_breaks, worst = _count_unit_breaks(df, max_growth)
    tolerance = int(0.01 * len(df))
    report.checks.append(
        (
            "no unit breaks",
            n_breaks <= tolerance,
            f"{n_breaks} region-years exceed {max_growth:.0%} real YoY growth "
            f"(tolerate {tolerance}); worst {worst}",
        )
    )

    return report


def _count_unit_breaks(df: pd.DataFrame, max_growth: float) -> tuple[int, str]:
    """Count region-years whose real YoY growth implies a lakh/crore units break.

    Coerces ``gdp_constant`` defensively. Callers upstream now guarantee float64,
    but this is the function where an object-dtype column historically surfaced
    as an opaque ufunc error, so it does not assume its input is clean.
    """
    work = df.copy()
    work["gdp_constant"] = pd.to_numeric(work["gdp_constant"], errors="coerce").astype("float64")
    work = work.dropna(subset=["gdp_constant"])
    work = work[work["gdp_constant"] > 0]
    if work.empty:
        return 0, "n/a"
    work = work.sort_values(["source_region_name", "parent_name", "year"])
    grp = work.groupby(["source_region_name", "parent_name"], sort=False)["gdp_constant"]
    log_growth = grp.transform(lambda s: np.log(s).diff())
    breaks = log_growth.abs() > np.log1p(max_growth)
    n_breaks = int(breaks.sum())
    if n_breaks:
        i = log_growth.abs().idxmax()
        worst = f"{work.loc[i, 'source_region_name']} {int(work.loc[i, 'year'])}"
    else:
        worst = "none"
    return n_breaks, worst


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest and deflate GDP labels")
    parser.add_argument("--country", default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = country_config(args.country)

    panel = load_labels(cfg)
    report = validate_labels(panel, cfg)
    print()
    print(report.render())
    print()

    if not args.no_write and report.ok:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"labels_{cfg['country']}.parquet"
        panel.to_parquet(out, index=False)
        print(f"Wrote {out}")

    if not report.ok:
        print("Label validation failed. Fix before matching.")
        return 1
    print("Labels ready. Next: python -m gdp_proxy.match --propose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
