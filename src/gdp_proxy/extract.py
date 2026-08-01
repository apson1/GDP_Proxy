"""Phase 2b: zonal extraction from Earth Engine.

Everything here is batch. There is no code path that pulls per-polygon values
through ``getInfo``. Interactive reduction over hundreds of districts across
hundreds of months is how you burn a month of EECU quota in an afternoon.

Workflow:

    python -m gdp_proxy.extract --pilot --submit     # 2 years, submit tasks
    python -m gdp_proxy.extract --status             # poll running tasks
    # download the CSVs from Drive into data/raw/exports/
    python -m gdp_proxy.extract --ingest             # CSV -> validated parquet
    python -m gdp_proxy.extract --years 2014-2025 --submit   # full run

``ee`` is imported lazily so the parsing and validation logic can be tested
without credentials.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATA_DIR, ConfigError, country_config
from .masks import mask_summary, validate_mask_config

logger = logging.getLogger(__name__)

EXPORT_DIR = DATA_DIR / "raw" / "exports"
OUT_DIR = DATA_DIR / "processed"

# Columns reduceRegions produces, mapped to the names used downstream.
COLUMN_MAP = {
    "rad_sum": "sol",
    "rad_mean": "mean_rad",
    "rad_p50": "median_rad",
    "rad_p90": "p90_rad",
    "lit_sum": "lit_pixels",
    "valid_sum": "valid_pixels",
}

REQUIRED_OUTPUT_COLUMNS = [
    "region_id",
    "year",
    "month",
    "sol",
    "mean_rad",
    "lit_pixels",
    "valid_pixels",
]


# --------------------------------------------------------------------------
# planning: what to run, what to skip
# --------------------------------------------------------------------------


def parse_years(spec: str) -> list[int]:
    """Accept '2019', '2014-2025' or '2019,2021,2023'."""
    years: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo_i, hi_i = int(lo), int(hi)
            if hi_i < lo_i:
                raise ConfigError(f"Bad year range '{part}': end is before start")
            years.update(range(lo_i, hi_i + 1))
        else:
            years.add(int(part))
    if not years:
        raise ConfigError(f"No years parsed from '{spec}'")
    return sorted(years)


def series_of(cfg: dict[str, Any]) -> str:
    """Which VIIRS product family this run uses: 'annual' or 'monthly'."""
    series = str(cfg.get("training_series", "monthly")).lower()
    if series not in {"annual", "monthly"}:
        raise ConfigError(f"training_series must be 'annual' or 'monthly', got {series!r}")
    return series


def dataset_id(cfg: dict[str, Any], series: str | None = None, year: int | None = None) -> str:
    """The GEE collection id for this series, and for this year if annual.

    The annual family is split by processing version: ANNUAL_V21 holds 2013-2021
    and V22 holds 2022 onward. Asking V22 for 2018 returns an empty collection
    and a null-image error several calls later, so the year has to be resolved
    here rather than assumed.
    """
    series = series or series_of(cfg)
    if series != "annual":
        return str(cfg["viirs_monthly"])

    versions = cfg.get("viirs_annual_versions") or []
    if year is not None and versions:
        for v in versions:
            if int(v["start_year"]) <= year <= int(v["end_year"]):
                return str(v["id"])
        covered = ", ".join(f"{v['start_year']}-{v['end_year']}" for v in versions)
        raise ConfigError(
            f"No annual VIIRS collection covers {year}. Configured coverage: {covered}."
        )
    return str(cfg["viirs_annual"])


def export_stem(cfg: dict[str, Any], year: int, series: str | None = None) -> str:
    """Output name. Annual and monthly never share a file: mixing product
    families mid-series creates growth that is not there (pitfalls.md)."""
    series = series or series_of(cfg)
    suffix = "_annual" if series == "annual" else ""
    return f"sol_{cfg['country']}_adm{cfg['admin_level']}{suffix}_{year}"


def output_path(cfg: dict[str, Any], year: int, series: str | None = None) -> Path:
    return OUT_DIR / f"{export_stem(cfg, year, series)}.parquet"


def years_to_run(cfg: dict[str, Any], years: list[int], force: bool = False) -> list[int]:
    """Drop years whose parquet already exists. Recomputation wastes quota."""
    if force:
        return years
    todo = []
    for year in years:
        path = output_path(cfg, year)
        if path.exists() and path.stat().st_size > 0:
            logger.info("Skipping %d, already extracted at %s", year, path.name)
        else:
            todo.append(year)
    return todo


def pilot_years(cfg: dict[str, Any], n: int = 2) -> list[int]:
    """The most recent n years in the configured range."""
    end = int(cfg["end_year"])
    start = max(int(cfg["start_year"]), end - n + 1)
    return list(range(start, end + 1))


# --------------------------------------------------------------------------
# Earth Engine side
# --------------------------------------------------------------------------


def boundaries_to_ee(cfg: dict[str, Any], asset_id: str | None = None):
    """Get the boundary FeatureCollection into Earth Engine.

    Preferred path is a GEE asset you uploaded once. The client-side fallback
    exists for small admin sets only: several hundred detailed district
    polygons will blow the request size limit, and the error you get back does
    not say so clearly.
    """
    # .env first: the asset path embeds a Cloud project id, which CLAUDE.md keeps
    # out of committed config. countries.yaml remains a fallback for forks that
    # prefer to pin it there.
    import os

    import ee

    from .boundaries import load_snapshot
    from .config import load_env

    load_env()
    asset_id = asset_id or os.getenv("GEE_BOUNDARY_ASSET", "").strip() or cfg.get("boundary_asset")
    if asset_id:
        logger.info("Using boundary asset %s", asset_id)
        return ee.FeatureCollection(asset_id)

    gdf = load_snapshot(cfg["country"])
    n = len(gdf)
    if n > 200:
        raise ConfigError(
            f"{n} polygons is too many to send inline to Earth Engine.\n"
            "Upload the boundaries once as a GEE asset, then set GEE_BOUNDARY_ASSET "
            "in .env (it holds a project id, so it does not belong in committed "
            "config):\n"
            "  1. Export data/processed/boundaries_*.parquet to a shapefile or GeoJSON\n"
            "  2. Code Editor -> Assets -> New -> Shapefile, upload it\n"
            "  3. GEE_BOUNDARY_ASSET=projects/<your-project>/assets/<name>\n"
            "This is a one-time step and makes every extraction faster and cheaper."
        )

    logger.warning("Sending %d polygons inline. Upload an asset before scaling up.", n)
    features = [
        ee.Feature(
            ee.Geometry(row.geometry.__geo_interface__),
            {"region_id": row.region_id},
        )
        for row in gdf.itertuples()
    ]
    return ee.FeatureCollection(features)


def _annual_mask_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Config view with the ANNUAL_V22 band names swapped in.

    The annual product does not carry ``avg_rad``; it exposes ``average`` and
    ``average_masked``. Everything else about the masking is unchanged.
    """
    return {
        **cfg,
        "radiance_band": str(cfg.get("annual_radiance_band", "average_masked")),
        "coverage_band": str(cfg.get("annual_coverage_band", "cf_cvg")),
    }


def images_for_year(cfg: dict[str, Any], year: int, series: str | None = None):
    """The masked image collection for one year, from the configured series.

    Annual returns a one-image collection (the inter-calibrated composite);
    monthly returns up to twelve. Downstream reduction is identical, so the only
    difference that reaches the parquet is how many rows per region-year.
    """
    import ee

    from .masks import apply_masks

    series = series or series_of(cfg)
    mask_cfg = _annual_mask_cfg(cfg) if series == "annual" else cfg
    coll = ee.ImageCollection(dataset_id(cfg, series, year)).filterDate(
        f"{year}-01-01", f"{year + 1}-01-01"
    )
    return coll.map(lambda img: apply_masks(img, mask_cfg))


def monthly_images(cfg: dict[str, Any], year: int):
    """Backwards-compatible alias for the monthly series."""
    return images_for_year(cfg, year, series="monthly")


def source_image_ids(cfg: dict[str, Any], year: int, series: str | None = None) -> list[str]:
    """Small list, safe to getInfo. Recorded as provenance in the output."""
    import ee

    coll = ee.ImageCollection(dataset_id(cfg, series, year)).filterDate(
        f"{year}-01-01", f"{year + 1}-01-01"
    )
    return list(coll.aggregate_array("system:index").getInfo())


def _reducer():
    import ee

    return (
        ee.Reducer.sum()
        .combine(ee.Reducer.mean(), sharedInputs=True)
        .combine(ee.Reducer.percentile([50, 90]), sharedInputs=True)
    )


def submit_year(
    cfg: dict[str, Any],
    year: int,
    asset_id: str | None = None,
    dry_run: bool = False,
    series: str | None = None,
):
    """Create and start one Export.table.toDrive task for a year."""
    import ee

    series = series or series_of(cfg)
    regions = boundaries_to_ee(cfg, asset_id)
    coll = images_for_year(cfg, year, series)
    scale = int(cfg.get("scale_m", 500))
    reducer = _reducer()

    def reduce_one(image):
        stats = image.reduceRegions(
            collection=regions,
            reducer=reducer,
            scale=scale,
            tileScale=int(cfg.get("tile_scale", 4)),
        )
        date = ee.Date(image.get("system:time_start"))
        return stats.map(
            lambda f: f.set(
                {
                    "year": date.get("year"),
                    "month": date.get("month"),
                    "image_id": image.get("system:index"),
                }
            )
        )

    table = coll.map(reduce_one).flatten()

    keep = [
        "region_id",
        "year",
        "month",
        "image_id",
        "rad_sum",
        "rad_mean",
        "rad_p50",
        "rad_p90",
        "lit_sum",
        "valid_sum",
    ]

    task = ee.batch.Export.table.toDrive(
        collection=table.select(keep, retainGeometry=False),
        description=export_stem(cfg, year, series),
        folder=str(cfg.get("export_folder", "gdp_proxy_exports")),
        fileNamePrefix=export_stem(cfg, year, series),
        fileFormat="CSV",
    )

    if dry_run:
        logger.info("Dry run: would submit %s", export_stem(cfg, year, series))
        return None

    task.start()
    logger.info("Submitted task %s (id %s)", export_stem(cfg, year, series), task.id)
    _write_manifest(cfg, year, task.id, series)
    return task


def _write_manifest(
    cfg: dict[str, Any], year: int, task_id: str | None, series: str | None = None
) -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "country": cfg["country"],
        "admin_level": cfg["admin_level"],
        "year": year,
        "series": series or series_of(cfg),
        "dataset_id": dataset_id(cfg, series, year),
        "scale_m": int(cfg.get("scale_m", 500)),
        "masking": mask_summary(cfg),
        "task_id": task_id,
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        manifest["image_ids"] = source_image_ids(cfg, year, series)
    except Exception as exc:  # noqa: BLE001 - provenance is best effort
        logger.warning("Could not record image ids for %d: %s", year, exc)
        manifest["image_ids"] = []
    (EXPORT_DIR / f"{export_stem(cfg, year, series)}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def task_status() -> list[dict[str, Any]]:
    """Summarise recent Earth Engine tasks."""
    import ee

    out = []
    for task in ee.batch.Task.list()[:25]:
        status = task.status()
        out.append(
            {
                "description": status.get("description"),
                "state": status.get("state"),
                "id": status.get("id"),
                "error": status.get("error_message", ""),
            }
        )
    return out


# --------------------------------------------------------------------------
# ingestion and validation (pure pandas, fully testable offline)
# --------------------------------------------------------------------------


@dataclass
class ExtractionReport:
    year: int
    n_rows: int
    n_regions: int
    n_months: int
    n_missing_cells: int
    pct_low_coverage: float
    checks: list[tuple[str, bool, str]]

    @property
    def ok(self) -> bool:
        return all(p for _, p, _ in self.checks)

    def render(self) -> str:
        lines = [
            f"Extraction {self.year}: {self.n_rows} rows, "
            f"{self.n_regions} regions x {self.n_months} months",
            f"  missing region-months  {self.n_missing_cells}",
            f"  low-coverage cells     {self.pct_low_coverage:.1f}%",
            "",
        ]
        for name, passed, detail in self.checks:
            lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        return "\n".join(lines)


def parse_export_csv(path: Path) -> pd.DataFrame:
    """Read one exported CSV and normalise it to the downstream schema."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ("system:index", ".geo") if c in df.columns])
    df = df.rename(columns=COLUMN_MAP)

    if "region_id" not in df.columns:
        raise ConfigError(f"{path.name} has no region_id column. Got {list(df.columns)}")

    for col in ("sol", "mean_rad", "median_rad", "p90_rad", "lit_pixels", "valid_pixels"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("year", "month"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # A region-month that GEE could not reduce comes back empty, not zero.
    # Keep it as NaN. Rule 10: missing is not dark.
    df["is_missing"] = df["valid_pixels"].isna() | (df["valid_pixels"].fillna(0) == 0)
    return df.sort_values(["region_id", "year", "month"]).reset_index(drop=True)


def validate_extraction(
    df: pd.DataFrame,
    expected_regions: int,
    year: int,
    cfg: dict[str, Any] | None = None,
    series: str = "monthly",
) -> ExtractionReport:
    """Assert the shape and sanity of one year of extracted data."""
    cfg = cfg or {}
    checks: list[tuple[str, bool, str]] = []

    n_regions = int(df["region_id"].nunique())
    n_months = int(df["month"].nunique())
    # The annual composite is one image per year, so one row per region.
    periods = 1 if series == "annual" else 12
    expected_rows = expected_regions * periods

    checks.append(
        (
            "region count matches boundaries",
            n_regions == expected_regions,
            f"got {n_regions}, boundary file has {expected_regions}",
        )
    )
    checks.append(
        (
            f"{periods} period(s) present",
            n_months == periods,
            f"got {n_months} distinct period(s), expected {periods} for {series}",
        )
    )
    checks.append(
        (
            "row count is regions x periods",
            len(df) == expected_rows,
            f"got {len(df)}, expected {expected_rows}",
        )
    )

    dupes = int(df.duplicated(subset=["region_id", "year", "month"]).sum())
    checks.append(("no duplicate region-months", dupes == 0, f"{dupes} duplicates"))

    missing_cols = [c for c in REQUIRED_OUTPUT_COLUMNS if c not in df.columns]
    checks.append(
        (
            "required columns present",
            not missing_cols,
            f"missing {missing_cols}" if missing_cols else "all present",
        )
    )

    if "sol" in df.columns:
        negative = int((df["sol"].fillna(0) < 0).sum())
        checks.append(("no negative SOL", negative == 0, f"{negative} rows below zero"))

    n_missing = int(df["is_missing"].sum())
    pct_missing = n_missing / len(df) * 100 if len(df) else 0.0
    # Monsoon and high-latitude winter legitimately produce gaps. A third of the
    # panel missing means the mask is wrong, not the weather.
    checks.append(
        (
            "missing share is plausible",
            pct_missing < 33.0,
            f"{pct_missing:.1f}% of region-months unusable",
        )
    )

    zero_but_present = int(((df["sol"].fillna(0) == 0) & (~df["is_missing"])).sum())
    checks.append(
        (
            "few genuinely dark region-months",
            zero_but_present < len(df) * 0.05,
            f"{zero_but_present} region-months are lit-free but well observed",
        )
    )

    return ExtractionReport(
        year=year,
        n_rows=len(df),
        n_regions=n_regions,
        n_months=n_months,
        n_missing_cells=n_missing,
        pct_low_coverage=pct_missing,
        checks=checks,
    )


def ingest(
    cfg: dict[str, Any], year: int, expected_regions: int, series: str | None = None
) -> ExtractionReport:
    """Turn a downloaded CSV into a validated, provenance-stamped parquet."""
    series = series or series_of(cfg)
    csv_path = EXPORT_DIR / f"{export_stem(cfg, year, series)}.csv"
    if not csv_path.exists():
        raise ConfigError(
            f"No export CSV at {csv_path}. Download it from the "
            f"'{cfg.get('export_folder', 'gdp_proxy_exports')}' folder in your Google Drive."
        )

    df = parse_export_csv(csv_path)
    report = validate_extraction(df, expected_regions, year, cfg, series=series)

    manifest_path = EXPORT_DIR / f"{export_stem(cfg, year, series)}_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )

    df["series"] = series
    df["dataset_id"] = dataset_id(cfg, series, year)
    df["n_source_images"] = len(manifest.get("image_ids", []))
    df["extracted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    df["noise_floor"] = float(cfg.get("noise_floor", 0.4))
    df["min_cf_cvg"] = int(cfg.get("min_cf_cvg", 2))

    if report.ok:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path(cfg, year, series), index=False)
        logger.info("Wrote %s", output_path(cfg, year, series).name)
    else:
        logger.error("Validation failed for %d, refusing to write parquet", year)

    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="VIIRS zonal extraction")
    parser.add_argument("--country", default=None)
    parser.add_argument("--years", default=None, help="e.g. 2023 or 2014-2025 or 2019,2021")
    parser.add_argument("--pilot", action="store_true", help="most recent 2 years only")
    parser.add_argument("--submit", action="store_true", help="start Earth Engine export tasks")
    parser.add_argument("--status", action="store_true", help="poll task status")
    parser.add_argument("--ingest", action="store_true", help="CSV -> parquet")
    parser.add_argument("--force", action="store_true", help="re-run years already extracted")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--series",
        default=None,
        choices=["annual", "monthly"],
        help="override training_series from config",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = country_config(args.country)
    if args.series:
        cfg = {**cfg, "training_series": args.series}
    series = series_of(cfg)

    problems = validate_mask_config(cfg)
    if problems:
        for p in problems:
            print(f"[FAIL] mask config: {p}")
        return 1

    if args.status:
        from .auth import init_ee

        init_ee()
        rows = task_status()
        if not rows:
            print("No recent Earth Engine tasks found for this project.")
            return 0
        for row in rows:
            print(f"{row['state']:<12} {row['description']} {row['error']}")
        return 0

    if args.pilot:
        years = pilot_years(cfg)
    elif args.years:
        years = parse_years(args.years)
    else:
        years = parse_years(f"{cfg['start_year']}-{cfg['end_year']}")

    if args.submit:
        from .auth import init_ee

        init_ee()
        todo = years_to_run(cfg, years, force=args.force)  # rule 4: never recompute
        if not todo:
            print("Nothing to do. All requested years are already extracted.")
            return 0
        print(f"Submitting {len(todo)} {series} export task(s): {todo}")
        for year in todo:
            submit_year(cfg, year, dry_run=args.dry_run, series=series)
        print("\nTasks submitted. Poll with --status, then download the CSVs from Drive.")
        return 0

    if args.ingest:
        from .boundaries import load_snapshot

        expected = len(load_snapshot(cfg["country"]))
        failed = 0
        for year in years:
            report = ingest(cfg, year, expected, series=series)
            print()
            print(report.render())
            failed += 0 if report.ok else 1
        print()
        if failed:
            print(f"{failed} year(s) failed validation. Do not proceed to Phase 3.")
            return 1
        print("Phase 2 complete.")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
