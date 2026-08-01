"""District population from GHSL, for GDP per capita.

Total GDP ranks districts by size as much as by prosperity: a populous poor
district out-produces a small rich one in absolute terms, which is true and also
not what anyone means by "a lagging district". Per capita is the metric that
answers the question people actually ask, and it needs a denominator.

GHSL P2023A (``JRC/GHSL/P2023A/GHS_POP``) is free in Earth Engine, 100 m, with
five-year epochs from 1975 to 2030. Intermediate years are linearly interpolated
between epochs and flagged as such, because an interpolated value is not an
observation and the difference matters when someone reads a per-capita trend.

**Known circularity**: GHSL disaggregates census counts onto a grid using
built-up-area layers derived partly from satellite imagery. That is optical and
radar imagery, not nighttime lights, so the coupling with our numerator is
indirect and weak. It is not zero. See the caveats in the README and dashboard.

Same discipline as ``extract.py``: batch exports only, skip work whose output
exists, provenance columns on every row, ``ee`` imported lazily.

CLI:
    python -m gdp_proxy.population --submit      # batch export the epochs
    python -m gdp_proxy.population --status
    python -m gdp_proxy.population --ingest      # CSV -> validated parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DATA_DIR, ConfigError, country_config

logger = logging.getLogger(__name__)

EXPORT_DIR = DATA_DIR / "raw" / "exports"
OUT_DIR = DATA_DIR / "processed"

# GHSL epochs available in P2023A. 2025 and 2030 are projections, not
# observations, which is recorded per row.
GHSL_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]
PROJECTED_FROM = 2025


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def bracketing_epochs(years: list[int]) -> list[int]:
    """The smallest set of GHSL epochs that brackets every requested year."""
    if not years:
        raise ConfigError("No years requested.")
    lo, hi = min(years), max(years)
    if lo < GHSL_EPOCHS[0] or hi > GHSL_EPOCHS[-1]:
        raise ConfigError(
            f"Years {lo}-{hi} fall outside GHSL coverage {GHSL_EPOCHS[0]}-{GHSL_EPOCHS[-1]}."
        )
    below = [e for e in GHSL_EPOCHS if e <= lo] or [GHSL_EPOCHS[0]]
    above = [e for e in GHSL_EPOCHS if e >= hi] or [GHSL_EPOCHS[-1]]
    return [e for e in GHSL_EPOCHS if below[-1] <= e <= above[0]]


def export_stem(cfg: dict[str, Any], epoch: int) -> str:
    return f"pop_{cfg['country']}_adm{cfg['admin_level']}_{epoch}"


def output_path(cfg: dict[str, Any], epoch: int) -> Path:
    return OUT_DIR / f"{export_stem(cfg, epoch)}.parquet"


def epochs_to_run(cfg: dict[str, Any], epochs: list[int], force: bool = False) -> list[int]:
    """Skip epochs already extracted. Recomputation wastes quota (rule 4)."""
    if force:
        return epochs
    todo = []
    for epoch in epochs:
        path = output_path(cfg, epoch)
        if path.exists() and path.stat().st_size > 0:
            logger.info("Skipping epoch %d, already extracted at %s", epoch, path.name)
        else:
            todo.append(epoch)
    return todo


# --------------------------------------------------------------------------
# Earth Engine
# --------------------------------------------------------------------------


def submit_epoch(cfg: dict[str, Any], epoch: int, dry_run: bool = False):
    """One batch export of zonal population sums for a single GHSL epoch."""
    import ee

    from .extract import boundaries_to_ee

    regions = boundaries_to_ee(cfg)
    collection = str(cfg.get("ghsl_collection", "JRC/GHSL/P2023A/GHS_POP"))
    band = str(cfg.get("ghsl_band", "population_count"))
    # Native 100 m on purpose: population_count is a per-pixel count, so
    # reducing at a coarser scale resamples and the summed total is wrong.
    scale = int(cfg.get("population_scale_m", 100))

    image = ee.Image(
        ee.ImageCollection(collection).filter(ee.Filter.eq("system:index", str(epoch))).first()
    )
    stats = image.select(band).reduceRegions(
        collection=regions,
        reducer=ee.Reducer.sum(),
        scale=scale,
        tileScale=int(cfg.get("tile_scale", 4)),
    )
    table = stats.map(lambda f: f.set({"epoch": epoch}))

    task = ee.batch.Export.table.toDrive(
        collection=table.select(["region_id", "epoch", "sum"], retainGeometry=False),
        description=export_stem(cfg, epoch),
        folder=str(cfg.get("export_folder", "gdp_proxy_exports")),
        fileNamePrefix=export_stem(cfg, epoch),
        fileFormat="CSV",
    )
    if dry_run:
        logger.info("Dry run: would submit %s", export_stem(cfg, epoch))
        return None

    task.start()
    logger.info("Submitted %s (id %s)", export_stem(cfg, epoch), task.id)
    _write_manifest(cfg, epoch, task.id)
    return task


def _write_manifest(cfg: dict[str, Any], epoch: int, task_id: str | None) -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    (EXPORT_DIR / f"{export_stem(cfg, epoch)}_manifest.json").write_text(
        json.dumps(
            {
                "country": cfg["country"],
                "admin_level": cfg["admin_level"],
                "epoch": epoch,
                "dataset_id": str(cfg.get("ghsl_collection", "JRC/GHSL/P2023A/GHS_POP")),
                "band": str(cfg.get("ghsl_band", "population_count")),
                "scale_m": int(cfg.get("population_scale_m", 100)),
                "task_id": task_id,
                "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# ingest (pure pandas, testable offline)
# --------------------------------------------------------------------------


def parse_export_csv(path: Path) -> pd.DataFrame:
    """Read one exported epoch CSV and normalise it."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ("system:index", ".geo") if c in df.columns])
    if "region_id" not in df.columns:
        raise ConfigError(f"{path.name} has no region_id column. Got {list(df.columns)}")
    if "sum" not in df.columns:
        raise ConfigError(f"{path.name} has no 'sum' column. Got {list(df.columns)}")

    out = pd.DataFrame(
        {
            "region_id": df["region_id"].astype(str),
            "epoch": pd.to_numeric(df["epoch"], errors="coerce").astype("Int64"),
            "population": pd.to_numeric(df["sum"], errors="coerce").astype("float64"),
        }
    )
    return out.sort_values(["region_id", "epoch"]).reset_index(drop=True)


def ingest_epoch(cfg: dict[str, Any], epoch: int, expected_regions: int) -> PopulationReport:
    """Turn a downloaded epoch CSV into a validated, provenance-stamped parquet."""
    csv_path = EXPORT_DIR / f"{export_stem(cfg, epoch)}.csv"
    if not csv_path.exists():
        raise ConfigError(
            f"No export CSV at {csv_path}. Download it from the "
            f"'{cfg.get('export_folder', 'gdp_proxy_exports')}' folder in your Google Drive."
        )

    df = parse_export_csv(csv_path)
    report = validate_population(df, expected_regions, epoch)

    df["dataset_id"] = str(cfg.get("ghsl_collection", "JRC/GHSL/P2023A/GHS_POP"))
    df["population_scale_m"] = int(cfg.get("population_scale_m", 100))
    df["is_projection"] = epoch >= PROJECTED_FROM
    df["extracted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if report.ok:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path(cfg, epoch), index=False)
        logger.info("Wrote %s", output_path(cfg, epoch).name)
    else:
        logger.error("Validation failed for epoch %d, refusing to write parquet", epoch)
    return report


def load_epochs(cfg: dict[str, Any]) -> pd.DataFrame:
    """Concatenate every extracted epoch parquet."""
    stem = f"pop_{cfg['country']}_adm{cfg['admin_level']}_"
    paths = sorted(OUT_DIR.glob(f"{stem}*.parquet"))
    if not paths:
        raise ConfigError(
            f"No population parquets matching {stem}*.parquet in {OUT_DIR}. "
            "Run: python -m gdp_proxy.population --submit, then --ingest."
        )
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["epoch"] = df["epoch"].astype(int)
    return df


# --------------------------------------------------------------------------
# interpolation
# --------------------------------------------------------------------------


def interpolate_population(epochs_df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Linear interpolation between GHSL epochs, flagged as interpolated.

    Returns one row per region-year with ``population``,
    ``population_interpolated`` (False only on an exact epoch) and
    ``population_is_projection`` (True when derived from a 2025+ epoch, which
    GHSL publishes as a projection rather than an observation).

    Linear is the honest choice at five-year spacing: anything fancier implies
    knowledge of within-epoch dynamics that GHSL does not provide.
    """
    if epochs_df.empty:
        raise ConfigError("No population epochs to interpolate from.")

    known = sorted(epochs_df["epoch"].unique())
    lo, hi = min(years), max(years)
    if lo < known[0] or hi > known[-1]:
        raise ConfigError(
            f"Years {lo}-{hi} are not bracketed by extracted epochs {known}. "
            "Extract the bracketing epochs first."
        )

    proj_epochs = set(epochs_df.loc[epochs_df.get("is_projection", False), "epoch"].unique())

    rows: list[dict[str, Any]] = []
    for region_id, grp in epochs_df.groupby("region_id"):
        g = grp.sort_values("epoch")
        xs = g["epoch"].to_numpy(dtype=float)
        ys = g["population"].to_numpy(dtype=float)
        for year in years:
            value = float(np.interp(year, xs, ys))
            exact = year in set(xs.astype(int))
            # A year is projection-derived if either bracketing epoch is.
            nearest_above = min([e for e in known if e >= year], default=known[-1])
            nearest_below = max([e for e in known if e <= year], default=known[0])
            rows.append(
                {
                    "region_id": region_id,
                    "year": int(year),
                    "population": value,
                    "population_interpolated": not exact,
                    "population_is_projection": bool({nearest_above, nearest_below} & proj_epochs),
                    "population_epoch_lo": int(nearest_below),
                    "population_epoch_hi": int(nearest_above),
                }
            )
    out = pd.DataFrame(rows)
    n_interp = int(out["population_interpolated"].sum())
    logger.info(
        "Population: %d region-years (%d interpolated between epochs, %d projection-derived)",
        len(out),
        n_interp,
        int(out["population_is_projection"].sum()),
    )
    return out


def load_population(cfg: dict[str, Any], years: list[int]) -> pd.DataFrame:
    """Extracted epochs, interpolated onto the requested years."""
    return interpolate_population(load_epochs(cfg), years)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@dataclass
class PopulationReport:
    epoch: int
    n_rows: int
    n_regions: int
    total_population: float
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(p for _, p, _ in self.checks)

    def render(self) -> str:
        lines = [
            f"Population {self.epoch}: {self.n_rows} rows, {self.n_regions} regions",
            f"  total population  {self.total_population:,.0f}",
            "",
        ]
        for name, passed, detail in self.checks:
            lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        return "\n".join(lines)


def validate_population(df: pd.DataFrame, expected_regions: int, epoch: int) -> PopulationReport:
    """Structural and plausibility checks on one epoch."""
    checks: list[tuple[str, bool, str]] = []
    n_regions = int(df["region_id"].nunique())
    total = float(df["population"].fillna(0).sum())

    checks.append(
        (
            "region count matches boundaries",
            n_regions == expected_regions,
            f"got {n_regions}, boundary file has {expected_regions}",
        )
    )
    dupes = int(df.duplicated(subset=["region_id", "epoch"]).sum())
    checks.append(("no duplicate region-epochs", dupes == 0, f"{dupes} duplicates"))

    n_missing = int(df["population"].isna().sum())
    checks.append(("no missing population", n_missing == 0, f"{n_missing} null values"))

    negative = int((df["population"].fillna(0) < 0).sum())
    checks.append(("no negative population", negative == 0, f"{negative} rows below zero"))

    # A district with zero people is possible only for uninhabited territory and
    # is far more likely to mean the zonal sum silently failed.
    zeros = int((df["population"].fillna(0) == 0).sum())
    checks.append(
        ("few zero-population districts", zeros <= max(1, int(0.01 * len(df))), f"{zeros} at zero")
    )

    checks.append(("total population positive", total > 0, f"{total:,.0f}"))
    return PopulationReport(
        epoch=epoch,
        n_rows=len(df),
        n_regions=n_regions,
        total_population=total,
        checks=checks,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="GHSL district population extraction")
    parser.add_argument("--country", default=None)
    parser.add_argument("--years", default=None, help="years needing population, e.g. 2014-2025")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = country_config(args.country)

    from .extract import parse_years

    years = (
        parse_years(args.years)
        if args.years
        else parse_years(f"{cfg['start_year']}-{cfg['end_year']}")
    )
    epochs = bracketing_epochs(years)

    if args.status:
        from .extract import task_status

        for row in task_status():
            if str(row["description"]).startswith("pop_"):
                print(f"{row['state']:<10} {row['description']} {row['error']}")
        return 0

    if args.submit:
        from .auth import init_ee

        init_ee()
        todo = epochs_to_run(cfg, epochs, force=args.force)
        if not todo:
            print("Nothing to do. All bracketing epochs are already extracted.")
            return 0
        print(f"Submitting {len(todo)} epoch export(s): {todo}")
        for epoch in todo:
            submit_epoch(cfg, epoch, dry_run=args.dry_run)
        print("\nTasks submitted. Poll with --status, then download the CSVs from Drive.")
        return 0

    if args.ingest:
        from .boundaries import load_snapshot

        expected = len(load_snapshot(cfg["country"]))
        failed = 0
        for epoch in epochs:
            if output_path(cfg, epoch).exists() and not args.force:
                print(f"Epoch {epoch}: parquet exists, skipping.")
                continue
            report = ingest_epoch(cfg, epoch, expected)
            print()
            print(report.render())
            failed += 0 if report.ok else 1
        print()
        if failed:
            print(f"{failed} epoch(s) failed validation.")
            return 1
        print("Population ready.")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
