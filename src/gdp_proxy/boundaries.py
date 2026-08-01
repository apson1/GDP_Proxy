"""Phase 1: administrative boundaries.

Two swappable sources behind one interface:

  gadm           free for academic and non-commercial use only, no redistribution
  geoboundaries  CC-BY / ODbL, safe for commercial use

The output is a GeoDataFrame with a stable synthetic ``region_id``. Nothing
downstream in this project may ever join on a name string. District names are
transliterated inconsistently, get renamed by state governments, and will
silently drop rows.

CLI:
    python -m gdp_proxy.boundaries --country india
    python -m gdp_proxy.boundaries --country india --source geoboundaries --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from pyproj import Geod
from shapely.validation import make_valid

from .config import DATA_DIR, ConfigError, country_config

logger = logging.getLogger(__name__)

GADM_GPKG_URL = "https://geodata.ucdavis.edu/gadm/gadm{ver}/gpkg/gadm{vernodot}_{iso3}.gpkg"
GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/ADM{level}/"

_GEOD = Geod(ellps="WGS84")

RAW_DIR = DATA_DIR / "raw" / "boundaries"
OUT_DIR = DATA_DIR / "processed"


# --------------------------------------------------------------------------
# region_id
# --------------------------------------------------------------------------


def make_region_id(iso3: str, level: int, source: str, source_code: str) -> str:
    """Build a deterministic, name-independent key.

    Stable for a given (source, source_code) pair, so re-running the loader
    never reshuffles ids. Changing boundary source or version deliberately
    produces different ids, which is correct: they are different geographies
    and must not be silently treated as the same units.
    """
    payload = f"{source}|{iso3}|{level}|{source_code}".encode()
    digest = hashlib.blake2b(payload, digest_size=5).hexdigest()
    return f"{iso3}{level}-{digest}"


# --------------------------------------------------------------------------
# download helpers
# --------------------------------------------------------------------------


def _download(url: str, dest: Path, force: bool = False, timeout: int = 300) -> Path:
    """Stream a file to disk. Skips if already present unless force."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        logger.info("Using cached download %s", dest.name)
        return dest

    logger.info("Downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.replace(dest)
    logger.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------


def load_gadm(cfg: dict[str, Any], force: bool = False) -> gpd.GeoDataFrame:
    """Load a GADM level-N layer for the target country."""
    iso3 = cfg["iso3"]
    level = int(cfg["admin_level"])
    version = str(cfg.get("boundary_version", "4.1"))
    url = GADM_GPKG_URL.format(ver=version, vernodot=version.replace(".", ""), iso3=iso3)
    path = _download(url, RAW_DIR / f"gadm{version.replace('.', '')}_{iso3}.gpkg", force=force)

    layer = f"ADM_ADM_{level}"
    gdf = gpd.read_file(path, layer=layer)

    code_col, name_col = f"GID_{level}", f"NAME_{level}"
    missing = [c for c in (code_col, name_col) if c not in gdf.columns]
    if missing:
        raise ConfigError(
            f"GADM layer {layer} is missing columns {missing}. Got {list(gdf.columns)}"
        )

    out = gpd.GeoDataFrame(
        {
            "source_code": gdf[code_col].astype(str),
            "name": gdf[name_col].astype(str),
            "parent_name": gdf["NAME_1"].astype(str) if "NAME_1" in gdf else pd.NA,
            "name_variants": gdf.get("VARNAME_" + str(level), pd.Series([pd.NA] * len(gdf))),
            "geometry": gdf.geometry,
        },
        crs=gdf.crs,
    )
    out.attrs["boundary_source"] = "gadm"
    out.attrs["boundary_version"] = version
    out.attrs["boundary_license"] = "GADM: academic and non-commercial use only"
    return out


def load_geoboundaries(cfg: dict[str, Any], force: bool = False) -> gpd.GeoDataFrame:
    """Load a geoBoundaries gbOpen release for the target country."""
    iso3 = cfg["iso3"]
    level = int(cfg["admin_level"])
    meta_url = GEOBOUNDARIES_API.format(iso3=iso3, level=level)

    logger.info("Querying geoBoundaries API %s", meta_url)
    meta = requests.get(meta_url, timeout=60).json()
    if isinstance(meta, list):
        meta = meta[0]

    gj_url = meta["gjDownloadURL"]
    path = _download(gj_url, RAW_DIR / f"geoboundaries_{iso3}_ADM{level}.geojson", force=force)
    gdf = gpd.read_file(path)

    code_col = "shapeID" if "shapeID" in gdf.columns else "shapeGroup"
    out = gpd.GeoDataFrame(
        {
            "source_code": gdf[code_col].astype(str),
            "name": gdf["shapeName"].astype(str),
            "parent_name": pd.NA,
            "name_variants": pd.NA,
            "geometry": gdf.geometry,
        },
        crs=gdf.crs or "EPSG:4326",
    )
    out.attrs["boundary_source"] = "geoboundaries"
    out.attrs["boundary_version"] = str(meta.get("boundaryID", "current"))
    out.attrs["boundary_license"] = str(meta.get("boundaryLicense", "see geoBoundaries"))
    out.attrs["boundary_year"] = str(meta.get("boundaryYearRepresented", ""))
    out.attrs["api_reported_count"] = int(meta.get("admUnitCount", 0) or 0)
    return out


LOADERS = {"gadm": load_gadm, "geoboundaries": load_geoboundaries}


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------


def clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Repair invalid geometries and drop empties. Multipart shapes are kept.

    Island districts and enclaves are legitimately multipolygons. Exploding
    them would inflate the polygon count and break the join cardinality.
    """
    gdf = gdf.copy()
    if gdf.crs is None:
        raise ConfigError("Boundary source returned no CRS. Refusing to guess.")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    invalid = ~gdf.geometry.is_valid
    n_invalid = int(invalid.sum())
    if n_invalid:
        logger.warning("Repairing %d invalid geometries", n_invalid)
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)

    empty = gdf.geometry.is_empty | gdf.geometry.isna()
    if empty.any():
        logger.warning("Dropping %d empty geometries", int(empty.sum()))
        gdf = gdf.loc[~empty].copy()

    gdf.attrs["n_repaired"] = n_invalid
    return gdf


def add_geodesic_area(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add area_km2 computed on the WGS84 ellipsoid.

    Uses pyproj.Geod rather than a projected CRS on purpose. Picking the wrong
    projection is easy: EPSG:7755 for India is conformal, not equal area, and
    silently distorts areas. Geodesic area needs no projection choice.
    """
    gdf = gdf.copy()
    areas = []
    for geom in gdf.geometry:
        area_m2, _ = _GEOD.geometry_area_perimeter(geom)
        areas.append(abs(area_m2) / 1e6)
    gdf["area_km2"] = areas
    return gdf


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@dataclass
class BoundaryReport:
    country: str
    source: str
    version: str
    n_polygons: int
    total_area_km2: float
    n_repaired: int
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.checks)

    def render(self) -> str:
        lines = [
            f"Boundaries: {self.country} from {self.source} {self.version}",
            f"  polygons     {self.n_polygons}",
            f"  total area   {self.total_area_km2:,.0f} km2",
            f"  repaired     {self.n_repaired} invalid geometries",
            "",
        ]
        for name, passed, detail in self.checks:
            lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        return "\n".join(lines)


def validate(gdf: gpd.GeoDataFrame, cfg: dict[str, Any]) -> BoundaryReport:
    """Structural checks plus optional checks against reference values in config."""
    report = BoundaryReport(
        country=cfg["country"],
        source=str(gdf.attrs.get("boundary_source", "unknown")),
        version=str(gdf.attrs.get("boundary_version", "unknown")),
        n_polygons=len(gdf),
        total_area_km2=float(gdf["area_km2"].sum()),
        n_repaired=int(gdf.attrs.get("n_repaired", 0)),
    )

    dupes = int(gdf["region_id"].duplicated().sum())
    report.checks.append(("region_id unique", dupes == 0, f"{dupes} duplicates"))

    null_names = int(gdf["name"].isna().sum() + (gdf["name"].astype(str).str.strip() == "").sum())
    report.checks.append(("names present", null_names == 0, f"{null_names} blank names"))

    still_invalid = int((~gdf.geometry.is_valid).sum())
    report.checks.append(
        ("geometries valid", still_invalid == 0, f"{still_invalid} still invalid after repair")
    )

    tiny = int((gdf["area_km2"] < 1.0).sum())
    report.checks.append(("no sliver polygons", tiny == 0, f"{tiny} polygons under 1 km2"))

    expected = cfg.get("expected_polygon_count")
    if expected:
        match = len(gdf) == int(expected)
        report.checks.append(
            (
                "polygon count matches config",
                match,
                f"got {len(gdf)}, expected {expected}",
            )
        )
    else:
        report.checks.append(
            (
                "polygon count matches config",
                True,
                "skipped: set expected_polygon_count in countries.yaml after your first run",
            )
        )

    ref_area = cfg.get("reference_land_area_km2")
    if ref_area:
        tol = float(cfg.get("area_tolerance_pct", 1.0))
        diff_pct = abs(report.total_area_km2 - float(ref_area)) / float(ref_area) * 100
        report.checks.append(
            (
                "total area within tolerance",
                diff_pct <= tol,
                f"{diff_pct:.2f}% from reference {float(ref_area):,.0f} km2 (tolerance {tol}%)",
            )
        )
    else:
        report.checks.append(
            ("total area within tolerance", True, "skipped: no reference_land_area_km2 in config")
        )

    api_count = gdf.attrs.get("api_reported_count")
    if api_count:
        report.checks.append(
            (
                "count matches source metadata",
                len(gdf) == int(api_count),
                f"loader got {len(gdf)}, source metadata says {api_count}",
            )
        )

    return report


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def build(
    country: str | None = None,
    source: str | None = None,
    force: bool = False,
    write: bool = True,
) -> tuple[gpd.GeoDataFrame, BoundaryReport]:
    """Download, clean, key, validate and snapshot the boundary set."""
    cfg = country_config(country)
    source = (source or cfg.get("boundary_source", "gadm")).lower()
    if source not in LOADERS:
        raise ConfigError(f"Unknown boundary_source '{source}'. Options: {sorted(LOADERS)}")

    gdf = LOADERS[source](cfg, force=force)
    attrs = dict(gdf.attrs)

    gdf = clean_geometries(gdf)
    attrs["n_repaired"] = gdf.attrs.get("n_repaired", 0)
    gdf = add_geodesic_area(gdf)

    level = int(cfg["admin_level"])
    gdf["region_id"] = [
        make_region_id(cfg["iso3"], level, source, code) for code in gdf["source_code"]
    ]

    gdf["iso3"] = cfg["iso3"]
    gdf["admin_level"] = level
    gdf["boundary_source"] = source
    gdf["boundary_version"] = attrs.get("boundary_version", "unknown")
    gdf["retrieved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    cols = [
        "region_id",
        "source_code",
        "name",
        "parent_name",
        "name_variants",
        "area_km2",
        "iso3",
        "admin_level",
        "boundary_source",
        "boundary_version",
        "retrieved_at",
        "geometry",
    ]
    gdf = gdf[[c for c in cols if c in gdf.columns]].sort_values("region_id").reset_index(drop=True)
    gdf.attrs.update(attrs)

    report = validate(gdf, cfg)

    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"boundaries_{cfg['country']}_adm{level}_{source}"
        gdf.to_parquet(OUT_DIR / f"{stem}.parquet", index=False)
        (OUT_DIR / f"{stem}_manifest.json").write_text(
            json.dumps(
                {
                    "country": cfg["country"],
                    "iso3": cfg["iso3"],
                    "admin_level": level,
                    "source": source,
                    "version": str(attrs.get("boundary_version")),
                    "license": str(attrs.get("boundary_license")),
                    "n_polygons": len(gdf),
                    "total_area_km2": round(float(gdf["area_km2"].sum()), 2),
                    "retrieved_at": gdf["retrieved_at"].iloc[0],
                    "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in report.checks],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Wrote %s", OUT_DIR / f"{stem}.parquet")

    return gdf, report


def load_snapshot(country: str | None = None, source: str | None = None) -> gpd.GeoDataFrame:
    """Read the committed snapshot. Downstream code uses this, never build()."""
    cfg = country_config(country)
    source = (source or cfg.get("boundary_source", "gadm")).lower()
    path = OUT_DIR / f"boundaries_{cfg['country']}_adm{cfg['admin_level']}_{source}.parquet"
    if not path.exists():
        raise ConfigError(f"No boundary snapshot at {path}. Run: python -m gdp_proxy.boundaries")
    return gpd.read_parquet(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the boundary snapshot")
    parser.add_argument("--country", default=None)
    parser.add_argument("--source", default=None, choices=sorted(LOADERS) + [None])
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _, report = build(
        country=args.country, source=args.source, force=args.force, write=not args.no_write
    )
    print()
    print(report.render())
    print()
    if not report.ok:
        print("Boundary validation failed. Do not proceed to Phase 2.")
        return 1
    print("Phase 1 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
