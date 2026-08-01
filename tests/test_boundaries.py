"""Phase 1 tests.

The unit tests run offline against synthetic geometries with known answers.
The integration test is marked ``network`` and hits the real source; run it
with ``pytest -m network``.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon

from gdp_proxy.boundaries import (
    add_geodesic_area,
    build,
    clean_geometries,
    make_region_id,
    validate,
)
from gdp_proxy.config import ConfigError


def _square(lon: float, lat: float, size: float = 1.0) -> Polygon:
    return Polygon([(lon, lat), (lon + size, lat), (lon + size, lat + size), (lon, lat + size)])


def _frame(geoms, names=None) -> gpd.GeoDataFrame:
    names = names or [f"region_{i}" for i in range(len(geoms))]
    gdf = gpd.GeoDataFrame(
        {
            "source_code": [f"X.{i}_1" for i in range(len(geoms))],
            "name": names,
            "parent_name": ["Parent"] * len(geoms),
            "name_variants": [pd.NA] * len(geoms),
            "geometry": geoms,
        },
        crs="EPSG:4326",
    )
    gdf.attrs["boundary_source"] = "test"
    gdf.attrs["boundary_version"] = "0"
    return gdf


# --------------------------------------------------------------------- ids


def test_region_id_is_deterministic():
    a = make_region_id("IND", 2, "gadm", "IND.16.20_1")
    b = make_region_id("IND", 2, "gadm", "IND.16.20_1")
    assert a == b
    assert a.startswith("IND2-")


def test_region_id_ignores_names_but_separates_sources():
    gadm = make_region_id("IND", 2, "gadm", "IND.16.20_1")
    geob = make_region_id("IND", 2, "geoboundaries", "IND.16.20_1")
    assert gadm != geob, "different sources must not collide into the same key"


def test_region_id_is_unique_across_codes():
    ids = {make_region_id("IND", 2, "gadm", f"IND.{i}_1") for i in range(5000)}
    assert len(ids) == 5000


# ---------------------------------------------------------------- geometry


def test_clean_repairs_bowtie_geometry():
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
    gdf = _frame([bowtie])
    assert not gdf.geometry.is_valid.all()

    cleaned = clean_geometries(gdf)
    assert cleaned.geometry.is_valid.all()
    assert cleaned.attrs["n_repaired"] == 1


def test_clean_drops_empty_but_keeps_multipart():
    islands = MultiPolygon([_square(0, 0), _square(5, 5)])
    gdf = _frame([islands, _square(10, 10), Polygon()])
    cleaned = clean_geometries(gdf)
    assert len(cleaned) == 2, "empty dropped, multipolygon kept as one row"


def test_clean_reprojects_to_wgs84():
    gdf = _frame([_square(0, 0)]).to_crs("EPSG:3857")
    cleaned = clean_geometries(gdf)
    assert cleaned.crs.to_epsg() == 4326


def test_clean_rejects_missing_crs():
    gdf = _frame([_square(0, 0)])
    gdf.crs = None
    with pytest.raises(ConfigError):
        clean_geometries(gdf)


# -------------------------------------------------------------------- area


def test_geodesic_area_matches_known_value():
    """One degree square at the equator is about 12,308 km2 on WGS84."""
    gdf = add_geodesic_area(_frame([_square(0, 0)]))
    assert 12200 < gdf["area_km2"].iloc[0] < 12400


def test_area_shrinks_with_latitude():
    """A projection mistake usually shows up as latitude-invariant area."""
    equator = add_geodesic_area(_frame([_square(0, 0)]))["area_km2"].iloc[0]
    high_lat = add_geodesic_area(_frame([_square(0, 60)]))["area_km2"].iloc[0]
    assert high_lat < equator * 0.6


def test_multipart_area_is_the_sum_of_parts():
    parts = add_geodesic_area(_frame([_square(0, 0), _square(5, 5)]))["area_km2"].sum()
    combined = add_geodesic_area(_frame([MultiPolygon([_square(0, 0), _square(5, 5)])]))[
        "area_km2"
    ].iloc[0]
    assert combined == pytest.approx(parts, rel=1e-9)


# -------------------------------------------------------------- validation


def _validatable(gdf: gpd.GeoDataFrame, source: str = "test") -> gpd.GeoDataFrame:
    gdf = add_geodesic_area(clean_geometries(gdf))
    gdf["region_id"] = [make_region_id("XXX", 2, source, c) for c in gdf["source_code"]]
    return gdf


def test_validate_passes_on_clean_input():
    gdf = _validatable(_frame([_square(0, 0), _square(2, 2)]))
    report = validate(gdf, {"country": "test", "iso3": "XXX", "admin_level": 2})
    assert report.ok, report.render()
    assert report.n_polygons == 2


def test_validate_catches_duplicate_region_ids():
    gdf = _validatable(_frame([_square(0, 0), _square(2, 2)]))
    gdf.loc[1, "region_id"] = gdf.loc[0, "region_id"]
    report = validate(gdf, {"country": "test", "iso3": "XXX", "admin_level": 2})
    assert not report.ok
    assert any("region_id unique" in n and not p for n, p, _ in report.checks)


def test_validate_catches_wrong_polygon_count():
    gdf = _validatable(_frame([_square(0, 0)]))
    cfg = {"country": "test", "iso3": "XXX", "admin_level": 2, "expected_polygon_count": 640}
    report = validate(gdf, cfg)
    assert not report.ok


def test_validate_catches_area_drift():
    gdf = _validatable(_frame([_square(0, 0)]))
    cfg = {
        "country": "test",
        "iso3": "XXX",
        "admin_level": 2,
        "reference_land_area_km2": 50000,
        "area_tolerance_pct": 1.0,
    }
    report = validate(gdf, cfg)
    assert not report.ok


def test_validate_skips_gracefully_when_no_reference_configured():
    gdf = _validatable(_frame([_square(0, 0)]))
    cfg = {"country": "test", "iso3": "XXX", "admin_level": 2}
    report = validate(gdf, cfg)
    assert report.ok
    assert any("skipped" in d for _, _, d in report.checks)


def test_validate_flags_slivers():
    tiny = Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)])
    gdf = _validatable(_frame([_square(0, 0), tiny]))
    report = validate(gdf, {"country": "test", "iso3": "XXX", "admin_level": 2})
    assert not report.ok
    assert any("sliver" in n and not p for n, p, _ in report.checks)


# ------------------------------------------------------------- integration


@pytest.mark.network
def test_build_real_boundaries_phase1_exit_test():
    """Phase 1 exit test. Requires network. Run with: pytest -m network"""
    gdf, report = build(write=True)
    print(report.render())
    assert report.ok, report.render()
    assert gdf["region_id"].is_unique
    assert gdf.crs.to_epsg() == 4326
    assert gdf["area_km2"].min() > 0
