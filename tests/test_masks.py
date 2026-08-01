"""Phase 2a tests. Pure config and geometry logic, no Earth Engine needed."""

import pytest

from gdp_proxy.config import country_config
from gdp_proxy.masks import flare_sites, mask_summary, validate_mask_config

BASE = {"noise_floor": 0.4, "lit_threshold": 0.5, "min_cf_cvg": 2, "flare_buffer_m": 7500}


def test_real_config_passes_validation():
    assert validate_mask_config(country_config("india")) == []
    assert validate_mask_config(country_config("kenya")) == []


def test_lit_threshold_below_noise_floor_is_rejected():
    """Otherwise every pixel zeroed by the noise floor still counts as lit."""
    cfg = BASE | {"noise_floor": 0.8, "lit_threshold": 0.3}
    problems = validate_mask_config(cfg)
    assert any("lit_threshold" in p for p in problems)


def test_zero_coverage_threshold_is_rejected():
    """min_cf_cvg of 0 turns 'we could not see' into a fake zero."""
    problems = validate_mask_config(BASE | {"min_cf_cvg": 0})
    assert any("missing data into fake zeros" in p for p in problems)


def test_absurd_noise_floor_is_rejected():
    problems = validate_mask_config(BASE | {"noise_floor": 5.0})
    assert any("very high" in p for p in problems)


def test_negative_noise_floor_is_rejected():
    problems = validate_mask_config(BASE | {"noise_floor": -1})
    assert any("positive" in p for p in problems)


def test_flare_sites_use_per_site_radius_when_given():
    cfg = BASE | {
        "manual_flare_regions": [
            {"name": "offshore", "lon": 71.5, "lat": 19.4, "radius_km": 60},
            {"name": "no_radius", "lon": 70.0, "lat": 20.0},
        ]
    }
    sites = flare_sites(cfg)
    assert sites[0]["radius_m"] == 60_000
    assert sites[1]["radius_m"] == 7500, "falls back to flare_buffer_m"


def test_flare_sites_empty_when_unconfigured():
    assert flare_sites(BASE) == []


def test_out_of_range_flare_coordinates_are_caught():
    cfg = BASE | {"manual_flare_regions": [{"name": "bad", "lon": 271.5, "lat": 19.4}]}
    assert any("out-of-range" in p for p in validate_mask_config(cfg))


def test_india_flare_sites_are_inside_india_bbox():
    sites = flare_sites(country_config("india"))
    assert sites, "India config should list petroleum basins to mask"
    for s in sites:
        assert 68 < s["lon"] < 98, s
        assert 6 < s["lat"] < 38, s


def test_mask_summary_is_recorded_for_provenance():
    summary = mask_summary(country_config("india"))
    assert summary["noise_floor"] > 0
    assert summary["n_flare_sites"] >= 1
    assert set(summary) == {
        "min_cf_cvg",
        "noise_floor",
        "lit_threshold",
        "water_occurrence_threshold",
        "n_flare_sites",
        "flare_buffer_m",
    }


@pytest.mark.network
def test_apply_masks_runs_on_earth_engine():
    """Requires credentials. Run with: pytest -m network"""
    import ee

    from gdp_proxy.auth import init_ee
    from gdp_proxy.masks import apply_masks

    init_ee()
    cfg = country_config("india")
    image = ee.ImageCollection(cfg["viirs_monthly"]).first()
    masked = ee.Image(apply_masks(image, cfg))
    assert masked.bandNames().getInfo() == ["rad", "lit", "valid"]
