"""Config tests. These run without network or Earth Engine access."""

import pytest

from gdp_proxy.config import ConfigError, country_config


def test_defaults_merge_into_country():
    cfg = country_config("india")
    assert cfg["iso3"] == "IND"
    assert cfg["admin_level"] == 2
    # inherited from defaults
    assert cfg["viirs_monthly"].startswith("NOAA/VIIRS/DNB/MONTHLY")
    assert cfg["noise_floor"] > 0


def test_country_block_overrides_defaults_where_set():
    india = country_config("india")
    kenya = country_config("kenya")
    assert india["admin_level"] != kenya["admin_level"]
    assert india["viirs_monthly"] == kenya["viirs_monthly"]


def test_unknown_country_lists_options():
    with pytest.raises(ConfigError) as exc:
        country_config("atlantis")
    assert "india" in str(exc.value)


def test_year_range_is_sane():
    cfg = country_config("india")
    # VCMSLCFG stray-light corrected series starts 2014
    assert cfg["start_year"] >= 2014
    assert cfg["end_year"] > cfg["start_year"]
