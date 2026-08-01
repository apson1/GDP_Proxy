"""Population tests. Epoch selection, interpolation and validation, all offline."""

import pandas as pd
import pytest

from gdp_proxy.config import ConfigError
from gdp_proxy.population import (
    bracketing_epochs,
    interpolate_population,
    parse_export_csv,
    validate_population,
)

# ----------------------------------------------------------------- epochs


def test_bracketing_epochs_covers_the_requested_range():
    assert bracketing_epochs([2014, 2019, 2025]) == [2010, 2015, 2020, 2025]
    assert bracketing_epochs([2015]) == [2015]
    assert bracketing_epochs([2016, 2017]) == [2015, 2020]


def test_bracketing_epochs_rejects_years_outside_coverage():
    with pytest.raises(ConfigError, match="outside GHSL coverage"):
        bracketing_epochs([1960])
    with pytest.raises(ConfigError, match="outside GHSL coverage"):
        bracketing_epochs([2050])


# ----------------------------------------------------------------- parsing


def test_parse_export_csv_normalises_the_sum_column(tmp_path):
    path = tmp_path / "pop.csv"
    pd.DataFrame(
        [
            {"system:index": "0", ".geo": "{}", "region_id": "A", "epoch": 2020, "sum": 1234.5},
            {"system:index": "1", ".geo": "{}", "region_id": "B", "epoch": 2020, "sum": 99.0},
        ]
    ).to_csv(path, index=False)
    df = parse_export_csv(path)
    assert list(df.columns) == ["region_id", "epoch", "population"]
    assert df.loc[df.region_id == "A", "population"].iloc[0] == pytest.approx(1234.5)


def test_parse_export_csv_rejects_a_missing_sum(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"region_id": "A", "epoch": 2020}]).to_csv(path, index=False)
    with pytest.raises(ConfigError, match="no 'sum' column"):
        parse_export_csv(path)


# ----------------------------------------------------------------- interpolation


def _epochs():
    return pd.DataFrame(
        [
            {"region_id": "A", "epoch": 2015, "population": 100.0, "is_projection": False},
            {"region_id": "A", "epoch": 2020, "population": 200.0, "is_projection": False},
            {"region_id": "A", "epoch": 2025, "population": 300.0, "is_projection": True},
        ]
    )


def test_interpolation_is_linear_between_epochs():
    out = interpolate_population(_epochs(), [2015, 2017, 2020]).set_index("year")
    assert out.loc[2015, "population"] == pytest.approx(100.0)
    assert out.loc[2020, "population"] == pytest.approx(200.0)
    # 2017 is two fifths of the way from 2015 to 2020
    assert out.loc[2017, "population"] == pytest.approx(140.0)


def test_exact_epochs_are_not_flagged_interpolated():
    """An interpolated value is not an observation and must say so."""
    out = interpolate_population(_epochs(), [2015, 2017, 2020]).set_index("year")
    assert bool(out.loc[2015, "population_interpolated"]) is False
    assert bool(out.loc[2020, "population_interpolated"]) is False
    assert bool(out.loc[2017, "population_interpolated"]) is True


def test_projection_derived_years_are_flagged():
    """GHSL publishes 2025+ as a projection, not an observation."""
    out = interpolate_population(_epochs(), [2017, 2022, 2025]).set_index("year")
    assert bool(out.loc[2017, "population_is_projection"]) is False
    assert bool(out.loc[2022, "population_is_projection"]) is True  # brackets 2025
    assert bool(out.loc[2025, "population_is_projection"]) is True


def test_interpolation_refuses_years_outside_the_extracted_epochs():
    with pytest.raises(ConfigError, match="not bracketed"):
        interpolate_population(_epochs(), [2010])


# ----------------------------------------------------------------- validation


def _pop(n=3, **over):
    df = pd.DataFrame(
        [{"region_id": f"R{i}", "epoch": 2020, "population": 1000.0 + i} for i in range(n)]
    )
    for k, v in over.items():
        df[k] = v
    return df


def test_validate_passes_on_a_clean_epoch():
    report = validate_population(_pop(), expected_regions=3, epoch=2020)
    assert report.ok, report.render()


def test_validate_catches_dropped_regions():
    report = validate_population(_pop(3), expected_regions=676, epoch=2020)
    assert not report.ok
    assert any("region count" in n and not p for n, p, _ in report.checks)


def test_validate_catches_negative_population():
    df = _pop()
    df.loc[0, "population"] = -5.0
    report = validate_population(df, expected_regions=3, epoch=2020)
    assert not report.ok
    assert any("negative population" in n and not p for n, p, _ in report.checks)


def test_validate_flags_widespread_zeros():
    """Zero people usually means the zonal sum failed, not empty land."""
    report = validate_population(_pop(n=10, population=0.0), expected_regions=10, epoch=2020)
    assert not report.ok
    assert any("zero-population" in n and not p for n, p, _ in report.checks)
