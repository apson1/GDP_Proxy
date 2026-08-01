"""Phase 4 tests.

Offline tests on synthetic panels cover annualisation, concentration and
dynamics. The exit test ``test_national_sol_tracks_national_gdp`` needs the real
extracted parquets and a WDI call, so it is marked ``needs_data`` and ``network``
and skips until those exist.
"""

import numpy as np
import pandas as pd
import pytest

from gdp_proxy.config import ConfigError, country_config
from gdp_proxy.features import (
    add_growth_features,
    add_level_features,
    annualise,
    gini,
    national_sol_by_year,
    validate_features,
)

CFG = {"country": "test", "admin_level": 2, "min_valid_months": 6, "baseline_year": 2014}


# ----------------------------------------------------------------- annualise


def _monthly(region, year, n_valid, sol=100.0):
    """12 months, the first (12 - n_valid) marked missing."""
    rows = []
    for m in range(1, 13):
        missing = m <= (12 - n_valid)
        rows.append(
            {
                "region_id": region,
                "year": year,
                "month": m,
                "sol": np.nan if missing else sol,
                "mean_rad": np.nan if missing else 2.0,
                "median_rad": np.nan if missing else 1.5,
                "p90_rad": np.nan if missing else 6.0,
                "lit_pixels": np.nan if missing else 30.0,
                "valid_pixels": np.nan if missing else 200.0,
                "is_missing": missing,
            }
        )
    return rows


def test_annualise_averages_only_valid_months():
    df = pd.DataFrame(_monthly("A", 2020, n_valid=8, sol=100.0))
    annual = annualise(df, CFG)
    assert len(annual) == 1
    assert annual["n_valid_months"].iloc[0] == 8
    # mean over the 8 valid months, not diluted by the 4 missing ones
    assert annual["sol"].iloc[0] == pytest.approx(100.0)


def test_annualise_drops_and_counts_low_month_region_years():
    rows = _monthly("A", 2020, n_valid=8) + _monthly("B", 2020, n_valid=3)
    annual = annualise(pd.DataFrame(rows), CFG)
    assert set(annual["region_id"]) == {"A"}
    assert annual.attrs["n_dropped_low_months"] == 1


def test_annualise_never_imputes_zero_for_missing():
    """Rule 10: a masked month must not become a zero that drags the mean down."""
    df = pd.DataFrame(_monthly("A", 2020, n_valid=6, sol=100.0))
    annual = annualise(df, CFG)
    assert annual["sol"].iloc[0] == pytest.approx(100.0)  # not 100*6/12


def test_annualise_requires_is_missing_column():
    df = pd.DataFrame(_monthly("A", 2020, n_valid=8)).drop(columns=["is_missing"])
    with pytest.raises(ConfigError):
        annualise(df, CFG)


# ----------------------------------------------------------------- gini


def test_gini_zero_for_uniform_light():
    assert gini([5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.0, abs=1e-9)


def test_gini_approaches_one_for_single_hot_pixel():
    values = [0.0] * 999 + [1000.0]
    assert gini(values) > 0.99


def test_gini_handles_empty_and_all_zero():
    assert gini([]) == 0.0
    assert gini([0.0, 0.0, 0.0]) == 0.0


# ----------------------------------------------------------------- level features


def _annual_panel():
    return pd.DataFrame(
        [
            {
                "region_id": "A",
                "year": 2020,
                "sol": 100.0,
                "mean_rad": 2.0,
                "median_rad": 1.5,
                "p90_rad": 6.0,
                "lit_pixels": 30.0,
                "valid_pixels": 200.0,
                "n_valid_months": 12,
            },
            {
                "region_id": "A",
                "year": 2021,
                "sol": 150.0,
                "mean_rad": 3.0,
                "median_rad": 2.0,
                "p90_rad": 8.0,
                "lit_pixels": 40.0,
                "valid_pixels": 200.0,
                "n_valid_months": 12,
            },
        ]
    )


def _boundaries():
    return pd.DataFrame([{"region_id": "A", "area_km2": 50.0, "population": 1000.0}])


def test_level_features_normalise_and_bound_lit_share():
    out = add_level_features(_annual_panel(), _boundaries(), CFG)
    assert out["sol_per_area"].iloc[0] == pytest.approx(2.0)
    assert out["sol_per_capita"].iloc[0] == pytest.approx(0.1)
    assert out["lit_share"].iloc[0] == pytest.approx(30.0 / 200.0)
    assert (out["lit_share"] <= 1.0).all()
    assert np.isfinite(out["log_sol"]).all()


def test_level_features_raise_on_region_id_mismatch():
    """A district present in the panel but absent from boundaries must not be
    silently dropped. Rule 6."""
    panel = _annual_panel()
    panel.loc[0, "region_id"] = "GHOST"
    with pytest.raises(ValueError, match="no boundary area"):
        add_level_features(panel, _boundaries(), CFG)


def test_zero_sol_becomes_nan_log_not_negative_inf():
    panel = _annual_panel()
    panel.loc[0, "sol"] = 0.0
    out = add_level_features(panel, _boundaries(), CFG)
    assert not np.isinf(out["log_sol"]).any()
    assert bool(np.isnan(out.loc[out.year == 2020, "log_sol"].iloc[0]))


# ----------------------------------------------------------------- growth features


def test_yoy_is_nan_in_first_year_not_zero():
    leveled = add_level_features(_annual_panel(), _boundaries(), CFG)
    out = add_growth_features(leveled, CFG)
    first = out.loc[out.year == 2020, "sol_yoy"].iloc[0]
    assert bool(np.isnan(first)), "first-year YoY must be NaN, not a fabricated 0"
    # 2021 is a real log difference
    second = out.loc[out.year == 2021, "sol_yoy"].iloc[0]
    assert second == pytest.approx(np.log(150.0) - np.log(100.0))


def test_newly_lit_is_zero_at_baseline_and_grows_after():
    leveled = add_level_features(_annual_panel(), _boundaries(), CFG)
    out = add_growth_features(leveled, {**CFG, "baseline_year": 2020})
    assert out.loc[out.year == 2020, "newly_lit"].iloc[0] == 0.0
    assert out.loc[out.year == 2021, "newly_lit"].iloc[0] == pytest.approx(10.0)


# ----------------------------------------------------------------- validation


def _feature_panel():
    leveled = add_level_features(_annual_panel(), _boundaries(), CFG)
    return add_growth_features(leveled, CFG)


def test_validate_passes_on_clean_features():
    report = validate_features(_feature_panel(), CFG)
    assert report.ok, report.render()


def test_validate_catches_duplicate_region_years():
    df = pd.concat([_feature_panel(), _feature_panel().head(1)], ignore_index=True)
    report = validate_features(df, CFG)
    assert not report.ok
    assert any("one row per region-year" in n and not p for n, p, _ in report.checks)


def test_validate_reports_valid_month_histogram():
    report = validate_features(_feature_panel(), CFG)
    assert report.valid_month_hist  # non-empty, so a masking regression is visible


# ----------------------------------------------------------------- exit test


# -------------------------------------------------- version boundary bridge


def _ratio_frame(rows):
    return pd.DataFrame(rows)


def test_version_boundary_flags_a_scale_shift(monkeypatch):
    """A silent offset between annual product versions must raise.

    V21 ends 2021 and V22 starts 2022 with no overlapping year, so the offset
    cannot be measured directly. The continuously-running monthly series is the
    bridge: annual/monthly should stay flat. If it steps, the two versions are
    on different scales and every district estimate shifts by a constant factor,
    which is invisible on a map.
    """
    import gdp_proxy.features as F

    stable = [
        {
            "year": y,
            "annual_sol": 100.0,
            "monthly_sol": 100.0,
            "ratio": 1.0,
            "annual_dataset": "X/ANNUAL_V21",
        }
        for y in (2014, 2015, 2016, 2017)
    ]
    shifted = stable + [
        {
            "year": 2022,
            "annual_sol": 130.0,
            "monthly_sol": 100.0,
            "ratio": 1.30,
            "annual_dataset": "X/ANNUAL_V22",
        }
    ]
    monkeypatch.setattr(F, "annual_monthly_ratio", lambda cfg: _ratio_frame(shifted))
    with pytest.raises(ConfigError, match="not on the same scale"):
        F.assert_version_boundary_consistent({}, baseline_end_year=2021)


def test_version_boundary_accepts_a_stable_ratio(monkeypatch):
    import gdp_proxy.features as F

    rows = [
        {
            "year": 2014,
            "annual_sol": 100.0,
            "monthly_sol": 100.0,
            "ratio": 1.02,
            "annual_dataset": "X/ANNUAL_V21",
        },
        {
            "year": 2015,
            "annual_sol": 100.0,
            "monthly_sol": 100.0,
            "ratio": 0.98,
            "annual_dataset": "X/ANNUAL_V21",
        },
        {
            "year": 2016,
            "annual_sol": 100.0,
            "monthly_sol": 100.0,
            "ratio": 1.00,
            "annual_dataset": "X/ANNUAL_V21",
        },
        {
            "year": 2022,
            "annual_sol": 100.0,
            "monthly_sol": 100.0,
            "ratio": 0.99,
            "annual_dataset": "X/ANNUAL_V22",
        },
    ]
    monkeypatch.setattr(F, "annual_monthly_ratio", lambda cfg: _ratio_frame(rows))
    result = F.assert_version_boundary_consistent({}, baseline_end_year=2021)
    assert result["ok"] is True
    assert result["checked_years"] == [2022]


def test_version_boundary_needs_enough_baseline_years(monkeypatch):
    import gdp_proxy.features as F

    rows = [
        {
            "year": 2014,
            "annual_sol": 1.0,
            "monthly_sol": 1.0,
            "ratio": 1.0,
            "annual_dataset": "X/ANNUAL_V21",
        },
        {
            "year": 2022,
            "annual_sol": 1.0,
            "monthly_sol": 1.0,
            "ratio": 1.0,
            "annual_dataset": "X/ANNUAL_V22",
        },
    ]
    monkeypatch.setattr(F, "annual_monthly_ratio", lambda cfg: _ratio_frame(rows))
    with pytest.raises(ConfigError, match="at least 2 baseline years"):
        F.assert_version_boundary_consistent({}, baseline_end_year=2021)


@pytest.mark.needs_data
def test_version_boundary_holds_on_real_data():
    """The live check: V22 must sit inside the V21 baseline band."""
    from gdp_proxy.features import check_version_boundary

    try:
        result = check_version_boundary(country_config())
    except ConfigError as exc:
        pytest.skip(f"Need both annual and monthly extractions: {exc}")

    if not result["checked_years"]:
        pytest.skip("No post-boundary year extracted in both series yet.")
    assert result["ok"], (
        f"annual/monthly ratio outside baseline for {result['offending_years']}: "
        f"{[(r['year'], round(r['ratio'], 4)) for r in result['ratios']]}"
    )


@pytest.mark.needs_data
@pytest.mark.network
def test_national_sol_tracks_national_gdp():
    """Phase 4 exit test. National SOL vs national GDP should correlate > 0.9.

    A cheap, strong check on the whole upstream chain: broken extraction, masking
    or annualisation collapses this correlation. Needs the extracted parquets and
    a WDI call, so it skips until both are available.
    """
    from gdp_proxy.features import build_features, load_national_gdp

    cfg = country_config()
    try:
        features = build_features(cfg)
    except ConfigError as exc:
        pytest.skip(f"No extracted feature panel yet: {exc}")

    sol = national_sol_by_year(features)
    gdp = load_national_gdp(cfg["iso3"], str(cfg.get("wdi_gdp_indicator", "NY.GDP.MKTP.KN")))
    merged = sol.merge(gdp, on="year", how="inner")

    # A correlation over 2 points is trivially +/-1 and proves nothing. This is a
    # pilot-run condition, not a failure: skip until the full series is extracted.
    # The 0.9 threshold below is NOT relaxed to accommodate a short series.
    if len(merged) < 5:
        pytest.skip(
            f"only {len(merged)} overlapping year(s) "
            f"({sorted(merged['year'])}); need >=5 for a meaningful correlation. "
            "Extract the full series: python -m gdp_proxy.extract --years 2014-2025"
        )

    corr = np.corrcoef(merged["national_sol"], merged["national_gdp"])[0, 1]
    assert corr > 0.9, f"national SOL vs GDP correlation {corr:.3f} is below 0.9"
