"""District GDDP tests: adapters, the training guard, and the allocation check."""

import numpy as np
import pandas as pd
import pytest

from gdp_proxy.config import country_config
from gdp_proxy.gddp import (
    _fiscal_to_calendar,
    _to_number,
    allocation_error_report,
    assert_no_validation_labels,
    match_gddp_to_regions,
)

# ----------------------------------------------------------------- parsing helpers


def test_fiscal_year_maps_to_its_dominant_calendar_year():
    """Indian FY 2018-19 runs Apr 2018 to Mar 2019, so it is mostly 2018."""
    assert _fiscal_to_calendar("2018-19") == 2018
    assert _fiscal_to_calendar("Real GDDP 2024-25+") == 2024
    assert _fiscal_to_calendar("District") is None


def test_to_number_survives_indian_table_formatting():
    assert _to_number("6,07,710") == pytest.approx(607710.0)  # lakh grouping
    assert _to_number("1,234.5") == pytest.approx(1234.5)
    assert np.isnan(_to_number("-"))
    assert np.isnan(_to_number(None))


# ----------------------------------------------------------------- the guard


def test_validation_labels_may_never_reach_training():
    """Training on district GDDP would make the allocation check circular."""
    frame = pd.DataFrame(
        {
            "region_id": ["a", "b"],
            "year": [2019, 2019],
            "validation_only": [False, True],
            "label_source": ["dose", "gddp_tamil_nadu"],
        }
    )
    with pytest.raises(ValueError, match="validation-only"):
        assert_no_validation_labels(frame)


def test_guard_is_silent_on_a_clean_training_frame():
    clean = pd.DataFrame({"region_id": ["a"], "year": [2019], "validation_only": [False]})
    assert_no_validation_labels(clean)
    assert_no_validation_labels(pd.DataFrame({"region_id": ["a"]}))  # no column at all


def test_real_training_frame_carries_no_validation_labels():
    """The wiring, not just the helper: build_training_frame must be clean."""
    from gdp_proxy.config import ConfigError

    cfg = country_config()
    try:
        from gdp_proxy.pipeline import build_training_frame

        frame = build_training_frame(cfg)
    except (ConfigError, FileNotFoundError, ImportError) as exc:
        pytest.skip(f"No modelling frame yet: {exc}")
    assert_no_validation_labels(frame)


# ----------------------------------------------------------------- matching


def _boundaries():
    return pd.DataFrame(
        [
            {
                "region_id": "r1",
                "name": "Chennai",
                "parent_name": "Tamil Nadu",
                "name_variants": None,
            },
            {
                "region_id": "r2",
                "name": "Bangalore",
                "parent_name": "Karnataka",
                "name_variants": None,
            },
            {
                "region_id": "r3",
                "name": "Bangalore Rural",
                "parent_name": "Karnataka",
                "name_variants": None,
            },
        ]
    )


def _gddp(rows):
    return pd.DataFrame(rows)


def test_matching_blocks_on_state():
    """A district name must never match across a state boundary."""
    g = _gddp(
        [
            {
                "source_region_name": "Tamil Nadu",
                "district_name": "Bangalore",
                "year": 2019,
                "gddp": 1.0,
            }
        ]
    )
    out = match_gddp_to_regions(g, _boundaries())
    assert out.empty, "Bangalore is not in Tamil Nadu; it must not match"


def test_alias_maps_a_renamed_district_exactly():
    """Bengaluru Urban is GADM's 'Bangalore'. Without the alias it scores 87
    against 'Bangalore Rural', which would be the wrong district entirely."""
    g = _gddp(
        [
            {
                "source_region_name": "Karnataka",
                "district_name": "Bengaluru Urban",
                "year": 2022,
                "gddp": 1.0,
            }
        ]
    )

    unaliased = match_gddp_to_regions(g, _boundaries())
    assert unaliased.empty, "should fall below threshold rather than match Bangalore Rural"

    aliased = match_gddp_to_regions(g, _boundaries(), aliases={"Bengaluru Urban": "Bangalore"})
    assert aliased["region_id"].iloc[0] == "r2"


def test_aggregate_rows_are_excluded():
    """Division subtotals sit alongside districts and would double-count."""
    g = _gddp(
        [
            {
                "source_region_name": "Tamil Nadu",
                "district_name": "Chennai",
                "year": 2019,
                "gddp": 1.0,
            },
            {
                "source_region_name": "Tamil Nadu",
                "district_name": "KONKAN DIV.",
                "year": 2019,
                "gddp": 99.0,
            },
        ]
    )
    out = match_gddp_to_regions(g, _boundaries())
    assert set(out["district_name"]) == {"Chennai"}


# ----------------------------------------------------------------- allocation error


def test_allocation_error_is_zero_when_light_tracks_gdp_exactly():
    matched = pd.DataFrame(
        [
            {"region_id": "r1", "source_region_name": "S", "year": 2019, "gddp": 60.0},
            {"region_id": "r2", "source_region_name": "S", "year": 2019, "gddp": 40.0},
        ]
    )
    features = pd.DataFrame(
        [
            {"region_id": "r1", "year": 2019, "sol": 6.0},
            {"region_id": "r2", "year": 2019, "sol": 4.0},
        ]
    )
    out = allocation_error_report(matched, features)
    assert out["abs_share_error"].max() == pytest.approx(0.0, abs=1e-12)
    assert out["allocation_ratio"].tolist() == pytest.approx([1.0, 1.0])


def test_allocation_error_signs_are_interpretable():
    """Negative error means light UNDER-allocates GDP to that district."""
    matched = pd.DataFrame(
        [
            {"region_id": "r1", "source_region_name": "S", "year": 2019, "gddp": 90.0},
            {"region_id": "r2", "source_region_name": "S", "year": 2019, "gddp": 10.0},
        ]
    )
    features = pd.DataFrame(
        [
            {"region_id": "r1", "year": 2019, "sol": 5.0},
            {"region_id": "r2", "year": 2019, "sol": 5.0},
        ]
    )
    out = allocation_error_report(matched, features).set_index("region_id")
    assert out.loc["r1", "share_error"] < 0  # rich district, light under-allocates
    assert out.loc["r2", "share_error"] > 0
    assert out.loc["r1", "allocation_ratio"] < 1


def test_shares_are_normalised_within_state_year():
    matched = pd.DataFrame(
        [
            {"region_id": "r1", "source_region_name": "S", "year": 2019, "gddp": 60.0},
            {"region_id": "r2", "source_region_name": "S", "year": 2019, "gddp": 40.0},
            {"region_id": "r1", "source_region_name": "S", "year": 2020, "gddp": 10.0},
        ]
    )
    features = pd.DataFrame(
        [
            {"region_id": "r1", "year": 2019, "sol": 1.0},
            {"region_id": "r2", "year": 2019, "sol": 1.0},
            {"region_id": "r1", "year": 2020, "sol": 7.0},
        ]
    )
    out = allocation_error_report(matched, features)
    sums = out.groupby(["source_region_name", "year"])[["sol_share", "gddp_share"]].sum()
    assert np.allclose(sums.to_numpy(), 1.0)


# ----------------------------------------------------------------- share correction


def _errors(rows):
    df = pd.DataFrame(rows)
    df["share_error"] = df["sol_share"] - df["gddp_share"]
    df["abs_share_error"] = df["share_error"].abs()
    return df


def test_power_correction_is_identity_at_alpha_one():
    from gdp_proxy.gddp import apply_power_correction

    df = _errors(
        [
            {"source_region_name": "S", "year": 2019, "sol_share": 0.6, "gddp_share": 0.5},
            {"source_region_name": "S", "year": 2019, "sol_share": 0.4, "gddp_share": 0.5},
        ]
    )
    assert apply_power_correction(df, 1.0).tolist() == pytest.approx([0.6, 0.4])


def test_power_correction_renormalises_within_state_year():
    from gdp_proxy.gddp import apply_power_correction

    df = _errors(
        [
            {"source_region_name": "S", "year": 2019, "sol_share": 0.6, "gddp_share": 0.5},
            {"source_region_name": "S", "year": 2019, "sol_share": 0.4, "gddp_share": 0.5},
            {"source_region_name": "T", "year": 2019, "sol_share": 1.0, "gddp_share": 1.0},
        ]
    )
    out = apply_power_correction(df, 2.0)
    sums = out.groupby([df["source_region_name"], df["year"]]).sum()
    assert np.allclose(sums.to_numpy(), 1.0)


def test_alpha_above_one_shifts_mass_to_large_districts():
    """The direction the pooled size gradient points, so the sign must be right."""
    from gdp_proxy.gddp import apply_power_correction

    df = _errors(
        [
            {"source_region_name": "S", "year": 2019, "sol_share": 0.7, "gddp_share": 0.8},
            {"source_region_name": "S", "year": 2019, "sol_share": 0.3, "gddp_share": 0.2},
        ]
    )
    out = apply_power_correction(df, 1.5)
    assert out.iloc[0] > 0.7, "alpha>1 must increase the large district's share"
    assert out.iloc[1] < 0.3


def test_loso_rejects_a_correction_that_does_not_generalise():
    """A correction is only accepted if alpha keeps its sign AND every held-out
    fold improves. Three states pulling in different directions must be rejected.
    """
    from gdp_proxy.gddp import leave_one_state_out_correction

    rng = np.random.default_rng(0)
    rows = []
    # Each state has a different true relationship, so no single alpha fits.
    for state, exponent in (("A", 0.6), ("B", 1.0), ("C", 1.7)):
        for _ in range(20):
            s = float(rng.uniform(0.02, 0.2))
            rows.append(
                {
                    "source_region_name": state,
                    "year": 2019,
                    "sol_share": s,
                    "gddp_share": s**exponent,
                }
            )
    df = pd.DataFrame(rows)
    for _key, grp in df.groupby(["source_region_name", "year"]):
        df.loc[grp.index, "sol_share"] = grp["sol_share"] / grp["sol_share"].sum()
        df.loc[grp.index, "gddp_share"] = grp["gddp_share"] / grp["gddp_share"].sum()
    df["share_error"] = df["sol_share"] - df["gddp_share"]

    result = leave_one_state_out_correction(df)
    assert result["accepted"] is False
    assert "REJECTED" in result["verdict"]
    assert len(result["folds"]) == 3


def test_loso_accepts_a_correction_that_does_generalise():
    """The guard must not be unconditionally negative: a genuinely shared bias
    should pass, or the test proves nothing."""
    from gdp_proxy.gddp import leave_one_state_out_correction

    rng = np.random.default_rng(1)
    rows = []
    for state in ("A", "B", "C"):  # same exponent everywhere
        for _ in range(30):
            s = float(rng.uniform(0.02, 0.2))
            rows.append(
                {
                    "source_region_name": state,
                    "year": 2019,
                    "sol_share": s,
                    "gddp_share": s**1.6,
                }
            )
    df = pd.DataFrame(rows)
    for _key, grp in df.groupby(["source_region_name", "year"]):
        df.loc[grp.index, "sol_share"] = grp["sol_share"] / grp["sol_share"].sum()
        df.loc[grp.index, "gddp_share"] = grp["gddp_share"] / grp["gddp_share"].sum()
    df["share_error"] = df["sol_share"] - df["gddp_share"]

    result = leave_one_state_out_correction(df)
    assert result["accepted"] is True, result
    assert result["alpha_sign_consistent"] is True


def test_estimates_keep_the_uncorrected_share():
    """Whatever is decided about correction, the raw allocation must be
    reconstructible from the published output."""
    from gdp_proxy.pipeline import ESTIMATES_PATH

    if not ESTIMATES_PATH.exists():
        pytest.skip("No estimates.parquet yet.")
    est = pd.read_parquet(ESTIMATES_PATH)
    assert "allocation_share_uncorrected" in est.columns
    assert "share_correction" in est.columns
    sums = est.groupby(["source_region_name", "year"])["allocation_share_uncorrected"].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-6)


# ----------------------------------------------------------------- live check


@pytest.mark.needs_data
def test_allocation_is_validated_against_real_gddp():
    """The split must be checked against published district GDP, not assumed.

    This asserts only that the check RUNS and finds a positive relationship. The
    magnitude of the bias is reported, not asserted: light share is known to
    under-allocate service-led and large districts, and pinning a threshold here
    would freeze today's bias in as acceptable.
    """
    from gdp_proxy.config import ConfigError
    from gdp_proxy.gddp import validate_allocation

    try:
        summary = validate_allocation(country_config())
    except ConfigError as exc:
        pytest.skip(f"No district GDDP or overlapping light years: {exc}")

    assert summary["n_district_years"] > 50, summary
    assert summary["pearson"] > 0.5, (
        f"light share barely relates to published GDP share (r={summary['pearson']:.3f}); "
        "the within-state allocation would be close to arbitrary"
    )
    print(
        f"\nAllocation validation: {summary['n_district_years']} district-years, "
        f"r={summary['pearson']:.3f}, MAE={summary['mae_share_points']:.2f} share points"
    )
