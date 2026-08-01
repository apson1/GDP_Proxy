"""Phase 2b tests. CSV parsing, skip logic and validation, all offline."""

import pandas as pd
import pytest

from gdp_proxy.config import ConfigError
from gdp_proxy.extract import (
    parse_export_csv,
    parse_years,
    pilot_years,
    validate_extraction,
)

CFG = {"country": "test", "admin_level": 2, "start_year": 2014, "end_year": 2025}


# --------------------------------------------------------------------- years


def test_parse_years_handles_single_range_and_list():
    assert parse_years("2019") == [2019]
    assert parse_years("2014-2016") == [2014, 2015, 2016]
    assert parse_years("2019,2021,2019") == [2019, 2021]
    assert parse_years("2014-2015,2020") == [2014, 2015, 2020]


def test_parse_years_rejects_backwards_range():
    with pytest.raises(ConfigError):
        parse_years("2025-2014")


def test_parse_years_rejects_empty():
    with pytest.raises(ConfigError):
        parse_years(" , ")


def test_pilot_is_the_two_most_recent_years():
    assert pilot_years(CFG) == [2024, 2025]


def test_pilot_never_runs_before_the_series_starts():
    assert pilot_years({**CFG, "start_year": 2025, "end_year": 2025}) == [2025]


def test_years_to_run_skips_existing_output(tmp_path, monkeypatch):
    """Rule 4: never recompute a year. Recomputation is how quota gets wasted."""
    import gdp_proxy.extract as ex

    monkeypatch.setattr(ex, "OUT_DIR", tmp_path)
    done = tmp_path / "sol_test_adm2_2019.parquet"
    done.write_bytes(b"x")

    assert ex.years_to_run(CFG, [2019, 2020]) == [2020]
    assert ex.years_to_run(CFG, [2019, 2020], force=True) == [2019, 2020]


def test_years_to_run_ignores_zero_byte_output(tmp_path, monkeypatch):
    """A truncated export must not be mistaken for a finished one."""
    import gdp_proxy.extract as ex

    monkeypatch.setattr(ex, "OUT_DIR", tmp_path)
    (tmp_path / "sol_test_adm2_2019.parquet").write_bytes(b"")
    assert ex.years_to_run(CFG, [2019]) == [2019]


# ----------------------------------------------------------------- csv parse


def _csv(tmp_path, rows) -> "pd.DataFrame":
    path = tmp_path / "export.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return parse_export_csv(path)


def test_parse_renames_gee_columns_and_drops_junk(tmp_path):
    df = _csv(
        tmp_path,
        [
            {
                "system:index": "0_0",
                ".geo": '{"type":"Point"}',
                "region_id": "IND2-aaa",
                "year": 2024,
                "month": 1,
                "rad_sum": 120.5,
                "rad_mean": 2.1,
                "rad_p50": 1.4,
                "rad_p90": 6.0,
                "lit_sum": 40,
                "valid_sum": 58,
            }
        ],
    )
    assert "system:index" not in df.columns
    assert ".geo" not in df.columns
    assert df["sol"].iloc[0] == 120.5
    assert df["lit_pixels"].iloc[0] == 40
    assert df["valid_pixels"].iloc[0] == 58


def test_parse_rejects_csv_without_region_id(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"year": 2024, "month": 1, "rad_sum": 1.0}]).to_csv(path, index=False)
    with pytest.raises(ConfigError):
        parse_export_csv(path)


def test_empty_reduction_is_missing_not_zero(tmp_path):
    """Rule 10. A cloud-blanked district must never become a dark district."""
    df = _csv(
        tmp_path,
        [
            {"region_id": "A", "year": 2024, "month": 7, "rad_sum": "", "valid_sum": ""},
            {"region_id": "B", "year": 2024, "month": 7, "rad_sum": 0.0, "valid_sum": 300},
        ],
    )
    assert bool(df.loc[df.region_id == "A", "is_missing"].iloc[0]) is True
    assert pd.isna(df.loc[df.region_id == "A", "sol"].iloc[0])
    assert bool(df.loc[df.region_id == "B", "is_missing"].iloc[0]) is False
    assert df.loc[df.region_id == "B", "sol"].iloc[0] == 0.0


def test_zero_valid_pixels_counts_as_missing(tmp_path):
    df = _csv(
        tmp_path,
        [{"region_id": "A", "year": 2024, "month": 7, "rad_sum": 0.0, "valid_sum": 0}],
    )
    assert bool(df["is_missing"].iloc[0]) is True


# ----------------------------------------------------------------- validation


def _panel(n_regions=3, n_months=12, **overrides) -> pd.DataFrame:
    rows = []
    for r in range(n_regions):
        for m in range(1, n_months + 1):
            rows.append(
                {
                    "region_id": f"R{r}",
                    "year": 2024,
                    "month": m,
                    "sol": 100.0 + r,
                    "mean_rad": 2.0,
                    "lit_pixels": 30,
                    "valid_pixels": 200,
                    "is_missing": False,
                }
            )
    df = pd.DataFrame(rows)
    for key, value in overrides.items():
        df[key] = value
    return df


def test_validate_passes_on_a_complete_panel():
    report = validate_extraction(_panel(), expected_regions=3, year=2024)
    assert report.ok, report.render()
    assert report.n_rows == 36


def test_validate_catches_dropped_regions():
    """The single most likely silent failure in this project."""
    report = validate_extraction(_panel(n_regions=3), expected_regions=640, year=2024)
    assert not report.ok
    assert any("region count" in n and not p for n, p, _ in report.checks)


def test_validate_catches_missing_months():
    report = validate_extraction(_panel(n_months=9), expected_regions=3, year=2024)
    assert not report.ok
    assert any("period(s) present" in n and not p for n, p, _ in report.checks)


def test_validate_catches_duplicate_region_months():
    df = pd.concat([_panel(), _panel().head(1)], ignore_index=True)
    report = validate_extraction(df, expected_regions=3, year=2024)
    assert not report.ok
    assert any("duplicate" in n and not p for n, p, _ in report.checks)


def test_validate_catches_negative_sol():
    df = _panel()
    df.loc[0, "sol"] = -5.0
    report = validate_extraction(df, expected_regions=3, year=2024)
    assert not report.ok
    assert any("negative SOL" in n and not p for n, p, _ in report.checks)


def test_validate_flags_mask_that_erased_the_country():
    """Half the panel missing means a broken mask, not bad weather."""
    df = _panel()
    df.loc[: len(df) // 2, "is_missing"] = True
    report = validate_extraction(df, expected_regions=3, year=2024)
    assert not report.ok
    assert any("missing share" in n and not p for n, p, _ in report.checks)


def test_validate_tolerates_a_realistic_monsoon_gap():
    df = _panel()
    df.loc[df["month"].isin([7, 8]), "is_missing"] = True  # 2 of 12 months
    report = validate_extraction(df, expected_regions=3, year=2024)
    assert report.ok, report.render()


def test_validate_flags_well_observed_but_dark_regions():
    """Well observed and zero light usually means the noise floor is too high."""
    df = _panel()
    df["sol"] = 0.0
    report = validate_extraction(df, expected_regions=3, year=2024)
    assert not report.ok
    assert any("dark region-months" in n and not p for n, p, _ in report.checks)
