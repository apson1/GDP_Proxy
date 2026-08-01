"""Phase 3a tests. DOSE reshape, deflation and label validation, all offline."""

import numpy as np
import pandas as pd
import pytest

from gdp_proxy.labels import (
    deflate,
    normalise_dose,
    validate_labels,
)

CFG = {
    "country": "test",
    "iso3": "IND",
    "base_year": 2015,
    "max_real_yoy_growth": 0.5,
}


# ----------------------------------------------------------------- DOSE reshape


def _dose_raw(**over):
    rows = [
        {
            "GID_0": "IND",
            "country": "India",
            "region": "Maharashtra",
            "year": 2014,
            "grp_lcu": 100.0,
            "grp_lcu_2015": 98.0,
            "pop": 1000,
        },
        {
            "GID_0": "IND",
            "country": "India",
            "region": "Bihar",
            "year": 2014,
            "grp_lcu": 50.0,
            "grp_lcu_2015": 49.0,
            "pop": 800,
        },
        {
            "GID_0": "KEN",
            "country": "Kenya",
            "region": "Nairobi",
            "year": 2014,
            "grp_lcu": 20.0,
            "grp_lcu_2015": 20.0,
            "pop": 400,
        },
    ]
    df = pd.DataFrame(rows)
    for k, v in over.items():
        df[k] = v
    return df


def test_normalise_dose_filters_country_and_keeps_schema():
    out = normalise_dose(_dose_raw(), CFG)
    assert set(out["source_region_name"]) == {"Maharashtra", "Bihar"}
    assert (out["admin_level"] == 1).all()
    assert (out["label_source"] == "dose").all()
    # DOSE's own constant-price level series is preferred when present.
    assert (out["deflation_method"] == "dose_constant_level_2015").all()
    assert out.loc[out.source_region_name == "Maharashtra", "gdp_constant"].iloc[0] == 98.0


def test_normalise_dose_marks_deflation_pending_without_constant_column():
    raw = _dose_raw().drop(columns=["grp_lcu_2015"])
    out = normalise_dose(raw, CFG)
    assert (out["deflation_method"] == "pending").all()
    assert out["gdp_constant"].isna().all()
    assert out["gdp_constant"].dtype == np.float64  # typed NaN, not object


# ------------------------------------------------- DOSE constant-price series


def test_dose_per_capita_constant_is_multiplied_by_population():
    """DOSE V2.14 ships grp_pc_lcu_2015; levels are per-capita x pop."""
    raw = _dose_raw().drop(columns=["grp_lcu_2015"])
    raw["grp_pc_lcu_2015"] = [0.098, 0.06125, 0.05]  # per-capita, 2015 LCU
    out = normalise_dose(raw, CFG)

    assert (out["deflation_method"] == "dose_constant_pc_x_pop_2015").all()
    assert out["gdp_constant"].dtype == np.float64
    # Maharashtra: 0.098 * 1000 pop
    mh = out.loc[out.source_region_name == "Maharashtra", "gdp_constant"].iloc[0]
    assert mh == pytest.approx(98.0)


def test_dose_per_capita_without_population_fails_loudly():
    from gdp_proxy.config import ConfigError

    raw = _dose_raw().drop(columns=["grp_lcu_2015", "pop"])
    raw["grp_pc_lcu_2015"] = [0.098, 0.06125, 0.05]
    with pytest.raises(ConfigError, match="population"):
        normalise_dose(raw, CFG)


def test_use_dose_constant_false_forces_the_wdi_path():
    """Sources without a validated constant series still deflate via WDI."""
    out = normalise_dose(_dose_raw(), {**CFG, "use_dose_constant": False})
    assert (out["deflation_method"] == "pending").all()
    assert out["gdp_constant"].isna().all()
    assert out["gdp_constant"].dtype == np.float64


def test_normalise_dose_raises_when_country_absent():
    from gdp_proxy.config import ConfigError

    with pytest.raises(ConfigError):
        normalise_dose(_dose_raw(), {**CFG, "iso3": "USA"})


# ----------------------------------------------------------------- deflation


def test_deflate_fills_missing_constant_and_leaves_existing():
    df = pd.DataFrame(
        {
            "source_region_name": ["A", "A", "B"],
            "parent_name": ["X", "X", "X"],
            "year": [2014, 2015, 2015],
            "gdp_nominal": [100.0, 110.0, 200.0],
            "gdp_constant": [np.nan, np.nan, 999.0],  # B already constant
            "deflation_method": ["pending", "pending", "dose_constant_2015"],
        }
    )
    deflator = pd.DataFrame({"year": [2014, 2015], "deflator": [90.0, 100.0]})
    out = deflate(df, deflator, base_year=2015)

    # base year factor is 1.0
    assert out.loc[out.year == 2015, "gdp_constant"].iloc[0] == pytest.approx(110.0)
    # 2014: 100 * (100/90)
    assert out.loc[(out.source_region_name == "A") & (out.year == 2014), "gdp_constant"].iloc[
        0
    ] == pytest.approx(100.0 * 100.0 / 90.0)
    # existing constant untouched
    assert out.loc[out.source_region_name == "B", "gdp_constant"].iloc[0] == 999.0


def test_deflate_rejects_base_year_absent_from_series():
    from gdp_proxy.config import ConfigError

    df = pd.DataFrame(
        {
            "year": [2014],
            "gdp_nominal": [1.0],
            "gdp_constant": [np.nan],
            "deflation_method": ["pending"],
        }
    )
    deflator = pd.DataFrame({"year": [2014], "deflator": [90.0]})
    with pytest.raises(ConfigError):
        deflate(df, deflator, base_year=2015)


# ------------------------------------------------------- dtype regression (object dtype)


def test_deflation_path_yields_float_gdp_constant_with_none_in_deflator():
    """Regression: gdp_constant must come out float64, never object.

    An object-dtype gdp_constant crashes np.log downstream with "loop of ufunc
    does not support argument 0 of type float which has no callable log method",
    and it crashes in _count_unit_breaks, far from where the dtype was set. Two
    ways object dtype can creep in are covered here: a None mixed into the
    deflator frame, and a DOSE file with no constant-price column at all (which
    used to broadcast a scalar pd.NA and silently make the column object).
    """
    raw = _dose_raw().drop(columns=["grp_lcu_2015"])  # forces the WDI deflation path
    panel = normalise_dose(raw, CFG)
    assert panel["gdp_constant"].dtype == np.float64, "pd.NA broadcast must not make it object"

    # A None mixed in with floats is exactly what the WDI JSON can produce.
    deflator = pd.DataFrame({"year": [2013, 2014, 2015], "deflator": [None, 90.0, 100.0]})
    out = deflate(panel, deflator, base_year=2015)

    assert out["gdp_constant"].dtype == np.float64, (
        f"gdp_constant came out {out['gdp_constant'].dtype}, not float64"
    )
    # and the downstream consumer that actually crashed must survive
    report = validate_labels(out, CFG)
    assert isinstance(report.n_rows, int)


def test_deflate_raises_clearly_if_gdp_constant_is_not_float():
    """A non-numeric gdp_constant must fail here, naming the column, rather than
    surfacing as an opaque ufunc error several functions later."""
    df = pd.DataFrame(
        {
            "source_region_name": ["A"],
            "parent_name": ["X"],
            "year": [2015],
            "gdp_nominal": [100.0],
            "gdp_constant": ["not-a-number"],
            "deflation_method": ["pending"],
        }
    )
    deflator = pd.DataFrame({"year": [2015], "deflator": [100.0]})
    with pytest.raises(ValueError, match="gdp_constant"):
        deflate(df, deflator, base_year=2015)


def test_load_national_deflator_drops_null_years(monkeypatch):
    """A WDI year with a null value is dropped, not carried as None into the
    arithmetic where it would poison the dtype."""
    import gdp_proxy.labels as lab

    payload = [
        {"page": 1},
        [
            {"date": "2014", "value": 90.0},
            {"date": "2015", "value": 100.0},
            {"date": "2016", "value": None},
        ],
    ]

    class _Resp:
        def json(self):
            return payload

    monkeypatch.setattr(lab.requests, "get", lambda *a, **k: _Resp())
    defl = lab.load_national_deflator("IND", 2015, "NY.GDP.DEFL.ZS")
    assert defl["deflator"].dtype == np.float64
    assert 2016 not in set(defl["year"])


def test_count_unit_breaks_survives_object_dtype():
    """The function that crashed must be robust regardless of what reaches it."""
    from gdp_proxy.labels import _count_unit_breaks

    df = pd.DataFrame(
        {
            "source_region_name": ["A", "A"],
            "parent_name": ["X", "X"],
            "year": [2014, 2015],
            "gdp_constant": pd.Series([100.0, 110.0], dtype=object),
        }
    )
    n_breaks, worst = _count_unit_breaks(df, 0.5)
    assert n_breaks == 0
    assert worst == "none"


# ----------------------------------------------------------------- validation


def _panel(**over):
    rows = []
    for region in ("A", "B"):
        for year in (2014, 2015, 2016):
            rows.append(
                {
                    "source_region_name": region,
                    "parent_name": "X",
                    "year": year,
                    "gdp_constant": 100.0 + year - 2014,
                    "gdp_nominal": 120.0 + year - 2014,
                    "population": 1000,
                    "admin_level": 1,
                    "label_source": "dose",
                    "ingested_at": "2026-07-30T00:00:00+00:00",
                }
            )
    df = pd.DataFrame(rows)
    for k, v in over.items():
        df[k] = v
    return df


def test_validate_passes_on_a_clean_panel():
    report = validate_labels(_panel(), CFG)
    assert report.ok, report.render()


def test_validate_fails_when_constant_missing():
    df = _panel()
    df.loc[0, "gdp_constant"] = np.nan
    report = validate_labels(df, CFG)
    assert not report.ok
    assert any("gdp_constant present" in n and not p for n, p, _ in report.checks)


def test_validate_flags_nominal_equals_constant():
    """If deflation silently did nothing, constant == nominal. Rule 9."""
    df = _panel()
    df["gdp_nominal"] = df["gdp_constant"]
    report = validate_labels(df, CFG)
    assert not report.ok
    assert any("constant differs from nominal" in n and not p for n, p, _ in report.checks)


def test_validate_flags_unit_break():
    """A 100x jump in one year is lakh-vs-crore, not growth."""
    df = _panel()
    df.loc[(df.source_region_name == "A") & (df.year == 2015), "gdp_constant"] = 10000.0
    report = validate_labels(df, {**CFG, "max_real_yoy_growth": 0.5})
    assert not report.ok
    assert any("unit breaks" in n and not p for n, p, _ in report.checks)


def test_validate_flags_base_year_outside_panel():
    report = validate_labels(_panel(), {**CFG, "base_year": 2030})
    assert not report.ok
    assert any("base year within panel" in n and not p for n, p, _ in report.checks)


def test_validate_catches_duplicate_region_years():
    df = pd.concat([_panel(), _panel().head(1)], ignore_index=True)
    report = validate_labels(df, CFG)
    assert not report.ok
    assert any("duplicate region-years" in n and not p for n, p, _ in report.checks)
