"""Phase 6 tests. Decision logic, the estimates store, and the training join.

All offline. The Earth Engine ``latest_available_month`` is monkeypatched; the
network path itself is exercised only under ``-m network`` elsewhere.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from gdp_proxy import pipeline
from gdp_proxy.config import ConfigError

CFG = {"country": "test", "admin_level": 2, "start_year": 2014, "viirs_monthly": "DS/ID"}


# ----------------------------------------------------------------- decision logic


def test_needs_run():
    assert pipeline.needs_run(date(2025, 6, 1), None) is True
    assert pipeline.needs_run(date(2025, 6, 1), date(2025, 5, 1)) is True
    assert pipeline.needs_run(date(2025, 5, 1), date(2025, 5, 1)) is False


def test_affected_years_full_range_when_nothing_processed():
    assert pipeline.affected_years(date(2016, 3, 1), None, 2014) == [2014, 2015, 2016]


def test_affected_years_empty_when_up_to_date():
    assert pipeline.affected_years(date(2025, 5, 1), date(2025, 5, 1), 2014) == []


def test_affected_years_spans_processed_to_available():
    assert pipeline.affected_years(date(2025, 2, 1), date(2024, 11, 1), 2014) == [2024, 2025]


# ----------------------------------------------------------------- latest_processed_month


def test_latest_processed_month_reads_parquets(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "PROCESSED_DIR", tmp_path)
    df = pd.DataFrame(
        {"region_id": ["A", "A"], "year": [2024, 2024], "month": [11, 12], "sol": [1.0, 2.0]}
    )
    df.to_parquet(tmp_path / "sol_test_adm2_2024.parquet", index=False)
    assert pipeline.latest_processed_month(CFG) == date(2024, 12, 1)


def test_latest_processed_month_none_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "PROCESSED_DIR", tmp_path)
    assert pipeline.latest_processed_month(CFG) is None


# ----------------------------------------------------------------- run (mocked EE)


def test_run_noop_when_up_to_date(monkeypatch):
    monkeypatch.setattr(pipeline, "latest_available_month", lambda cfg: date(2025, 5, 1))
    monkeypatch.setattr(pipeline, "latest_processed_month", lambda cfg: date(2025, 5, 1))
    result = pipeline.run(CFG, check_only=True)
    assert result.new_data is False
    assert result.action == "noop"


def test_run_check_only_reports_new_month(monkeypatch):
    monkeypatch.setattr(pipeline, "latest_available_month", lambda cfg: date(2025, 6, 1))
    monkeypatch.setattr(pipeline, "latest_processed_month", lambda cfg: date(2025, 5, 1))
    result = pipeline.run(CFG, check_only=True)
    assert result.new_data is True
    assert result.action == "new-month-available"
    assert result.affected_years == [2025]


# ----------------------------------------------------------------- estimates store


def _estimates():
    return pd.DataFrame(
        {
            "region_id": ["A", "B"],
            "year": [2024, 2024],
            "prediction": [4.0, 5.0],
            "gdp_estimate": [54.6, 148.4],
            "gdp_lower": [40.0, 120.0],
            "gdp_upper": [70.0, 180.0],
        }
    )


def test_append_estimates_stamps_vintage(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "ESTIMATES_PATH", tmp_path / "estimates.parquet")
    out = pipeline.append_estimates(_estimates(), CFG, vintage="v1")
    assert (out["vintage"] == "v1").all()
    assert len(out) == 2


def test_append_estimates_is_idempotent_for_same_vintage(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "ESTIMATES_PATH", tmp_path / "estimates.parquet")
    pipeline.append_estimates(_estimates(), CFG, vintage="v1")
    out = pipeline.append_estimates(_estimates(), CFG, vintage="v1")
    assert len(out) == 2, "re-running the same vintage must not duplicate rows"


def test_append_estimates_keeps_both_vintages(tmp_path, monkeypatch):
    """A reprocessed composite gets a new vintage; the old row is kept for audit."""
    monkeypatch.setattr(pipeline, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "ESTIMATES_PATH", tmp_path / "estimates.parquet")
    pipeline.append_estimates(_estimates(), CFG, vintage="v1")
    out = pipeline.append_estimates(_estimates(), CFG, vintage="v2")
    assert len(out) == 4
    assert set(out["vintage"]) == {"v1", "v2"}


# ----------------------------------------------------------------- training frame


def test_build_training_frame_asserts_join_and_adds_target(monkeypatch):
    """features + labels join is 1:1 on region_id/year and yields log_gdp."""
    # Balanced light panel: both districts in both years. build_training_frame
    # asserts this, because an unbalanced district sum turns coverage gaps into
    # fake growth. The label side may still be short; that is checked separately.
    features = pd.DataFrame(
        {
            "region_id": ["A", "A", "B", "B"],
            "year": [2015, 2016, 2015, 2016],
            "log_sol": [4.0, 4.2, 3.0, 3.1],
            "lit_share": [0.5, 0.6, 0.3, 0.3],
        }
    )
    labels = pd.DataFrame(
        {
            "region_id": ["A", "A", "B"],
            "year": [2015, 2016, 2015],
            "gdp_constant": [100.0, 110.0, 50.0],
        }
    )
    monkeypatch.setattr(pipeline, "build_training_frame", pipeline.build_training_frame)

    import gdp_proxy.features as feats
    import gdp_proxy.labels as labs
    import gdp_proxy.match as match

    monkeypatch.setattr(feats, "build_features", lambda cfg: features)
    monkeypatch.setattr(labs, "load_labels", lambda cfg: labels)
    monkeypatch.setattr(match, "load_crosswalk", lambda cfg: pd.DataFrame())
    monkeypatch.setattr(match, "apply_crosswalk", lambda labels, cw: labels)

    frame = pipeline.build_training_frame({"admin_level": 2})
    assert len(frame) == 3
    assert "log_gdp" in frame.columns
    assert frame.loc[frame.region_id == "A", "log_gdp"].iloc[0] == pytest.approx(np.log(100.0))


def test_build_training_frame_raises_on_empty_overlap(monkeypatch):
    """The ADM1/ADM2 region_id mismatch must fail loudly, not return an empty frame."""
    features = pd.DataFrame({"region_id": ["A"], "year": [2015], "log_sol": [4.0]})
    labels = pd.DataFrame({"region_id": ["STATE_X"], "year": [2015], "gdp_constant": [100.0]})

    import gdp_proxy.features as feats
    import gdp_proxy.labels as labs
    import gdp_proxy.match as match

    monkeypatch.setattr(feats, "build_features", lambda cfg: features)
    monkeypatch.setattr(labs, "load_labels", lambda cfg: labels)
    monkeypatch.setattr(match, "load_crosswalk", lambda cfg: pd.DataFrame())
    monkeypatch.setattr(match, "apply_crosswalk", lambda labels, cw: labels)

    with pytest.raises(ConfigError, match="zero rows"):
        pipeline.build_training_frame({"admin_level": 2})
