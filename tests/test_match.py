"""Phase 3b tests.

The crosswalk is **one row per boundary polygon**, and with ADM1 labels against
ADM2 boundaries it is deliberately one-to-many: one DOSE state covers many GADM
districts. The invariant is the other direction, that each district maps to
exactly one state, and that is what the exit test asserts.

Most tests run offline on synthetic frames. The exit test needs the real boundary
snapshot, label panel and committed crosswalk, so it is marked ``needs_data``.
"""

import pandas as pd
import pytest

from gdp_proxy.config import ConfigError, country_config
from gdp_proxy.match import (
    adm1_gid_from_source_code,
    apply_crosswalk,
    normalise,
    propose_crosswalk,
    validate_crosswalk,
)


@pytest.fixture(autouse=True)
def _isolate_review_path(tmp_path, monkeypatch):
    """Never let a test write over the real data/crosswalk_review.csv.

    propose_crosswalk writes to REVIEW_PATH as a side effect, so without this
    every test that calls it silently replaces the reviewed artefact for the real
    country with a two-row synthetic frame.
    """
    import gdp_proxy.match as m

    monkeypatch.setattr(m, "REVIEW_PATH", tmp_path / "crosswalk_review.csv")


# ----------------------------------------------------------------- normalisation


def test_normalise_strips_admin_suffix_and_case():
    assert normalise("Pune District") == "pune"
    assert normalise("PUNE") == "pune"
    assert normalise("Pune  zilla") == "pune"


def test_normalise_strips_diacritics_and_punctuation():
    assert normalise("Bengalūru!") == "bengaluru"
    assert normalise("N.C.T. of Delhi") == "n c t of delhi".replace("  ", " ")


def test_normalise_keeps_urban_rural_distinction():
    """Bangalore Urban and Bangalore Rural are different districts; do not merge."""
    assert normalise("Bangalore Urban") != normalise("Bangalore Rural")
    assert normalise("Bangalore Urban") == "bangalore urban"


# ----------------------------------------------------------------- GID derivation


def test_adm1_gid_derived_from_gid2():
    """GADM nests its codes, so the state is a string operation on the district."""
    assert adm1_gid_from_source_code("IND.16.20_1") == "IND.16_1"
    assert adm1_gid_from_source_code("IND.5.9_1") == "IND.5_1"


def test_adm1_gid_passes_through_an_adm1_code():
    assert adm1_gid_from_source_code("IND.16_1") == "IND.16_1"


def test_adm1_gid_rejects_non_gadm_codes():
    for bad in (None, "", "nan", "shapeID-123", float("nan")):
        assert adm1_gid_from_source_code(bad) is None


# ----------------------------------------------------------------- proposal: gid pass


def _districts(rows):
    return pd.DataFrame(rows)


def _dose_labels(rows):
    return pd.DataFrame(rows)


def test_gid_join_is_exact_and_needs_no_review():
    """DOSE carries GID_1, so state matching is a code join, not a name guess."""
    boundaries = _districts(
        [
            {
                "region_id": "IND2-pune",
                "source_code": "IND.16.20_1",
                "name": "Pune",
                "parent_name": "Maharashtra",
                "name_variants": None,
            },
            {
                "region_id": "IND2-nagp",
                "source_code": "IND.16.17_1",
                "name": "Nagpur",
                "parent_name": "Maharashtra",
                "name_variants": None,
            },
        ]
    )
    # DOSE spells it differently; the code join does not care.
    labels = _dose_labels(
        [{"source_region_name": "Maharastra", "parent_name": "India", "source_gid": "IND.16_1"}]
    )
    review = propose_crosswalk(boundaries, labels, {"country": "test"})

    assert set(review["method"]) == {"gid"}
    assert (review["score"] == 100.0).all()
    assert not review["needs_review"].any()
    # one row per district, both pointing at the same state: one-to-many
    assert len(review) == 2
    assert set(review["source_region_name"]) == {"Maharastra"}


def test_gid_pass_runs_before_fuzzy_and_beats_a_closer_name():
    """A district whose GID matches must not be handed a better-spelled state."""
    boundaries = _districts(
        [
            {
                "region_id": "IND2-a",
                "source_code": "IND.16.20_1",
                "name": "Pune",
                "parent_name": "Maharashtra",
                "name_variants": None,
            },
        ]
    )
    labels = _dose_labels(
        [
            # exact name match on the district's parent, but the WRONG state code
            {"source_region_name": "Maharashtra", "parent_name": "India", "source_gid": "IND.99_1"},
            # ugly name, but the RIGHT code
            {"source_region_name": "Mahrshtr", "parent_name": "India", "source_gid": "IND.16_1"},
        ]
    )
    review = propose_crosswalk(boundaries, labels, {"country": "test"})
    row = review.iloc[0]
    assert row["method"] == "gid"
    assert row["source_region_name"] == "Mahrshtr", "the code join must win over the name"


def test_districts_without_a_gid_match_fall_through_to_fuzzy():
    """Telangana exists in GADM but not DOSE, so its districts get fuzzy-scored
    and flagged for a human rather than silently assigned."""
    boundaries = _districts(
        [
            {
                "region_id": "IND2-pune",
                "source_code": "IND.16.20_1",
                "name": "Pune",
                "parent_name": "Maharashtra",
                "name_variants": None,
            },
            {
                "region_id": "IND2-hyd",
                "source_code": "IND.32.3_1",
                "name": "Hyderabad",
                "parent_name": "Telangana",
                "name_variants": None,
            },
        ]
    )
    labels = _dose_labels(
        [{"source_region_name": "Maharashtra", "parent_name": "India", "source_gid": "IND.16_1"}]
    )
    review = propose_crosswalk(boundaries, labels, {"country": "test"})

    pune = review[review.region_id == "IND2-pune"].iloc[0]
    hyd = review[review.region_id == "IND2-hyd"].iloc[0]
    assert pune["method"] == "gid"
    assert hyd["method"] != "gid"
    assert bool(hyd["needs_review"]) is True, "an unmatched state must reach a human"


def test_fuzzy_fallback_matches_on_the_parent_state_name():
    """With ADM1 labels and ADM2 boundaries, the label should be compared to the
    district's parent, not the district itself."""
    boundaries = _districts(
        [
            {
                "region_id": "IND2-x",
                "source_code": "XXX.9.1_1",
                "name": "Some District",
                "parent_name": "Karnataka",
                "name_variants": None,
            },
        ]
    )
    labels = _dose_labels(
        [{"source_region_name": "Karnataka", "parent_name": "India", "source_gid": ""}]
    )
    review = propose_crosswalk(boundaries, labels, {"country": "test"})
    row = review.iloc[0]
    assert row["source_region_name"] == "Karnataka"
    assert row["method"] == "parent"
    assert row["score"] == 100.0


def test_review_file_sorts_worst_first(tmp_path, monkeypatch):
    import gdp_proxy.match as m

    monkeypatch.setattr(m, "REVIEW_PATH", tmp_path / "crosswalk_review.csv")
    boundaries = _districts(
        [
            {
                "region_id": "IND2-a",
                "source_code": "IND.16.20_1",
                "name": "Pune",
                "parent_name": "Maharashtra",
                "name_variants": None,
            },
            {
                "region_id": "IND2-z",
                "source_code": "IND.99.1_1",
                "name": "Zzxq",
                "parent_name": "Zzxq",
                "name_variants": None,
            },
        ]
    )
    labels = _dose_labels(
        [{"source_region_name": "Maharashtra", "parent_name": "India", "source_gid": "IND.16_1"}]
    )
    review = propose_crosswalk(boundaries, labels, {"country": "test"})
    assert bool(review["needs_review"].iloc[0]) is True
    assert review["score"].iloc[0] <= review["score"].iloc[-1]
    assert (tmp_path / "crosswalk_review.csv").exists()


# ----------------------------------------------------------------- apply_crosswalk


def _state_labels():
    """Two years of one state's GDP."""
    return pd.DataFrame(
        [
            {
                "source_region_name": "Maharashtra",
                "parent_name": "India",
                "year": 2015,
                "gdp_constant": 100.0,
            },
            {
                "source_region_name": "Maharashtra",
                "parent_name": "India",
                "year": 2016,
                "gdp_constant": 110.0,
            },
        ]
    )


def _state_to_district_crosswalk():
    """One state, three districts. The one-to-many shape."""
    return pd.DataFrame(
        [
            {"region_id": "IND2-a", "source_region_name": "Maharashtra", "parent_name": "India"},
            {"region_id": "IND2-b", "source_region_name": "Maharashtra", "parent_name": "India"},
            {"region_id": "IND2-c", "source_region_name": "Maharashtra", "parent_name": "India"},
        ]
    )


def test_apply_crosswalk_fans_one_state_out_to_its_districts():
    out = apply_crosswalk(_state_labels(), _state_to_district_crosswalk())
    # 2 state-years x 3 districts
    assert len(out) == 6
    assert set(out["region_id"]) == {"IND2-a", "IND2-b", "IND2-c"}
    # every district-year in a state carries that state's value
    y2015 = out[out.year == 2015]
    assert (y2015["gdp_constant"] == 100.0).all()


def test_apply_crosswalk_records_the_fanout_factor():
    """Downstream must be able to tell N copies of a state total from N
    independent district observations."""
    out = apply_crosswalk(_state_labels(), _state_to_district_crosswalk())
    assert (out["n_districts_in_label"] == 3).all()


def test_apply_crosswalk_raises_on_label_absent_from_crosswalk():
    """A label region with no districts assigned must raise, not vanish. Rule 6."""
    labels = pd.concat(
        [
            _state_labels(),
            pd.DataFrame(
                [
                    {
                        "source_region_name": "Kerala",
                        "parent_name": "India",
                        "year": 2015,
                        "gdp_constant": 50.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="not in the crosswalk"):
        apply_crosswalk(labels, _state_to_district_crosswalk())


def test_apply_crosswalk_ignores_districts_the_reviewer_left_unassigned():
    cw = pd.concat(
        [
            _state_to_district_crosswalk(),
            pd.DataFrame(
                [{"region_id": "IND2-telangana", "source_region_name": "", "parent_name": ""}]
            ),
        ],
        ignore_index=True,
    )
    out = apply_crosswalk(_state_labels(), cw)
    assert "IND2-telangana" not in set(out["region_id"])
    assert len(out) == 6


# ----------------------------------------------------------------- validate_crosswalk


def _bnd_for_validate():
    return pd.DataFrame(
        [
            {"region_id": "IND2-a", "name": "Pune", "parent_name": "Maharashtra"},
            {"region_id": "IND2-b", "name": "Nagpur", "parent_name": "Maharashtra"},
            {"region_id": "IND2-c", "name": "Thane", "parent_name": "Maharashtra"},
        ]
    )


def test_validate_accepts_one_state_mapping_to_many_districts():
    """The whole point of the reshape: one-to-many is correct here, not an error."""
    report = validate_crosswalk(
        _state_to_district_crosswalk(), _bnd_for_validate(), _state_labels(), {"country": "t"}
    )
    assert report.ok, report.render()
    assert report.n_labels == 1
    assert report.n_polygons == 3
    assert report.n_matched == 3


def test_validate_catches_a_district_mapped_to_two_states():
    """The real cardinality error: a district drawing GDP from two states."""
    cw = pd.concat(
        [
            _state_to_district_crosswalk(),
            pd.DataFrame(
                [{"region_id": "IND2-a", "source_region_name": "Gujarat", "parent_name": "India"}]
            ),
        ],
        ignore_index=True,
    )
    labels = pd.concat(
        [
            _state_labels(),
            pd.DataFrame(
                [
                    {
                        "source_region_name": "Gujarat",
                        "parent_name": "India",
                        "year": 2015,
                        "gdp_constant": 80.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    report = validate_crosswalk(cw, _bnd_for_validate(), labels, {"country": "t"})
    assert not report.ok
    assert any("exactly one label region" in n and not p for n, p, _ in report.checks)


def test_validate_catches_a_polygon_missing_from_the_crosswalk():
    cw = _state_to_district_crosswalk().head(2)  # IND2-c never reviewed
    report = validate_crosswalk(cw, _bnd_for_validate(), _state_labels(), {"country": "t"})
    assert not report.ok
    assert any("every polygon reviewed" in n and not p for n, p, _ in report.checks)


def test_validate_catches_crosswalk_id_not_in_boundaries():
    cw = pd.concat(
        [
            _state_to_district_crosswalk(),
            pd.DataFrame(
                [
                    {
                        "region_id": "IND2-ghost",
                        "source_region_name": "Maharashtra",
                        "parent_name": "India",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    report = validate_crosswalk(cw, _bnd_for_validate(), _state_labels(), {"country": "t"})
    assert not report.ok
    assert any("exist in boundaries" in n and not p for n, p, _ in report.checks)


def test_validate_catches_a_ghost_label_region():
    cw = _state_to_district_crosswalk().copy()
    cw.loc[0, "source_region_name"] = "Atlantis"
    report = validate_crosswalk(cw, _bnd_for_validate(), _state_labels(), {"country": "t"})
    assert not report.ok
    assert any("exist in labels" in n and not p for n, p, _ in report.checks)


def test_validate_requires_a_reason_for_unmatched_districts(tmp_path, monkeypatch):
    """An unmatched district is allowed; an undocumented one is not."""
    import gdp_proxy.match as m

    monkeypatch.setattr(m, "CONFIG_DIR", tmp_path)
    boundaries = pd.concat(
        [
            _bnd_for_validate(),
            pd.DataFrame(
                [{"region_id": "IND2-hyd", "name": "Hyderabad", "parent_name": "Telangana"}]
            ),
        ],
        ignore_index=True,
    )
    cw = pd.concat(
        [
            _state_to_district_crosswalk(),
            pd.DataFrame([{"region_id": "IND2-hyd", "source_region_name": "", "parent_name": ""}]),
        ],
        ignore_index=True,
    )

    report = validate_crosswalk(cw, boundaries, _state_labels(), {"country": "t"})
    assert not report.ok, "undocumented unmatched district must fail"

    (tmp_path / "unmatched_t.csv").write_text(
        "name,reason\nTelangana,Split from Andhra Pradesh in 2014\n", encoding="utf-8"
    )
    report = validate_crosswalk(cw, boundaries, _state_labels(), {"country": "t"})
    assert report.ok, report.render()
    assert report.n_unmatched == 1


def test_committed_unmatched_india_lists_the_units_dose_omits():
    """The ADM1 units genuinely absent from DOSE V2.14, each with a reason.

    Deliberately NOT Telangana or Ladakh. Verified against the real data on
    2026-07-31: DOSE V2.14 carries Telangana as IND.32_1 and it matches GADM
    exactly, while Ladakh is an ADM1 in neither source (GADM 4.1 still uses the
    pre-2019 Jammu and Kashmir). The real gaps are three small union territories.
    """
    from gdp_proxy.match import load_unmatched

    rows = load_unmatched({"country": "india"})
    names = {normalise(n) for n in rows["name"]}
    assert normalise("Lakshadweep") in names
    assert normalise("Dadra and Nagar Haveli") in names
    assert normalise("Daman and Diu") in names
    assert normalise("Telangana") not in names, "Telangana matches by GID; do not list it"
    assert normalise("Ladakh") not in names, "Ladakh is an ADM1 in neither source"
    for reason in rows["reason"]:
        assert len(str(reason).strip()) > 20, f"reason too thin: {reason!r}"


# ------------------------------------------------- disputed-territory GID pass


def test_disputed_gid_rewrites_to_the_claiming_country_code():
    from gdp_proxy.match import national_gid_from_disputed

    assert national_gid_from_disputed("Z01.14_1", "IND") == "IND.14_1"
    assert national_gid_from_disputed("Z07.3_1", "IND") == "IND.3_1"
    assert national_gid_from_disputed("Z09.13_1", "IND") == "IND.13_1"
    # a normal code is not a disputed code
    assert national_gid_from_disputed("IND.16_1", "IND") is None
    assert national_gid_from_disputed(None, "IND") is None


def test_disputed_territory_districts_match_but_are_flagged_for_review():
    """GADM files Jammu and Kashmir under Z01.14_1 while DOSE uses IND.14_1.
    Same unit, two code spaces. Match it, but never silently."""
    boundaries = _districts(
        [
            {
                "region_id": "IND2-jk1",
                "source_code": "Z01.14.3_1",
                "name": "Srinagar",
                "parent_name": "Jammu and Kashmir",
                "name_variants": None,
            },
        ]
    )
    labels = _dose_labels(
        [
            {
                "source_region_name": "Jammu & Kashmir",
                "parent_name": "India",
                "source_gid": "IND.14_1",
            }
        ]
    )
    review = propose_crosswalk(boundaries, labels, {"country": "test", "iso3": "IND"})
    row = review.iloc[0]
    assert row["method"] == "gid_disputed"
    assert row["source_region_name"] == "Jammu & Kashmir"
    assert bool(row["needs_review"]) is True, "a disputed-territory match must reach a human"


def test_disputed_matching_can_be_turned_off():
    boundaries = _districts(
        [
            {
                "region_id": "IND2-jk1",
                "source_code": "Z01.14.3_1",
                "name": "Srinagar",
                "parent_name": "Jammu and Kashmir",
                "name_variants": None,
            },
        ]
    )
    labels = _dose_labels(
        [
            {
                "source_region_name": "Jammu & Kashmir",
                "parent_name": "India",
                "source_gid": "IND.14_1",
            }
        ]
    )
    review = propose_crosswalk(
        boundaries, labels, {"country": "test", "iso3": "IND", "match_disputed_territory": False}
    )
    assert review.iloc[0]["method"] != "gid_disputed"


# ----------------------------------------------------------------- exit test


@pytest.mark.needs_data
def test_every_district_maps_to_exactly_one_label_region():
    """Phase 3 exit test.

    NOT a bijection. DOSE is ADM1 (33 Indian regions) and the boundaries are ADM2
    (676 districts), so one state legitimately covers many districts. The
    invariant asserted here is the one that would actually corrupt the panel if
    violated: every district maps to exactly one label region, every polygon has
    been reviewed, and unmatched districts are documented by ADM1 unit with a
    reason in config/unmatched_<country>.csv.
    """
    from gdp_proxy.boundaries import load_snapshot
    from gdp_proxy.labels import load_labels
    from gdp_proxy.match import crosswalk_path, load_crosswalk

    cfg = country_config()
    if not crosswalk_path(cfg).exists():
        pytest.skip(f"No committed crosswalk at {crosswalk_path(cfg)}; run --propose and review.")

    try:
        boundaries = load_snapshot(cfg["country"])
        labels = load_labels(cfg)
    except ConfigError as exc:
        pytest.skip(f"Missing upstream artefact: {exc}")

    crosswalk = load_crosswalk(cfg)
    report = validate_crosswalk(crosswalk, boundaries, labels, cfg)
    assert report.ok, report.render()
