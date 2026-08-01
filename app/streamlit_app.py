"""Phase 6b: the dashboard.

The dashboard's real job is to make the uncertainty as visible as the estimate. A
choropleth of point estimates alone invites a reader to treat a 15% gap between
two districts as real when their intervals overlap completely. So every estimate
is shown with its band, extrapolated districts are flagged, and the diagnostics
(elasticity, holdout scores, coverage, unmatched count) are published rather than
hidden.

Performance rules (pitfalls.md, roadblock 10): the app reads precomputed parquet
only. It never calls Earth Engine, never fits a model at request time, and uses
simplified geometries for display. Loaders are cached.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"
ESTIMATES_PATH = PROCESSED / "estimates.parquet"
DIAGNOSTICS_PATH = PROCESSED / "diagnostics.json"


# --------------------------------------------------------------------------
# cached loaders (files only, never Earth Engine)
# --------------------------------------------------------------------------


@st.cache_data
def load_estimates() -> pd.DataFrame:
    if not ESTIMATES_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(ESTIMATES_PATH)
    # Show only the newest vintage per region-year; older vintages stay for audit.
    if "vintage" in df.columns:
        df = df.sort_values("vintage").drop_duplicates(["region_id", "year"], keep="last")
    return df


@st.cache_data
def load_boundaries() -> pd.DataFrame | None:
    paths = sorted(PROCESSED.glob("boundaries_*.parquet"))
    if not paths:
        return None
    import geopandas as gpd

    gdf = gpd.read_parquet(paths[0])
    # Simplify for display only; the full-resolution geometry stays in extraction.
    gdf["geometry"] = gdf.geometry.simplify(0.01, preserve_topology=True)
    return gdf


@st.cache_data
def load_diagnostics() -> dict:
    if DIAGNOSTICS_PATH.exists():
        return json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def load_map_geojson() -> tuple[dict, list[str]]:
    """Serialise the display geometry once, not on every widget change.

    ``to_json`` on 676 simplified districts costs ~0.26 s and produces ~2.7 MB.
    Doing it inside the render path meant every year-slider nudge and metric
    toggle paid that again. Cached here, the render path only rewrites the small
    per-feature properties, which is roughly three orders of magnitude cheaper.
    """
    gdf = load_boundaries()
    if gdf is None:
        return {}, []
    geojson = json.loads(gdf.to_json())
    order = [f["properties"].get("region_id") for f in geojson["features"]]
    return geojson, order


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------


def page_map(estimates: pd.DataFrame) -> None:
    st.header("Estimated district GDP")
    if estimates.empty:
        st.info("No estimates yet. Run the pipeline to populate data/processed/estimates.parquet.")
        return

    st.caption(
        "Estimated **level**, not growth. Year-on-year changes in this number are not a "
        "growth estimate: the within-district relationship is not identified on a "
        "6-year label overlap. See Caveats."
    )
    years = sorted(estimates["year"].unique())
    year = st.select_slider("Year", options=years, value=years[-1])

    # Per capita is the default on purpose. Total GDP ranks big districts high
    # regardless of prosperity, and the two orderings are very different: a
    # populous poor district is high on total and low on per capita.
    has_pc = "gdp_per_capita_estimate" in estimates.columns
    metric = st.radio(
        "Metric",
        (
            ["GDP per capita (how well off)", "Total GDP (how much output)"]
            if has_pc
            else ["Total GDP (how much output)"]
        )
        + ["Interval width (uncertainty)"],
        horizontal=True,
        help=(
            "Per capita answers 'how well off are people here'. Total answers 'how much "
            "output does this district produce'. A populous poor district ranks high on "
            "total and low on per capita. They are not interchangeable."
        ),
    )

    year_df = estimates[estimates["year"] == year].copy()
    if metric.startswith("Interval") and {"gdp_upper", "gdp_lower"}.issubset(year_df):
        year_df["value"] = year_df["gdp_upper"] - year_df["gdp_lower"]
        st.caption("Darker = wider band = the model is less sure here.")
    elif metric.startswith("Total"):
        year_df["value"] = year_df.get("gdp_estimate", year_df.get("prediction"))
        st.caption(
            "**Total** district GDP. Big districts rank high because they are big. "
            "Switch to per capita to compare prosperity."
        )
    else:
        year_df["value"] = year_df["gdp_per_capita_estimate"]
        st.caption(
            "**GDP per capita** (GHSL population denominator). This is the headline "
            "metric and ranks districts very differently from total GDP."
        )

    if "state_gdp_source" in year_df.columns:
        n_pred = int((year_df["state_gdp_source"] == "predicted").sum())
        if n_pred:
            st.info(
                f"{n_pred} of {len(year_df)} districts in {year} are downscaled from a "
                "**predicted** state total (no published GDP for this year), rather than "
                "from a published one.",
                icon="ℹ️",
            )

    geojson, order = load_map_geojson()
    if not geojson:
        st.warning("No boundary snapshot found; showing a table instead of a map.")
        st.dataframe(year_df[["region_id", "value"]].sort_values("value", ascending=False))
        return

    import pydeck as pdk

    values = year_df.set_index("region_id")["value"]
    aligned = values.reindex(order)
    # Districts with no estimate are NOT zero. Filling them with 0 would paint
    # them as the poorest in the country; they are drawn grey and labelled.
    has_value = aligned.notna()
    lo, hi = aligned.quantile(0.05), aligned.quantile(0.95)
    span = (hi - lo) or 1.0
    fill = ((aligned - lo) / span).clip(0, 1)

    n_missing = int((~has_value).sum())
    for feature, value, shade, present in zip(
        geojson["features"], aligned, fill, has_value, strict=False
    ):
        props = feature["properties"]
        props["fill"] = float(shade) if present else 0.0
        props["has_value"] = bool(present)
        props["value"] = f"{value:,.0f}" if present else "no estimate"

    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson,
        get_fill_color=(
            "properties.has_value "
            "? [255 * properties.fill, 140, 255 * (1 - properties.fill), 160] "
            ": [200, 200, 200, 70]"
        ),
        pickable=True,
        stroked=True,
        get_line_color=[60, 60, 60],
        line_width_min_pixels=0.5,
    )
    view = pdk.ViewState(latitude=22.0, longitude=79.0, zoom=3.5)
    st.pydeck_chart(
        pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "{name}\n{value}"})
    )
    if n_missing:
        st.caption(
            f"{n_missing} district(s) shown grey have no estimate for {year} "
            "(no matching label region). They are not zero."
        )


def page_district(estimates: pd.DataFrame) -> None:
    st.header("District detail")
    if estimates.empty:
        st.info("No estimates yet.")
        return

    gdf = load_boundaries()
    names = {}
    if gdf is not None:
        names = dict(zip(gdf["region_id"], gdf["name"], strict=False))

    region_ids = sorted(estimates["region_id"].unique())
    label = st.selectbox(
        "District",
        region_ids,
        format_func=lambda r: f"{names.get(r, r)} ({r})",
    )
    d = estimates[estimates["region_id"] == label].sort_values("year")

    if "extrapolation_flag" in d.columns and bool(d["extrapolation_flag"].any()):
        st.warning(
            "No district in this project has a published GDP. This figure is its "
            "state's total, **allocated by light share**, so it inherits both the "
            "state-level model error and the allocation bias. Treat the band, not "
            "the point, as the answer."
        )
    else:
        st.caption("This district's estimate is shown with its 90% interval.")

    # Match the map's default: per capita is the headline metric, so the district
    # chart must not quietly show total instead.
    has_pc = "gdp_per_capita_estimate" in d.columns and d["gdp_per_capita_estimate"].notna().any()
    basis = st.radio(
        "Show",
        ["GDP per capita", "Total GDP"] if has_pc else ["Total GDP"],
        horizontal=True,
        key="district_basis",
    )
    prefix = "gdp_per_capita" if basis == "GDP per capita" else "gdp"
    cols = [f"{prefix}_lower", f"{prefix}_estimate", f"{prefix}_upper"]

    if set(cols).issubset(d.columns):
        st.line_chart(d.set_index("year")[cols])
    elif {"gdp_estimate", "gdp_lower", "gdp_upper"}.issubset(d.columns):
        st.line_chart(d.set_index("year")[["gdp_lower", "gdp_estimate", "gdp_upper"]])
    else:
        st.line_chart(d.set_index("year")[["prediction"]])

    if "state_gdp_source" in d.columns:
        predicted_years = sorted(d.loc[d["state_gdp_source"] == "predicted", "year"].tolist())
        if predicted_years:
            st.caption(
                f"Years {predicted_years} are downscaled from a **predicted** state "
                "total; the rest from a published one."
            )

    if "n_valid_months" in d.columns:
        st.subheader("Data quality: valid months per year")
        st.bar_chart(d.set_index("year")[["n_valid_months"]])

    st.dataframe(d)


def page_diagnostics(estimates: pd.DataFrame) -> None:
    st.header("Model diagnostics")
    st.caption("Published on purpose. A dashboard that hides these is decoration.")
    diag = load_diagnostics()
    if not diag:
        st.info(
            "No diagnostics.json yet. Write one from evaluate.report() so the "
            "elasticity, holdout scores, coverage and unmatched count are visible here."
        )
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Cross-sectional elasticity",
            f"{diag.get('cross_elasticity', float('nan')):.3f}",
            help="The identified quantity. Expect 0.6-1.0. This is what the estimates rest on.",
        )
        col2.metric("Spatial-holdout R2", f"{diag.get('spatial_log_r2', float('nan')):.3f}")
        col3.metric(
            "Interval coverage",
            f"{diag.get('interval_coverage', float('nan')):.2f}",
            help=f"Nominal {diag.get('nominal_coverage', 0.9):.2f}",
        )
        st.caption(
            f"Within elasticity {diag.get('within_elasticity', float('nan')):+.3f} "
            f"(se {diag.get('within_se', float('nan')):.3f}) is **reported, not asserted**: "
            "six overlapping label years do not identify it. It is not a defect in the "
            "extraction."
        )

        contaminated = diag.get("contaminated_diagnostics") or {}
        if contaminated.get("sign_flips_across_break"):
            st.error(
                f"Entity-FE-only elasticity "
                f"{contaminated.get('within_entity_only', float('nan')):+.3f} flips sign "
                f"across {contaminated.get('break_year')} "
                f"({contaminated.get('entity_only_pre_break', float('nan')):+.3f} before, "
                f"{contaminated.get('entity_only_post_break', float('nan')):+.3f} after). "
                "It retains the VIIRS calibration discontinuity and is NOT a validation "
                "signal, despite landing near the literature's 0.3.",
                icon="🚫",
            )

        alloc = diag.get("allocation_validation") or {}
        if alloc.get("n_district_years"):
            st.subheader("Within-state allocation, validated")
            a1, a2, a3 = st.columns(3)
            a1.metric(
                "Light share vs true GDP share",
                f"r = {alloc.get('pearson', float('nan')):.3f}",
                help=f"{alloc['n_district_years']} district-years across "
                f"{', '.join(alloc.get('states', []))}, held out of training.",
            )
            a2.metric("Mean absolute error", f"{alloc.get('mae_share_points', 0):.2f} pts")
            a3.metric(
                "Size gradient",
                f"{alloc.get('size_gradient_corr', float('nan')):+.2f}",
                help="Negative means large districts are under-allocated (sensor saturation).",
            )
            corr = alloc.get("share_correction") or {}
            if corr.get("verdict"):
                folds = corr.get("folds") or []
                if corr.get("accepted"):
                    st.success(f"Share correction: {corr['verdict']}", icon="✅")
                else:
                    st.warning(
                        f"**Share correction {corr['verdict']}** — "
                        f"alpha ranged {corr['alpha_range'][0]:.3f} to "
                        f"{corr['alpha_range'][1]:.3f} across folds and held-out error "
                        f"worsened in {sum(1 for f in folds if not f['improved'])} of "
                        f"{len(folds)}. Published shares are uncorrected.",
                        icon="⚠️",
                    )
                if folds:
                    st.dataframe(
                        pd.DataFrame(folds)[
                            [
                                "held_out_state",
                                "n_test",
                                "alpha_fitted_on_others",
                                "mae_uncorrected_pts",
                                "mae_corrected_pts",
                                "delta_mae_pts",
                                "pearson_uncorrected",
                                "pearson_corrected",
                            ]
                        ].round(3),
                        hide_index=True,
                    )

            sect = alloc.get("sectoral_gradient") or {}
            if sect:
                st.caption(
                    "Sectoral bias (uncorrected): services "
                    f"{sect.get('services', float('nan')):+.2f}, agriculture "
                    f"{sect.get('agriculture', float('nan')):+.2f}, industry "
                    f"{sect.get('industry', float('nan')):+.2f}. Negative means light "
                    "under-allocates GDP to districts led by that sector."
                )

        dropped = diag.get("dropped_label_regions") or []
        if dropped:
            st.warning(
                f"Excluded from training entirely: **{', '.join(dropped)}** "
                f"({diag.get('n_dropped_districts', 0)} districts), for incomplete year "
                "coverage. No estimate for these rests on their own label.",
                icon="⚠️",
            )

        st.write("Unmatched regions:", diag.get("n_unmatched", "unknown"))
        st.json(diag)

    if not estimates.empty and "vintage" in estimates.columns:
        st.subheader("Estimate vintages")
        st.write(
            sorted(pd.read_parquet(ESTIMATES_PATH)["vintage"].unique())
            if ESTIMATES_PATH.exists()
            else []
        )


def page_caveats() -> None:
    st.header("What this does and does not measure")
    st.markdown(
        """
### Levels, not growth

**This model estimates how large a district's economy is, not how fast it is growing.**

The cross-sectional relationship is well identified: brighter districts really are
richer districts, at an elasticity of 0.83 that matches the published literature.
The *within* relationship, how a district's GDP moves when its own lights move, is
**not** identified here. The GDP label series ends in 2019 and the satellite series
begins in 2014, leaving six overlapping years, and once state and year effects are
removed almost no variation survives.

So a district's estimated level carries real information. Year-on-year *changes* in
that estimate should not be read as growth, and a ranking of districts by growth
rate would be noise. This is a limitation of the label data, not of the satellite.

### District GDP is an allocation, not a measurement

Training labels exist only at state level, so a state's total is **split across its
districts in proportion to their share of the state's light**. District estimates
sum exactly to the state total, and the national total is correct by construction.

That split has been **validated against published district GDDP** from Tamil Nadu,
Maharashtra and Karnataka — 321 district-years across 95 districts, held out of
training entirely. Light share correlates **0.81** with true GDP share (mean
absolute error 0.99 share points). So the split carries real information.

**It is also biased, systematically, and the bias has not been corrected:**

- **Service economies are under-allocated.** In Karnataka, where sector shares are
  published, the correlation between a district's services share and its
  allocation error is **−0.59**. An office district produces far more output per
  lumen than a factory or a highway.
- **Agricultural districts are over-allocated** (correlation **+0.54**). Spread-out
  village lighting buys more light share than it does output.
- **Large districts are under-allocated** — the DNB sensor saturates in dense
  cores. Top-decile districts by GDP share receive on average **0.80×** their true
  share; bottom-half districts receive **1.22×**.

Worst cases: Thane receives 0.31× its true share, Dakshina Kannada 0.25×, Chennai
0.35×; Ahmadnagar receives 2.01×, Bangalore Rural 1.95×.

Read a district's figure with that direction in mind: if it is a service or port
economy, the estimate is probably low; if it is agricultural or peri-urban, high.

**We tried to correct this and could not.** A one-parameter reweighting
(`share^α`) was fitted and tested leave-one-state-out. It made the held-out error
*worse* in all three folds, and the fitted α straddled 1.0, meaning the states
disagree about which way to correct: Karnataka's own optimum is 1.376, Tamil
Nadu's is 0.904. No single exponent fits them.

Notably, the α that most improves the face-validity check (1.376) is the one the
generalisation test rejects hardest. Applying it would mean fitting to 30
Karnataka districts and imposing that on all 676.

So the shipped figures are **uncorrected**. `allocation_share_uncorrected` is
published on every row so the raw split can always be reconstructed. This is a
negative result, reported rather than hidden.

### The population denominator is not fully independent

GDP per capita uses GHSL population as its denominator. GHSL distributes census
counts across a grid using **built-up-area layers derived partly from satellite
imagery**. That imagery is optical and radar, *not* nighttime lights, so the
coupling with our light-based numerator is indirect and weak — but it is not
zero. Built-up area and lit area correlate, so a district whose built-up extent
GHSL overestimates may receive both more people and more light.

The practical consequence is mild: it can compress per-capita differences between
districts slightly. It is stated here rather than left for a reader to discover.

### What nightlights miss

Nightlights track energy-intensive and urban activity. They are a **proxy**, not a
measurement, and they miss things that matter:

- **Subsistence agriculture, informal services and remittance-driven consumption**
  are badly underrepresented. In an agrarian district, household income can rise
  while measured light barely moves.
- **Dense urban cores saturate** the sensor. Growth in an already-bright city is
  understated, so the biggest places are the least reliable.
- **LED conversion** lowers emitted light in the DNB band even as activity rises.
  Estimates for rapidly-LED-ifying cities are a **lower bound**.
- **The ADM1 to ADM2 jump.** The model trains on state-level GDP and predicts
  districts, a roughly twentyfold resolution gap. District estimates are
  extrapolations, flagged as such, and carry wide intervals.

Use the intervals. A point estimate here is the middle of a range, not a figure to
quote to three significant digits.
        """
    )


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Nightlights GDP proxy", layout="wide")
    st.title("Subnational GDP from nighttime lights")
    st.caption("Research and non-commercial use. Estimates are a proxy with uncertainty.")
    # The project's stated position, on every page, not buried in Caveats.
    st.warning(
        "**This estimates district GDP levels, not growth.** How large a district's "
        "economy is, is well identified (cross-sectional elasticity 0.81). How fast it "
        "is growing is not: six overlapping label years do not identify the "
        "within-district relationship. Do not read year-on-year changes as growth or "
        "rank districts by growth rate.",
        icon="⚠️",
    )

    estimates = load_estimates()
    page = st.sidebar.radio("Page", ["Map", "District detail", "Diagnostics", "Caveats"])
    if page == "Map":
        page_map(estimates)
    elif page == "District detail":
        page_district(estimates)
    elif page == "Diagnostics":
        page_diagnostics(estimates)
    else:
        page_caveats()


if __name__ == "__main__":
    main()
