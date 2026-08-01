# Pitfalls

Each of these produces a result that looks fine. That is what makes them
expensive. Check the relevant ones before declaring a phase done.

## Data and joins

**Silent row loss.** A merge on names drops the districts whose spelling differs
and reports no error. The panel shrinks, the model still fits, the map still
renders. Assert row counts after every join and name the dropped regions.

**Duplicate district names across states.** India has multiple Aurangabads,
Bilaspurs and Hamirpurs. Matching without blocking on the parent state maps
economic data to the wrong place, and the error is invisible on a map. Block on
`parent_name` before scoring.

**District splits and renames.** India created dozens of districts in the last
decade. A 2014 boundary set and 2023 labels will not align, and neither source
announces this. Freeze one boundary vintage, document the mapping to newer ones,
and treat a district that appears mid-series as a fact to record rather than a
row to drop.

**Nominal values.** Training on nominal GDP teaches the model the inflation
series. Deflate first, always.

**Unit breaks.** Indian statistics move between lakh, crore and rupees, sometimes
mid-table. A 100x jump in one year is a units break, not growth. Flag year-on-year
real growth above 50 percent.

## Satellite and masking

**Missing treated as zero.** The most consequential bug available in this
project. A cloudy month becomes a dark district, the district looks poor, and
the estimate is confidently wrong. `cf_cvg` at or below 1 is missing data.

**Product mixing.** VCMCFG and VCMSLCFG differ in stray-light handling; NOAA
VIIRS and NASA Black Marble differ in atmospheric correction. Splicing them
mid-series creates growth that is not there. Pick one family for the whole
series; if you must switch, estimate the offset on an overlap period and record
it.

**Gas flares.** An oil district with flaring reads like a megacity. Mask flare
buffers. Until the EOG licence arrives, hand-mask the known basins from config.

**Boat and rig lights.** Fishing fleets light up coastal waters. The JRC water
mask removes most of it. Without it, coastal districts show phantom activity
that moves with fishing seasons.

**Saturation.** Dense urban cores max out the sensor, so SOL understates growth
exactly where growth is largest. `lit_share` and `p90_rad` keep responding after
radiance flattens. Check residuals against baseline brightness; a downward slope
means saturation is biasing you.

**LED conversion.** Cities replacing sodium lamps with LEDs emit less in the DNB
band while activity rises. This biases estimates downward in exactly the
middle-income places the project targets. Look for step changes, and treat
affected districts as a lower bound rather than pretending it is not happening.

**Seasonality.** Monsoon and high-latitude winter produce months with almost no
valid observations. Use annual composites for training and require a minimum
valid-month count before annualising.

## Modelling

**Random K-fold.** Leaks through panel autocorrelation and reports R2 above 0.95.
If you see that number, look for the split before believing the model.

**Feature leakage of region identity.** `area_km2` and unnormalised `sol` encode
which district a row is. In a spatial holdout this is a leak. Normalise, or
exclude them from the tree model.

**Overfitting a tiny panel.** Thirty states times ten years is a few hundred
rows. A deep boosted forest memorises it. Keep depth shallow and expect only
modest gains over the log-log baseline.

**Point estimates without intervals.** Not shippable. The use case is capital
allocation decisions; a number with no error bar invites false precision.

**Implicit extrapolation.** Predicting 640 districts from 36 state labels is a
twentyfold jump. It is defensible if stated and quantified, and misleading if
presented as a measurement. Flag extrapolated districts explicitly.

## Infrastructure

**Quota burn.** Interactive `getInfo` over hundreds of polygons and months
exhausts the monthly EECU allowance and throttles the account until the calendar
month resets. Batch exports only.

**Recomputation.** Re-running an already-extracted year is the most common way
quota disappears. Check for the output parquet and skip.

**GEE reprocessing.** Dataset versions change and results move. Record the
dataset id, image ids and extraction timestamp in every output, which the
existing extractor already does. Without it you cannot explain why a number
moved.

**Dashboard calling Earth Engine on page load.** Every visitor spends your
quota. Precompute to parquet.

**GADM licence.** Academic and non-commercial only, no redistribution. Fine for
the current scope. If the project ever turns commercial, swap the boundary
source to geoBoundaries, which the loader already supports, and re-key the
panel, because the region ids will change.
