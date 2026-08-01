"""Phase 2a: masking.

Order matters and the reasons are not obvious, so they are written down here.

  1. Coverage    cf_cvg below the threshold means the sensor did not get a clean
                 look. Those pixels are MASKED (missing), never set to zero. A
                 cloudy month over a district is not a dark district.
  2. Water       JRC surface water kills boat, fishing and offshore rig lights,
                 which otherwise show up as phantom coastal economic activity.
  3. Flares      Gas flaring is combustion, not human settlement. Left in, oil
                 districts look like megacities.
  4. Noise floor Applied LAST, and it sets values to zero rather than masking.
                 A genuinely dark rural pixel is real information: it means no
                 measurable activity. Masking it would bias SOL upward.

The distinction in steps 1 and 4 is the whole ballgame. Masked means "we do not
know". Zero means "we looked and there was nothing".

``ee`` is imported inside the functions so the pure-Python helpers stay
testable without Earth Engine credentials.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

JRC_WATER = "JRC/GSW1_4/GlobalSurfaceWater"

EARTH_RADIUS_KM = 6371.0088


# --------------------------------------------------------------------------
# pure helpers (no Earth Engine)
# --------------------------------------------------------------------------


def flare_sites(cfg: dict[str, Any]) -> list[dict[str, float]]:
    """Return flare sites as {lon, lat, radius_m} from config.

    Until the EOG Nightfire licence comes through, this reads the hand-listed
    ``manual_flare_regions`` block. When the licence arrives, point this at the
    real flare point file and the rest of the pipeline does not change.
    """
    default_buffer_m = float(cfg.get("flare_buffer_m", 7500))
    sites: list[dict[str, float]] = []
    for entry in cfg.get("manual_flare_regions") or []:
        radius_km = entry.get("radius_km")
        radius_m = float(radius_km) * 1000 if radius_km else default_buffer_m
        sites.append(
            {
                "name": str(entry.get("name", "unnamed")),
                "lon": float(entry["lon"]),
                "lat": float(entry["lat"]),
                "radius_m": radius_m,
            }
        )
    return sites


def mask_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    """Human-readable record of the masking configuration.

    Written into every output so a parquet from March can be told apart from
    one produced after you retuned the noise floor.
    """
    return {
        "min_cf_cvg": int(cfg.get("min_cf_cvg", 2)),
        "noise_floor": float(cfg.get("noise_floor", 0.4)),
        "lit_threshold": float(cfg.get("lit_threshold", 0.5)),
        "water_occurrence_threshold": int(cfg.get("water_occurrence_threshold", 50)),
        "n_flare_sites": len(flare_sites(cfg)),
        "flare_buffer_m": float(cfg.get("flare_buffer_m", 7500)),
    }


def validate_mask_config(cfg: dict[str, Any]) -> list[str]:
    """Return a list of problems with the masking config. Empty means fine."""
    problems: list[str] = []

    noise_floor = float(cfg.get("noise_floor", 0.4))
    lit_threshold = float(cfg.get("lit_threshold", 0.5))
    min_cf_cvg = int(cfg.get("min_cf_cvg", 2))

    if noise_floor <= 0:
        problems.append("noise_floor must be positive; VIIRS has real negative-radiance noise")
    if noise_floor > 2.0:
        problems.append(
            f"noise_floor {noise_floor} is very high and will erase small towns entirely"
        )
    if lit_threshold < noise_floor:
        problems.append(
            f"lit_threshold {lit_threshold} is below noise_floor {noise_floor}, so every "
            "zeroed pixel would still count as lit"
        )
    if min_cf_cvg < 1:
        problems.append(
            "min_cf_cvg below 1 accepts pixels with no cloud-free observation, which "
            "turns missing data into fake zeros"
        )
    for site in flare_sites(cfg):
        if not -180 <= site["lon"] <= 180 or not -90 <= site["lat"] <= 90:
            problems.append(f"flare site {site['name']} has out-of-range coordinates")
        if site["radius_m"] <= 0:
            problems.append(f"flare site {site['name']} has a non-positive radius")
    return problems


# --------------------------------------------------------------------------
# Earth Engine layers
# --------------------------------------------------------------------------


def water_mask_image(cfg: dict[str, Any]):
    """Boolean image: 1 where land, 0 where persistent water."""
    import ee

    threshold = int(cfg.get("water_occurrence_threshold", 50))
    occurrence = ee.Image(JRC_WATER).select("occurrence").unmask(0)
    return occurrence.lt(threshold)


def flare_mask_image(cfg: dict[str, Any]):
    """Boolean image: 1 outside flare buffers, 0 inside."""
    import ee

    sites = flare_sites(cfg)
    if not sites:
        return ee.Image.constant(1)

    buffers = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point([s["lon"], s["lat"]]).buffer(s["radius_m"])) for s in sites]
    )
    inside = ee.Image.constant(0).paint(buffers, 1).unmask(0)
    return inside.Not()


def apply_masks(image, cfg: dict[str, Any]):
    """Return a 3-band image: masked radiance, lit flag, valid flag.

    Bands
      rad    radiance after all masking, zeroed below the noise floor
      lit    1 where rad exceeds lit_threshold, else 0
      valid  1 where the pixel survived masking, used to count usable pixels
    """
    import ee

    rad_band = cfg.get("radiance_band", "avg_rad")
    cvg_band = cfg.get("coverage_band", "cf_cvg")
    min_cvg = int(cfg.get("min_cf_cvg", 2))
    noise_floor = float(cfg.get("noise_floor", 0.4))
    lit_threshold = float(cfg.get("lit_threshold", 0.5))

    image = ee.Image(image)
    rad = image.select(rad_band)

    # 1. coverage: mask, do not zero
    coverage_ok = image.select(cvg_band).gte(min_cvg)
    rad = rad.updateMask(coverage_ok)

    # 2 and 3. water and flares: mask
    rad = rad.updateMask(water_mask_image(cfg)).updateMask(flare_mask_image(cfg))

    # 4. noise floor: zero, do not mask. Also clamps negative retrievals.
    rad = rad.where(rad.lt(noise_floor), 0).rename("rad")

    lit = rad.gt(lit_threshold).rename("lit")
    valid = rad.mask().rename("valid")

    return rad.addBands(lit).addBands(valid).copyProperties(image, ["system:time_start"])
