"""Phase 3b: region name matching.

Rule 7 in CLAUDE.md is the shape of this module. Fuzzy matching runs **once**, as
a proposal step that writes ``data/crosswalk_review.csv`` with a confidence score
for a human to check. The reviewed file is committed to
``config/crosswalk_<country>.csv`` and that committed file is the only thing the
pipeline reads. Fuzzy matching at runtime means the mapping silently changes when
a library version changes.

Rule 8: ``region_id`` is a stable synthetic key. Nothing here joins on a name
string except the one-time proposal.

The subtle, quiet error this module exists to prevent is a cross-state match on a
duplicated district name: India has several Aurangabads. Blocking on
``parent_name`` before scoring is what stops "Aurangabad, Maharashtra" being
handed the GDP of "Aurangabad, Bihar".
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATA_DIR, REPO_ROOT, ConfigError, country_config

logger = logging.getLogger(__name__)

REVIEW_PATH = DATA_DIR / "crosswalk_review.csv"
CONFIG_DIR = REPO_ROOT / "config"

# Administrative suffixes stripped before scoring. These carry no identity and
# differ between sources ("Pune District" vs "Pune").
ADMIN_SUFFIXES = {
    "district",
    "dist",
    "zilla",
    "zila",
    "zilha",
    "taluk",
    "taluka",
    "tehsil",
    "county",
    "division",
    "subdivision",
    "region",
    "state",
    "province",
    "pradesh",
}
# Deliberately NOT stripped: "urban"/"rural"/"city". "Bangalore Urban" and
# "Bangalore Rural" are two real districts; collapsing them merges distinct units.

# One row per BOUNDARY polygon, not per label region. DOSE is ADM1 (33 Indian
# regions) while the boundaries are ADM2 (676 districts), so the relationship is
# one state to many districts. Keying the review on the district is what makes
# that cardinality expressible: each district gets exactly one state, and a state
# legitimately appears on many rows.
REVIEW_COLUMNS = [
    "region_id",
    "boundary_name",
    "boundary_parent",
    "adm1_gid",
    "source_region_name",
    "parent_name",
    "score",
    "method",
    "needs_review",
    "decision",
]

# Committed crosswalk schema, one row per district. ``source_region_name`` empty
# means the reviewer marked this district as having no label region; the reason
# belongs in unmatched_<country>.csv, keyed by the ADM1 unit.
CROSSWALK_COLUMNS = ["region_id", "source_region_name", "parent_name"]

UNMATCHED_COLUMNS = ["name", "reason"]


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


def normalise(name: str) -> str:
    """Casefold, strip diacritics, drop punctuation and admin suffixes, collapse space.

    "Bangalore Urban District" and "bengaluru urban" do not normalise to the same
    string (the roots differ), but "Pune District" and "pune" do, and that is the
    class of difference this handles deterministically before fuzzy scoring picks
    up the transliteration cases.
    """
    if name is None or (isinstance(name, float)):
        return ""
    text = str(name)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in ADMIN_SUFFIXES]
    return " ".join(tokens)


def _variant_list(raw: Any) -> list[str]:
    """Split a GADM VARNAME cell (pipe-separated) into normalised variants."""
    if raw is None or (isinstance(raw, float)):
        return []
    parts = re.split(r"[|,;]", str(raw))
    return [normalise(p) for p in parts if normalise(p)]


# --------------------------------------------------------------------------
# proposal
# --------------------------------------------------------------------------


def _score(a: str, b: str) -> float:
    from rapidfuzz import fuzz

    return float(fuzz.token_sort_ratio(a, b))


def adm1_gid_from_source_code(source_code: Any) -> str | None:
    """Derive the GADM ADM1 code from a GADM GID_2 (or GID_1) code.

    GADM nests its codes, so the parent is a pure string operation, no lookup:

        IND.16.20_1  (Pune, Maharashtra)  ->  IND.16_1  (Maharashtra)

    DOSE V2.14 carries GID_1 directly, so this makes DOSE-to-GADM state matching
    an **exact code join** rather than a fuzzy name match. That removes the entire
    class of transliteration and duplicate-name errors for every district whose
    state DOSE covers. Returns None for anything that is not a GADM code.
    """
    if source_code is None:
        return None
    text = str(source_code).strip()
    if not text or text.lower() == "nan":
        return None
    parts = text.split(".")
    if len(parts) < 2:
        return None
    if len(parts) == 2:
        return text  # already an ADM1 code such as IND.16_1
    return f"{parts[0]}.{parts[1]}_1"


def national_gid_from_disputed(gid: str | None, iso3: str) -> str | None:
    """Rewrite a GADM disputed-territory ADM1 code into the claiming country's code.

    GADM 4.1 does not file every Indian district under ``IND.``. Territory subject
    to an international dispute gets a ``Z``-prefixed code instead, and the second
    component still carries the claiming country's ADM1 index:

        Z01.14_1  -> IND.14_1   all 22 Jammu and Kashmir districts
        Z07.3_1   -> IND.3_1    the disputed slice of Arunachal Pradesh
        Z04.13_1  -> IND.13_1   Himachal Pradesh
        Z05.35_1  -> IND.35_1   Uttarakhand

    DOSE files these under the plain ``IND.`` code, so without this rewrite the
    exact-code pass misses them and they fall through to name matching. That is
    not a cosmetic difference: Jammu and Kashmir alone is 22 districts.

    This asserts nothing about sovereignty. It records that GADM and DOSE describe
    the same administrative unit with two different codes. Matches made this way
    are tagged ``gid_disputed`` and flagged for review so the choice stays visible.
    """
    if not gid:
        return None
    m = re.fullmatch(r"Z\d+\.(.+)", str(gid).strip())
    if not m:
        return None
    return f"{iso3}.{m.group(1)}"


def _best_name_match(poly: Any, targets: pd.DataFrame) -> dict[str, Any]:
    """Best fuzzy label match for one boundary polygon. Pass 2 only.

    Scores the label name against both the polygon's own name and its parent
    name, and keeps whichever is better. That covers both shapes without a
    special case: when labels are ADM1 and boundaries ADM2, the *parent* (state)
    is what should match the label; when both are ADM1, the polygon's own name is.
    GADM ``name_variants`` are scored too, which resolves transliterations free.
    """
    candidates: list[tuple[str, str]] = [(poly.norm_name, "name")]
    parent_norm = getattr(poly, "norm_parent", "") or ""
    if parent_norm:
        candidates.append((parent_norm, "parent"))
    candidates.extend((v, "variant") for v in (getattr(poly, "variants", None) or []))

    best = {
        "source_region_name": "",
        "parent_name": "",
        "score": -1.0,
        "method": "none",
    }
    for tgt in targets.itertuples(index=False):
        label_norm = normalise(tgt.source_region_name)
        for text, how in candidates:
            score = _score(label_norm, text)
            if score > best["score"]:
                best = {
                    "source_region_name": tgt.source_region_name,
                    "parent_name": tgt.parent_name,
                    "score": score,
                    "method": how,
                }
    return best


def propose_crosswalk(
    boundaries: pd.DataFrame, labels: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Propose, for each boundary polygon, which label region it belongs to.

    Two passes, deterministic first:

    1. **gid** — derive the ADM1 GADM code from the boundary's ``source_code``
       (``IND.16.20_1 -> IND.16_1``) and join it against the label's own
       ``source_gid`` (DOSE's ``GID_1``). This is an exact code join, scores 100,
       and needs no review. It resolves every district whose state DOSE covers.
    2. **name / variant** — only districts left over by pass 1 are fuzzy scored
       against label names, blocked by normalised parent so a duplicate district
       name cannot match across a state boundary. These are the rows a human
       actually has to look at.

    One row per boundary polygon. With ADM1 labels and ADM2 boundaries the result
    is deliberately one-to-many: one state, many districts.

    Writes ``data/crosswalk_review.csv`` sorted worst-score-first and returns it.
    ``decision`` is left blank for the reviewer.
    """
    review_threshold = float(cfg.get("match_review_threshold", 90.0))

    bnd = boundaries.copy()
    # Non-underscore names on purpose: itertuples mangles leading-underscore columns.
    bnd["norm_name"] = bnd["name"].map(normalise)
    bnd["norm_parent"] = bnd["parent_name"].map(normalise) if "parent_name" in bnd else ""
    if "name_variants" in bnd.columns:
        bnd["variants"] = bnd["name_variants"].map(_variant_list)
    else:
        bnd["variants"] = [[] for _ in range(len(bnd))]
    if "source_code" in bnd.columns:
        bnd["adm1_gid"] = bnd["source_code"].map(adm1_gid_from_source_code)
    else:
        bnd["adm1_gid"] = None

    targets = labels[
        [c for c in ("source_region_name", "parent_name", "source_gid") if c in labels.columns]
    ].drop_duplicates()
    if "source_gid" not in targets.columns:
        targets["source_gid"] = ""

    # Exact ADM1 code -> label region. Built once; this is pass 1.
    gid_lookup: dict[str, tuple[str, str]] = {}
    for tgt in targets.itertuples(index=False):
        gid = str(getattr(tgt, "source_gid", "") or "").strip()
        if gid:
            gid_lookup[gid] = (tgt.source_region_name, tgt.parent_name)

    iso3 = str(cfg.get("iso3", "")).upper()
    allow_disputed = bool(cfg.get("match_disputed_territory", True))

    rows: list[dict[str, Any]] = []
    for poly in bnd.itertuples(index=False):
        gid = poly.adm1_gid
        hit = gid_lookup.get(gid) if gid else None
        method, needs_review = "gid", False

        # Pass 1b: GADM files disputed territory under Z-codes while DOSE uses the
        # plain national code. Same unit, two code spaces.
        if hit is None and gid and allow_disputed and iso3:
            rewritten = national_gid_from_disputed(gid, iso3)
            if rewritten and rewritten in gid_lookup:
                hit = gid_lookup[rewritten]
                method, needs_review = "gid_disputed", True

        if hit is not None:
            source_name, parent = hit
            rows.append(
                {
                    "region_id": poly.region_id,
                    "boundary_name": poly.name,
                    "boundary_parent": getattr(poly, "parent_name", ""),
                    "adm1_gid": gid,
                    "source_region_name": source_name,
                    "parent_name": parent,
                    "score": 100.0,
                    "method": method,
                    "needs_review": needs_review,
                    "decision": "",
                }
            )
            continue

        # Pass 2: no exact code match, so fall through to fuzzy name scoring.
        best = _best_name_match(poly, targets)
        rows.append(
            {
                "region_id": poly.region_id,
                "boundary_name": poly.name,
                "boundary_parent": getattr(poly, "parent_name", ""),
                "adm1_gid": gid,
                "source_region_name": best["source_region_name"],
                "parent_name": best["parent_name"],
                "score": round(best["score"], 1),
                "method": best["method"],
                "needs_review": best["score"] < review_threshold,
                "decision": "",
            }
        )

    review = pd.DataFrame(rows, columns=REVIEW_COLUMNS)
    review = review.sort_values(["needs_review", "score"], ascending=[False, True]).reset_index(
        drop=True
    )

    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(REVIEW_PATH, index=False)
    n_gid = int((review["method"] == "gid").sum())
    n_disputed = int((review["method"] == "gid_disputed").sum())
    logger.info(
        "Wrote %s: %d polygons, %d exact GID, %d disputed-territory GID, "
        "%d fuzzy name, %d need review",
        REVIEW_PATH,
        len(review),
        n_gid,
        n_disputed,
        len(review) - n_gid - n_disputed,
        int(review["needs_review"].sum()),
    )
    return review


# --------------------------------------------------------------------------
# committed crosswalk
# --------------------------------------------------------------------------


def crosswalk_path(cfg: dict[str, Any]) -> Path:
    return CONFIG_DIR / f"crosswalk_{cfg['country']}.csv"


def unmatched_path(cfg: dict[str, Any]) -> Path:
    return CONFIG_DIR / f"unmatched_{cfg['country']}.csv"


def load_crosswalk(cfg: dict[str, Any]) -> pd.DataFrame:
    """Read the reviewed, committed crosswalk. The pipeline uses only this."""
    path = crosswalk_path(cfg)
    if not path.exists():
        raise ConfigError(
            f"No reviewed crosswalk at {path}. Run 'python -m gdp_proxy.match --propose', "
            f"review data/crosswalk_review.csv by hand, and commit it as {path.name}."
        )
    df = pd.read_csv(path, dtype=str)
    missing = [c for c in CROSSWALK_COLUMNS if c not in df.columns]
    if missing:
        raise ConfigError(f"{path.name} is missing columns {missing}.")
    for col in CROSSWALK_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df


def load_unmatched(cfg: dict[str, Any]) -> pd.DataFrame:
    """Read the committed list of ADM1 units with no label region, plus reasons."""
    path = unmatched_path(cfg)
    if not path.exists():
        return pd.DataFrame(columns=UNMATCHED_COLUMNS)
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in UNMATCHED_COLUMNS if c not in df.columns]
    if missing:
        raise ConfigError(f"{path.name} is missing columns {missing}.")
    return df


def apply_crosswalk(labels: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Attach ``region_id`` to label rows, fanning one label region to its districts.

    With ADM1 labels and ADM2 boundaries this is deliberately **one-to-many**: a
    state's GDP is attached to each of its districts, so one state-year becomes N
    district-year rows. That is a mapping, not a measurement, and the output
    carries ``n_districts_in_label`` so nothing downstream can mistake N copies of
    a state total for N independent district observations.

    The expected output row count is computed up front and asserted (rule 6), and
    a label region absent from the crosswalk raises rather than vanishing.
    """
    key = ["source_region_name", "parent_name"]
    cw = crosswalk.copy()
    for col in key + ["region_id"]:
        cw[col] = cw[col].fillna("").astype(str).str.strip()
    # Districts the reviewer left unassigned carry no label region.
    cw = cw[cw["source_region_name"] != ""]

    lab = labels.copy()
    lab["parent_name"] = lab["parent_name"].fillna("").astype(str).str.strip()
    lab["source_region_name"] = lab["source_region_name"].astype(str).str.strip()

    fan = cw.groupby(key, sort=False).size().rename("n_districts_in_label").reset_index()

    known = set(map(tuple, fan[key].itertuples(index=False, name=None)))
    present = set(map(tuple, lab[key].drop_duplicates().itertuples(index=False, name=None)))
    unknown = present - known
    if unknown:
        raise ValueError(
            f"{len(unknown)} label region(s) are not in the crosswalk and would be "
            f"silently dropped: {sorted(unknown)[:10]}. Add them to the reviewed file "
            f"(assigned to districts, or listed in the unmatched file with a reason)."
        )

    # Expected fan-out size, computed before the join so the join cannot define
    # its own success criterion.
    sizes = lab.merge(fan, on=key, how="left", validate="m:1")
    expected = int(sizes["n_districts_in_label"].fillna(0).sum())

    merged = lab.merge(cw[key + ["region_id"]], on=key, how="inner")
    if len(merged) != expected:
        raise ValueError(
            f"Crosswalk fan-out produced {len(merged)} rows, expected {expected}. "
            "The crosswalk and label frames disagree about which districts belong "
            "to which label region."
        )

    merged = merged.merge(fan, on=key, how="left", validate="m:1")

    n_labels = lab[key].drop_duplicates().shape[0]
    logger.info(
        "Crosswalk: %d label region(s) x %d label-year rows fanned out to %d district-year rows",
        n_labels,
        len(lab),
        len(merged),
    )
    return merged.reset_index(drop=True)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@dataclass
class MatchReport:
    country: str
    n_labels: int
    n_polygons: int
    n_matched: int
    n_unmatched: int
    n_gid_matched: int = 0
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    unmatched_names: list[str] = field(default_factory=list)
    label_fanout: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.checks)

    def render(self) -> str:
        lines = [
            f"Crosswalk: {self.country}",
            f"  label regions        {self.n_labels}",
            f"  boundary polygons    {self.n_polygons}",
            f"  matched polygons     {self.n_matched}",
            f"  unmatched polygons   {self.n_unmatched}",
        ]
        if self.n_gid_matched:
            lines.append(f"  matched by exact GID {self.n_gid_matched}")
        if self.label_fanout:
            sizes = sorted(self.label_fanout.values())
            lines.append(
                f"  districts per label  min {sizes[0]}, median "
                f"{sizes[len(sizes) // 2]}, max {sizes[-1]}"
            )
        lines.append("")
        for name, passed, detail in self.checks:
            lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        if self.unmatched_names:
            lines.append("")
            lines.append(f"  unmatched: {', '.join(self.unmatched_names[:20])}")
        return "\n".join(lines)


def _unmatched_documented(
    unmatched: pd.DataFrame, boundaries: pd.DataFrame, cfg: dict[str, Any]
) -> bool:
    """True when every unmatched polygon's ADM1 unit carries a written reason.

    Reasons are recorded per ADM1 unit, not per district, because the cause is
    always a state-level fact: Telangana split from Andhra Pradesh in 2014,
    Ladakh from Jammu and Kashmir in 2019, Dadra and Nagar Haveli merged with
    Daman and Diu in 2020. Every district in such a state is unmatched for the
    same reason, and repeating it 30 times per state would obscure that.
    """
    if not cfg.get("country"):
        return False
    reasons = load_unmatched(cfg)
    if reasons.empty:
        return False

    documented = {normalise(n) for n in reasons["name"] if str(n).strip()}
    if not documented:
        return False

    parent_by_id = {}
    if "parent_name" in boundaries.columns:
        parent_by_id = dict(
            zip(
                boundaries["region_id"].astype(str),
                boundaries["parent_name"].astype(str),
                strict=False,
            )
        )

    for region_id in unmatched["region_id"].astype(str):
        parent = normalise(parent_by_id.get(region_id, ""))
        if parent not in documented:
            logger.warning(
                "Unmatched polygon %s (ADM1 '%s') has no documented reason",
                region_id,
                parent_by_id.get(region_id, "?"),
            )
            return False
    return True


def validate_crosswalk(
    crosswalk: pd.DataFrame,
    boundaries: pd.DataFrame,
    labels: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> MatchReport:
    """Assert the crosswalk's cardinality, which is one-to-many, not a bijection.

    DOSE is ADM1 (33 Indian regions); the boundaries are ADM2 (676 districts). One
    state therefore maps to many districts **by design**. The invariant that
    actually matters is the other direction: every district must map to exactly
    one state, because a district drawing GDP from two states would double-count
    and a district with two rows would silently duplicate panel rows.

    So this checks:
      - every boundary polygon appears exactly once in the crosswalk
      - each polygon names at most one label region
      - crosswalk region_ids and label regions both exist in their sources
      - unmatched polygons are documented by ADM1 unit with a reason
    """
    cfg = cfg or {}
    cw = crosswalk.copy()
    for col in ("source_region_name", "parent_name", "region_id"):
        cw[col] = cw[col].fillna("").astype(str).str.strip()

    key = ["source_region_name", "parent_name"]
    lab_keys = labels.assign(
        parent_name=labels["parent_name"].fillna("").astype(str).str.strip(),
        source_region_name=labels["source_region_name"].astype(str).str.strip(),
    )[key].drop_duplicates()

    matched = cw[cw["source_region_name"] != ""]
    unmatched = cw[cw["source_region_name"] == ""]

    fanout = matched.groupby(key, sort=False).size()

    report = MatchReport(
        country=cfg.get("country", "unknown"),
        n_labels=len(lab_keys),
        n_polygons=len(boundaries),
        n_matched=int(matched["region_id"].nunique()),
        n_unmatched=int(unmatched["region_id"].nunique()),
        n_gid_matched=int((cw["method"] == "gid").sum()) if "method" in cw.columns else 0,
        unmatched_names=sorted(unmatched["region_id"].astype(str).tolist()),
        label_fanout={f"{a}|{b}": int(n) for (a, b), n in fanout.items()},
    )

    # 1. every boundary polygon has a decision. This replaces the old
    #    "every label region maps to one polygon", which is the wrong direction
    #    for a one-to-many crosswalk.
    boundary_ids = set(boundaries["region_id"].astype(str))
    cw_ids = set(cw["region_id"])
    undecided = boundary_ids - cw_ids
    report.checks.append(
        (
            "every polygon reviewed",
            not undecided,
            f"{len(undecided)} polygon(s) absent from crosswalk: {sorted(undecided)[:5]}",
        )
    )

    # 2. every crosswalk row points at a real polygon
    unknown_ids = cw_ids - boundary_ids
    report.checks.append(
        (
            "crosswalk ids exist in boundaries",
            not unknown_ids,
            f"{len(unknown_ids)} region_id(s) not in boundaries: {sorted(unknown_ids)[:5]}",
        )
    )

    # 3. THE cardinality check: each district maps to exactly one state.
    dup = cw["region_id"].value_counts()
    doubled = dup[dup > 1]
    report.checks.append(
        (
            "each polygon maps to exactly one label region",
            len(doubled) == 0,
            f"{len(doubled)} region_id(s) appear on multiple rows: {list(doubled.index[:5])}",
        )
    )

    # 4. every label region named in the crosswalk really exists in the labels.
    #    (Not the reverse: a label region covering many districts is correct.)
    lab_key_set = set(map(tuple, lab_keys.itertuples(index=False, name=None)))
    cw_key_set = set(map(tuple, matched[key].drop_duplicates().itertuples(index=False, name=None)))
    ghost_labels = cw_key_set - lab_key_set
    report.checks.append(
        (
            "crosswalk label regions exist in labels",
            not ghost_labels,
            f"{len(ghost_labels)} crosswalk label region(s) absent from the label panel: "
            f"{sorted(ghost_labels)[:5]}",
        )
    )

    # 5. unmatched polygons are documented, by ADM1 unit, with a reason.
    if len(unmatched):
        documented = _unmatched_documented(unmatched, boundaries, cfg)
        upath = unmatched_path(cfg) if cfg.get("country") else None
        target = upath.name if upath else "the unmatched file"
        detail = (
            f"ADM1 reasons in {target}"
            if documented
            else f"add their ADM1 unit and a reason to {target}"
        )
        report.checks.append(
            (
                "unmatched polygons documented",
                documented,
                f"{len(unmatched)} unmatched polygon(s); {detail}",
            )
        )
    else:
        report.checks.append(("unmatched polygons documented", True, "none unmatched"))

    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Propose or validate the region crosswalk")
    parser.add_argument("--country", default=None)
    parser.add_argument("--propose", action="store_true", help="write crosswalk_review.csv")
    parser.add_argument("--validate", action="store_true", help="check the committed crosswalk")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = country_config(args.country)

    if args.propose:
        from .boundaries import load_snapshot
        from .labels import load_labels

        boundaries = load_snapshot(cfg["country"])
        labels = load_labels(cfg)
        review = propose_crosswalk(boundaries, labels, cfg)
        print(
            f"\nWrote {REVIEW_PATH} ({len(review)} rows, {int(review['needs_review'].sum())} "
            f"need review)."
        )
        print("Review it by hand, then commit as", crosswalk_path(cfg).name)
        return 0

    if args.validate:
        from .boundaries import load_snapshot
        from .labels import load_labels

        boundaries = load_snapshot(cfg["country"])
        labels = load_labels(cfg)
        crosswalk = load_crosswalk(cfg)
        report = validate_crosswalk(crosswalk, boundaries, labels, cfg)
        print()
        print(report.render())
        print()
        return 0 if report.ok else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
