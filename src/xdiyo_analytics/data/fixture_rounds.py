"""Audited fixture annotations, separate from model features and source evidence.

V59 keys are workbook match IDs, never native/canonical event IDs. Integration
does not certify historical availability or enable any experiment capability.
"""
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path

import pandas as pd


RELEASE = "v59_20261108"
SELECTED = {
    "season_round": "v59_selected_season_round",
    "phase_round": "v44_selected_phase_round",
    "phase_key": "v47_selected_phase_key",
    "stage_label": "v44_selected_stage_label",
    "knockout_label": "v44_selected_knockout_label",
    "leg_number": "v44_selected_leg_number",
    "series_game_number": "v45_series_game_number",
}
NUMERIC = {"season_round", "phase_round", "leg_number", "series_game_number"}
DESCRIPTORS = ("country", "league", "season", "kickoff_date", "home_team", "away_team")
SOURCE_IDS = ("match_id", "league_id", "home_team_id", "away_team_id")
IDENTITY = (*SOURCE_IDS, *DESCRIPTORS, "start_datetime")
FALLBACKS = {73239: (4, 9209891), 73217: (7, 9324717),
             73205: (10, 9351183), 73186: (13, 9382095)}
SEMANTICS = "USER_SELECTED_SOFASCORE_DISPLAYED_ROUND_FALLBACK"
COMPACT = "v59_selected_round_mapping_REVIEW_NOT_APPROVED.parquet"
LEDGER = "v59_field_lineage_REVIEW_ONLY.parquet"


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(left, right):
    return bool((left.eq(right).fillna(False) | (left.isna() & right.isna())).all())


def _ids(values, name):
    # Float IDs (even integral-looking ones) may already have lost precision.
    _require(not pd.api.types.is_float_dtype(values.dtype), f"{name}: floating IDs are forbidden")
    valid = values.map(lambda x: not isinstance(x, bool) and
                       (isinstance(x, str) and x.isascii() and x.isdecimal() or
                        isinstance(x, Integral)))
    _require(bool(valid.all()), f"{name}: exact positive integer IDs required")
    result = values.astype("Int64")
    _require(bool(result.gt(0).all()), f"{name}: exact positive integer IDs required")
    return result


def _identity(frame):
    _require(not frame.columns.duplicated().any(), "Duplicate input columns")
    required = set(IDENTITY) - {"kickoff_date"}
    _require(required <= set(frame), f"Missing fixture identity columns: {sorted(required-set(frame))}")
    result = frame.loc[:, list(required)].copy().reset_index(drop=True)
    for col in SOURCE_IDS:
        result[col] = _ids(result[col], col)
    _require(result.match_id.is_unique, "Duplicate match_id in fixture identities")
    for col in set(DESCRIPTORS) - {"kickoff_date"}:
        result[col] = result[col].astype("string")
        _require(bool(result[col].notna().all() and result[col].str.strip().ne("").all()),
                 f"Missing fixture identity: {col}")
    stamp = pd.to_datetime(result.start_datetime, errors="raise")
    _require(stamp.dt.tz is None, "Workbook start_datetime must remain local civil time")
    _require(bool(stamp.notna().all()), "Missing fixture datetime")
    result["start_datetime"] = stamp.map(lambda value: value.isoformat()).astype("string")
    result["kickoff_date"] = stamp.dt.strftime("%Y-%m-%d").astype("string")
    if "kickoff_date" in frame:
        _require(_same(result.kickoff_date, frame.kickoff_date.reset_index(drop=True).astype("string")),
                 "kickoff_date disagrees with start_datetime")
    return result.loc[:, list(IDENTITY)]


def _typed(values, field):
    if field in NUMERIC:
        result = pd.to_numeric(values.replace("", pd.NA), errors="raise").astype("Int64")
        _require(bool(result.dropna().gt(0).all()), f"Invalid {field}")
        return result
    result = values.astype("string")
    return result.mask(result.str.strip().eq(""), pd.NA)


def project_selections(compact, target_identity):
    """Whitelist current selections; historical candidates can never fill a null."""
    identity = _identity(target_identity)
    _require(compact.match_id.is_unique and compact.match_id.notna().all(), "Duplicate/null compact identity")
    current = compact.set_index("match_id").reindex(identity.match_id).reset_index()
    _require(set(compact.match_id) == set(identity.match_id), "Target/compact population mismatch")
    for col in DESCRIPTORS:
        _require(_same(current[col].astype("string"), identity[col]), f"Target/compact identity mismatch: {col}")
    for field, source in SELECTED.items():
        identity[field] = _typed(current[source], field)
    return identity


def validate_audit(compact, ledger, queue):
    """Verify the complete frozen V59 contract, including typed field lineage."""
    _require(len(compact) == 61488 and compact.match_id.is_unique, "Expected 61,488 unique fixtures")
    _require(len(ledger) == 368928 and not ledger.duplicated(["match_id", "field"]).any(),
             "Expected 368,928 unique fixture/field rows")
    _require(set(ledger.match_id) == set(compact.match_id), "Ledger population mismatch")
    fields = set(SELECTED) - {"series_game_number"}
    _require(set(ledger.field) == fields, "Unexpected ledger fields")
    indexed = compact.set_index("match_id")
    for field in fields:
        part = ledger.loc[ledger.field.eq(field)].set_index("match_id").reindex(indexed.index)
        for col in DESCRIPTORS:
            _require(_same(part[col].astype("string"), indexed[col].astype("string")),
                     f"Ledger identity mismatch: {field}/{col}")
        selected = _typed(indexed[SELECTED[field]], field)
        _require(_same(selected.astype("string"), _typed(part.v59_selected_value, "text")),
                 f"Ledger value mismatch: {field}")
        typed = part.v59_selected_numeric if field in NUMERIC else part.v59_selected_text
        _require(_same(selected, _typed(typed, field)), f"Typed ledger mismatch: {field}")
        other = part.v59_selected_text if field in NUMERIC else part.v59_selected_numeric
        _require(bool(other.isna().all()), f"Spurious ledger value type: {field}")
    held = set(compact.loc[compact.original_794_hold, "match_id"])
    _require(len(queue) == 794 and queue.match_id.is_unique and len(held) == 794 and
             held == set(queue.match_id), "Historical 794 hold population changed")
    for frame, cols in [(compact, ("integration_approved", "feature_asof_approved")),
                        (ledger, ("integration_approved", "model_input_approved"))]:
        for col in cols:
            _require(bool(frame[col].eq(False).all()), f"Historical approval changed: {col}")
    _require(bool(ledger.metadata_must_not_be_model_input.all()), "Provenance/model separation changed")
    selected_fallbacks = compact.v59_selection_semantics.notna()
    _require(set(compact.loc[selected_fallbacks, "match_id"]) == set(FALLBACKS), "Fallback scope changed")
    _require(set(ledger.loc[ledger.v59_selected_user_fallback, "match_id"]) == set(FALLBACKS) and
             int(ledger.v59_selected_user_fallback.sum()) == 4, "Ledger fallback scope changed")
    for mid, (value, event) in FALLBACKS.items():
        row = indexed.loc[mid]
        line = ledger.loc[ledger.match_id.eq(mid) & ledger.field.eq("season_round")].iloc[0]
        _require(row.v59_selected_season_round == value and row.v58_sofascore_event_id == event and
                 row.v58_sofascore_displayed_round == value and row.v59_selection_semantics == SEMANTICS,
                 f"Fallback value/provider semantics changed: {mid}")
        _require(pd.isna(row.v53_selected_season_round) and row.v59_original_draw_round_verified == False and
                 line.v59_original_draw_round_verified == False and SEMANTICS in line.v59_decision_semantics and
                 "NOT_VERIFIED_ORIGINAL_DRAW_ROUND" in line.v59_decision_semantics and
                 "HELD_AUSTRALIA_ORIGINAL_DRAW_CONFLICT_V46" in line.base_field_conflict_gate_status,
                 f"Fallback uncertainty lost: {mid}")
    _require(bool(compact.loc[~selected_fallbacks, "v59_original_draw_round_verified"].isna().all()),
             "Unassessed original verification flags must be null")
    _require(bool(ledger.loc[~ledger.v59_selected_user_fallback, "v59_original_draw_round_verified"].isna().all()),
             "Unassessed ledger verification flags must be null")
    clears = ledger.loc[ledger.v47_selected_explicit_clear]
    _require(len(clears) == 3 and clears.country.eq("Bulgaria").all() and
             clears.v59_selected_value.isna().all(), "Three explicit Bulgarian clears changed")
    _require(int(compact.v45_series_game_number.notna().sum()) == 60, "MLS series ordinals changed")
    _require(int(compact.v59_selected_season_round.isna().sum()) == 13168, "Season null population changed")
    named = compact.v59_selected_season_round.isna() & compact.v44_selected_phase_round.isna()
    _require(int(named.sum()) == 745 and
             bool((_typed(compact.loc[named, "v44_selected_stage_label"], "stage_label").notna() |
                   _typed(compact.loc[named, "v44_selected_knockout_label"], "knockout_label").notna()).all()),
             "Named stages must survive without invented numeric rounds")
    return dict(fixtures=61488, fields=368928, historical_holds=794, explicit_clears=3,
                series_ordinals=60, season_round_nulls=13168, named_without_numeric_round=745,
                user_selected_provider_fallbacks=4)


@dataclass(frozen=True)
class FixtureRounds:
    """Values plus an explicit reference to the separately stored audit evidence."""
    selections: pd.DataFrame
    provenance: dict


def load_fixture_rounds(root):
    """Load a versioned release, verifying every payload and its V59 contract."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    _require(manifest["schema"] == 1 and manifest["release"] == RELEASE and
             manifest["selected_columns"] == SELECTED, "Unsupported fixture-round release")
    required = {"fixtures.parquet", f"provenance/{COMPACT}", f"provenance/{LEDGER}",
                "provenance/v59_held_794_queue_REVIEW.csv"}
    _require(required <= set(manifest["files"]), "Incomplete round provenance")
    for name, info in manifest["files"].items():
        path = (root / name).resolve()
        _require(path.is_relative_to(root.resolve()), "Invalid round payload path")
        _require(file_sha256(path) == info["sha256"], f"Round payload checksum mismatch: {name}")
    compact = pd.read_parquet(root / "provenance" / COMPACT)
    ledger = pd.read_parquet(root / "provenance" / LEDGER)
    queue = pd.read_csv(root / "provenance/v59_held_794_queue_REVIEW.csv")
    counts = validate_audit(compact, ledger, queue)
    selections = pd.read_parquet(root / "fixtures.parquet")
    _require(set(selections) == set(IDENTITY) | set(SELECTED), "Nonselected fields in dataset annotations")
    expected = project_selections(compact, selections)
    pd.testing.assert_frame_equal(selections.reset_index(drop=True), expected, check_dtype=True)
    return FixtureRounds(selections, dict(release=RELEASE, manifest_sha256=file_sha256(root / "manifest.json"),
        identity_namespace="workbook_match_id", counts=counts, evidence_directory=str(root.resolve() / "provenance"),
        selection_semantics="mixed_original_and_provider_conventions", feature_asof_approved=False,
        model_input_approved=False))


def assign_fixture_rounds(matches, mapping, *, require_all_targets=True):
    """Attach only seven selected columns by checked workbook identity.

    Preserve input order, index, IDs, kickoff, results, and out-of-scope values.
    Replace in-scope fields even when the authoritative value is null. No
    season/phase coalescing, stage-number invention, or native-ID matching.
    """
    identity = _identity(matches)
    selected = mapping.selections
    _require(set(selected) == set(IDENTITY) | set(SELECTED), "Nonselected fields in annotation input")
    checked = _identity(selected)
    source = selected.set_index("match_id")
    covered = identity.match_id.isin(source.index)
    if require_all_targets:
        _require(set(source.index) <= set(identity.match_id), "Dataset is missing audited target IDs")
    joined = checked.set_index("match_id").reindex(identity.loc[covered, "match_id"])
    actual = identity.loc[covered].set_index("match_id")
    for col in set(IDENTITY) - {"match_id"}:
        _require(_same(actual[col], joined[col]), f"Fixture identity mismatch: {col}")
    result = matches.copy()
    for field in SELECTED:
        values = (_typed(matches[field], field).reset_index(drop=True) if field in matches else
                  pd.Series(pd.NA, index=range(len(matches)), dtype="Int64" if field in NUMERIC else "string"))
        incoming = _typed(source.loc[identity.loc[covered, "match_id"], field], field)
        values.loc[covered] = incoming.array
        result[field] = pd.Series(values.array, index=result.index)
    result.attrs["fixture_rounds"] = dict(mapping.provenance, assigned_rows=int(covered.sum()))
    return result


def load_round_enriched_matches(matches_path, mapping_root, *, require_all_targets=True):
    """Load raw match data with audited annotations; build no features or experiment."""
    return assign_fixture_rounds(pd.read_parquet(matches_path), load_fixture_rounds(mapping_root),
                                 require_all_targets=require_all_targets)
