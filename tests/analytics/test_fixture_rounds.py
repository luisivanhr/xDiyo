"""V59 selection, identity and provenance regression tests; no feature preparation."""
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from xdiyo_analytics.data import assign_fixture_rounds, load_fixture_rounds, load_round_enriched_matches
from xdiyo_analytics.data.fixture_rounds import (
    COMPACT, LEDGER, FALLBACKS, IDENTITY, NUMERIC, SELECTED, FixtureRounds, project_selections,
)

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "data/fixture_rounds/v59_20261108"


@pytest.fixture(scope="module")
def mapping():
    return load_fixture_rounds(RELEASE)


@pytest.fixture(scope="module")
def audit():
    return (pd.read_parquet(RELEASE / "provenance" / COMPACT),
            pd.read_parquet(RELEASE / "provenance" / LEDGER))


def target(mapping, ids=None):
    frame = mapping.selections
    if ids is not None:
        frame = frame.loc[frame.match_id.isin(ids)]
    return frame[list(IDENTITY)].copy().reset_index(drop=True)


def test_full_population_and_typed_roundtrip(mapping, audit, tmp_path):
    compact, ledger = audit
    merged = assign_fixture_rounds(target(mapping), mapping)
    assert len(merged) == 61488 and merged.match_id.is_unique
    pd.testing.assert_frame_equal(merged, mapping.selections)
    path = tmp_path / "rounds.parquet"
    merged.to_parquet(path, index=False)
    restored = pd.read_parquet(path)
    pd.testing.assert_frame_equal(merged, restored)
    assert restored.attrs == merged.attrs
    for field in SELECTED:
        assert str(restored[field].dtype) == ("Int64" if field in NUMERIC else "string")
    assert int(restored.season_round.isna().sum()) == 13168
    named = restored.season_round.isna() & restored.phase_round.isna()
    assert named.sum() == 745
    assert (restored.loc[named, "stage_label"].notna() | restored.loc[named, "knockout_label"].notna()).all()
    assert restored.series_game_number.notna().sum() == 60
    assert len(ledger) == 368928


def test_four_user_fallbacks_are_not_original_verified(mapping, audit):
    compact, ledger = audit
    values = mapping.selections.set_index("match_id")
    for mid, (value, event_id) in FALLBACKS.items():
        assert values.loc[mid, "season_round"] == value
        row = compact.set_index("match_id").loc[mid]
        assert row.v58_sofascore_event_id == event_id
        assert str(event_id) in row.v58_sofascore_event_url
        assert row.v59_original_draw_round_verified == False
        assert pd.isna(row.v53_selected_season_round)
        line = ledger.loc[ledger.match_id.eq(mid) & ledger.field.eq("season_round")].iloc[0]
        assert "NOT_VERIFIED_ORIGINAL_DRAW_ROUND" in line.v59_decision_semantics
        assert "HELD_AUSTRALIA_ORIGINAL_DRAW_CONFLICT_V46" in line.base_field_conflict_gate_status
        assert pd.notna(line.v54_source_specific_claims_json)
    assert compact.loc[~compact.match_id.isin(FALLBACKS), "v59_original_draw_round_verified"].isna().all()


def test_no_coalescing_and_explicit_clears(mapping, audit):
    compact, ledger = audit
    source = target(mapping)
    for field in SELECTED:
        source[field] = 999 if field in NUMERIC else "superseded"
    result = assign_fixture_rounds(source, mapping).set_index("match_id")
    pd.testing.assert_frame_equal(result[list(SELECTED)], mapping.selections.set_index("match_id")[list(SELECTED)])
    clears = ledger.loc[ledger.v47_selected_explicit_clear]
    assert len(clears) == 3
    for row in clears.itertuples():
        assert pd.isna(result.loc[row.match_id, row.field])
    # Even a populated earlier selection must not restore a current null.
    changed = compact.copy()
    nulls = changed.v59_selected_season_round.isna()
    changed.loc[nulls, "v53_selected_season_round"] = 999
    projected = project_selections(changed, target(mapping))
    assert projected.season_round.isna().sum() == nulls.sum()


def test_preserves_order_index_results_kickoff_and_out_of_scope(mapping):
    frame = target(mapping, list(FALLBACKS)).iloc[::-1].reset_index(drop=True)
    outside = frame.iloc[[0]].copy()
    outside["match_id"] = 999999999
    frame = pd.concat([frame, outside], ignore_index=True)
    frame["home_goals_ft"] = ["1", "0", "3", "2", "4"]
    frame["season_round"] = pd.Series([99] * len(frame), dtype="Int64")
    frame.index = pd.Index([8, 6, 6, 3, 1], name="original")
    before = frame.copy(deep=True)
    result = assign_fixture_rounds(frame, mapping, require_all_targets=False)
    assert result.index.equals(frame.index)
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(result[list(IDENTITY) + ["home_goals_ft"]], before[list(IDENTITY) + ["home_goals_ft"]])
    assert result.iloc[-1].season_round == 99
    assert result.attrs["fixture_rounds"]["assigned_rows"] == 4


@pytest.mark.parametrize("column", ["country", "league", "season", "home_team", "away_team", "league_id", "home_team_id", "away_team_id", "start_datetime"])
def test_identity_mismatches_fail_closed(mapping, column):
    frame = target(mapping, [73239])
    if column.endswith("_id"):
        frame.loc[0, column] += 1
    elif column == "start_datetime":
        frame.loc[0, column] = frame.loc[0, column] + ".000001"
    else:
        frame.loc[0, column] = "wrong identity"
    with pytest.raises(ValueError, match="identity mismatch"):
        assign_fixture_rounds(frame, mapping, require_all_targets=False)


def test_date_contradiction_and_reversed_home_away_fail(mapping):
    frame = target(mapping, [73239])
    frame.loc[0, "kickoff_date"] = "2021-01-20"
    with pytest.raises(ValueError, match="kickoff_date disagrees"):
        assign_fixture_rounds(frame, mapping, require_all_targets=False)
    frame = target(mapping, [73239])
    for suffix in ("team", "team_id"):
        frame[["home_" + suffix, "away_" + suffix]] = frame[["away_" + suffix, "home_" + suffix]].to_numpy()
    with pytest.raises(ValueError, match="identity mismatch"):
        assign_fixture_rounds(frame, mapping, require_all_targets=False)


@pytest.mark.parametrize("bad", [73239.0, True, None, "73239.1", "-73239"])
def test_nonexact_ids_rejected(mapping, bad):
    frame = target(mapping, [73239])
    frame["match_id"] = pd.Series([bad], dtype="object")
    with pytest.raises(ValueError, match="integer IDs"):
        assign_fixture_rounds(frame, mapping, require_all_targets=False)


def test_integer_string_ids_match_without_changing_input_dtype(mapping):
    frame = target(mapping, [73239])
    for col in ("match_id", "league_id", "home_team_id", "away_team_id"):
        frame[col] = frame[col].astype("string")
    result = assign_fixture_rounds(frame, mapping, require_all_targets=False)
    assert result.season_round.tolist() == [4]
    assert str(result.match_id.dtype) == "string"


def test_duplicate_missing_and_native_ids_do_not_join(mapping):
    frame = target(mapping, [73239])
    with pytest.raises(ValueError, match="Duplicate match_id"):
        assign_fixture_rounds(pd.concat([frame, frame]), mapping, require_all_targets=False)
    with pytest.raises(ValueError, match="missing audited target"):
        assign_fixture_rounds(frame, mapping)
    with pytest.raises(ValueError, match="Missing fixture identity"):
        assign_fixture_rounds(frame.rename(columns={"match_id": "event_id"}), mapping)
    duplicate = FixtureRounds(pd.concat([mapping.selections, mapping.selections.iloc[[0]]]), {})
    with pytest.raises(ValueError, match="Duplicate match_id"):
        assign_fixture_rounds(frame, duplicate, require_all_targets=False)


def test_only_whitelisted_annotations_no_evidence_or_outcomes(mapping):
    result = assign_fixture_rounds(target(mapping), mapping)
    assert set(result) == set(IDENTITY) | set(SELECTED)
    assert not any(token in col for col in result for token in ("evidence", "review", "goals", "approved", "source_url", "v59_"))
    polluted = FixtureRounds(mapping.selections.assign(evidence="result-bearing text"), {})
    with pytest.raises(ValueError, match="Nonselected fields"):
        assign_fixture_rounds(target(mapping), polluted)
    assert result.attrs["fixture_rounds"]["model_input_approved"] is False
    assert result.attrs["fixture_rounds"]["feature_asof_approved"] is False


def test_complete_claims_preserved_and_historical_gates_unchanged(audit):
    compact, ledger = audit
    manifest = json.loads((RELEASE / "manifest.json").read_text())
    source = RELEASE / "provenance/all_source_candidates_NOT_FOR_INTEGRATION.csv.gz"
    with gzip.open(source, "rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == manifest["original_candidate_claims_sha256"]
    assert manifest["base_conflict_gates_equal_v58"] is True
    assert len(ledger.loc[ledger.field.eq("season_round") & ledger.base_field_conflict_gate_status.str.contains("HELD_AUSTRALIA_ORIGINAL_DRAW_IDENTITY_UNRESOLVED_V56", regex=False)]) == 26
    assert not compact.integration_approved.any()
    assert not compact.feature_asof_approved.any()
    assert not ledger.integration_approved.any()
    assert not ledger.model_input_approved.any()
    assert compact.original_794_hold.sum() == 794


def test_payload_tampering_is_rejected(tmp_path):
    manifest = json.loads((RELEASE / "manifest.json").read_text())
    # Stop at the first bad payload: no need to duplicate the entire release.
    first = next(iter(manifest["files"]))
    path = tmp_path / first
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not the audited artifact")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_fixture_rounds(tmp_path)


def test_loader_and_empty_slice_are_compatible(mapping, tmp_path):
    raw = target(mapping, [73239])
    path = tmp_path / "raw.parquet"
    raw.to_parquet(path, index=False)
    result = load_round_enriched_matches(path, RELEASE, require_all_targets=False)
    assert result.season_round.tolist() == [4]
    assert "fixture_rounds" in result.attrs
    empty = assign_fixture_rounds(raw.iloc[:0], mapping, require_all_targets=False)
    assert len(empty) == 0 and str(empty.season_round.dtype) == "Int64"
