"""Exact workbook-to-native round annotations; no forecast or feature preparation.

Native event IDs occur only in audit lineage. Public annotation rows retain the
workbook identity and the same seven selected fields as the V59 release.
"""
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from .fixture_rounds import (
    FixtureRounds, IDENTITY, SELECTED, NUMERIC, _identity, _ids, _require,
    _typed, file_sha256, load_fixture_rounds,
)
from .catalogue.registry import name_key


RELEASE = "original13_native_20261108"
SEMANTICS = "RETAINED_NATIVE_ROUND_NOT_NEW_ORIGINAL_DRAW_VERIFICATION"
# Workbook ID, country, exact workbook label, native stem, competition, main stage.
LEAGUES = (
    (2, "England", "Premier League", "Premier_League", 17, 1),
    (6, "England", "Championship", "Championship", 18, 2),
    (4, "Germany", "Bundesliga", "Bundesliga", 35, 42),
    (12, "Germany", "2. Bundesliga", "Bundesliga_2", 44, 41),
    (1, "Spain", "LaLiga", "La_Liga", 8, 36),
    (11, "Spain", "LaLiga2", "La_Liga_2", 54, 37),
    (5, "France", "Ligue 1", "Ligue_1", 34, 4),
    (14, "France", "Ligue 2", "Ligue_2", 182, 19),
    (3, "Italy", "Serie A", "Serie_A", 23, 33),
    (13, "Italy", "Serie B", "Serie_B", 53, 34),
    (15, "Netherlands", "Eredivisie", "Eredivisie", 37, 39),
    (20, "Scotland", "Premiership", "Premiership", 36, 54),
    (24, "Belgium", "Jupiler Pro League", "Pro_League", 38, 38),
)
BY_ID = {row[0]: row for row in LEAGUES}
BY_COMP = {row[4]: row for row in LEAGUES}
NATIVE_COLUMNS = (
    "event_id", "competition_id", "season_id", "season", "home_id", "away_id",
    "home_name", "away_name", "kickoff_utc", "native_date", "native_civil_datetime",
    "tournament_id", "round", "stage_json", "publication", "matches_sha256",
)
KEY = ("competition_id", "season", "home_id", "away_id", "native_date")


@dataclass(frozen=True)
class NativeRoundBuild:
    selections: pd.DataFrame
    lineage: pd.DataFrame
    candidates: pd.DataFrame
    counts: dict


def _native(frame):
    _require(set(frame) == set(NATIVE_COLUMNS), "Unexpected native annotation input fields")
    result = frame.loc[:, list(NATIVE_COLUMNS)].copy().reset_index(drop=True)
    for col in ("event_id", "competition_id", "season_id", "home_id", "away_id", "tournament_id"):
        result[col] = _ids(result[col], col)
    _require(result.event_id.is_unique, "Duplicate native event ID; publications must be reconciled")
    _require(bool(result.home_id.ne(result.away_id).all()), "Native self-match")
    _require(bool(result.competition_id.isin(BY_COMP).all()), "Unexpected native competition")
    result["round"] = _typed(result["round"], "season_round")
    for col in ("season", "home_name", "away_name", "native_date", "native_civil_datetime", "publication", "matches_sha256"):
        result[col] = result[col].astype("string")
        _require(bool(result[col].notna().all()), f"Missing native {col}")
    _require(bool(result.season.str.fullmatch(r"\d{4}/\d{4}").all()), "Invalid native season")
    stamp = pd.to_datetime(result.kickoff_utc, unit="s", utc=True, errors="raise")
    _require(bool(stamp.notna().all()), "Missing native kickoff")
    civil = stamp.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    _require(bool(result.native_date.eq(civil.dt.strftime("%Y-%m-%d")).all()), "Native date projection changed")
    _require(bool(pd.to_datetime(result.native_civil_datetime).eq(civil).all()), "Native civil time projection changed")
    result["stage_json"] = result.stage_json.astype("string")
    for value in result.stage_json.dropna():
        _require(isinstance(json.loads(value), dict), "Native stage metadata must be an object")
    return result


def _projection(native):
    """Retain numeric scope; do not turn a reset phase ordinal into a season round."""
    values = {field: pd.NA for field in SELECTED}
    comp, tournament = int(native.competition_id), int(native.tournament_id)
    if pd.isna(native["round"]):
        return values, "unresolved_native_round_missing"
    if tournament == BY_COMP[comp][5]:
        values.update(season_round=int(native["round"]), phase_key="regular_season")
    elif comp == 36 and tournament in (92066, 92067):
        # The retained Scottish split labels continue at 34..38. No subtraction.
        values.update(season_round=int(native["round"]), phase_key=f"native_tournament_{tournament}")
    elif comp == 38 and tournament in (12656, 15545, 92614):
        values.update(phase_round=int(native["round"]), phase_key=f"native_tournament_{tournament}")
    else:
        return values, "unresolved_native_round_scope"
    stage = json.loads(native.stage_json) if pd.notna(native.stage_json) else {}
    # Preserve an actual retained name if present; no invented stage titles.
    if isinstance(stage.get("name"), str) and stage["name"].strip():
        values["stage_label"] = stage["name"]
    return values, "matched_exact_league_season_teams_date"


def build_native_fixture_rounds(workbook, native, directory):
    """Resolve only exact season/ordered-team/date identities, without score matching.

    Different-date candidates, repeated pairs, and reversed sides are evidence,
    never fallback matches. Every target survives, with typed nulls when held.
    """
    identity = _identity(workbook)
    identity = identity.loc[identity.league_id.isin(BY_ID)].reset_index(drop=True)
    _require(len(identity) > 0, "No original-13 workbook targets")
    for lid, country, league, *_ in LEAGUES:
        rows = identity.loc[identity.league_id.eq(lid)]
        _require(bool(rows.country.eq(country).all() and rows.league.eq(league).all()),
                 f"Workbook league identity conflict: {lid}")
    _require(bool(identity.season.str.fullmatch(r"\d{4}/\d{4}").all()), "Invalid workbook season")
    source = _native(native)
    aliases = defaultdict(list)
    for team in directory["teams"]:
        for alias in team["names"]:
            aliases[(team["country"], name_key(alias["name"]))].append(
                (alias["start"], alias["end"], int(team["team_id"])))
    exact, pair, reverse = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in source.to_dict("records"):
        k = tuple(row[c] for c in KEY)
        exact[k].append(row)
        pair[k[:-1]].append(row)
        reverse[(k[0], k[1], k[3], k[2])].append(row)
    output, audit, candidates = [], [], []
    selected_events = {}
    for row in identity.to_dict("records"):
        comp = BY_ID[int(row["league_id"])][4]
        teams = []
        for side in ("home", "away"):
            ids = {tid for start, end, tid in aliases[(row["country"], name_key(row[side + "_team"]))]
                   if start <= row["kickoff_date"] <= end}
            teams.append(next(iter(ids)) if len(ids) == 1 else None)
        values = {field: pd.NA for field in SELECTED}
        claim = dict(match_id=row["match_id"], selection_status="unresolved_team_identity",
                     selection_semantics=SEMANTICS, competition_id=comp,
                     canonical_home_id=teams[0], canonical_away_id=teams[1],
                     native_event_id=None, native_tournament_id=None, native_round=None,
                     native_stage_json=None, native_date=None, native_civil_datetime=None,
                     native_publication=None, native_matches_sha256=None,
                     civil_clock_equal=None, candidate_count=0, reversed_pair_candidate_count=0,
                     original_draw_round_verified=None, metadata_must_not_be_model_input=True)
        matches = []
        if None not in teams and teams[0] != teams[1]:
            k = (comp, row["season"], *teams, row["kickoff_date"])
            matches = exact[k]
            other = pair[k[:-1]]
            rev = reverse[k[:-1]]
            claim.update(candidate_count=len(other), reversed_pair_candidate_count=len(rev))
            if len(matches) == 1:
                n = pd.Series(matches[0])
                values, status = _projection(n)
                claim.update(selection_status=status, native_event_id=int(n.event_id),
                    native_tournament_id=int(n.tournament_id), native_round=n["round"],
                    native_stage_json=n.stage_json, native_date=n.native_date,
                    native_civil_datetime=n.native_civil_datetime,
                    native_publication=n.publication, native_matches_sha256=n.matches_sha256,
                    civil_clock_equal=pd.Timestamp(row["start_datetime"]) == pd.Timestamp(n.native_civil_datetime))
                selected_events.setdefault(int(n.event_id), []).append(len(output))
            elif len(matches) > 1:
                claim["selection_status"] = "unresolved_multiple_exact_native_fixtures"
            elif len(other) > 1:
                claim["selection_status"] = "unresolved_repeated_pair_other_dates"
            elif other:
                claim["selection_status"] = "unresolved_same_pair_other_date"
            else:
                claim["selection_status"] = "unresolved_no_native_pair_in_season"
            if len(matches) != 1:
                for n, orientation in [(n, "ordered") for n in other] + [(n, "reversed") for n in rev]:
                    candidates.append(dict(match_id=row["match_id"], orientation=orientation,
                        native_event_id=int(n["event_id"]), native_date=n["native_date"],
                        date_difference_days=(pd.Timestamp(n["native_date"])-pd.Timestamp(row["kickoff_date"])).days,
                        native_round=n["round"], native_tournament_id=int(n["tournament_id"]),
                        native_publication=n["publication"], native_matches_sha256=n["matches_sha256"],
                        decision="REVIEW_ONLY_NEVER_AUTOSELECT"))
        elif teams[0] is not None and teams[0] == teams[1]:
            claim["selection_status"] = "unresolved_canonical_self_match"
        output.append(dict(row, **values))
        audit.append(claim)
    # Distinct workbook rows must not acquire the same native fixture annotation.
    for positions in selected_events.values():
        if len(positions) > 1:
            for pos in positions:
                audit[pos]["selection_status"] = "unresolved_multiple_workbook_rows_for_native_fixture"
                output[pos].update({field: pd.NA for field in SELECTED})
    selections = pd.DataFrame(output)
    for field in SELECTED:
        selections[field] = _typed(selections[field], field)
    for col in IDENTITY:
        selections[col] = selections[col].astype(identity[col].dtype)
    lineage = pd.DataFrame(audit)
    for col in ("match_id", "competition_id", "canonical_home_id", "canonical_away_id", "native_event_id", "native_tournament_id", "native_round", "candidate_count", "reversed_pair_candidate_count"):
        lineage[col] = lineage[col].astype("Int64")
    for col in ("civil_clock_equal", "original_draw_round_verified", "metadata_must_not_be_model_input"):
        lineage[col] = lineage[col].astype("boolean")
    for col in set(lineage)-set(lineage.select_dtypes(include=["number", "bool"]).columns):
        if col not in ("civil_clock_equal", "original_draw_round_verified", "metadata_must_not_be_model_input"):
            lineage[col] = lineage[col].astype("string")
    candidate_columns = ("match_id", "orientation", "native_event_id", "native_date", "date_difference_days",
                         "native_round", "native_tournament_id", "native_publication", "native_matches_sha256", "decision")
    candidate_frame = pd.DataFrame(candidates, columns=candidate_columns)
    for col in candidate_columns:
        candidate_frame[col] = candidate_frame[col].astype("Int64" if col in ("match_id", "native_event_id", "date_difference_days", "native_round", "native_tournament_id") else "string")
    accepted = lineage.selection_status.eq("matched_exact_league_season_teams_date")
    window = identity.kickoff_date.ge("2019-07-01") & identity.kickoff_date.lt("2026-07-01")
    counts = dict(targets=len(identity), matched=int(accepted.sum()), unresolved=int((~accepted).sum()),
        statuses={str(k): int(v) for k,v in lineage.selection_status.value_counts().items()},
        experiment_window_targets=int(window.sum()), experiment_window_matched=int((accepted & window).sum()),
        experiment_window_unresolved=int((~accepted & window).sum()),
        civil_clock_differences=int(lineage.civil_clock_equal.eq(False).sum()),
        season_rounds=int(selections.season_round.notna().sum()), phase_rounds=int(selections.phase_round.notna().sum()))
    return NativeRoundBuild(selections, lineage, candidate_frame, counts)


def load_native_fixture_rounds(root):
    """Verify pinned inputs and reproduce the full selection and hold decisions."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    _require(manifest["schema"] == 1 and manifest["release"] == RELEASE, "Unsupported native-round release")
    _require(manifest["selected_columns"] == list(SELECTED), "Native round selection schema changed")
    required = {"fixtures.parquet", "provenance/workbook_targets.parquet", "provenance/native_annotations.parquet",
                "provenance/reviewed_team_directory.json", "provenance/lineage.parquet", "provenance/candidates.parquet"}
    _require(required <= set(manifest["files"]), "Native round release lacks provenance")
    for name, info in manifest["files"].items():
        path = (root/name).resolve()
        _require(path.is_relative_to(root.resolve()), "Native round payload escapes release")
        _require(path.stat().st_size == info["bytes"] and file_sha256(path) == info["sha256"], f"Native round checksum mismatch: {name}")
    built = build_native_fixture_rounds(pd.read_parquet(root/"provenance/workbook_targets.parquet"),
        pd.read_parquet(root/"provenance/native_annotations.parquet"),
        json.loads((root/"provenance/reviewed_team_directory.json").read_text(encoding="utf-8")))
    for name, expected in (("fixtures.parquet", built.selections), ("provenance/lineage.parquet", built.lineage), ("provenance/candidates.parquet", built.candidates)):
        pd.testing.assert_frame_equal(pd.read_parquet(root/name), expected)
    _require(manifest["counts"] == built.counts, "Native round counts changed")
    return FixtureRounds(built.selections, dict(release=RELEASE,
        manifest_sha256=file_sha256(root/"manifest.json"), identity_namespace="workbook_match_id",
        selection_semantics=SEMANTICS, evidence_directory=str((root/"provenance").resolve()),
        counts=built.counts, model_input_approved=False, feature_asof_approved=False))


def combine_fixture_rounds(v59, original13):
    """Combine disjoint annotation populations without coalescing any value."""
    _require(not set(v59.selections.match_id) & set(original13.selections.match_id), "Round layers overlap")
    _require(not v59.selections.league_id.isin(BY_ID).any(), "V59 contains original-13 targets")
    _require(bool(original13.selections.league_id.isin(BY_ID).all()), "Native layer contains other leagues")
    frame = pd.concat([v59.selections, original13.selections], ignore_index=True)
    _identity(frame)
    return FixtureRounds(frame, dict(release="v59_plus_" + RELEASE, identity_namespace="workbook_match_id",
        layers=[v59.provenance, original13.provenance], selection_semantics="MIXED_SOURCE_CONVENTIONS_RETAINED",
        counts=dict(targets=len(frame), v59_targets=len(v59.selections), original13_targets=len(original13.selections)),
        model_input_approved=False, feature_asof_approved=False))


def load_combined_fixture_rounds(v59_root, original13_root):
    return combine_fixture_rounds(load_fixture_rounds(v59_root), load_native_fixture_rounds(original13_root))
