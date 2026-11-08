"""Exact native annotation matching, conservative holds, and V59 coexistence."""
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from xdiyo_analytics.data import assign_fixture_rounds, load_fixture_rounds, load_native_fixture_rounds
from xdiyo_analytics.data.fixture_rounds import IDENTITY, NUMERIC, SELECTED, FixtureRounds
from xdiyo_analytics.data.native_fixture_rounds import (
    NATIVE_COLUMNS, SEMANTICS, build_native_fixture_rounds, combine_fixture_rounds,
)

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT/"data/fixture_rounds/original13_native_20261108"


def inputs():
    workbook = pd.DataFrame([dict(match_id=14919, league_id=2, country="England", league="Premier League",
        season="2024/2025", home_team_id=67, away_team_id=64, home_team="Manchester Utd", away_team="Fulham",
        start_datetime="2024-08-16T21:00:00", home_goals_ft=1, away_goals_ft=0)])
    native = pd.DataFrame([dict(event_id=12436870, competition_id=17, season_id=61627, season="2024/2025",
        home_id=35, away_id=43, home_name="Manchester United", away_name="Fulham",
        kickoff_utc=pd.Timestamp("2024-08-16T19:00:00Z").timestamp(), native_date="2024-08-16",
        native_civil_datetime="2024-08-16T21:00:00", tournament_id=1, round=1,
        stage_json='{"round":1}', publication="Premier_League_24_25.manifest.json", matches_sha256="a"*64)])
    directory = dict(teams=[dict(team_id=tid, country="England", names=[dict(name=name, start="2000-01-01", end="2030-01-01")])
                           for tid,name in [(35,"Manchester Utd"),(43,"Fulham")]])
    return workbook, native[list(NATIVE_COLUMNS)], directory


def build(workbook=None, native=None, directory=None):
    w,n,d = inputs()
    return build_native_fixture_rounds(w if workbook is None else workbook,
                                     n if native is None else native, d if directory is None else directory)


def test_exact_match_preserves_inputs_and_separates_audit():
    w,n,d = inputs(); before=(w.copy(deep=True), n.copy(deep=True), deepcopy(d))
    result = build_native_fixture_rounds(w,n,d)
    pd.testing.assert_frame_equal(w,before[0]); pd.testing.assert_frame_equal(n,before[1]); assert d==before[2]
    assert result.selections.season_round.tolist()==[1]
    assert set(result.selections)==set(IDENTITY)|set(SELECTED)
    assert result.lineage.native_event_id.tolist()==[12436870]
    assert result.lineage.selection_semantics.eq(SEMANTICS).all()
    assert result.lineage.original_draw_round_verified.isna().all()
    assert result.lineage.civil_clock_equal.all()
    for name in SELECTED:
        assert str(result.selections[name].dtype)==("Int64" if name in NUMERIC else "string")
    assert result.selections.phase_round.isna().all()


@pytest.mark.parametrize("column,value", [("league","Championship"),("country","Scotland")])
def test_league_descriptors_cannot_override_numeric_scope(column,value):
    w,_,_=inputs();w[column]=value
    with pytest.raises(ValueError,match="league identity conflict"):build(workbook=w)


def test_date_difference_is_never_accepted_from_unique_pair_or_scores():
    w,_,_=inputs();w.start_datetime="2024-08-17T21:00:00"
    result=build(workbook=w)
    assert result.counts["matched"]==0
    assert result.lineage.selection_status.tolist()==["unresolved_same_pair_other_date"]
    assert result.selections[list(SELECTED)].isna().all().all()
    assert result.candidates.date_difference_days.tolist()==[-1]
    assert result.candidates.decision.eq("REVIEW_ONLY_NEVER_AUTOSELECT").all()


def test_season_and_orientation_are_mandatory():
    w,n,_=inputs();w.season="2023/2024"
    result=build(workbook=w);assert result.counts["matched"]==0 and result.candidates.empty
    n[["home_id","away_id"]]=n[["away_id","home_id"]].to_numpy()
    result=build(native=n)
    assert result.counts["matched"]==0 and result.candidates.orientation.tolist()==["reversed"]


def test_dated_alias_does_not_extend_its_authority():
    w,n,d=inputs();d["teams"][0]["names"][0]["end"]="2024-08-15"
    assert build_native_fixture_rounds(w,n,d).lineage.selection_status.tolist()==["unresolved_team_identity"]


def test_ambiguous_alias_and_canonical_self_match_hold():
    w,n,d=inputs();d["teams"].append(dict(team_id=44,country="England",names=deepcopy(d["teams"][0]["names"])))
    assert build_native_fixture_rounds(w,n,d).counts["matched"]==0
    d=inputs()[2];d["teams"][1]["team_id"]=35
    assert build_native_fixture_rounds(w,n,d).lineage.selection_status.tolist()==["unresolved_canonical_self_match"]


def test_multiple_exact_native_candidates_and_duplicate_workbook_matches_hold():
    w,n,_=inputs();other=n.copy();other.event_id=12436871;n=pd.concat([n,other],ignore_index=True)
    result=build(native=n)
    assert result.lineage.selection_status.tolist()==["unresolved_multiple_exact_native_fixtures"]
    assert result.selections.season_round.isna().all()
    other=w.copy();other.match_id=14920
    result=build(workbook=pd.concat([w,other],ignore_index=True))
    assert result.lineage.selection_status.eq("unresolved_multiple_workbook_rows_for_native_fixture").all()
    assert result.selections[list(SELECTED)].isna().all().all()


def test_repeated_pair_on_other_dates_is_explicitly_ambiguous():
    w,n,_=inputs();w.start_datetime="2024-08-18T21:00:00"
    extra=n.copy();extra.event_id=12436871;extra.kickoff_utc+=86400
    extra.native_date="2024-08-17";extra.native_civil_datetime="2024-08-17T21:00:00"
    result=build(workbook=w,native=pd.concat([n,extra],ignore_index=True))
    assert result.lineage.selection_status.tolist()==["unresolved_repeated_pair_other_dates"]
    assert result.selections.season_round.isna().all() and len(result.candidates)==2


def test_raw_stage_cannot_restore_superseded_native_round():
    _,n,_=inputs();n["round"]=4;n.stage_json='{"round":0}'
    result=build(native=n)
    assert result.selections.season_round.tolist()==[4]
    assert result.lineage.native_stage_json.tolist()==['{"round":0}']


def test_same_day_clock_difference_is_retained_without_rewriting_workbook():
    w,_,_=inputs();w.start_datetime="2024-08-16T20:30:00"
    result=build(workbook=w)
    assert result.counts["matched"]==1 and result.counts["civil_clock_differences"]==1
    assert result.selections.start_datetime.tolist()==["2024-08-16T20:30:00"]


def test_belgian_reset_scope_uses_phase_round_and_unknown_scope_holds():
    w,n,d=inputs();w.country="Belgium";w.league="Jupiler Pro League";w.league_id=24
    for t in d["teams"]:t["country"]="Belgium"
    n.competition_id=38;n.tournament_id=12656;n["round"]=6
    result=build_native_fixture_rounds(w,n,d)
    assert result.selections.season_round.isna().all()
    assert result.selections.phase_round.tolist()==[6]
    assert result.selections.phase_key.tolist()==["native_tournament_12656"]
    n.tournament_id=999999
    assert build_native_fixture_rounds(w,n,d).lineage.selection_status.tolist()==["unresolved_native_round_scope"]


@pytest.mark.parametrize("change", ["float_id","zero_round","false_date","duplicate_event"])
def test_invalid_native_input_fails_closed(change):
    _,n,_=inputs()
    if change=="float_id":n.event_id=n.event_id.astype(float)
    elif change=="zero_round":n["round"]=0
    elif change=="false_date":n.native_date="2024-08-15"
    else:n=pd.concat([n,n],ignore_index=True)
    with pytest.raises(ValueError):build(native=n)


def test_unresolved_nulls_clear_old_annotations_and_preserve_original_data():
    w,_,_=inputs();w.start_datetime="2024-08-17T21:00:00"
    result=build(workbook=w)
    w["season_round"]=999;w["phase_key"]="old"
    enriched=assign_fixture_rounds(w,FixtureRounds(result.selections,{"selection_semantics":SEMANTICS}))
    assert enriched[list(SELECTED)].isna().all().all()
    pd.testing.assert_frame_equal(enriched[list(inputs()[0])],w[list(inputs()[0])])


@pytest.fixture(scope="module")
def published():
    return load_native_fixture_rounds(RELEASE)


def test_published_population_and_provenance_are_reproduced(published):
    assert len(published.selections)==52022 and published.selections.match_id.is_unique
    assert published.provenance["counts"]["targets"]==52022
    assert published.provenance["model_input_approved"] is False
    assert published.provenance["feature_asof_approved"] is False
    lineage=pd.read_parquet(RELEASE/"provenance/lineage.parquet")
    held=~lineage.selection_status.eq("matched_exact_league_season_teams_date")
    assert published.selections.loc[held,list(SELECTED)].isna().all().all()
    assert not lineage.loc[~held,"native_event_id"].duplicated().any()


def test_combination_preserves_every_v59_annotation_and_is_disjoint(published):
    v59=load_fixture_rounds(ROOT/"data/fixture_rounds/v59_20261108")
    combined=combine_fixture_rounds(v59,published)
    assert len(combined.selections)==113510
    pd.testing.assert_frame_equal(combined.selections.iloc[:61488].reset_index(drop=True),v59.selections)
    assert combined.provenance["layers"]==[v59.provenance,published.provenance]
    with pytest.raises(ValueError,match="overlap"):combine_fixture_rounds(published,published)


def test_tampered_release_rejected(tmp_path):
    (tmp_path/"manifest.json").write_text(json.dumps(dict(schema=1,release="original13_native_20261108",
        selected_columns=list(SELECTED),files={})))
    with pytest.raises(ValueError,match="lacks provenance"):load_native_fixture_rounds(tmp_path)
