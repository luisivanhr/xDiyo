"""Exact played-fixture supplements never become fabricated provider captures."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sqlite3

import pytest
import pyarrow.parquet as pq

from scraping.sofascore.config import Profile, Scope
from scraping.sofascore.export import publish_tables, read_tables
from scraping.sofascore.planner import component_jobs
from scraping.sofascore.schema import TABLES, to_arrow
from scraping.sofascore.store import json_bytes
from scraping.sofascore_web_v3.archive_formats import validate_schedule
from scraping.sofascore_web_v3.campaign import prepare_campaign
from scraping.sofascore_web_v3.publish import hash_file, publish_captured_season
from scraping.sofascore_web_v3.reconcile_publication import _bundle
from scraping.sofascore_web_v3 import source_gaps as sg


def inventory():
    raw = json.loads((Path(__file__).parents[1] / "fixtures/sofascore/greek_2020_21_source_gaps.json").read_text(encoding="utf-8"))
    return raw["events"], raw["spec"], {int(k): v for k, v in raw["standings"].items()}


def test_real_incomplete_inventory_validates_only_with_pinned_evidence(monkeypatch, tmp_path):
    events, spec, standings = inventory()
    before = deepcopy(events)
    assert validate_schedule(events, spec, standings) == []
    assert events == before and len(events) == 179
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(sg, "REVIEW", missing)
    with pytest.raises(FileNotFoundError):
        validate_schedule(events, spec, standings)
    missing.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_schedule(events, spec, standings)


@pytest.mark.parametrize("damage", ["fourth_gap", "extra_event", "round", "side", "team", "season", "competition", "unfinished", "standings", "scope"])
def test_review_does_not_excuse_other_inventory_defects(damage):
    events, spec, standings = inventory()
    if damage == "fourth_gap": events.pop()
    elif damage == "extra_event": events.append(deepcopy(events[0]))
    elif damage == "round": events[0]["roundInfo"]["round"] = 18
    elif damage == "side": events[0]["homeTeam"], events[0]["awayTeam"] = events[0]["awayTeam"], events[0]["homeTeam"]
    elif damage == "team": events[0]["homeTeam"]["id"] = 99999
    elif damage == "season": events[0]["season"]["id"] += 1
    elif damage == "competition": events[0]["tournament"]["uniqueTournament"]["id"] += 1
    elif damage == "unfinished": events[0]["status"]["type"] = "postponed"
    elif damage == "standings": standings[next(iter(standings))] = 25
    elif damage == "scope": spec["season"] = "2019/2020"
    assert validate_schedule(events, spec, standings)


def test_source_rows_preserve_native_ids_but_not_invented_event_ids():
    metadata = sg.review_metadata()
    tables = sg.supplemental_tables(metadata)
    assert len(tables["matches"]) == 3 and len(tables["statistics"]) == 174
    assert len({r["event_id"] for r in tables["matches"]}) == 3
    for match in tables["matches"]:
        assert match["event_id"] >= 10**15 and match["event_id"] < 2**53
        assert match["competition_id"] == 185 and match["season_id"] == 32553
        assert match["custom_id"] is match["tournament_id"] is match["is_awarded"] is None
    assert not {"shots", "heatmap_points", "lineup_players"} & tables.keys()
    assert {r["period"] for r in tables["statistics"]} == {"ALL", "1ST", "2ND"}
    assert all("flashscore" in r["source_key"] and r["raw_hash"] for r in tables["statistics"])
    assert not any(r["key"] in {"expectedGoals", "redCards"} for r in tables["statistics"])
    pan = sg.flashscore_event_id("z99UJF7d")
    assert not any(r["key"] in {"yellowCards", "goalkeeperSaves"} for r in tables["statistics"] if r["event_id"] == pan)
    sg.verify_supplemental_tables({k: to_arrow(k, v).to_pylist() for k, v in tables.items()}, metadata)


@pytest.mark.parametrize("damage", ["source_id", "url", "home", "score", "kickoff", "timezone", "duplicate", "percentage", "unknown_metric", "half_total", "period_score"])
def test_source_identity_and_statistics_are_strict(damage):
    o = deepcopy(sg.review_metadata()["review"]["observations"][-1])
    if damage == "source_id": o["source_match_id"] = "fakeFAKE"
    elif damage == "url": o["url"] = o["url"].replace("dIAQKehj", "other123")
    elif damage == "home": o["home_name"] = "AEK"
    elif damage == "score": o["home_score"] = 3
    elif damage == "kickoff": o["kickoff_utc"] = "2021-01-25T17:30:00Z"
    elif damage == "timezone": o["display_timezone"] = "UTC"
    elif damage == "duplicate": o["periods"]["ALL"].append("8Remates totales9")
    elif damage == "percentage": o["periods"]["ALL"][0] = "55%Posesión46%"
    elif damage == "unknown_metric": o["periods"]["ALL"].append("1New invented metric2")
    elif damage == "half_total": o["periods"]["1ST"][-1] = "4Paradas2"
    elif damage == "period_score": o["period_scores"]["1ST"] = [2, 1]
    with pytest.raises((KeyError, ValueError)):
        sg.observation_tables(o)


def test_complete_publication_preserves_179_captures_and_three_external_rows(tmp_path):
    events, _, _ = inventory()
    profile = Profile(include_pregame=False, include_statistics=True, include_shotmap=False,
        include_heatmaps=False, include_lineups=False, include_injuries=False, duration_mode="off")
    metadata = sg.review_metadata()
    cohort = dict(stem=sg.STEM, competition_id=185, season_id=32553, selected_matches=179,
        expected_dataset_matches=182, snapshot=False, schedule_verified=True, reviewed_source_gaps=metadata)
    references = {e["id"]: dict(event=e, observed_at=1700000000, season_stem=sg.STEM) for e in events}
    result = prepare_campaign(list(references), references=references, base=tmp_path / "campaigns",
        profile=profile, source_metadata=dict(collection="archive_discovery", seasons=[cohort]))
    directory = Path(result["directory"])
    original_rows = []
    with sqlite3.connect(directory / "queue.sqlite") as db:
        for event in events:
            eid = event["id"]
            match = dict.fromkeys(TABLES["matches"])
            match.update(event_id=eid, competition_id=185, season_id=32553, season_year="20/21",
                status="finished", status_code=100, is_awarded=False, round=event["roundInfo"]["round"],
                home_id=event["homeTeam"]["id"], away_id=event["awayTeam"]["id"], kickoff_utc=float(event["startTimestamp"]))
            original_rows.append(match)
            root = directory / "imports" / str(eid)
            canonical = publish_tables({"matches": [match], "statistics": []}, root,
                scope=Scope(185, 32553, (match["round"],)), profile=profile, provenance="offline test fixture")
            expected_paths = {j.path for j in component_jobs(event, profile, 1700000000)} | {f"/event/{eid}"}
            report = root / "report.json"
            report.write_bytes(json_bytes(dict(complete=True, manifest=str(canonical),
                coverage=[dict(path=p, status="empty") for p in sorted(expected_paths)])))
            (root / "feature_coverage.json").write_bytes(json_bytes(dict(features={})))
            db.execute("UPDATE queue SET state='complete',report=? WHERE event_id=?", (str(report), eid))
    queue_hash = hash_file(directory / "queue.sqlite")
    output = tmp_path / "published"
    report = publish_captured_season(directory, sg.STEM, output=output)
    assert report["matches"] == 182 and report["reconciliation"]["verified_capture_imports"] == 179
    assert report["reconciliation"]["missing_sofascore_fixtures"] == 3
    assert report["reconciliation"]["source_capture_complete"] is False
    saved = read_tables(output / report["manifest"])
    assert saved["matches"][:179] == original_rows
    assert len(saved["statistics"]) == 174
    nested = pq.ParquetFile(output / report["season_file"]).read().to_pylist()
    assert len(nested) == 182 and all(len(r["statistics"]) in {54, 66} for r in nested[-3:])
    expected = {r["event_id"]: tuple(r[k] for k in ("event_id", "competition_id", "season_id", "home_id", "away_id")) for r in original_rows}
    assert _bundle(output, sg.STEM, expected)["matches"] == 182
    assert publish_captured_season(directory, sg.STEM, output=output)["publication_state"] == "already_published"
    assert hash_file(directory / "queue.sqlite") == queue_hash
    damaged = deepcopy(saved)
    damaged["statistics"][0]["value"] += 1
    with pytest.raises(ValueError, match="pinned observations"):
        sg.verify_supplemental_tables(damaged, metadata)
    with pytest.raises(ValueError, match="native queue"):
        _bundle(output, sg.STEM, dict(list(expected.items())[1:]))
    # After authorized raw-source cleanup, revalidation still checks all182
    # dataset rows against179 retained native identities and the pinned review.
    (directory / "imports").rename(directory / "retained_test_imports")
    assert publish_captured_season(directory, sg.STEM, output=output)["publication_state"] == "already_published"
