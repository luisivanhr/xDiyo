"""Awarded-result projection and offline, source-preserving season revisions."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scraping.sofascore import awarded_backfill
from scraping.sofascore.awarded_backfill import backfill_season
from scraping.sofascore.campaign import publish_season
from scraping.sofascore.config import Profile, Scope
from scraping.sofascore.export import normalize_store, read_tables
from scraping.sofascore.legacy import legacy_view, normalize_legacy
from scraping.sofascore.models import Job
from scraping.sofascore.parsers import awarded_flag, parse_match
from scraping.sofascore.schema import MATCH_AWARDED_CONTRACT, arrow_schema
from scraping.sofascore.store import Store, json_bytes
from scraping.sofascore_web_v3.awarded import SavedAwardedFlags
from scraping.sofascore_web_v3.publish import hash_file

STEM = "Bundesliga_24_25"


def event(eid=12499586):
    # An awarded match can still have ordinary finished/100 status.
    return {"id": eid, "isAwarded": True, "status": {"type": "finished", "code": 100},
            "tournament": {"uniqueTournament": {"id": 35}}, "season": {"id": 63516},
            "homeTeam": {"id": 42}, "awayTeam": {"id": 44}, "roundInfo": {"round": 14},
            "homeScore": {"display": 1}, "awayScore": {"display": 1}}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Awarded backfill must not send network requests")
    monkeypatch.setattr(socket, "create_connection", reject)
    monkeypatch.setattr(socket.socket, "connect", reject)


@pytest.mark.parametrize("value,expected", [(True, True), (False, False), (None, None),
                                           (1, None), (0, None), ("true", None), ("false", None)])
def test_explicit_flag_is_not_coerced_or_inferred_from_status(value, expected):
    raw = {**event(), "isAwarded": value}
    row = parse_match(raw)
    assert row["is_awarded"] is expected
    assert row["status"] == "finished" and row["status_code"] == 100
    field = arrow_schema("matches").field("is_awarded")
    assert field.type == pa.bool_() and field.nullable
    assert field.metadata[b"source_field"] == b"event.isAwarded"


def test_omitted_optional_flag_vs_absent_metadata():
    raw = event()
    del raw["isAwarded"]
    assert awarded_flag(raw) is False
    assert awarded_flag(None) is None
    assert awarded_flag({}) is None
    assert awarded_flag({"id": True}) is None
    raw["status"] = {"type": "finished", "code": 100, "description": "Awarded"}
    assert parse_match(raw)["is_awarded"] is False  # Dedicated source flag, not label inference.


def test_shared_store_and_publication_keep_flag_in_both_layouts_and_loader(tmp_path):
    raw = event()
    scope = Scope(35, 63516, (14,))
    profile = Profile(include_pregame=False, include_statistics=False, include_shotmap=False,
                      include_heatmaps=False, include_lineups=False, include_injuries=False, duration_mode="off")
    with Store(tmp_path / "cache") as store:
        job = Job(f"/event/{raw['id']}", "event", raw["id"])
        store.record(job, "success", 1700000000, payload={"event": raw})
        store.add_events([raw], scope, job, 1700000000)
        tables = normalize_store(store, scope, profile)
    output = tmp_path / "output"
    report = publish_season(tables, output, scope, profile, STEM, {})
    restored = read_tables(output / report["manifest"])
    assert restored["matches"][0]["is_awarded"] is True
    assert pq.ParquetFile(output / report["season_file"]).read(columns=["is_awarded"]).to_pylist() == [{"is_awarded": True}]
    assert legacy_view(restored)["is_awarded"].to_list() == [True]
    assert json.loads((output / report["manifest"]).read_text())["match_awarded_contract"] == MATCH_AWARDED_CONTRACT


@pytest.mark.parametrize("stored,expected", [("", None), ("True", True), ("False", False)])
def test_legacy_reconstruction_only_uses_an_explicit_stored_flag(tmp_path, stored, expected):
    path = tmp_path / "legacy.csv"
    path.write_text("match_id,season_id,season_year,round,home_id,away_id,is_awarded\n"
                    f"1,63516,24/25,14,42,44,{stored}\n")
    tables, _ = normalize_legacy(path, Scope(35, 63516, (14,)), include_lineups=False)
    assert tables["matches"][0]["is_awarded"] is expected


def add_observation(path, raw, *, observed=1700000000, status=200, payload=None, suffix=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"event": raw} if payload is None else payload
    entry = {"startedDateTime": datetime.fromtimestamp(observed, timezone.utc).isoformat(), "time": 0,
             "request": {"method": "GET", "url": f"https://www.sofascore.com/api/v1/event/{raw['id']}{suffix}"},
             "response": {"status": status, "content": {"text": json.dumps(body)}},
             "_xdiyoSelected": True, "_queryPresent": False}
    encoded = json_bytes(entry)
    digest = hashlib.sha256(encoded).hexdigest()
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS observations(digest TEXT PRIMARY KEY,event_id INTEGER,"
                           "path TEXT,observed REAL,entry_json TEXT)")
        connection.execute("INSERT INTO observations VALUES (?,?,?,?,?)",
                           (digest, raw["id"], f"/event/{raw['id']}{suffix}", observed, encoded.decode()))


def test_sources_latest_success_across_campaigns_and_missing_is_unknown(tmp_path):
    first, second = tmp_path / "first.sqlite", tmp_path / "second.sqlite"
    add_observation(first, {**event(), "isAwarded": False})
    add_observation(second, event(), observed=1700000010)
    add_observation(second, event(), observed=1700000020, status=403)
    add_observation(second, event(), observed=1700000030, payload={"error": "missing"})
    before = {p: p.read_bytes() for p in (first, second)}
    with SavedAwardedFlags([first, second]) as flags:
        evidence = flags.get(parse_match(event()))
        assert evidence.value is True and evidence.observed_at == 1700000010
        assert evidence.source == str(second.resolve())
        assert flags.get(parse_match(event(2))).value is None
    assert {p: p.read_bytes() for p in before} == before


def test_later_null_flag_is_unknown_not_an_older_boolean(tmp_path):
    db = tmp_path / "queue.sqlite"
    add_observation(db, event())
    add_observation(db, {**event(), "isAwarded": None}, observed=1700000010)
    with SavedAwardedFlags([db]) as flags:
        evidence = flags.get(parse_match(event()))
        assert evidence.value is None and evidence.observed_at == 1700000010


@pytest.mark.parametrize("same_database", [True, False])
def test_conflicting_equally_recent_flags_are_rejected(tmp_path, same_database):
    first = tmp_path / "first.sqlite"
    second = first if same_database else tmp_path / "second.sqlite"
    add_observation(first, event())
    add_observation(second, {**event(), "isAwarded": False})
    with SavedAwardedFlags([first, second]) as flags:
        with pytest.raises(ValueError, match="equally recent"):
            flags.get(parse_match(event()))


@pytest.mark.parametrize("defect", ["identity", "hash", "time"])
def test_corrupt_or_wrong_fixture_metadata_rejected(tmp_path, defect):
    db = tmp_path / "queue.sqlite"
    raw = event()
    if defect == "identity":
        raw["homeTeam"]["id"] += 1
    add_observation(db, raw)
    if defect != "identity":
        with sqlite3.connect(db) as connection:
            connection.execute("UPDATE observations SET " + ("digest='wrong'" if defect == "hash" else "observed=observed+1"))
    with SavedAwardedFlags([db]) as flags:
        with pytest.raises(ValueError, match="mismatch"):
            flags.get(parse_match(event()))


def old_publication(root):
    rows = [parse_match(event()), parse_match(event(2))]
    # Include a legitimate zero and earlier statistics backfill provenance.
    tables = {"matches": rows, "coverage": [],
              "statistics": [{"event_id": rows[0]["event_id"], "period": "ALL", "side": "home", "key": "cornerKicks",
                              "value": 0, "source_key": "https://www.besoccer.com/match/saved"}],
              "heatmap_points": [{"event_id": 2, "kind": "player", "team_id": 42, "point_order": 0, "x": 1, "y": 2}]}
    report = publish_season(tables, root, Scope(35, 63516, (14,)), Profile(), STEM,
                            {"collection": "chrome_campaign_with_statistics_backfill"})
    canonical = root / report["manifest"]
    manifest = json.loads(canonical.read_text())
    del manifest["match_awarded_contract"]
    match_path = canonical.parent / "matches.parquet"
    pq.write_table(pq.ParquetFile(match_path).read().drop(["is_awarded"]), match_path)
    manifest["tables"]["matches"].update(sha256=hash_file(match_path), bytes=match_path.stat().st_size)
    canonical.write_bytes(json_bytes(manifest))
    season_path = root / report["season_file"]
    pq.write_table(pq.ParquetFile(season_path).read().drop(["is_awarded"]), season_path, use_compliant_nested_type=False)
    report.update(sha256=hash_file(season_path), bytes=season_path.stat().st_size)
    (root / f"{STEM}.manifest.json").write_bytes(json_bytes(report))
    return report


def sources(tmp_path):
    db = tmp_path / "queue.sqlite"
    add_observation(db, event())
    ordinary = event(2)
    del ordinary["isAwarded"]
    add_observation(db, ordinary)
    return db


def test_backfill_preserves_all_other_data_prior_version_and_is_idempotent(tmp_path):
    output = tmp_path / "output"
    prior = old_publication(output)
    old_manifest = output / prior["manifest"]
    immutable = {p: p.read_bytes() for p in old_manifest.parent.iterdir()}
    before = pq.ParquetFile(output / prior["season_file"]).read()
    db = sources(tmp_path)
    original_db = hash_file(db)
    with SavedAwardedFlags([db]) as flags:
        preview = backfill_season(output, STEM, flags)
        assert preview["state"] == "preview" and preview["awarded"] == preview["ordinary"] == 1
        assert json.loads((output / f"{STEM}.manifest.json").read_text()) == prior
        result = backfill_season(output, STEM, flags, apply=True)
        after = pq.ParquetFile(output / prior["season_file"]).read()
        assert after["is_awarded"].to_pylist() == [True, False]
        assert after.drop(["is_awarded"]).replace_schema_metadata(before.schema.metadata).equals(before, check_metadata=True)
        new_manifest = json.loads(Path(result["manifest"]).read_text())
        old_info = json.loads(old_manifest.read_text())
        assert new_manifest["is_awarded_backfill"]["previous_manifest"] == prior["manifest"]
        assert new_manifest["provenance"] == old_info["provenance"]
        for name, info in new_manifest["tables"].items():
            if name != "matches":
                assert info == old_info["tables"][name]
        before_matches = pq.ParquetFile(old_manifest.parent / "matches.parquet").read()
        after_matches = pq.ParquetFile(Path(result["manifest"]).parent / "matches.parquet").read()
        assert after_matches.drop(["is_awarded"]).equals(before_matches, check_metadata=True)
        assert [r["is_awarded"] for r in read_tables(result["manifest"])["matches"]] == [True, False]
        installed = {p: hash_file(p) for p in output.rglob("*") if p.is_file()}
        assert backfill_season(output, STEM, flags, apply=True)["state"] == "unchanged"
        assert {p: hash_file(p) for p in output.rglob("*") if p.is_file()} == installed
    assert {p: p.read_bytes() for p in immutable} == immutable
    assert hash_file(db) == original_db


def test_missing_source_does_not_publish_a_false_flag(tmp_path):
    output = tmp_path / "output"
    old_publication(output)
    db = sources(tmp_path)
    with sqlite3.connect(db) as connection:
        connection.execute("DELETE FROM observations WHERE event_id=2")
    before = {p: hash_file(p) for p in output.rglob("*") if p.is_file()}
    with SavedAwardedFlags([db]) as flags:
        assert backfill_season(output, STEM, flags)["unknown"] == 1
        with pytest.raises(ValueError, match="usable saved awarded metadata"):
            backfill_season(output, STEM, flags, apply=True)
    assert all(hash_file(path) == digest for path, digest in before.items())
    assert len(list(output.rglob("versions/*/manifest.json"))) == 1


def test_failed_install_restores_public_entry_files(tmp_path, monkeypatch):
    output = tmp_path / "output"
    old_publication(output)
    paths = [output / f"{STEM}.parquet", output / f"{STEM}.manifest.json",
             output / "_tables/competition=35/season=63516/CURRENT.json"]
    before = {p: p.read_bytes() for p in paths}
    original = awarded_backfill.atomic_write_status
    calls = 0

    def fail_once(path, data):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("Simulated interrupted install")
        return original(path, data)

    monkeypatch.setattr(awarded_backfill, "atomic_write_status", fail_once)
    with SavedAwardedFlags([sources(tmp_path)]) as flags:
        with pytest.raises(OSError, match="interrupted install"):
            backfill_season(output, STEM, flags, apply=True)
    assert {p: p.read_bytes() for p in paths} == before
