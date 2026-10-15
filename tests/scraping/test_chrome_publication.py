"""Season assembly preserves captured values and refuses incomplete/conflicting data."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import socket
import sqlite3

import pyarrow.parquet as pq
import pytest

from scraping.sofascore.export import read_tables
from scraping.sofascore.schema import to_arrow
from scraping.sofascore_web_v3.campaign import Campaign, prepare_campaign
from scraping.sofascore_web_v3.publish import completed_season_stems, hash_file, publish_captured_season
from utils.data_loading import merge_seasons

FIXTURES = Path(__file__).parents[1] / "fixtures/sofascore"
STEM = "Premier_League_24_25"
FIRST = 12436870


def fixture(name):
    return json.loads((FIXTURES / f"{FIRST}_{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Season assembly must stay offline")
    monkeypatch.setattr(socket, "create_connection", reject)
    monkeypatch.setattr(socket.socket, "connect", reject)


@pytest.fixture
def captured(tmp_path):
    references = {}
    for offset in range(3):
        event = deepcopy(fixture("event")["event"])
        event["id"] = FIRST + offset
        event["roundInfo"]["round"] = offset + 1
        if offset < 2:
            event["isAwarded"] = offset == 0
        references[event["id"]] = {"event": event, "observed_at": 1700000000, "season_stem": STEM}
    result = prepare_campaign(list(references), base=tmp_path / "campaigns", references=references)
    directory = Path(result["directory"])
    reports = []
    with Campaign(directory) as campaign:
        for offset, (eid, reference) in enumerate(references.items()):
            job = campaign.next_match()
            event = reference["event"]
            responses = [
                ("", {"event": event}, 200),
                ("/pregame-form", {"homeTeam": {"position": 1 if offset == 0 else 20},
                                   "awayTeam": {"position": 18 if offset == 0 else None}}, 404 if offset == 2 else 200),
                ("/statistics", {"statistics": []}, 200),
                ("/shotmap", fixture("shotmap"), 200),
                ("/lineups", fixture("lineups"), 200),
                ("/incidents", fixture("incidents"), 200),
                *[(f"/heatmap/{event[side]['id']}", {"playerPoints": [{"x": 1, "y": 2}], "goalkeeperPoints": []}, 200)
                  for side in ("homeTeam", "awayTeam")],
            ]
            for index, (suffix, payload, status) in enumerate(responses):
                campaign.observe(eid, job["lease"], {
                    "startedDateTime": f"2026-09-12T02:{offset:02d}:{index:02d}Z", "time": 10,
                    "request": {"method": "GET", "url": f"https://www.sofascore.com/api/v1/event/{eid}{suffix}"},
                    "response": {"status": status, "content": {"text": json.dumps(payload)} if status == 200 else {}},
                })
            finished = campaign.finish(eid, job["lease"], page_event_id=eid)
            reports.append(Path(finished["report"]))
    return directory, reports


def test_complete_cohort_preserves_every_table_and_loads_legacy_standings(captured, tmp_path):
    directory, reports = captured
    assert completed_season_stems(directory) == [STEM]
    original = {name: hash_file(directory / name) for name in ("config.json", "queue.sqlite")}
    expected = defaultdict(list)
    for report_path in reports:
        for name, rows in read_tables(json.loads(report_path.read_text())["manifest"]).items():
            expected[name].extend(rows)
    output = tmp_path / "published"
    result = publish_captured_season(directory, STEM, output=output)
    assert result["publication_state"] == "published"
    assert result["season_file"] == STEM + ".parquet"
    assert result["matches"] == 3
    assert read_tables(output / result["manifest"]) == dict(expected)
    nested = pq.ParquetFile(output / result["season_file"]).read().to_pylist()
    assert [row["event_id"] for row in nested] == [FIRST, FIRST + 1, FIRST + 2]
    assert [row["is_awarded"] for row in nested] == [True, False, False]
    for row in nested:
        for name in expected:
            if name != "matches":
                assert row[name] == [item for item in expected[name] if item["event_id"] == row["event_id"]]
    loaded, _ = merge_seasons([output / result["season_file"]], include_lineups=False)
    loaded = loaded.set_index("match_id")
    assert loaded.loc[FIRST, "home_standing"] == .05
    assert loaded.loc[FIRST + 1, "home_standing"] == 1
    assert loaded.loc[FIRST, "away_standing"] == 1
    assert loaded["is_awarded"].to_list() == [True, False, False]
    assert loaded["away_standing"].isna().sum() == 2
    assert loaded.loc[FIRST + 2, "home_standing"] != loaded.loc[FIRST + 2, "home_standing"]
    assert publish_captured_season(directory, STEM, output=output)["publication_state"] == "already_published"
    assert original == {name: hash_file(directory / name) for name in original}
    lineage = output / result["reconciliation"]["source_lineage"]
    lineage.write_text("{}")
    with pytest.raises(ValueError, match="source lineage mismatch"):
        publish_captured_season(directory, STEM, output=output)


@pytest.mark.parametrize("damage", [None, "team", "flat", "lineage"])
def test_cleaned_captures_reuse_only_verified_existing_publication(captured, tmp_path, damage):
    import shutil

    directory, _ = captured
    output = tmp_path / "published"
    published = publish_captured_season(directory, STEM, output=output)
    flat = output / published["season_file"]
    before = flat.read_bytes()
    shutil.rmtree(directory / "imports")
    if damage == "team":
        config = json.loads((directory / "config.json").read_text())
        config["items"][0]["reference"]["event"]["homeTeam"]["id"] += 1
        (directory / "config.json").write_text(json.dumps(config))
    elif damage == "flat":
        flat.write_bytes(before + b"corrupt")
    elif damage == "lineage":
        path = output / published["reconciliation"]["source_lineage"]
        lineage = json.loads(path.read_text())
        lineage["sources"][0]["report_sha256"] = "wrong"
        path.write_text(json.dumps(lineage))
    if damage:
        with pytest.raises(ValueError):
            publish_captured_season(directory, STEM, output=output)
    else:
        assert publish_captured_season(directory, STEM, output=output)["publication_state"] == "already_published"
        assert flat.read_bytes() == before
        assert not (directory / "imports").exists()


def test_older_capture_imports_get_awarded_from_saved_metadata(captured, tmp_path):
    directory, reports = captured
    for report_path in reports:
        manifest_path = Path(json.loads(report_path.read_text())["manifest"])
        manifest = json.loads(manifest_path.read_text())
        info = manifest["tables"]["matches"]
        path = manifest_path.parent / info["file"]
        pq.write_table(pq.ParquetFile(path).read().drop(["is_awarded"]), path)
        info.update(sha256=hash_file(path), bytes=path.stat().st_size)
        manifest.pop("match_awarded_contract")
        manifest_path.write_text(json.dumps(manifest))
    queue_hash = hash_file(directory / "queue.sqlite")
    output = tmp_path / "published"
    result = publish_captured_season(directory, STEM, output=output)
    assert [row["is_awarded"] for row in read_tables(output / result["manifest"])["matches"]] == [True, False, False]
    assert hash_file(directory / "queue.sqlite") == queue_hash


@pytest.mark.parametrize("state", ["pending", "running", "deferred"])
def test_unfinished_cohort_is_not_published(captured, tmp_path, state):
    directory, _ = captured
    with sqlite3.connect(directory / "queue.sqlite") as connection:
        connection.execute("UPDATE queue SET state=? WHERE event_id=?", (state, FIRST))
    assert completed_season_stems(directory) == []
    output = tmp_path / "published"
    with pytest.raises(ValueError, match="unfinished"):
        publish_captured_season(directory, STEM, output=output)
    assert not output.exists()


def test_existing_output_is_preserved(captured, tmp_path):
    directory, _ = captured
    output = tmp_path / "published"
    output.mkdir()
    path = output / (STEM + ".parquet")
    path.write_bytes(b"other collector output")
    with pytest.raises(ValueError, match="output pair"):
        publish_captured_season(directory, STEM, output=output)
    assert path.read_bytes() == b"other collector output"
    (output / (STEM + ".manifest.json")).write_text("{}")
    with pytest.raises(ValueError, match="different provenance"):
        publish_captured_season(directory, STEM, output=output)
    assert path.read_bytes() == b"other collector output"


def test_modified_source_table_is_rejected(captured, tmp_path):
    directory, reports = captured
    manifest_path = Path(json.loads(reports[0].read_text())["manifest"])
    manifest = json.loads(manifest_path.read_text())
    path = manifest_path.parent / manifest["tables"]["shots"]["file"]
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="table hash mismatch"):
        publish_captured_season(directory, STEM, output=tmp_path / "published")


def test_changed_team_with_valid_table_hash_is_rejected(captured, tmp_path):
    directory, reports = captured
    manifest_path = Path(json.loads(reports[0].read_text())["manifest"])
    manifest = json.loads(manifest_path.read_text())
    info = manifest["tables"]["matches"]
    path = manifest_path.parent / info["file"]
    rows = read_tables(manifest_path, names={"matches"})["matches"]
    rows[0]["away_id"] += 1
    pq.write_table(to_arrow("matches", rows), path)
    info.update(sha256=hash_file(path), bytes=path.stat().st_size)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fixture identity mismatch"):
        publish_captured_season(directory, STEM, output=tmp_path / "published")
    assert not (tmp_path / "published" / (STEM + ".parquet")).exists()


def test_missing_requested_component_accounting_is_rejected(captured, tmp_path):
    directory, reports = captured
    report = json.loads(reports[0].read_text())
    report["coverage"] = report["coverage"][1:]
    reports[0].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="component accounting"):
        publish_captured_season(directory, STEM, output=tmp_path / "published")


def test_watcher_waits_for_complete_cohort(captured, tmp_path):
    from scraping.sofascore_web_v3.publish_watch import watch_publications
    directory, _ = captured
    output = tmp_path / "published"
    with sqlite3.connect(directory / "queue.sqlite") as connection:
        connection.execute("UPDATE queue SET state='running' WHERE event_id=?", (FIRST,))
    waits = []

    def complete_during_wait(seconds):
        waits.append(seconds)
        assert not (output / (STEM + ".parquet")).exists()
        with sqlite3.connect(directory / "queue.sqlite") as connection:
            connection.execute("UPDATE queue SET state='complete' WHERE event_id=?", (FIRST,))

    result = watch_publications(directory, output=output, watch=True, sleep=complete_during_wait)
    assert result["state"] == "finished"
    assert waits == [60]
    assert result["verified_in_this_run"][STEM]["matches"] == 3
    assert pq.ParquetFile(output / (STEM + ".parquet")).metadata.num_rows == 3


def test_watcher_does_not_repeat_published_cohort_while_waiting(captured, tmp_path):
    from scraping.sofascore_web_v3.publish_watch import watch_publications
    directory, _ = captured
    output = tmp_path / "published"
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text())
    extra = deepcopy(config["items"][0])
    extra["event_id"] = FIRST + 1000
    extra["reference"]["season_stem"] = "Other_League_24_25"
    config["items"].append(extra)
    config_path.write_text(json.dumps(config))
    with sqlite3.connect(directory / "queue.sqlite") as connection:
        connection.execute("INSERT INTO queue(event_id,ordinal,state) VALUES(?,?,?)", (FIRST + 1000, 3, "pending"))
    before = {name: hash_file(directory / name) for name in ("config.json", "queue.sqlite")}
    snapshots = []

    def two_waits(seconds):
        snapshots.append(hash_file(output / (STEM + ".manifest.json")))
        if len(snapshots) == 2:
            raise KeyboardInterrupt

    result = watch_publications(directory, output=output, watch=True, sleep=two_waits)
    assert result["state"] == "stopped"
    assert len(snapshots) == 2 and snapshots[0] == snapshots[1]
    events = [json.loads(line) for line in (output / "_collection/chrome_publication_events.jsonl").read_text().splitlines()]
    assert len([row for row in events if row["state"] == "published"]) == 1
    assert before == {name: hash_file(directory / name) for name in before}


def test_watcher_preserves_conflicting_output_and_queue(captured, tmp_path):
    from scraping.sofascore_web_v3.publish_watch import watch_publications
    directory, _ = captured
    before = {name: hash_file(directory / name) for name in ("config.json", "queue.sqlite")}
    output = tmp_path / "published"
    output.mkdir()
    existing = output / (STEM + ".parquet")
    existing.write_bytes(b"different collector output")
    with pytest.raises(ValueError, match="output pair"):
        watch_publications(directory, output=output)
    assert existing.read_bytes() == b"different collector output"
    assert before == {name: hash_file(directory / name) for name in before}
    status = json.loads((output / "_collection/chrome_publication_watch.json").read_text())
    assert status["state"] == "failed" and status["active_stem"] == STEM


def test_watcher_catches_final_completion_between_queue_reads(captured, tmp_path, monkeypatch):
    from scraping.sofascore_web_v3 import publish_watch
    directory, _ = captured
    output = tmp_path / "published"
    with sqlite3.connect(directory / "queue.sqlite") as connection:
        connection.execute("UPDATE queue SET state='running' WHERE event_id=?", (FIRST,))
    original_scan = publish_watch.completed_season_stems
    scans = []

    def finish_after_first_scan(path):
        result = original_scan(path)
        scans.append(result)
        if len(scans) == 1:
            with sqlite3.connect(directory / "queue.sqlite") as connection:
                connection.execute("UPDATE queue SET state='complete' WHERE event_id=?", (FIRST,))
        return result

    monkeypatch.setattr(publish_watch, "completed_season_stems", finish_after_first_scan)
    result = publish_watch.watch_publications(directory, output=output, watch=True,
                                               sleep=lambda _: pytest.fail("Complete queue should not wait"))
    assert result["state"] == "finished"
    assert result["verified_in_this_run"][STEM]["matches"] == 3
    assert pq.ParquetFile(output / (STEM + ".parquet")).metadata.num_rows == 3
