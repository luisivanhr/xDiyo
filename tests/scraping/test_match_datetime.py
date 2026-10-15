"""Match chronology is provider event time, separate from collection time."""
from datetime import datetime, timezone
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from scraping.sofascore.config import Profile, Scope
from scraping.sofascore.export import normalize_store, publish_tables, read_tables
from scraping.sofascore.models import Job
from scraping.sofascore.parsers import parse_match
from scraping.sofascore.schema import arrow_schema
from scraping.sofascore.store import Store


def test_observed_historical_provider_start_preserved_with_utc_contract(tmp_path):
    path = Path(__file__).parents[1]/"fixtures"/"sofascore"/"12436870_event.json"
    event = json.loads(path.read_text(encoding="utf-8"))["event"]
    fetched = 1789146705.0
    row = parse_match(event,observed_at=fetched)
    assert row["kickoff_utc"] == 1723834800
    assert datetime.fromtimestamp(row["kickoff_utc"],timezone.utc).isoformat() == "2024-08-16T19:00:00+00:00"
    assert row["event_metadata_observed_at"] == fetched
    assert row["kickoff_utc"] != row["event_metadata_observed_at"]
    manifest = publish_tables({"matches":[row]},tmp_path,scope=Scope(17,61627,(1,)))
    restored = read_tables(manifest)["matches"][0]
    assert restored["kickoff_utc"] == row["kickoff_utc"]
    assert restored["event_metadata_observed_at"] == fetched
    metadata = pq.ParquetFile(manifest.parent/"matches.parquet").schema_arrow.field("kickoff_utc").metadata
    assert metadata[b"timezone"] == b"UTC"
    assert metadata[b"unit"] == b"seconds_since_unix_epoch"
    assert b"actual kickoff unverified" in metadata[b"meaning"]
    assert json.loads(manifest.read_text())["match_time_contract"]["kickoff_utc"]["source_field"] == "event.startTimestamp"


@pytest.mark.parametrize("value",[None,float("nan"),float("inf"),True,"not a timestamp"])
def test_unknown_date_never_filled_from_other_times(value):
    event = {"id":1,"startTimestamp":value,"status":{"type":"finished"},
             "currentPeriodStartTimestamp":1700000000,"time":{"currentPeriodStartTimestamp":1700000000},
             "changes":{"changeTimestamp":1700007200},"season":{"year":"2024/2025"},"roundInfo":{"round":1}}
    row = parse_match(event,observed_at=1800000000)
    assert row["kickoff_utc"] is None
    assert row["event_metadata_observed_at"] == 1800000000
    assert parse_match(event)["event_metadata_observed_at"] is None
    assert arrow_schema("matches").field("kickoff_utc").nullable


def test_rescheduling_updates_latest_time_but_preserves_observed_versions(tmp_path):
    scope = Scope(17,500,(1,))
    event = {"id":1,"startTimestamp":1700000000,"status":{"type":"notstarted"},
             "season":{"id":500},"tournament":{"uniqueTournament":{"id":17}}}
    job = Job("/event/1","event",1)
    profile = Profile(include_pregame=False,include_statistics=False,include_shotmap=False,
                       include_heatmaps=False,include_lineups=False,include_injuries=False,duration_mode="off")
    with Store(tmp_path/"cache") as store:
        store.record(job,"success",1699900000,payload={"event":event})
        store.add_events([event],scope,job,1699900000)
        revised = {**event,"startTimestamp":1700172800}
        store.record(job,"success",1699910000,payload={"event":revised})
        store.add_events([revised],scope,job,1699910000)
        # Replaying an older cached discovery response must not undo a revision.
        store.add_events([event],scope,job,1699900000)
        rows = normalize_store(store,scope,profile,now=1800000000)["matches"]
        assert len(rows) == 1 and rows[0]["event_id"] == 1
        assert rows[0]["kickoff_utc"] == revised["startTimestamp"]
        assert rows[0]["event_metadata_observed_at"] == 1699910000
        versions = store.snapshots(job)
        assert [s["payload"]["event"]["startTimestamp"] for s in versions] == [1700000000,1700172800]
        assert [s["fetched"] for s in versions] == [1699900000,1699910000]
