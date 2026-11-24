"""Missing MLS round labels remain null through capture and publication."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scraping.sofascore.config import Scope
from scraping.sofascore.discovery import round_jobs
from scraping.sofascore.export import read_tables
from scraping.sofascore_web_v3.campaign import Campaign, prepare_campaign
from scraping.sofascore_web_v3.fixture_scope import saved_fixture_scope, event_scope_row
from scraping.sofascore_web_v3.publish import publish_captured_season

FIXTURES = Path(__file__).parents[1] / "fixtures/sofascore"


def original_event():
    return json.loads((FIXTURES / "10025391_event.json").read_text(encoding="utf-8"))["event"]


def test_explicit_scope_does_not_generate_round_requests():
    event = original_event()
    assert "roundInfo" not in event
    scope = saved_fixture_scope(242, 40071, [event_scope_row(event, None, event["season"])])
    assert scope.rounds == () and scope.event_ids == (10025391,)
    with pytest.raises(ValueError, match="cannot enumerate"):
        round_jobs(scope)
    for rounds in [(), (None,), (0,), (True,)]:
        with pytest.raises(ValueError):
            Scope(17, 1, rounds)
    for ids in [(1, 1), (0,), (True,), ("1",)]:
        with pytest.raises(ValueError):
            Scope(242, 40071, (), event_ids=ids)


@pytest.mark.parametrize("change", ["league", "season", "stage", "mixed", "duplicate"])
def test_roundless_exception_cannot_expand_scope(change):
    row = event_scope_row(original_event(), None, original_event()["season"])
    if change == "league": row["competition_id"] = 17
    elif change == "season": row["season_id"] = 1
    elif change == "stage": row["tournament_id"] = 7048
    rows = [row]
    if change == "mixed": rows.append(dict(row, event_id=2, round=1))
    elif change == "duplicate": rows.append(deepcopy(row))
    with pytest.raises(ValueError):
        saved_fixture_scope(row["competition_id"], row["season_id"], rows)


def test_roundless_original_metadata_survives_collection_and_publication(tmp_path):
    event = original_event()
    original = deepcopy(event)
    eid, stem = event["id"], "MLS_22_22"
    prepared = prepare_campaign([eid], base=tmp_path / "campaigns", references={eid: {
        "event": event, "observed_at": 1794640000, "season_stem": stem}})
    directory = Path(prepared["directory"])
    with Campaign(directory) as campaign:
        job = campaign.next_match()
        responses = [("", {"event": event}, 200), ("/pregame-form", {}, 404),
            ("/statistics", {"statistics": []}, 200), ("/shotmap", {"shotmap": []}, 200),
            ("/lineups", {}, 404), ("/incidents", {"incidents": []}, 200),
            *[(f"/heatmap/{event[s]['id']}", {}, 404) for s in ("homeTeam", "awayTeam")]]
        for i, (suffix, payload, code) in enumerate(responses):
            campaign.observe(eid, job["lease"], {"startedDateTime": f"2026-11-14T09:00:{i:02d}Z",
                "request": {"method": "GET", "url": f"https://www.sofascore.com/api/v1/event/{eid}{suffix}"},
                "response": {"status": code, "content": {"text": json.dumps(payload)}}})
        result = campaign.finish(eid, job["lease"], page_event_id=eid)
        assert result["state"] == "complete"
    output = tmp_path / "published"
    published = publish_captured_season(directory, stem, output=output)
    matches = read_tables(output / published["manifest"])["matches"]
    assert len(matches) == 1 and matches[0]["round"] is None and matches[0]["event_id"] == eid
    manifest = json.loads((output / published["manifest"]).read_text())
    assert manifest["scope"]["rounds"] == [] and manifest["scope"]["event_ids"] == [eid]
    assert event == original
