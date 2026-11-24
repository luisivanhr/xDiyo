from copy import deepcopy
import json
from pathlib import Path

import pytest

from scraping.sofascore.parsers import parse_match
from scraping.sofascore.event_scores import reviewed_score_correction

FIXTURES = Path(__file__).parents[1] / "fixtures/sofascore"


def original():
    return json.loads((FIXTURES / "11031965_event.json").read_text(encoding="utf-8"))["event"]


def test_original_full_event_agrees_with_incidents_after_exact_projection():
    event = original()
    before = deepcopy(event)
    incidents = json.loads((FIXTURES / "11031965_incidents.json").read_text(encoding="utf-8"))["incidents"]
    ft = [i for i in incidents if i.get("incidentType") == "period" and i.get("text") == "FT"]
    goals = [i for i in incidents if i.get("incidentType") == "goal"]
    assert [(i["homeScore"], i["awayScore"]) for i in ft] == [(2, 3)]
    assert (sum(i["isHome"] for i in goals), sum(not i["isHome"] for i in goals)) == (2, 3)
    row = parse_match(event)
    assert (row["home_score_current"], row["away_score_current"], row["away_score_display"]) == (2, 3, 3)
    assert row["away_score_period1"] + row["away_score_period2"] == 3
    assert event == before and event["awayScore"]["current"] == 4


@pytest.mark.parametrize("path,value", [
    (("homeTeam", "id"), 999), (("awayTeam", "id"), 999),
    (("season", "id"), 999), (("tournament", "id"), 999),
    (("roundInfo", "round"), 8), (("status", "type"), "inprogress"),
    (("awayScore", "normaltime"), 4), (("awayScore", "period2"), 2),
    (("awayScore", "penalties"), 1),
])
def test_changed_evidence_requires_review(path, value):
    event = original()
    event[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        parse_match(event)


def test_other_events_and_provider_repair_are_not_changed():
    event = original()
    event["id"] += 1
    assert parse_match(event)["away_score_current"] == 4
    event = original()
    event["awayScore"].update(current=3, display=3)
    assert not reviewed_score_correction(event)
    assert parse_match(event)["away_score_current"] == 3
