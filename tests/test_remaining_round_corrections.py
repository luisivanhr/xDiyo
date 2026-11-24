from copy import deepcopy
import json
from pathlib import Path

import pytest

from scraping.sofascore.event_round import match_round

EVENTS = json.loads((Path(__file__).parent / "fixtures/sofascore/remaining_round_corrections.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("event,expected", list(zip(EVENTS, (30, 16, 8, 23))))
def test_original_round_projection_keeps_source(event, expected):
    original = deepcopy(event)
    assert match_round(event) == expected
    assert event == original


@pytest.mark.parametrize("event", EVENTS)
@pytest.mark.parametrize("field", ["id", "season", "tournament", "homeTeam", "awayTeam", "startTimestamp", "homeScore", "status", "roundInfo"])
def test_correction_does_not_extend_to_changed_identity(event, field):
    changed = deepcopy(event)
    if field in {"id", "startTimestamp"}:
        changed[field] += 1
    elif field in {"season", "tournament", "homeTeam", "awayTeam"}:
        changed[field]["id"] += 1
    elif field == "status":
        changed[field]["type"] = "notstarted"
    elif field == "roundInfo":
        changed[field] = {"round": 99}
    else:
        changed[field]["current"] += 1
    assert match_round(changed) == (changed.get("roundInfo") or {}).get("round")
