"""Checks for original Chrome season-event pages and complete page chains."""
from copy import deepcopy
import hashlib
import json

from .discovery import positive


def page_event_fingerprint(event):
    """Exclude only follower counters, never fixture identity or match data."""
    value = deepcopy(event)
    for path in (("homeTeam",), ("awayTeam",), ("tournament", "uniqueTournament")):
        item = value
        for key in path:
            item = item.get(key, {})
        item.pop("userCount", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def review_boundary_overlap(pages, *, competition_id, season_id, event_id):
    """Describe one explicitly reviewed adjacent reverse-page boundary overlap.

    Returning this record does not validate the season's fixture count. Its
    independent standings/schedule checks must still detect an omitted fixture.
    """
    found = []
    for page in pages:
        events = validate_event_page(page["payload"], competition_id=competition_id, season_id=season_id)
        found.extend((page["page"], i, len(events), event) for i, event in enumerate(events)
                     if event["id"] == positive(event_id))
    if (len(found) != 2 or found[1][0] != found[0][0] + 1 or found[0][1] != 0
            or found[1][1] != found[1][2] - 1
            or page_event_fingerprint(found[0][3]) != page_event_fingerprint(found[1][3])):
        raise ValueError("Overlap is not one equivalent fixture at an adjacent reverse-page boundary")
    return dict(competition_id=competition_id, season_id=season_id, event_id=event_id,
                pages=[x[0] for x in found], fingerprint=page_event_fingerprint(found[0][3]))


def validate_event_page(payload, *, competition_id, season_id):
    """Check one page before its successor is scheduled, without selecting scope."""
    positive(competition_id)
    positive(season_id)
    if (not isinstance(payload, dict) or type(payload.get("hasNextPage")) is not bool
            or not isinstance(payload.get("events"), list)):
        raise ValueError("Missing explicit pagination flag or event list")
    values = payload["events"]
    if payload["hasNextPage"] and not values:
        raise ValueError("Empty intermediate page")
    seen = set()
    for event in values:
        if not isinstance(event, dict):
            raise ValueError("Malformed event in captured page")
        eid = positive(event.get("id"))
        if eid in seen:
            raise ValueError("Repeated fixture within a page")
        if (event.get("tournament", {}).get("uniqueTournament", {}).get("id") != competition_id
                or event.get("season", {}).get("id") != season_id):
            raise ValueError("Foreign event in the selected season page")
        seen.add(eid)
    return values


def validate_event_pages(pages, *, competition_id, season_id, reviewed_overlaps=()):
    positive(competition_id)
    positive(season_id)
    if not isinstance(pages, list) or not pages or len(pages) > 4096:
        raise ValueError("Missing or excessive page chain")
    reviews = {}
    for review in reviewed_overlaps:
        eid = positive(review.get("event_id"))
        if eid in reviews or review != review_boundary_overlap(pages, competition_id=competition_id,
                                                               season_id=season_id, event_id=eid):
            raise ValueError("Changed or unused page-overlap review")
        reviews[eid] = review
    events, seen, used = [], set(), set()
    for index, record in enumerate(pages):
        if not isinstance(record, dict) or type(record.get("page")) is not int or record["page"] != index:
            raise ValueError("Page chain has a missing, duplicate or unordered page")
        payload = record.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("Malformed captured page")
        more, values = payload.get("hasNextPage"), payload.get("events")
        if type(more) is not bool or not isinstance(values, list):
            raise ValueError("Missing explicit pagination flag or event list")
        if more != (index < len(pages) - 1):
            raise ValueError("Page chain ends early or continues after its terminal page")
        if more and not values:
            raise ValueError("Empty intermediate page")
        for event in values:
            if (not isinstance(event, dict) or not isinstance(event.get("tournament"), dict)
                    or not isinstance(event.get("season"), dict)
                    or not isinstance(event["tournament"].get("uniqueTournament"), dict)):
                raise ValueError("Malformed event in captured page")
            eid = positive(event.get("id"))
            if eid in seen:
                if eid not in reviews or eid in used:
                    raise ValueError("Repeated fixture across pages; review the capture boundary")
                used.add(eid)
                continue
            if (event.get("tournament", {}).get("uniqueTournament", {}).get("id") != competition_id
                    or event.get("season", {}).get("id") != season_id):
                raise ValueError("Foreign event in the selected season page chain")
            seen.add(eid)
            events.append(event)
    if not events:
        raise ValueError("No fixture evidence in page chain")
    if used != set(reviews):
        raise ValueError("Unused page-overlap review")
    return events
