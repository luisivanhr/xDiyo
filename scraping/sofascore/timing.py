"""Keep timing evidence without promoting nominal clocks to measured durations."""
from __future__ import annotations

from .parsers import encoded, number


def timing_evidence(event, incidents=None):
    rows = []
    time = event.get("time") or {}
    for period in range(1,5):
        field = f"injuryTime{period}"
        if field not in time:
            continue
        value = number(time[field])
        valid = value is not None and 0 <= value < 999
        rows.append({"event_id":event["id"],"period":str(period),"source_field":"time."+field,
            "source_value_json":encoded(time[field]),"announced_added_minutes":None,
            "candidate_added_minutes":value if valid else None,"elapsed_seconds":None,
            "quality":"candidate_announcement_unvalidated" if valid else "invalid_or_sentinel"})
    for index,item in enumerate((incidents or {}).get("incidents",[])):
        if item.get("incidentType") not in {"period","injuryTime"}:
            continue
        added = number(item.get("addedTime"))
        length = number(item.get("length"))
        announcement = item.get("incidentType") == "injuryTime" and length is not None and 0 <= length < 999
        period_label = str(item.get("period",item.get("text","unknown")))
        # Only map regulation/extra-time boundaries for the observed standard
        # 45-minute competition contract. Other clocks remain raw evidence.
        if announcement and event.get("defaultPeriodLength") == 45:
            period_label = {45:"1",90:"2",105:"3",120:"4"}.get(item.get("time"), period_label)
        rows.append({"event_id":event["id"],"period":str(item.get("period",item.get("text","unknown"))),
            "source_field":f"incidents[{index}]","source_value_json":encoded(item),
            "announced_added_minutes":length if announcement else None,"candidate_added_minutes":None,"elapsed_seconds":None,
            "quality":"announcement_only" if announcement else "invalid_or_sentinel" if added is not None and added >= 999 else "unvalidated_period_clock"})
        rows[-1]["period"] = period_label
    if not rows:
        rows.append({"event_id":event["id"],"period":None,"source_field":None,
            "source_value_json":None,"announced_added_minutes":None,"candidate_added_minutes":None,
            "elapsed_seconds":None,"quality":"unavailable"})
    return rows
