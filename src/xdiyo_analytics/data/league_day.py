"""League calendar dates for daily standings and temporal folds (v1)."""

from zoneinfo import ZoneInfo

import pandas as pd


# Project calendar conventions, including one clock for multi-zone competitions.
_NATIVE_ZONES = {
    "Europe/Madrid": (8, 54), "Europe/London": (17, 18, 36, 206),
    "Europe/Oslo": (20,), "Europe/Rome": (23, 53), "Europe/Paris": (34, 182),
    "Europe/Berlin": (35, 44), "Europe/Amsterdam": (37, 131), "Europe/Brussels": (38,),
    "Europe/Copenhagen": (39,), "Europe/Stockholm": (40,), "Europe/Vienna": (45,),
    "Europe/Istanbul": (52,), "Australia/Sydney": (136,), "Europe/Bucharest": (152,),
    "America/Argentina/Buenos_Aires": (155,), "Europe/Zagreb": (170,), "Europe/Prague": (172,),
    "Europe/Athens": (185,), "Asia/Tokyo": (196, 402), "Europe/Warsaw": (202,),
    "Europe/Moscow": (203,), "Europe/Belgrade": (210,), "Europe/Ljubljana": (212,),
    "Europe/Zurich": (215,), "Europe/Kyiv": (218,), "Europe/Lisbon": (238, 239),
    "America/Guayaquil": (240,), "America/New_York": (242,), "Europe/Sofia": (247,),
    "America/Sao_Paulo": (325, 390), "America/Lima": (406,), "Asia/Shanghai": (649,),
    "Africa/Cairo": (808,), "Asia/Riyadh": (955,), "America/Bogota": (11536, 11539),
    "America/Mexico_City": (11620, 11621), "America/Santiago": (11653,),
}


def league_zones(mapping):
    if mapping == "native":
        return {str(cid): zone for zone, ids in _NATIVE_ZONES.items() for cid in ids}

    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("league_timezones must map competition IDs to IANA timezones")
    zones = {}
    for key, value in mapping.items():
        if not str(key).isascii() or not str(key).isdecimal() or str(int(key)) != str(key) or int(key) <= 0:
            raise ValueError(f"Invalid competition ID in league_timezones: {key!r}")
        if not isinstance(value, str) or str(key) in zones:
            raise ValueError(f"Invalid or duplicated timezone for competition {key}")
        zones[str(key)] = ZoneInfo(value).key
    return zones


def day_boundary(frame, times, mapping, *, after_match=False):
    """Convert local civil dates to UTC instants, preserving row alignment."""
    zones = league_zones(mapping)
    named = frame.competition_id.map(lambda key: zones.get(str(key)))
    if named.isna().any():
        missing = frame.loc[named.isna(), "competition_id"].unique()
        raise ValueError(f"Missing league timezone for competition(s) {missing}")
    if not times.index.equals(frame.index):
        raise ValueError("League calendar times must align to the input rows")
    result = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    for zone in named.unique():
        take = named.eq(zone)
        local = times.loc[take].dt.tz_convert(zone).dt.tz_localize(None).dt.normalize()
        if after_match:
            if "completion_date" in frame:
                dates = pd.to_datetime(frame.loc[take, "completion_date"], errors="raise")
                local = local.where(dates.isna() | local.ge(dates), dates)
            local = local + pd.Timedelta(days=1)
        # A civil day occasionally begins with a DST gap or two midnights.
        midnight = local.dt.tz_localize(zone, nonexistent="shift_forward", ambiguous=True)
        result.loc[take] = midnight.dt.tz_convert("UTC").astype(result.dtype)
    return result
