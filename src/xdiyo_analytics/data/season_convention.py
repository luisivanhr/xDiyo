"""The existing scope year-pair convention used by new experiments."""
POLICY = "sofascore_year_pair_v1"


def season_pair(start, end):
    """Calendar years repeat; never invent an adjacent-year season."""
    if (type(start) is not int or type(end) is not int
            or not 1000 <= start <= 9999 or end not in (start, start + 1) or end > 9999):
        raise ValueError("Season scope requires full start/end years, equal or consecutive.")
    return f"{start:04d}/{end:04d}"


def attach_season_convention(matches, scope):
    """Add verified scope metadata without overwriting provider names or source keys."""
    values = dict(experiment_season=season_pair(scope["season_start"], scope["season_end"]),
                  season_start=scope["season_start"], season_end=scope["season_end"])
    result = matches.copy()
    for key, value in values.items():
        if key in result and (result[key].isna().any() or not result[key].eq(value).all()):
            raise ValueError(f"Existing {key} disagrees with verified season scope.")
        result[key] = value
    return result


def validate_season_pair(value):
    """Validate a full-year pair without coercing single-year or display labels."""
    import re
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}/[0-9]{4}", value):
        raise ValueError("experiment_season must be a YYYY/YYYY year pair.")
    start, end = (int(p) for p in value.split("/"))
    season_pair(start, end)
    return value
