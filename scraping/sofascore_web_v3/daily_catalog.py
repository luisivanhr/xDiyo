"""Daily coverage selection, independent of historical archive configurations.

Names match the reviewed coverage-expansion output names. Season selectors are
explicit: a football cycle and a calendar year, never 'the latest' provider ID.
"""
from .discovery import league_catalog, normalize_leagues
from scraping.sofascore.store import season_year_pair


# name, provider ID, calendar/split, schedule shape. See DAILY_COLLECTION_README.
NEW_LEAGUES = (
    ("Liga_Profesional", 155, "calendar", "stages"),
    ("A_League", 136, "split", "uneven"),
    ("Austrian_Bundesliga", 45, "split", "split"),
    ("Brazil_Serie_A", 325, "calendar", "round_robin"),
    ("Parva_Liga", 247, "split", "split"),
    ("Chile_Primera_Division", 11653, "calendar", "round_robin"),
    ("Chinese_Super_League", 649, "calendar", "round_robin"),
    # 241 is the obsolete Liga Postobon inventory. The current combined 2026
    # competition lives at 11539 (both ordinary stages, playoffs excluded).
    ("Colombia_Primera_A", 11539, "calendar", "stages"),
    ("HNL", 170, "split", "round_robin"),
    ("Czech_First_League", 172, "split", "split"),
    ("Danish_Superliga", 39, "split", "split"),
    ("LigaPro", 240, "calendar", "split"),
    ("Egyptian_Premier_League", 808, "split", "split"),
    ("Greek_Super_League", 185, "split", "split"),
    ("Eliteserien", 20, "calendar", "round_robin"),
    ("Peru_Liga_1", 406, "calendar", "stages"),
    ("Ekstraklasa", 202, "split", "round_robin"),
    ("Liga_Portugal", 238, "split", "round_robin"),
    ("Romanian_Superliga", 152, "split", "split"),
    ("Russian_Premier_League", 203, "split", "round_robin"),
    ("Saudi_Pro_League", 955, "split", "round_robin"),
    ("Serbian_Superliga", 210, "split", "split"),
    ("PrvaLiga", 212, "split", "round_robin"),
    ("Allsvenskan", 40, "calendar", "round_robin"),
    ("Swiss_Super_League", 215, "split", "split"),
    ("Super_Lig", 52, "split", "round_robin"),
    ("MLS", 242, "calendar", "conferences"),
    ("Ukrainian_Premier_League", 218, "split", "round_robin"),
    ("Brazil_Serie_B", 390, "calendar", "round_robin"),
    ("Eerste_Divisie", 131, "split", "round_robin"),
    ("Liga_Portugal_2", 239, "split", "round_robin"),
    ("Scottish_Championship", 206, "split", "round_robin"),
)


def daily_league_catalog():
    """The original 13 plus all 32 leagues in the November expansion."""
    original = [{**v, "season_kind": "split", "schedule": "split" if v["competition_id"] in {36, 38}
                 else "round_robin"} for v in league_catalog()]
    added = [dict(name=name, competition_id=cid, season_kind=kind, schedule=shape,
                  meetings_per_pair=4 if cid in {170, 212, 206} else 2)
             for name, cid, kind, shape in NEW_LEAGUES]
    return original + added


def daily_selection(season, leagues=None, *, calendar_season=None, season_overrides=None):
    years = season_year_pair(season)
    if years is None or years[1] not in {years[0], years[0] + 1}:
        raise ValueError("Choose an explicit calendar year or consecutive football cycle")
    calendar = season_year_pair(str(calendar_season) if calendar_season is not None else str(years[0]))
    if calendar is None or calendar[0] != calendar[1]:
        raise ValueError("calendar_season must be one calendar year")
    catalog = daily_league_catalog()
    by_name = {v["name"]: v for v in catalog}
    if leagues is None:
        selected = catalog
    elif isinstance(leagues, list) and all(isinstance(v, str) for v in leagues):
        unknown = set(leagues) - by_name.keys()
        if unknown:
            raise ValueError(f"Unknown daily leagues: {sorted(unknown)}")
        selected = [by_name[name] for name in leagues]
    else:
        if not isinstance(leagues, list) or not all(isinstance(v, dict) for v in leagues):
            raise ValueError("Select a list of catalog names or explicit league dictionaries")
        selected = []
        for spec in leagues:
            known = by_name.get(spec.get("name"), {})
            selected.append({**known, **spec})
    normalize_leagues(selected)
    overrides = season_overrides or {}
    if not isinstance(overrides, dict) or set(overrides) - {v["name"] for v in selected}:
        raise ValueError("Season overrides must name selected leagues")
    result = []
    for spec in selected:
        kind = spec.get("season_kind", "split")
        if kind not in {"calendar", "split"}:
            raise ValueError("Unknown season_kind")
        choice = overrides.get(spec["name"], spec.get("season"))
        pair = season_year_pair(str(choice)) if choice is not None else (calendar if kind == "calendar" else years)
        if pair is None or pair[1] != pair[0] + (kind == "split"):
            raise ValueError(f"{spec['name']} needs an explicit {kind} season")
        shape = spec.get("schedule", "round_robin")
        if shape not in {"round_robin", "split", "stages", "uneven", "conferences"}:
            raise ValueError("Unknown daily schedule shape")
        result.append({**spec, "season_kind": kind, "schedule": shape, "years": list(pair)})
    return result
