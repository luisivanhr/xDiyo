"""Composable statistic bundles; periods filter a shared category definition."""

from .loading import SeasonData


# Each category is defined once, independently of period. Group names are part
# of the identity: e.g. Shots/expectedGoals and Match overview/expectedGoals
# remain distinct observations. These initial football classifications can change.
STAT_CATEGORIES = {
    "attack": (
        *(("Attack", key) for key in (
            "accurateThroughBall", "bigChanceMissed", "bigChanceScored",
            "fouledFinalThird", "offsides", "touchesInOppBox",
        )),
        *(("Shots", key) for key in (
            "blockedScoringAttempt", "expectedGoals", "expectedGoalsOnTarget",
            "hitWoodwork", "shotsOffGoal", "shotsOnGoal", "totalShotsInsideBox",
            "totalShotsOnGoal", "totalShotsOutsideBox",
        )),
        *(("Match overview", key) for key in (
            "bigChanceCreated", "cornerKicks", "expectedGoals", "totalShotsOnGoal",
        )),
        *(("Passes", key) for key in (
            "accurateCross", "finalThirdEntries", "finalThirdPhaseStatistic",
        )),
    ),
    "defense": (
        *(("Defending", key) for key in (
            "ballRecovery", "errorsLeadToGoal", "errorsLeadToShot",
            "interceptionWon", "totalClearance", "totalTackle", "wonTacklePercent",
        )),
        *(("Goalkeeping", key) for key in (
            "diveSaves", "goalkeeperSaves", "goalsPrevented", "highClaims",
            "penaltySaves", "punches",
        )),
        ("Match overview", "goalkeeperSaves"),
        ("Match overview", "totalTackle"),
    ),
}

_PERIODS = {"all": None, "totals": "ALL", "first_half": "1ST", "second_half": "2ND"}
_ALIASES = {
    "all_stats": "all_all",
    "all_totals_only": "all_totals",
    "all_first_period_only": "all_first_half",
    "all_second_period_only": "all_second_half",
}


def _bundle_specs(categories):
    specs = {
        f"{category}_{suffix}": (category, period)
        for category in ("all", *categories)
        for suffix, period in _PERIODS.items()
    }
    specs.update({alias: specs[name] for alias, name in _ALIASES.items()})
    return specs


def list_stat_bundles(*, categories=None):
    """Return bundle names and descriptions, including optional custom categories.

    categories maps category names to (group_name, key) pairs. Supplied entries
    add to or replace STAT_CATEGORIES; no per-period lists are needed.
    """
    categories = {**STAT_CATEGORIES, **(categories or {})}
    descriptions = {
        name: f"{category} statistics; period={period or 'all available periods'}"
        for name, (category, period) in _bundle_specs(categories).items()
    }
    descriptions["standings"] = "Pregame team positions, kept in a separate table."
    return descriptions


def select_stats(data: SeasonData, *, bundles=(), stats=(), categories=None) -> SeasonData:
    """Select the union of bundles and explicit (period, group_name, key) triples.

    Accepts a load_season/load_seasons result. Returns another SeasonData with
    matches (when loaded), selected statistics and/or standings in pregame.
    No pivot, join, aggregation or feature calculation occurs. Observed rows,
    values, nulls, order and group identities are preserved. Shots are separate.

    all_stats includes uncategorized statistics and any available periods.
    totals/first_half/second_half filter ALL/1ST/2ND respectively; they never
    derive values from other periods. Missing observations are not fabricated.
    An explicit statistic absent from the entire input raises KeyError; category
    members absent from that input are simply unavailable.

    categories optionally adds/replaces period-independent category definitions:
        {"attack": [("Match overview", "cornerKicks"), ...]}
    stats is a sequence of exact triples, e.g.:
        [("1ST", "Match overview", "cornerKicks")]
    """
    import pandas as pd

    if isinstance(bundles, str):
        bundles = (bundles,)
    bundles = tuple(dict.fromkeys(bundles))
    stats = tuple(tuple(stat) for stat in stats)
    if not bundles and not stats:
        raise ValueError("Choose at least one bundle or explicit statistic.")
    if any(len(stat) != 3 or not all(isinstance(x, str) for x in stat) for stat in stats):
        raise ValueError("Each statistic must be a (period, group_name, key) triple.")
    categories = {**STAT_CATEGORIES, **(categories or {})}
    if "all" in categories:
        raise ValueError("'all' is reserved for every available statistic.")
    specs = _bundle_specs(categories)
    unknown = set(bundles) - set(specs) - {"standings"}
    if unknown:
        raise KeyError(f"Unknown bundles: {sorted(unknown)}. Use list_stat_bundles().")

    tables = {}
    if data.matches is not None:
        tables["matches"] = data.matches.copy(deep=False)
    statistic_bundles = [name for name in bundles if name != "standings"]
    selected_fields = []
    if statistic_bundles or stats:
        if data.statistics is None:
            raise KeyError("Load the statistics table before selecting statistics.")
        frame = data.statistics
        fields = ["period", "group_name", "key"]
        # Match distinct identities once, then select all corresponding rows.
        catalogue = frame[fields].drop_duplicates()
        selected = pd.Series(False, index=catalogue.index)
        group_keys = pd.MultiIndex.from_frame(catalogue[["group_name", "key"]])
        for name in statistic_bundles:
            category, period = specs[name]
            mask = pd.Series(True, index=catalogue.index)
            if category != "all":
                mask &= group_keys.isin(categories[category])
            if period is not None:
                mask &= catalogue["period"].eq(period)
            selected |= mask
        identities = pd.MultiIndex.from_frame(catalogue)
        missing = [stat for stat in stats if stat not in identities]
        if missing:
            raise KeyError(f"Statistics absent from the input: {missing}")
        if stats:
            selected |= identities.isin(stats)
        chosen = catalogue.loc[selected]
        selected_fields = list(chosen.itertuples(index=False, name=None))
        row_mask = pd.MultiIndex.from_frame(frame[fields]).isin(selected_fields)
        tables["statistics"] = frame.loc[row_mask].copy()

    if "standings" in bundles:
        if data.pregame is None:
            raise KeyError("Load the pregame table to select standings.")
        required = ["event_id", "side", "position"]
        missing = set(required) - set(data.pregame.columns)
        if missing:
            raise KeyError(f"Pregame is missing standings columns: {sorted(missing)}")
        optional = ["team_id", "observed_at", "source_league", "source_season"]
        columns = required + [name for name in optional if name in data.pregame.columns]
        tables["pregame"] = data.pregame.loc[:, columns].copy()

    provenance = {**data.provenance, "selection": {
        "bundles": bundles, "statistics": tuple(selected_fields),
        "context": ("standings",) if "standings" in bundles else (),
    }}
    return SeasonData(tables=tables, provenance=provenance)
