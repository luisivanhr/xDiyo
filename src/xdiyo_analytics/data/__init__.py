"""Inspect, load and select experiment tables."""

from .loading import SeasonData, load_season, load_seasons, load_season_table
from .inspection import inspect_season
from .selection import STAT_CATEGORIES, list_stat_bundles, select_stats
from .prediction import PredictionFixtures, load_prediction_fixtures, select_prediction_fixtures

__all__ = [
    "SeasonData", "load_season", "load_seasons", "inspect_season",
    "STAT_CATEGORIES", "list_stat_bundles", "select_stats",
    "PredictionFixtures", "load_prediction_fixtures", "select_prediction_fixtures",
]
