"""Inspect, load and select experiment tables."""

from .loading import SeasonData, load_season, load_seasons, load_season_table
from .inspection import inspect_season
from .selection import STAT_CATEGORIES, list_stat_bundles, select_stats
from .prediction import PredictionFixtures, load_prediction_fixtures, select_prediction_fixtures
from .movements import derive_movements, save_movement_enrichment

__all__ = [
    "SeasonData", "load_season", "load_seasons", "inspect_season",
    "STAT_CATEGORIES", "list_stat_bundles", "select_stats",
    "PredictionFixtures", "load_prediction_fixtures", "select_prediction_fixtures",
    "derive_movements", "save_movement_enrichment",
]

from .catalogue import list_datasets, inspect_coverage, select_data, resolve_selection
__all__ += ["list_datasets", "inspect_coverage", "select_data", "resolve_selection"]

from .fixture_rounds import FixtureRounds, load_fixture_rounds, assign_fixture_rounds, load_round_enriched_matches
__all__ += ["FixtureRounds", "load_fixture_rounds", "assign_fixture_rounds", "load_round_enriched_matches"]
