"""Composable feature expressions and their chronological evaluator."""

from .expressions import (
    EMA, H2H, ForAgainst, IsHome, Lag, NormalizedStanding,
    RollingMean, RollingStd, RollingSkewness, RollingZScore, Stat, MatchScore,
)
from .evaluation import evaluate_features
from .history import eligible_history_rows, league_season_team_counts
from .ratings import BayesianFixture, BayesianRating, Rating, MatchResultGlicko, StatGlicko
from .league import League, LeaveOneOut, LeaguePopulation
from .warmup import WarmStart, SeededEMA, Hard, LinearFade, ObservationCount, blend_moments
from .transitions import build_team_seasons, TransitionContext
from .composition import Column, Constant, Sum, Difference, Ratio, combine_features, IdentityIndicators
from .contextual import RestDays, CalendarFeature, SeasonProgress, evaluate_context_features
from .presets import FeatureBankPreset
from .preparation import NumericFeatures, IdentityFeatureSpec
from .spatial import Heatmap, RegionMass, heatmap_grid
from .movement import TeamMovement

__all__ = [
    "TeamMovement",
    "RegionMass",
    "Stat", "MatchScore", "ForAgainst", "H2H", "IsHome", "NormalizedStanding", "Lag",
    "RollingMean", "RollingStd", "RollingSkewness", "RollingZScore", "EMA", "evaluate_features",
    "eligible_history_rows", "league_season_team_counts",
    "Rating", "MatchResultGlicko", "StatGlicko", "BayesianRating", "BayesianFixture",
    "League", "LeaveOneOut", "LeaguePopulation", "WarmStart", "SeededEMA",
    "Hard", "LinearFade", "ObservationCount", "blend_moments",
    "build_team_seasons", "TransitionContext",
    "Column", "Constant", "Sum", "Difference", "Ratio", "combine_features", "IdentityIndicators",
    "RestDays", "CalendarFeature", "SeasonProgress", "evaluate_context_features", "FeatureBankPreset",
    "NumericFeatures", "IdentityFeatureSpec",
    "Heatmap", "heatmap_grid",
]
