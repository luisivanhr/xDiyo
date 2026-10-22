"""Rating engines, independent state histories and reusable numeric snapshots."""

from .glicko import Glicko2, PairwiseRatingEngine, glicko_state
from .replay import build_ratings
from .snapshots import RatingRun
from .transitions import GlickoTransition, RatingTransition
from .bayesian_core import BayesianParameters, GammaState, TeamState
from .bayesian import BayesianConfig, BayesianModel, BayesianRatingRun, build_bayesian_ratings
from .bayesian_training import (
    BayesianTrainingResult, BayesianScoreAdapter, BayesianScoreSerializer, train_bayesian,
)

__all__ = ["Glicko2", "PairwiseRatingEngine", "glicko_state", "build_ratings", "RatingRun",
           "GlickoTransition", "RatingTransition", "BayesianParameters", "GammaState", "TeamState",
           "BayesianConfig", "BayesianModel", "BayesianRatingRun", "build_bayesian_ratings",
           "BayesianTrainingResult", "BayesianScoreAdapter", "BayesianScoreSerializer", "train_bayesian"]

from .gap_core import GAPParameters, GAPState, GAPPrediction, WheatcroftGAP
from .gap_data import GAPStatistic, GAPSeason, derive_shots_off_target, gap_season_from_context
from .gap import GAPVintage, GAPRatingRun, build_gap_ratings
from .gap_training import GAPFitResult, gap_mae, tune_gap

__all__ += ["GAPParameters", "GAPState", "GAPPrediction", "WheatcroftGAP", "GAPStatistic",
            "GAPSeason", "derive_shots_off_target", "gap_season_from_context", "GAPVintage",
            "GAPRatingRun", "build_gap_ratings", "GAPFitResult", "gap_mae", "tune_gap"]
