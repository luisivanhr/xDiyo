"""Reusable reporter contracts, study library and standalone HTML presentation."""

from .contracts import Artifact, StudyResult, AnalysisContext, Reporter, StudyRun, AnalysisReport, FeatureSelection, PostTrainingContext
from .studies import FeatureDistributionReporter, CorrelationAnalysis, FeatureTimeline
from .selection import TopKCorrelationSelector
from .selectors import FeatureSelector, vote_selections, resolve_selection_count
from .post_training import (PredictionReporter, PerformanceReporter, ResidualAnalysisReporter,
                            CalibrationReporter, PredictionTimelineReporter, PredictionDistributionReporter,
                            LabelPredictionDistributionReporter)
from .betting import BetPerformanceReporter
from .leaderboard import ExperimentLeaderboardReporter
from .model_diagnostics import LearningCurveReporter, CoefficientReporter
from .match_results import MatchResultReporter
from .teams import TeamCatalog

__all__ = ["Artifact", "StudyResult", "AnalysisContext", "Reporter", "StudyRun", "AnalysisReport",
           "FeatureDistributionReporter", "CorrelationAnalysis", "FeatureTimeline",
           "FeatureSelection", "FeatureSelector", "vote_selections", "resolve_selection_count", "TopKCorrelationSelector",
           "PostTrainingContext", "PredictionReporter", "PerformanceReporter", "ResidualAnalysisReporter",
           "CalibrationReporter", "PredictionTimelineReporter", "PredictionDistributionReporter",
           "LabelPredictionDistributionReporter", "BetPerformanceReporter", "ExperimentLeaderboardReporter",
           "LearningCurveReporter", "CoefficientReporter", "MatchResultReporter", "TeamCatalog"]
