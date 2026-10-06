"""Reusable reporter contracts, study library and standalone HTML presentation."""

from .contracts import Artifact, StudyResult, AnalysisContext, Reporter, StudyRun, AnalysisReport, FeatureSelection, PostTrainingContext
from .studies import FeatureDistributionReporter, CorrelationAnalysis, FeatureTimeline
from .selection import TopKCorrelationSelector
from .class_weights import ClassWeightReporter
from .selectors import FeatureSelector, vote_selections, resolve_selection_count
from .post_training import (PredictionReporter, PerformanceReporter, ResidualAnalysisReporter,
                            CalibrationReporter, PredictionTimelineReporter, PredictionDistributionReporter,
                            LabelPredictionDistributionReporter)
from .classifier_diagnostics import CountClassificationReporter
from .betting import BetPerformanceReporter
from .bet_outcomes import BetOutcomeReporter
from .leaderboard import ExperimentLeaderboardReporter
from .model_diagnostics import LearningCurveReporter, CoefficientReporter, FeatureImportanceReporter
from .match_results import MatchResultReporter
from .teams import TeamCatalog
from .heatmaps import HeatmapReporter, axial_direction_summary
from .ratings import RatingReporter

__all__ = ["ClassWeightReporter", "Artifact", "StudyResult", "AnalysisContext", "Reporter", "StudyRun", "AnalysisReport",
           "RatingReporter", "HeatmapReporter", "axial_direction_summary", "FeatureDistributionReporter", "CorrelationAnalysis", "FeatureTimeline",
           "FeatureSelection", "FeatureSelector", "vote_selections", "resolve_selection_count", "TopKCorrelationSelector",
           "PostTrainingContext", "PredictionReporter", "PerformanceReporter", "ResidualAnalysisReporter",
           "CalibrationReporter", "PredictionTimelineReporter", "PredictionDistributionReporter",
           "LabelPredictionDistributionReporter", "BetPerformanceReporter", "ExperimentLeaderboardReporter",
           "LearningCurveReporter", "CoefficientReporter", "FeatureImportanceReporter", "CountClassificationReporter", "MatchResultReporter", "TeamCatalog", "BetOutcomeReporter"]
