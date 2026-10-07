"""Native prediction composition; outer evaluation remains owned by TrainingRunner."""
from .contracts import OutputSchema, OutputRef, PredictionBundle, TargetSpec
from .graph import ModelNode, TransformNode, ReducerNode, PredictionGraphSpec
from .training import TrainingPlan
from .adapter import CompositeModelAdapter
from .ensemble import ModelEnsemble
from .stack import ModelStack
from .reducers import Mean, HardVote, DistributionMixture
from .transforms import OutputFeatures
from .residual import ResidualModel
from .targets import LearnedTargetAdapter, UtilityTarget, SavedUtilityModel
from .estimator import Estimator
