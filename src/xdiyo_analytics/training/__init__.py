"""Fixed-configuration fitting and reusable, identity-preserving predictions."""

from .contracts import FitContext, FoldResult, ModelAdapter, PredictionContext, TrainingResult
from ..weighting import ClassWeightPolicy, ObservationWeights
from .estimators import EstimatorAdapter
from .targets import TargetTransformAdapter
from .calibration import ProbabilityCalibrator, CalibratedAdapter
from .runner import TrainingRunner, fit_predict, split_training_rows
from .control import EarlyStopping, ReduceOnPlateau, TrainingControl, TrainingEvent, ValidationTail, IterativeAdapter, run_iterations
from .incremental import PartialFitBackend
from .live import LiveLossPlot
from .refit import FittedModel, refit_model
from .checkpoints import CheckpointPolicy
from .persistence import JoblibSerializer, load_model, save_model
from .execution import ExecutionPolicy, DeviceAdapter, torch_devices

__all__ = ["SpatialPCA", "SpatialClusters", "ClassWeightPolicy", "ObservationWeights", "ProbabilityCalibrator", "CalibratedAdapter", "NegativeBinomialRegressor", "EstimatorAdapter", "TargetTransformAdapter", "FitContext", "FoldResult", "ModelAdapter",
           "PredictionContext", "TrainingResult", "TrainingRunner", "fit_predict", "split_training_rows",
           "EarlyStopping", "ReduceOnPlateau", "TrainingControl", "TrainingEvent", "ValidationTail",
           "IterativeAdapter", "run_iterations", "PartialFitBackend", "LiveLossPlot", "FittedModel", "refit_model", "CheckpointPolicy",
           "JoblibSerializer", "load_model", "save_model", "ExecutionPolicy", "DeviceAdapter", "torch_devices"]


def __getattr__(name):
    if name in ('SpatialPCA', 'SpatialClusters'):
        from .spatial import SpatialPCA, SpatialClusters
        return {'SpatialPCA':SpatialPCA, 'SpatialClusters':SpatialClusters}[name]
    if name == 'NegativeBinomialRegressor':
        from .counts import NegativeBinomialRegressor
        return NegativeBinomialRegressor
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
