"""Native standalone CNN declaration for historically prepared heatmap grids."""
from dataclasses import dataclass, field
from math import isfinite
from .config import HeatmapCNNConfig, positive_integer
from ..training.control import TrainingControl


@dataclass(frozen=True)
class HeatmapCNNModel:
    """Predict from native grid feature groups, optionally with numeric features.

    grid_features selects full group names (without _yXX_xXX); empty discovers
    every native heatmap group. All selected grids need the same resolution.
    Use Lag, RollingMean or EMA grids for forecasting, never current-match maps.
    Classification learns its class order from fitting labels only. Regression
    supports one or more numeric targets. No preprocessing is fit on validation.
    """
    task: str = 'classification'
    encoder: HeatmapCNNConfig = field(default_factory=HeatmapCNNConfig)
    grid_features: tuple = ()
    numeric_features: tuple = ()
    hidden_width: int = 64
    learning_rate: float = .001
    weight_decay: float = .001
    batch_size: int = 128
    control: TrainingControl = field(default_factory=TrainingControl)

    def __post_init__(self):
        if self.task not in ('classification', 'regression'):
            raise ValueError('task must be classification or regression.')
        if not isinstance(self.encoder, HeatmapCNNConfig):
            raise TypeError('encoder must be HeatmapCNNConfig.')
        if not isinstance(self.control, TrainingControl):
            raise TypeError('control must be TrainingControl.')
        for name in ('grid_features', 'numeric_features'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
            if len(set(getattr(self, name))) != len(getattr(self, name)):
                raise ValueError(f'{name} must be distinct.')
        for name in ('hidden_width', 'batch_size'):
            positive_integer(name, getattr(self, name))
        if isinstance(self.learning_rate, bool) or not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive and finite.')
        if isinstance(self.weight_decay, bool) or not isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('weight_decay must be nonnegative and finite.')

    def build(self):
        from .adapter import HeatmapCNNAdapter
        return HeatmapCNNAdapter(self)
