"""Declarative CNN settings; importing these types never imports PyTorch."""
from dataclasses import dataclass
from math import isfinite


def positive_integer(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer.')


@dataclass(frozen=True)
class Reconstruction:
    """Optional masked reconstruction objective, separate from prediction loss."""
    weight: float = .1
    hidden_width: int = 64
    loss: str = 'mse'

    def __post_init__(self):
        if isinstance(self.weight, bool) or not isfinite(self.weight) or self.weight <= 0:
            raise ValueError('Reconstruction weight must be finite and positive; disable the branch with None.')
        positive_integer('hidden_width', self.hidden_width)
        if self.loss not in ('mse', 'kl'):
            raise ValueError('Reconstruction loss must be mse or kl.')


@dataclass(frozen=True)
class HeatmapCNNConfig:
    """Shared encoder: convolutions, average pooling, flattening and projection.

    Spatial position is retained. Input channel count and flattened width are
    inferred from the selected maps and grid dimensions, never guessed.
    Defaults give 4,096 encoder parameters for two 10x10 maps plus their masks.
    reconstruction=None creates no decoder and no auxiliary training loss.
    """
    channels: tuple = (8, 8)
    kernel_size: int = 3
    pool_size: int = 2
    embedding_dim: int = 16
    activation: str = 'relu'
    reconstruction: Reconstruction | None = None

    def __post_init__(self):
        object.__setattr__(self, 'channels', tuple(self.channels))
        if not self.channels:
            raise ValueError('At least one convolution is required.')
        for value in self.channels:
            positive_integer('channels', value)
        for name in ('kernel_size', 'pool_size', 'embedding_dim'):
            positive_integer(name, getattr(self, name))
        if self.kernel_size % 2 != 1:
            raise ValueError('kernel_size must be odd to preserve grid dimensions.')
        if self.activation not in ('relu', 'gelu', 'tanh'):
            raise ValueError('activation must be relu, gelu or tanh.')
        if self.reconstruction is not None and not isinstance(self.reconstruction, Reconstruction):
            raise TypeError('reconstruction must be Reconstruction or None.')

    @classmethod
    def from_dict(cls, state):
        state = dict(state)
        if state.get('reconstruction') is not None:
            state['reconstruction'] = Reconstruction(**state['reconstruction'])
        return cls(**state)
