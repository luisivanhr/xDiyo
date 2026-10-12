"""Native deterministic historical-match grids, never learned feature caches."""
from dataclasses import dataclass, field, asdict
import json
import numpy as np
import pandas as pd
from ..features.spatial import Heatmap, heatmap_values
from ..features.point_geometry import validate_point_identities
from ..neural.config import HeatmapCNNConfig, positive_integer
from ..neural.inputs import neural_maps


@dataclass(frozen=True)
class GraphHeatmaps:
    grid: Heatmap = field(default_factory=lambda: Heatmap(side='both', kinds=('player',), orientation='team'))
    encoder: HeatmapCNNConfig = field(default_factory=HeatmapCNNConfig)
    batch_size: int = 256

    def __post_init__(self):
        if not isinstance(self.grid, Heatmap) or self.grid.side != 'both' or self.grid.orientation != 'team':
            raise ValueError('Graph heatmaps require both maps in the receiving team frame (side=both, orientation=team).')
        if not isinstance(self.encoder, HeatmapCNNConfig):
            raise TypeError('encoder must be HeatmapCNNConfig.')
        if self.encoder.pool_size > self.grid.grid_size:
            raise ValueError('Pooling cannot exceed grid size.')
        positive_integer('batch_size', self.batch_size)

    @classmethod
    def from_dict(cls, state):
        return cls(grid=Heatmap(**state['grid']), encoder=HeatmapCNNConfig.from_dict(state['encoder']), batch_size=state['batch_size'])


def grid_identity(config):
    return json.loads(json.dumps(asdict(config.grid)))


def columns(config):
    n = config.grid.grid_size
    return [f'cnn::{side}::y{y}::x{x}' for side in ('for', 'against') for y in range(n) for x in range(n)]


def validate_grid_contract(store, config):
    """Check semantic grid identity independently of the table fingerprint.

    Different grid recipes can coincidentally produce identical old cells.
    Such stores are not interchangeable for future evidence or saved models.
    """
    if config is None:
        return
    if store.provenance.get('heatmap_grid') != grid_identity(config):
        raise ValueError('Heatmap grid settings change preparation; use a matching prepared recipe or event store.')
    actual = {c for c in store._history.columns if c.startswith('cnn::')}
    if actual != set(columns(config)):
        raise ValueError('Event store CNN grid columns do not match the configured resolution.')


def prepare_heatmaps(history, points, config):
    keys = validate_point_identities(history, points)
    for name in ('raw_hash', 'source_key'):
        if name in points and (points.groupby(keys, dropna=False)[name].nunique(dropna=True) > 1).any():
            raise ValueError(f'Conflicting spatial {name}; select one source version per map.')
    grids = heatmap_values(history, points, config.grid)
    # Native heatmap_values performs the opponent's 180-degree rotation.
    grids.columns = columns(config)
    if set(grids) & set(history):
        raise ValueError('History already contains CNN grids; do not overwrite prepared evidence.')
    neural_maps(grids.to_numpy().reshape(len(history), 2, config.grid.grid_size, config.grid.grid_size))
    grids.attrs = {}
    return pd.concat((history, grids), axis=1)


def tensors(frame, config, device):
    import torch
    names = columns(config)
    if not set(names) <= set(frame):
        raise ValueError('CNN-enabled GAT requires native prepared historical heatmap grids.')
    maps, available = neural_maps(frame[names].to_numpy(dtype=float, na_value=np.nan).reshape(
        len(frame), 2, config.grid.grid_size, config.grid.grid_size))
    return torch.as_tensor(maps, device=device), torch.as_tensor(available, device=device)
