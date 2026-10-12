"""Portable data-only configuration and tensor-only CNN weights."""
from dataclasses import asdict
import json
from pathlib import Path


class HeatmapCNNSerializer:
    format_id = 'xdiyo.heatmap-cnn.v1'

    def __init__(self, *, device='cpu'):
        self.device = device

    def save(self, adapter, directory):
        import torch
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        b = adapter.backend_
        state = dict(spec=asdict(adapter.spec), inputs=b.inputs.state(), classes=b.classes,
                     targets=b.target_names, training_summary=adapter.training_summary_, inner_threads=adapter.inner_threads)
        (path/'cnn.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
        torch.save({k: v.detach().cpu() for k, v in b.model.state_dict().items()}, path/'weights.pt')
        adapter.training_history_.to_parquet(path/'training_history.parquet', compression='zstd', index=False)

    def load(self, directory):
        import torch
        import pandas as pd
        from .config import HeatmapCNNConfig
        from .model import HeatmapCNNModel
        from .adapter import HeatmapCNNAdapter, CNNBackend, CNNInputs, make_network
        from ..training.control import TrainingControl, EarlyStopping, ReduceOnPlateau
        path = Path(directory)
        state = json.loads((path/'cnn.json').read_text(encoding='utf-8'))
        spec = state['spec']
        spec['encoder'] = HeatmapCNNConfig.from_dict(spec['encoder'])
        control = spec['control']
        for name, cls in [('early_stopping', EarlyStopping), ('scheduler', ReduceOnPlateau)]:
            if control.get(name) is not None:
                control[name] = cls(**control[name])
        spec['control'] = TrainingControl(**control)
        spec = HeatmapCNNModel(**spec)
        adapter = HeatmapCNNAdapter(spec, inner_threads=state['inner_threads'])
        adapter.configure_device(self.device)
        b = CNNBackend(spec, 0, adapter.device)
        b.inputs = CNNInputs.from_state(spec, state['inputs'])
        b.classes, b.target_names = state['classes'], state['targets']
        with torch.random.fork_rng(devices=[]):
            b.model = make_network(spec, b.inputs, len(b.classes) if b.classes is not None else len(b.target_names)).to(adapter.device)
        b.model.load_state_dict(torch.load(path/'weights.pt', map_location=adapter.device, weights_only=True))
        adapter.backend_ = b
        adapter.training_summary_ = state['training_summary']
        adapter.training_history_ = pd.read_parquet(path/'training_history.parquet')
        return adapter
