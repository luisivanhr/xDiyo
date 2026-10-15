"""Fresh-process checkpoint fixture worker (no production experiments)."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(Path(__file__).parent)]
import json
import os
import random
import numpy as np
import torch
from xdiyo_analytics.training import TrainingControl, CheckpointPolicy, ReduceOnPlateau, EarlyStopping
from xdiyo_analytics.training.native_resume import capture_backend, rng_state
from xdiyo_analytics.experiments.recovery import pack


def fixture(kind):
    if kind.startswith('gat'):
        from test_temporal_graph import sample, config, context
        from xdiyo_analytics.graphs import TemporalGraphAdapter
        store = sample()
        train = context(store, [1, 2, 3, 4]); train.validation = context(store, [5, 6])
        cfg = config(deterministic=True, fixture_features=('unused',) if kind == 'gat_fixture' else ())
        if cfg.fixture_features:
            for c in (train, train.validation):
                c.metadata['feature_cutoff_at'] = [cfg.history_cutoff(t) for t in store.issue_times(c.metadata, cfg)]
        query = context(store, [7, 8], fit=False)
        if cfg.fixture_features:
            query.metadata['feature_cutoff_at'] = [cfg.history_cutoff(t) for t in store.issue_times(query.metadata, cfg)]
        return TemporalGraphAdapter(cfg, store), train, query
    from test_heatmap_cnn import standalone_context
    from xdiyo_analytics.neural import HeatmapCNNModel, HeatmapCNNConfig, Reconstruction
    c = standalone_context('regression' if kind == 'cnn_regression' else 'classification')
    from dataclasses import replace
    train = replace(c, X=c.X.iloc[:8].copy(), y=c.y.iloc[:8].copy(), metadata=c.metadata.iloc[:8].copy())
    valid = replace(c, X=c.X.iloc[8:].copy(), y=c.y.iloc[8:].copy(), metadata=c.metadata.iloc[8:].copy())
    train.validation = valid
    model = HeatmapCNNModel(task='regression' if kind == 'cnn_regression' else 'classification',
        encoder=HeatmapCNNConfig(channels=(3, 5), embedding_dim=7, reconstruction=Reconstruction(loss='kl')),
        numeric_features=('strength',), batch_size=2)
    return model.build(), train, valid


def main(directory, kind, mode):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    random.seed(321); np.random.seed(321); torch.manual_seed(321)
    adapter, train, query = fixture(kind)
    device = os.environ.get('XDIYO_RESUME_DEVICE', 'cpu')
    adapter.configure_device(device)
    if os.environ.get('XDIYO_RESUME_CACHE') == '1':
        adapter.configure_input_cache(256 * 1024**2, 1024**3)
    control = TrainingControl(max_steps=4, restarts=1, seed=19,
        scheduler=ReduceOnPlateau(patience=1), early_stopping=EarlyStopping(patience=3))
    updates = []
    step = torch.optim.AdamW.step
    def observe(opt, *args, **kwargs):
        gradients = [p.grad.detach().cpu().clone() if p.grad is not None else None for g in opt.param_groups for p in g['params']]
        result = step(opt, *args, **kwargs)
        from xdiyo_analytics.training.native_resume import _cpu
        updates.append(dict(gradients=gradients, optimizer=_cpu(opt.state_dict()),
                            parameters=[p.detach().cpu().clone() for g in opt.param_groups for p in g['params']]))
        return result
    torch.optim.AdamW.step = observe
    # Consume all global RNGs in callbacks; resume must restore their exact next values.
    def observer(event):
        random.random(); np.random.random(); torch.rand(1); torch.rand(1, device=device) if device.startswith("cuda") else None
    train.observer = observer
    if mode == 'reference':
        adapter.fit_controlled(train, control)
    else:
        if mode == 'interrupt':
            original = Path.replace
            count = 0
            def replace(self, target):
                nonlocal count
                result = original(self, target)
                if Path(target).name == 'latest.json':
                    count += 1
                    if count == 2:
                        torch.save(updates, directory/'prefix.pt')
                        os._exit(73)
                return result
            Path.replace = replace
        wrapped = CheckpointPolicy(every=1).wrap(lambda: adapter, directory/'checkpoints', 'fixture')()
        wrapped.fit_controlled(train, control)
    torch.save(dict(backend=capture_backend(adapter.backend_), updates=updates, rng=rng_state(device)),
               directory/(mode+'.pt'))
    (directory/(mode+'.json')).write_text(json.dumps(pack(dict(history=adapter.training_history_,
        summary=adapter.training_summary_, prediction=adapter.predict(query))), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main(*sys.argv[1:])
