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
        if kind in ('gat_dependency', 'gat_dependency_statistics', 'gat_dependency_fallback'):
            from graph_execution_fixture import execution_fixture
            requested = os.environ.get('XDIYO_RESUME_EXECUTION_MODE', 'dependency_packed') if kind == 'gat_dependency_statistics' else 'dependency_packed'
            adapter, train, query = execution_fixture(mode=requested, statistics=kind == 'gat_dependency_statistics')
            if kind == 'gat_dependency_fallback':
                from dataclasses import replace
                adapter.config = replace(adapter.config, execution=replace(adapter.config.execution, host_memory_bytes=1))
            return adapter, train, query
        from test_temporal_graph import sample, config, context
        from xdiyo_analytics.graphs import TemporalGraphAdapter
        if kind in ('gat_memory', 'gat_statistics', 'gat_combined'):
            from test_graph_integration_gates import extension_fixture
            return extension_fixture(kind)
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
    if kind in ('gat_dependency', 'gat_dependency_statistics', 'gat_dependency_fallback'):
        # Keep the real validation outcome CE. A large significance threshold
        # exercises nonzero patience and scheduler state before interruption.
        control = TrainingControl(max_steps=4, restarts=1, seed=19,
            scheduler=ReduceOnPlateau(patience=1), early_stopping=EarlyStopping(patience=3, min_delta=10.))
    scenario = os.environ.get('XDIYO_RESUME_BOUNDARY', '')
    if scenario == 'numpy':
        control = TrainingControl(max_steps=np.int64(2), restarts=np.int64(1), seed=np.int64(19),
            scheduler=ReduceOnPlateau(patience=np.int64(1), factor=np.float32(.5)),
            early_stopping=EarlyStopping(patience=np.int64(3), min_delta=np.float64(0)))
    elif scenario in ('early_stopping', 'nonfinite_monitor'):
        control = TrainingControl(max_steps=4, restarts=1, seed=19, monitor='train_loss',
                                  early_stopping=EarlyStopping(patience=1))
        from xdiyo_analytics.graphs.adapter import _Backend
        from xdiyo_analytics.neural.adapter import CNNBackend
        cls = _Backend if kind.startswith('gat') else CNNBackend
        checkpoint_metrics = cls.checkpoint_metrics
        def fixed_metrics(self, monitor, values):
            values = checkpoint_metrics(self, monitor, values)
            step_number = max(int(v['step']) for v in self.optimizer.state.values())
            return dict(values, train_loss=float('nan') if scenario == 'nonfinite_monitor' and step_number == 2 else 1.)
        cls.checkpoint_metrics = fixed_metrics
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
    if kind in ('gat_dependency', 'gat_dependency_statistics', 'gat_dependency_fallback'):
        from copy import deepcopy
        from xdiyo_analytics.graphs.adapter import _Backend
        backend_step = _Backend.step
        def traced_step(self):
            values = backend_step(self)
            trace = deepcopy(self._execution_trace)
            requested = self.config.execution.mode
            assert trace['requested_mode'] == requested
            if requested == 'native':
                assert trace['whole_pass_native'] is True, trace
                assert trace['fallback_reasons'] == {'native_requested': 1}, trace
            elif kind == 'gat_dependency_fallback':
                assert trace['fallback_groups'] == trace['groups'], trace
                assert trace['fallback_reasons'] == {'descriptor_budget': trace['groups']}, trace
            else:
                assert trace['fallback_groups'] == 0, trace
                if requested == 'dependency_packed':
                    assert trace['batches'] < trace['groups'], trace
                else:
                    assert requested in ('dependency', 'dependency_lean')
                    assert trace['batches'] == trace['groups'], trace
            updates[-1]['execution_trace'] = trace
            return values
        _Backend.step = traced_step
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
                    stop = count == 2
                    if scenario in ('restart', 'early_stopping', 'nonfinite_monitor'):
                        latest = json.loads(Path(target).read_text())
                        from xdiyo_analytics.training.native_resume import read_checkpoint
                        saved, _ = read_checkpoint(Path(target).parent/latest['directory'])
                        if scenario == 'restart':
                            stop = saved['attempt'] == 1 and saved['active'] is None
                        else:
                            stop = saved['active'] is not None and saved['active']['stopped'] and saved['active']['reason'] == scenario
                    if stop:
                        if kind in ('gat_dependency', 'gat_dependency_statistics', 'gat_dependency_fallback'):
                            latest = json.loads(Path(target).read_text())
                            from xdiyo_analytics.training.native_resume import read_checkpoint
                            saved, _ = read_checkpoint(Path(target).parent/latest['directory'])
                            active = saved['active']
                            torch.save({name: active[name] for name in (
                                'best', 'best_step', 'progress_best', 'stale', 'schedule_stale', 'last_value')},
                                directory/'prefix-control.pt')
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
