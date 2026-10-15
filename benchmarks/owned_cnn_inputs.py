"""Isolated CPU comparison; synthetic population, no frozen experiments opened."""
from pathlib import Path
import argparse, gc, hashlib, json, os, statistics, subprocess, sys, time, types
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src')]
import numpy as np
import pandas as pd
import psutil
import torch
from xdiyo_analytics.neural import HeatmapCNNModel, HeatmapCNNConfig, Reconstruction
from xdiyo_analytics.training.contracts import FitContext


def worker(kind, output):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    sources = {}
    if kind == 'reference':
        modules = {}
        for name in ('network', 'adapter'):
            source = subprocess.check_output(['git', 'show', f'2e3b967:src/xdiyo_analytics/neural/{name}.py'], cwd=ROOT)
            sources[name] = hashlib.sha256(source).hexdigest()
            module = types.ModuleType('xdiyo_analytics.neural._frozen_'+name)
            module.__package__ = 'xdiyo_analytics.neural'
            exec(compile(source, name, 'exec'), module.__dict__); modules[name] = module
        modules['adapter'].make_network = lambda spec, inputs, outputs: modules['network'].HeatmapNetwork(spec, len(inputs.group_names), inputs.size, outputs)
        Backend = modules['adapter'].CNNBackend
    else:
        from xdiyo_analytics.neural.adapter import CNNBackend as Backend
        for name in ('adapter', 'network', 'prepared'):
            sources[name] = hashlib.sha256((ROOT/f'src/xdiyo_analytics/neural/{name}.py').read_bytes()).hexdigest()
    rng = np.random.default_rng(19)
    maps = rng.random((2000, 2, 10, 10)); maps /= maps.sum(axis=(2,3), keepdims=True)
    maps[::13,0] = np.nan; maps[::17,1] = np.nan
    columns = [f'{side}::recent::heatmap_for_mass_10_y{y:02}_x{x:02}' for side in ('home','away') for y in range(10) for x in range(10)]
    X = pd.DataFrame(maps.reshape(2000,-1), columns=columns)
    c = FitContext(X, pd.DataFrame(index=X.index), 'match', ('event_id',), 0, y=pd.DataFrame({'target':np.arange(2000)%3}))
    spec = HeatmapCNNModel(encoder=HeatmapCNNConfig(reconstruction=Reconstruction(loss='kl')), batch_size=128)
    process = psutil.Process(); rss_before = process.memory_info().rss
    t = time.perf_counter(); backend = Backend(spec, 11, 'cpu'); backend.initialize(c)
    cold = time.perf_counter()-t
    timings, metrics, digests = [], [], []
    for _ in range(5):
        t = time.perf_counter(); values = backend.step(); values = backend.checkpoint_metrics('train_loss', values)
        timings.append(time.perf_counter()-t); metrics.append(values)
        h = hashlib.sha256()
        def hash_state(value):
            if isinstance(value, torch.Tensor):
                h.update(str((value.dtype,tuple(value.shape))).encode()); h.update(value.detach().cpu().numpy().tobytes())
            elif isinstance(value, dict):
                for key, item in value.items(): h.update(repr(key).encode()); hash_state(item)
            elif isinstance(value, (list,tuple)):
                for item in value: hash_state(item)
            else: h.update(repr(value).encode())
        hash_state(backend.model.state_dict()); hash_state(backend.optimizer.state_dict())
        hash_state([p.grad for p in backend.model.parameters()]); digests.append(h.hexdigest())
    info = process.memory_info()
    result = dict(kind=kind, sources=sources, candidate_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), command=[sys.executable, *sys.argv], python=sys.version, torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__,
                  rows=len(X), maps_shape=list(maps.shape), inputs_sha256=hashlib.sha256(maps.tobytes()).hexdigest(),
                  cold_seconds=cold, update_and_post_evaluation_seconds=timings, warm_median_seconds=statistics.median(timings[1:]),
                  metrics=metrics, per_update_state_sha256=digests, rss_before=rss_before, rss_after=info.rss,
                  peak_working_set=getattr(info,'peak_wset',None), gc_tracked_objects=len(gc.get_objects()),
                  cuda_allocated=None, cuda_reserved=None, file_cache=None)
    Path(output).write_text(json.dumps(result, indent=2),encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--kind', choices=['reference','candidate']); parser.add_argument('--output', required=True)
    args=parser.parse_args(); worker(args.kind,args.output)
