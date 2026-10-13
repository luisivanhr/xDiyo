"""Bounded, disposable speed audit of a completed fold (never writes to it).

Run reference/native in separate processes with identical --fold/--device.
Reads saved preprocessing and weights, makes two disposable full updates from
the same saved weights and a fresh AdamW, and discards them. This is not a study
continuation. Input hashes are rechecked after the benchmark. The reference is
frozen 11ffa446 from tests/analytics/reference. Requires psutil and torch.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import platform
import os
import sys
import threading
import time
import zipfile
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'tests/analytics')]


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=Path, required=True)
    parser.add_argument('--mode', choices=('native','reference'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--profile', action='store_true', help='One extra instrumented update to measure overhead.')
    parser.add_argument('--cold-pass', action='store_true', help='Time one complete preparation pass before warming samples or updating.')
    parser.add_argument('--deterministic', action='store_true', help='Require deterministic CUDA operations for exact paired updates.')
    parser.add_argument('--native-deterministic', action='store_true', help='Exercise the native opt-in with caller deterministic flags disabled (native mode only).')
    parser.add_argument('--evidence', type=Path, help='Ayre support ZIP containing pinned input/source hashes.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.native_deterministic and (args.mode != 'native' or args.deterministic):
        parser.error('--native-deterministic requires native mode without --deterministic.')
    if args.repeats < 1 or args.repeats > 3:
        parser.error('Use 1 to 3 bounded full-update repeats.')
    if args.output.resolve().is_relative_to(args.fold.resolve()):
        parser.error('Output must be outside the completed fold.')
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import pandas as pd
    import psutil
    import torch
    from test_graph_speed import reference
    from xdiyo_analytics.graphs.config import TemporalGraphConfig, CLASS_ORDER
    from xdiyo_analytics.graphs.execution import prepare_determinism
    if args.native_deterministic:
        prepare_determinism(TemporalGraphConfig('UTC', deterministic=True), args.device)
    from xdiyo_analytics.graphs.network import TemporalNetwork
    from xdiyo_analytics.graphs import GraphProfile
    from xdiyo_analytics.training.contracts import FitContext
    from xdiyo_analytics.graphs import adapter, store, preprocessing, streaming
    modules = reference() if args.mode == 'reference' else dict(adapter=adapter,store=store,preprocessing=preprocessing,streaming=streaming)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    paths = [args.fold/'train_metadata.parquet', *[args.fold/'model'/n for n in
             ('graph.json','history.parquet','team_seasons.parquet','schedule.parquet','weights.pt')]]
    hashes = {str(p.relative_to(args.fold)):digest(p) for p in paths}
    commitment = json.loads((args.fold/'model/MODEL_COMMITTED.json').read_text())
    for name, expected in commitment['files'].items():
        path = (args.fold/'model'/name).resolve()
        if not path.is_relative_to((args.fold/'model').resolve()) or digest(path) != expected:
            raise ValueError(f'Saved model commitment mismatch: {name}')
    if args.evidence:
        with zipfile.ZipFile(args.evidence) as archive:
            evidence = json.loads(archive.read('support/RESULTS.json'))
        for name, expected in evidence['input_files'].items():
            relative = name.split('/refit-fold-02/')[1]
            assert digest(args.fold/relative) == expected, relative
        for name, expected in evidence['source_files'].items():
            assert digest(ROOT/'tests/analytics/reference'/f'gat_{Path(name).stem}_11ffa446.txt') == expected
    process = psutil.Process()
    rss = {'baseline': process.memory_info().rss, 'peak':process.memory_info().rss}
    stop = threading.Event()
    def sample():
        while not stop.wait(.01):
            rss['peak'] = max(rss['peak'],process.memory_info().rss)
    sampler = threading.Thread(target=sample,daemon=True); sampler.start()
    if args.device.startswith('cuda'):
        torch.cuda.synchronize(args.device)  # Initialize the allocator before resetting counters (Torch 2.9).
        torch.cuda.reset_peak_memory_stats(args.device)
    setup_start = time.perf_counter()
    meta = json.loads((args.fold/'model/graph.json').read_text())
    if meta['config']['heatmaps'] is not None or meta['config']['fixture_features']:
        raise ValueError('This audit harness expects the supplied scalar expanded-totals fold.')
    cfg = TemporalGraphConfig(**meta['config'])
    event_store = modules['store'].TemporalEventStore(pd.read_parquet(args.fold/'model/history.parquet'),
        pd.read_parquet(args.fold/'model/team_seasons.parquet'), schedule=pd.read_parquet(args.fold/'model/schedule.parquet'))
    assert event_store.version == meta['event_store_version']
    frame = pd.read_parquet(args.fold/'train_metadata.parquet')
    ctx = FitContext(X=pd.DataFrame(index=frame.index), metadata=frame, layout='match', match_columns=(), fold_id=2,
                     y=pd.DataFrame({meta['target_name']:np.sign(frame.goals_for-frame.goals_against).astype(int)}))
    pre = modules['preprocessing'].GraphPreprocessor.from_state(cfg,meta['preprocessing'])
    assert tuple(meta['edge_feature_order']) == pre.edge_columns
    assert tuple(meta['fixture_feature_order']) == pre.fixture_columns
    plan = modules['streaming'].GroupPlan(event_store,cfg,ctx,event_store.issue_times(frame,cfg))
    groups = modules['streaming'].PreparedGroups(plan,pre,args.device)
    setup_seconds = time.perf_counter()-setup_start
    def sync():
        if args.device.startswith('cuda'):
            torch.cuda.synchronize(args.device)
    def measure(fn):
        sync(); start=time.perf_counter(); value=fn(); sync()
        return value,time.perf_counter()-start
    def tensor_digest(group):
        h=sha256()
        for t in [*group[1].values(),*group[2:]]:
            h.update(t.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    try:
        cold_pass = None
        if args.cold_pass:
            def prepare_one(index):
                group = groups[index]
                return len(group[0]), group[1]['x'].shape[0]
            def prepare_all():
                fixtures = edges = 0
                for index in range(len(groups)):
                    rows, directed = prepare_one(index)
                    fixtures += rows; edges += directed
                return dict(fixtures=fixtures, directed_edges=edges)
            population, seconds = measure(prepare_all)
            cold_pass = dict(seconds=seconds,**population)
            print(json.dumps(dict(mode=args.mode,cold_preparation=cold_pass)),flush=True)
        samples=[]
        for label,index in zip(('early','middle','late'),(0,len(plan)//2,len(plan)-1)):
            cold,cold_seconds=measure(lambda:groups[index])
            rows,width=cold[1]['x'].shape
            checksum=tensor_digest(cold); del cold
            times=[]
            for _ in range(3):
                group,seconds=measure(lambda:groups[index])
                assert tensor_digest(group)==checksum
                times.append(seconds); del group
            samples.append(dict(label=label,index=index,edges=rows,width=width,
                issuance=str(plan.descriptors[index][0]),cold_seconds=cold_seconds,
                warm_seconds=times,tensor_sha256=checksum))
        print(json.dumps(dict(mode=args.mode,samples=samples)),flush=True)
        backend=modules['adapter']._Backend(cfg,event_store,17,args.device)
        backend.preprocessor=pre
        with torch.random.fork_rng(devices=[]):
            backend.model=TemporalNetwork(cfg,len(pre.leagues),2*len(pre.edge_columns)+len(pre.leagues)+2,
                2*len(pre.fixture_columns)+len(pre.leagues)+1).to(args.device)
        saved=torch.load(args.fold/'model/weights.pt',map_location=args.device,weights_only=True)
        backend.model.load_state_dict(saved)
        backend.refresh_version()
        assert backend.model_version == meta['model_version']
        if args.native_deterministic:
            # Verify the frozen artifact first; alter only the disposable backend's
            # execution policy, never the saved file or model architecture.
            cfg = replace(cfg, deterministic=True)
            backend.config = cfg
            pre.config = cfg
        backend.output_classes=CLASS_ORDER
        backend.train_groups=groups
        backend.train_targets,backend.train_weights=backend._targets(ctx)
        backend.validation_groups=None
        records=[]; saved_states=[]
        for repeat in range(args.repeats + int(args.profile)):
            backend.model.load_state_dict(saved)
            backend.optimizer=torch.optim.AdamW(backend.model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
            profile=GraphProfile(args.device) if repeat==args.repeats else None
            rng_before=torch.get_rng_state().clone()
            cuda_rng_before=torch.cuda.get_rng_state(args.device).clone() if args.device.startswith('cuda') else None
            with profile if profile else nullcontext():
                values,training=measure(backend.step)
                print(json.dumps(dict(mode=args.mode,repeat=repeat,phase='training_complete',seconds=training)),flush=True)
                metrics,replay=measure(lambda:backend.checkpoint_metrics('train_loss',values))
            assert torch.equal(rng_before,torch.get_rng_state())
            if cuda_rng_before is not None:
                assert torch.equal(cuda_rng_before,torch.cuda.get_rng_state(args.device))
            record=dict(repeat=repeat,profiled=profile is not None,training_seconds=training,
                metric_seconds=replay,total_seconds=training+replay,values=values,metrics=metrics,
                phases=profile.timings if profile else {},rng_unchanged=True)
            records.append(record)
            if repeat==0:
                saved_states=deepcopy(dict(model={k:v.detach().cpu() for k,v in backend.model.state_dict().items()},
                    gradients=[p.grad.detach().cpu() if p.grad is not None else None for p in backend.model.parameters()],
                    optimizer=backend.optimizer.state_dict(),rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(args.device) if args.device.startswith('cuda') else None))
            print(json.dumps(dict(mode=args.mode,**record)),flush=True)
        assert hashes=={str(p.relative_to(args.fold)):digest(p) for p in paths}
        for name, expected in commitment['files'].items():
            assert digest(args.fold/'model'/name) == expected
        cache=getattr(pre,'_edge_cache',None)
        report=dict(mode=args.mode,baseline='11ffa446',input_sha256=hashes,
            command=[sys.executable,*sys.argv], executable=sys.executable,
            harness_sha256=digest(Path(__file__)),setup_seconds=setup_seconds,cold_preparation=cold_pass,
            source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'src/xdiyo_analytics/graphs').glob('*.py')},
            reference_sha256={p.name:digest(p) for p in (ROOT/'tests/analytics/reference').glob('gat_*_11ffa446.txt')},
            runtime=dict(python=platform.python_version(),torch=torch.__version__,pandas=pd.__version__,numpy=np.__version__,
                device=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'cpu',threads=1,
                deterministic=args.deterministic or args.native_deterministic,tf32=False),
            native_deterministic=args.native_deterministic,
            caller_deterministic_after=torch.are_deterministic_algorithms_enabled(),
            samples=samples,updates=records,issuances=len(plan),source_edges=len(event_store._history),rss=rss,
            peak_device_allocated=torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else 0,
            peak_device_reserved=torch.cuda.max_memory_reserved(args.device) if args.device.startswith('cuda') else 0,
            cache_bytes=cache[1].nbytes+cache[2].nbytes if cache else 0,
            inputs_unchanged=True,disposable_updates_only=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        torch.save(saved_states,args.output.with_suffix('.pt'))
    finally:
        stop.set();sampler.join()


if __name__=='__main__':
    main()
