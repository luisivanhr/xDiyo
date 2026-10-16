"""Bounded CPU/GPU extension benchmark on a READ-ONLY saved fold.

Fresh disposable fits, one update. This compares different models, not parity
or predictive benefit. Never writes to the source fold or a research runner.
"""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def digest(path):
    h=sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold',type=Path,required=True)
    p.add_argument('--mode',choices=['default','memory','statistics','combined'],required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.resolve().is_relative_to(args.fold.resolve()):raise ValueError('Output must be outside the saved fold.')
    import numpy as np
    import pandas as pd
    import psutil
    import torch
    from xdiyo_analytics.graphs import TemporalGraphConfig, TemporalEventStore, BoundaryMemory, StatisticsForecast, StatisticTarget
    from xdiyo_analytics.graphs.adapter import _Backend, torch_threads, inference_mode
    from xdiyo_analytics.training.contracts import FitContext, PredictionContext
    timings={};process=psutil.Process();peak=[process.memory_info().rss];stop=threading.Event()
    def watch():
        while not stop.wait(.01):peak[0]=max(peak[0],process.memory_info().rss)
    sampler=threading.Thread(target=watch,daemon=True);sampler.start()
    source_start=time.perf_counter()
    files=[args.fold/'train_metadata.parquet',*[args.fold/'model'/name for name in ('graph.json','history.parquet','team_seasons.parquet','schedule.parquet','weights.pt')]]
    hashes={str(f.relative_to(args.fold)):digest(f) for f in files}
    commitment=json.loads((args.fold/'model/MODEL_COMMITTED.json').read_text())
    for name,expected in commitment['files'].items():
        f=(args.fold/'model'/name).resolve()
        if not f.is_relative_to((args.fold/'model').resolve()) or digest(f)!=expected:raise ValueError('Saved input commitment mismatch: '+name)
    meta=json.loads((args.fold/'model/graph.json').read_text())
    cfg=TemporalGraphConfig(**meta['config'])
    if cfg.heatmaps is not None or cfg.fixture_features:raise ValueError('Requires scalar saved fold with no external fixture head features.')
    if args.mode in ('memory','combined'):cfg=replace(cfg,boundary_memory=BoundaryMemory('fixed_days',180))
    if args.mode in ('statistics','combined'):cfg=replace(cfg,statistics_forecast=StatisticsForecast(tuple(StatisticTarget('goals',side,family='poisson') for side in ('home','away')),route='combined'))
    store=TemporalEventStore(pd.read_parquet(files[2]),pd.read_parquet(files[3]),schedule=pd.read_parquet(files[4]))
    if store.version!=meta['event_store_version']:raise ValueError('Saved store identity mismatch.')
    frame=pd.read_parquet(files[0]);issue=store.issue_times(frame,cfg);times=issue.drop_duplicates().sort_values().tolist()
    def context(selected,fit=True):
        data=frame.loc[issue.isin(selected)].copy()
        kwargs=dict(X=pd.DataFrame(index=data.index),metadata=data,layout='match',match_columns=('competition_id','season_id','event_id'),fold_id=2)
        if fit:return FitContext(**kwargs,y=pd.DataFrame({meta['target_name']:np.sign(data.goals_for-data.goals_against).astype(int)},index=data.index))
        return PredictionContext(**kwargs)
    train=context(times[-80:-60]);train.validation=context(times[-40:-30]);pred=context(times[-10:],False)
    timings['source_setup']=time.perf_counter()-source_start
    if args.device.startswith('cuda'):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    def measure(name,fn):
        if args.device.startswith('cuda'):torch.cuda.synchronize()
        start=time.perf_counter();value=fn()
        if args.device.startswith('cuda'):torch.cuda.synchronize()
        timings[name]=time.perf_counter()-start
        return value
    with torch_threads(1):
        backend=_Backend(cfg,store,17,args.device)
        measure('initialize',lambda:backend.initialize(train))
        def prepare():
            replay=backend._replay(backend.train_groups)
            for i in range(len(backend.train_groups)):
                group=replay.group(backend.train_groups,i) if replay else backend.train_groups[i]
                del group
        measure('cold_preparation_pass',prepare)
        validation=backend.validation_groups;backend.validation_groups=None
        measure('training_update',backend.step);backend.validation_groups=validation
        with inference_mode(backend.model):
            measure('post_update_training_loss',lambda:backend._loss(backend.train_groups,backend.train_targets,backend.train_weights))
            measure('validation_loss',lambda:backend._loss(validation,backend.validation_targets,backend.validation_weights))
        out=measure('prediction',lambda:backend.predict(pred))
    stop.set();sampler.join()
    for f in files:
        if digest(f)!=hashes[str(f.relative_to(args.fold))]:raise ValueError('Read-only source changed during benchmark.')
    receipt=dict(mode=args.mode,device=args.device,gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,deterministic=cfg.deterministic,python=platform.python_version(),torch=torch.__version__,command=subprocess.list2cmdline([sys.executable,*sys.argv]),
        cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes={str(f.relative_to(ROOT)):digest(f) for f in sorted((ROOT/'src/xdiyo_analytics/graphs').glob('*.py'))},
        input_hashes=hashes,counts=dict(history_team_rows=len(store._history),train=len(train.X),validation=len(train.validation.X),prediction=len(pred.X),issuances=len(backend.train_groups)),
        seconds=timings,peak_rss_bytes=peak[0],peak_vram_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
        peak_vram_reserved_bytes=torch.cuda.max_memory_reserved() if args.device.startswith('cuda') else None,
        finite_predictions=bool(np.isfinite(out['predict_proba'].to_numpy()).all()),limitations='One bounded update; new modes are different models. No accuracy or optimizer-resume claim. VRAM is allocated tensor peak (not all device reservations); null for CPU.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('mode','counts','seconds','peak_rss_bytes','finite_predictions')},indent=2))

if __name__=='__main__':main()
