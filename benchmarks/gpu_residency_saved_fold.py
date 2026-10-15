"""Isolated saved-fold CUDA residency benchmark. Refuses to overlap native GPU workers."""
from pathlib import Path
import argparse,gc,hashlib,json,os,shutil,statistics,subprocess,sys,tempfile,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/analytics')]
import numpy as np
import pandas as pd
import psutil


def digest_tree(value):
    import torch
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.detach().cpu().numpy().tobytes())
        elif isinstance(v,np.ndarray):
            h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k,x in v.items():h.update(repr(k).encode());visit(x)
        elif isinstance(v,(list,tuple)):
            for x in v:visit(x)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()


def run(args):
    if os.environ.get('XDIYO_CUDA_CHECKS')!='1':raise RuntimeError('Explicit free-GPU gate required')
    for process in psutil.process_iter(['pid','cmdline']):
        if process.pid==os.getpid():continue
        if any('gpu_worker.py' in s for s in (process.info['cmdline'] or [])):
            raise RuntimeError(f'Active native GPU worker {process.pid}; wait without interruption')
    import torch
    from dataclasses import replace
    from xdiyo_analytics.graphs import TemporalGraphSerializer
    from xdiyo_analytics.graphs.adapter import _Backend
    from xdiyo_analytics.training.contracts import FitContext
    from xdiyo_analytics.training.residency import InputCachePolicy
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    fold=Path(args.fold);model=fold/'model'
    manifest=json.loads((model/'MODEL_COMMITTED.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files'].items():
        if hashlib.sha256((model/name).read_bytes()).hexdigest()!=expected:raise ValueError(f'Frozen model hash mismatch: {name}')
    metadata_sha=hashlib.sha256((fold/'train_metadata.parquet').read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='xdiyo-isolated-residency-') as temporary:
        isolated=Path(temporary)/'model';shutil.copytree(model,isolated)
        shutil.copy2(fold/'train_metadata.parquet',Path(temporary)/'train_metadata.parquet')
        for name,expected in manifest['files'].items():
            assert hashlib.sha256((isolated/name).read_bytes()).hexdigest()==expected
        loaded=TemporalGraphSerializer(device='cpu').load(isolated)
        assert loaded.backend_.model_version==manifest['model_version']
        cfg=replace(loaded.config,deterministic=True)
        store=loaded.event_store
        metadata=pd.read_parquet(Path(temporary)/'train_metadata.parquet')
        X=metadata.loc[:,list(cfg.fixture_features)].copy()
        y=pd.DataFrame({'result':np.where(metadata.goals_for>metadata.goals_against,1,np.where(metadata.goals_for<metadata.goals_against,-1,0))},index=metadata.index)
        context=FitContext(X,metadata,'match',('competition_id','season_id','event_id'),2,y=y)
        del loaded
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        process=psutil.Process();rss_before=process.memory_info().rss
        torch.cuda.synchronize();start=time.perf_counter()
        backend=_Backend(cfg,store,17,'cuda:0',InputCachePolicy(args.cache_bytes,args.headroom_bytes));backend.initialize(context)
        torch.cuda.synchronize();cold=time.perf_counter()-start
        # Evidence pass uses the uncached native stream, so it does not warm GPU residency.
        prepared=hashlib.sha256()
        for i in range(len(backend.train_groups)):
            prepared.update(digest_tree(backend.train_groups[i]).encode())
        timings=[];metrics=[];states=[]
        for _ in range(args.updates):
            torch.cuda.synchronize();start=time.perf_counter()
            value=backend.step();value=backend.checkpoint_metrics('train_loss',value)
            torch.cuda.synchronize();timings.append(time.perf_counter()-start);metrics.append(value)
            states.append(digest_tree((backend.model.state_dict(),backend.optimizer.state_dict(),[p.grad for p in backend.model.parameters()])))
        pool=backend.train_groups._resident.pool
        info=process.memory_info()
        from xdiyo_analytics.experiments.recovery import training_source_files
        source=ROOT/'src/xdiyo_analytics'
        receipt=dict(command=[sys.executable,*sys.argv],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
            source_files={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in training_source_files(source)},
            frozen_model_manifest=manifest,train_metadata_sha256=metadata_sha,rows=len(metadata),issuances=len(backend.train_groups),
            fixture_features=list(cfg.fixture_features),prepared_sha256=prepared.hexdigest(),cache_bytes=args.cache_bytes,headroom_bytes=args.headroom_bytes,
            cold_preparation_seconds=cold,update_and_post_evaluation_seconds=timings,warm_median_seconds=statistics.median(timings[1:] or timings),
            metrics=metrics,per_update_state_sha256=states,cache=None if pool is None else dict(hits=pool.hits,misses=pool.misses,resident=pool.resident,transient=pool.transient),
            python=sys.version,torch=str(torch.__version__),pandas=pd.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),
            rss_before=rss_before,rss_after=info.rss,peak_working_set=getattr(info,'peak_wset',None),live_gc_objects=len(gc.get_objects()),
            cuda_allocated_peak=torch.cuda.max_memory_allocated(),cuda_reserved_peak=torch.cuda.max_memory_reserved(),file_cache=None,
            notes='Same candidate code with residency disabled/enabled isolates cache effect. Initialized fresh using saved history and fixture population, not an experiment continuation. Timing excludes verification hashes. OS file-cache attribution unavailable.')
        Path(args.output).write_text(json.dumps(receipt,indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True);p.add_argument('--cache-bytes',type=int,default=0)
    p.add_argument('--headroom-bytes',type=int,default=2*1024**3);p.add_argument('--updates',type=int,default=3);p.add_argument('--output',required=True)
    run(p.parse_args())
