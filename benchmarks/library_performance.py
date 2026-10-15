"""Bounded CPU comparisons against the audited commit; never opens experiment runs.

Run from the checkout: python benchmarks/library_performance.py --output receipt.json
Timings exclude fixture construction/imports. Allocation and RSS probes are separate
from timing; RSS is allocator-dependent within this process, not isolated peak RAM.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import tracemalloc
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests/analytics')]
BASE = '01617da9f414d8929f4a2afd5d6ccd35d2d78512'
SOURCES = {}


def reference(relative):
    name = 'xdiyo_analytics.'+relative.replace('/', '.').removesuffix('.py')
    source = subprocess.check_output(['git','show',BASE+':src/xdiyo_analytics/'+relative], cwd=ROOT)
    SOURCES[relative] = hashlib.sha256(source).hexdigest()
    module = types.ModuleType(name.rsplit('.',1)[0]+'._perf_reference_'+name.rsplit('.',1)[1])
    module.__package__ = name.rsplit('.',1)[0]
    sys.modules[module.__name__] = module
    exec(compile(source,BASE+':'+relative,'exec'),module.__dict__)
    return module


def measure(call):
    import psutil
    call()  # warm imports/allocators
    samples=[]
    for _ in range(3):
        gc.collect();start=time.perf_counter();value=call();samples.append(time.perf_counter()-start);del value
    gc.collect();tracemalloc.start();value=call();_,peak=tracemalloc.get_traced_memory();tracemalloc.stop();del value
    gc.collect();process=psutil.Process();initial=process.memory_info().rss;rss=[initial];stop=threading.Event()
    def sample():
        while not stop.wait(.002):rss.append(process.memory_info().rss)
    thread=threading.Thread(target=sample);thread.start()
    try:value=call();rss.append(process.memory_info().rss)
    finally:stop.set();thread.join()
    return dict(seconds=statistics.median(samples),samples=samples,tracemalloc_peak_bytes=peak,
                rss_initial_bytes=initial,rss_sampled_peak_bytes=max(rss)),value


def main(output):
    import numpy as np
    import pandas as pd
    import torch
    from xdiyo_analytics.features.history import _eligible_history_index
    from xdiyo_analytics.features import evaluate_features, EMA, MatchScore, RollingMean
    from xdiyo_analytics.experiments import recovery
    from xdiyo_analytics.data import load_season
    from xdiyo_analytics.histories import build_team_history
    from xdiyo_analytics.splits.cpcv import _closed_overlap
    from transition_samples import history_from_games, STAT
    baseline_history=reference('features/history.py')
    baseline_evaluation=reference('features/evaluation.py')
    baseline_evaluation.eligible_history_rows=baseline_history.eligible_history_rows
    baseline_recovery=reference('experiments/recovery.py')
    baseline_bayesian=reference('ratings/bayesian.py')
    baseline_bayesian.RatingRun=reference('ratings/snapshots.py').RatingRun
    from xdiyo_analytics.ratings import build_bayesian_ratings
    from test_bayesian_training import synthetic_history
    rating_history=synthetic_history(leagues=(10,20,30,40),seasons=1,matches=24)
    rating_run=build_bayesian_ratings(rating_history)
    history=history_from_games([dict(day=i, values=(float(i%11),float(i%7))) for i in range(1600)])
    real=build_team_history(load_season(ROOT/'data/xDiyo_data','Premier_League_10_11',verify_hashes=True))
    features={'mean':RollingMean(MatchScore(),window=20), 'ema':EMA(MatchScore(),span=20)}
    data=pd.DataFrame(np.arange(200000,dtype=float).reshape(20000,10));data.attrs={'coverage':list(range(1000))}
    start=pd.date_range('2024-01-01',periods=2000,freq='h',tz='UTC');end=start+pd.Timedelta(minutes=20)
    merged=list(zip(start,end))
    def old_intervals():
        mask=np.zeros(len(start),dtype=bool)
        for a,b in merged:mask |= (start <= b) & (end >= a)
        return mask
    import tempfile
    temporary = tempfile.TemporaryDirectory(prefix='xdiyo-recovery-bench-')
    def save(module, name):
        path=Path(temporary.name)/name
        path.unlink(missing_ok=True)
        module.dump_bundle(path,data)
        return path
    cases={
      'bayesian_four_league_lookup':(lambda:baseline_bayesian.BayesianRatingRun.features(rating_run,rating_history), lambda:rating_run.features(rating_history)),
      'bounded_history_3200_rows':(lambda:[x[-20:] for x in baseline_history.eligible_history_rows(history)],
         lambda:bounded(history)),
      'long_ema_3200_rows':(lambda:baseline_evaluation.evaluate_features(history,{'ema':EMA(STAT,span=20)}), lambda:evaluate_features(history,{'ema':EMA(STAT,span=20)})),
      'real_premier_2010_11_mean_ema':(lambda:baseline_evaluation.evaluate_features(real,features),lambda:evaluate_features(real,features)),
      'recovery_20000_by_10':(lambda:save(baseline_recovery,'reference.json'),lambda:save(recovery,'candidate.json')),
      'cpcv_2000_by_2000':(old_intervals,lambda:_closed_overlap(start,end,merged)),
    }
    records={}
    for name,(old,new) in cases.items():
        a,expected=measure(old);b,actual=measure(new)
        if isinstance(expected,pd.DataFrame):
            pd.testing.assert_frame_equal(actual,expected,check_exact=True)
            assert actual.attrs == expected.attrs
        elif isinstance(expected,list):
            for x,y in zip(actual,expected):np.testing.assert_array_equal(x,y)
        elif isinstance(expected,Path):assert actual.read_bytes()==expected.read_bytes()
        elif isinstance(expected,str):assert actual==expected
        else:np.testing.assert_array_equal(actual,expected)
        records[name]={'reference':a,'candidate':b,'speedup':a['seconds']/b['seconds'],'exact_parity':True}
    # Source provenance includes new private helpers as well as modified modules.
    candidate={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted((ROOT/'src/xdiyo_analytics').rglob('*.py'))}
    manifests=[ROOT/'data/xDiyo_data/Premier_League_10_11.manifest.json']
    receipt=dict(base=BASE,command=sys.argv,interpreter=sys.executable,python=sys.version,
        candidate_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        benchmark_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        packages={name:importlib.metadata.version(name) for name in ('numpy','pandas','scipy','scikit-learn','torch','pyarrow','psutil')},
        cuda_available=torch.cuda.is_available(),cuda_build=torch.version.cuda,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        limits='CPU-only bounded probes; no training/update/full-fold/GPU speedup or resume claim. RSS includes allocator reuse.',
        real_fixture_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in manifests},
        synthetic_fixture_hash=hashlib.sha256(history.to_json().encode()).hexdigest(),
        rating_fixture_hash=hashlib.sha256(rating_history.to_json().encode()).hexdigest(),
        reference_source_hashes=SOURCES,candidate_source_hashes=candidate,results=records)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({name:{'speedup':r['speedup'],'reference_seconds':r['reference']['seconds'],
        'candidate_seconds':r['candidate']['seconds'],'reference_allocated_peak':r['reference']['tracemalloc_peak_bytes'],
        'candidate_allocated_peak':r['candidate']['tracemalloc_peak_bytes']} for name,r in records.items()},indent=2))


def bounded(history):
    from xdiyo_analytics.features.history import _eligible_history_index
    index=_eligible_history_index(history)
    return [index.window(i,20) for i in range(len(history))]


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
