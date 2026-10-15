"""Reproducible CPU identity/search/index/retention measurements on disposable data."""
from pathlib import Path
import argparse, gc, hashlib, json, os, platform, statistics, subprocess, sys, tempfile, time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/analytics'),str(ROOT)]
import numpy as np
import pandas as pd
import psutil
from xdiyo_analytics.experiments.recovery import execution_key, training_source_files
from xdiyo_analytics.experiments.frozen import FrozenExecution
from xdiyo_analytics.experiments.run_index import RunIndex
from test_performance_remaining import record

def measure(fn, count=1):
    timings=[]; value=None
    for _ in range(count):
        start=time.perf_counter(); value=fn(); timings.append(time.perf_counter()-start)
    return value, dict(seconds=timings, median_seconds=statistics.median(timings))

def run(output):
    process=psutil.Process(); start_rss=process.memory_info().rss
    rng=np.random.default_rng(82)
    frame=pd.DataFrame(rng.normal(size=(20000,24)),columns=[f'feature_{i}' for i in range(24)])
    frame.iloc[::19,0]=np.nan
    candidates=[dict(alpha=x) for x in (.001,.01,.1,1,10,100)]
    reference, ref=measure(lambda:[execution_key(frame,c) for c in candidates])
    snapshot, cold=measure(lambda:FrozenExecution(frame))
    actual, warm=measure(lambda:[snapshot.key(None,c,shared_positions=((0,0),)) for c in candidates],3)
    assert reference==actual
    with tempfile.TemporaryDirectory(prefix='xdiyo-completion-benchmark-') as temp:
        root=Path(temp)
        for i in range(600): record(root,i,key=str(i%9))
        index=RunIndex(root,'e')
        _, index_cold=measure(lambda:index.read('trial',key='4',completed=True))
        def legacy():
            rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'runs').glob('*/run.json'))]
            rows=[r for r in rows if r['recovery_key']=='4' and r['status']=='complete']
            return sorted(rows,key=lambda r:(r['created_at'],r['run_id']))[-1]
        legacy_value, legacy_times=measure(legacy,5)
        indexed_value, indexed_times=measure(lambda:index.read('trial',key='4',completed=True),5)
        assert legacy_value==indexed_value
        from football_experiment_samples import prepared,ridge,post
        from model_selection_samples import plan
        from xdiyo_analytics.experiments import FootballExperiment
        from xdiyo_analytics.selection import ModelSelection
        from xdiyo_analytics.ui.server import BuilderState, render_result_report
        from xdiyo_analytics.ui.retention import payload_bytes
        bundle, preparation=measure(lambda:prepared(holdout=True))
        experiment=FootballExperiment('completion-benchmark',output_dir=root/'artifacts')
        search=lambda:ModelSelection([ridge(a) for a in (.01,.1,1)],metrics='mse')
        # Candidate names must be distinct while their native configurations stay exact.
        from dataclasses import replace
        def selection(): return ModelSelection([replace(ridge(a),name=f'Ridge {a}') for a in (.01,.1,1)],metrics='mse')
        result, fit=measure(lambda:experiment.run(bundle,model_selection=selection(),selection_plan=plan(bundle.dataset),post_analysis=post()))
        loaded, reload_time=measure(lambda:experiment.load(result.record['run_id']))
        _, reuse=measure(lambda:experiment.run(bundle,model_selection=selection(),selection_plan=plan(bundle.dataset),post_analysis=post()))
        state=BuilderState(root,None,retention_bytes=0)
        recipe={'name':'completion-benchmark','output_dir':str(root/'artifacts')}
        job=dict(object=result,recipe=recipe,status='complete',action='run'); state.jobs['run']=job
        _, render=measure(lambda:state._report(job))
        cached_html, cached=measure(lambda:state._report(job),3)
        assert cached_html==render_result_report(result,recipe,spatial_transport=state.spatial_transport())
        before=payload_bytes(job['object'])+payload_bytes(job.get('html',''))
        _, eviction=measure(state._trim_results)
        assert 'object' not in job,job.get('retention')
        after=payload_bytes(job.get('object'))+payload_bytes(job.get('html',''))
        html_after, reopen=measure(lambda:state._report(job))
        assert html_after==cached_html
        _, export=measure(lambda:state.dispatch('save_model',{'id':'run','path':str(root/'export'),'fold_id':0}))
        state.executor.shutdown()
        lifecycle=dict(preparation=preparation,three_candidate_two_inner_fold_fit=fit,reload=reload_time,reuse=reuse,
            render=render,cached_document=cached,verified_eviction=eviction,reopen_after_eviction=reopen,model_export=export,
            retained_payload_before=before,retained_payload_after=after,
            qualification='Synthetic 16-fixture lifecycle smoke benchmark; not a realistic training speed claim.')
    info=process.memory_info()
    source=ROOT/'src/xdiyo_analytics'
    receipt=dict(command=[sys.executable,*sys.argv],commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
        sources={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in training_source_files(source)},
        python=sys.version,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),
        identity=dict(shape=list(frame.shape),candidates=len(candidates),reference=ref,snapshot_construction=cold,frozen_keys=warm,exact_keys=True),
        run_index=dict(manifests=600,cold=index_cold,legacy=legacy_times,indexed=indexed_times,exact_winner=True),lifecycle=lifecycle,
        rss_before=start_rss,rss_after=info.rss,peak_working_set=getattr(info,'peak_wset',None),live_gc_objects=len(gc.get_objects()),
        file_cache=None,file_cache_note='OS file cache cannot be attributed to this process.',cuda_allocated=None,cuda_reserved=None)
    Path(output).write_text(json.dumps(receipt,indent=2),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);run(parser.parse_args().output)
