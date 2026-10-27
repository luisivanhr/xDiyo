"""Disposable end-to-end worker and independent report/paired-statistic checks."""
from pathlib import Path
from dataclasses import asdict
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from test_temporal_graph import sample,context
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'examples'))
from gat_suite import core,metrics,reporting
from xdiyo_analytics.graphs.adapter import _Backend
from compare_gat_temporal import write_json,digest


def test_paired_bootstrap_averages_seed_losses_not_probabilities(tmp_path):
    run=tmp_path;folder=run/'shared/bootstrap/fold_1';folder.mkdir(parents=True)
    counts=np.array([[1,1,1,1],[2,0,2,0],[0,1,2,1],[1,0,0,3]],dtype=np.int16)
    np.save(folder/'blocks_4.npy',counts);write_json(folder/'rounds.json',[1,2,3,4])
    y=np.array([0,1,0,1,2,1,2,1]);rounds=np.repeat([1,2,3,4],2);entries=[]
    for seed in (17,42,73):
        p=np.random.default_rng(seed).dirichlet(np.ones(3),size=8)
        frame=pd.DataFrame(dict(y=y,round=rounds));frame[metrics.P_COLUMNS]=p
        for name,value in metrics.outcome_rows(p,y).items():frame[f'raw_{name}']=value
        path=run/f'{seed}.parquet';frame.to_parquet(path,index=False);entries.append((frame,path))
    point,samples,per_seed,fixture=reporting.aggregate_sample(run,'fold_1','test','raw',entries,4)
    expected=np.mean([metrics.outcome_metrics(p[metrics.P_COLUMNS],y)['log_loss'] for p,_ in entries])
    np.testing.assert_allclose(point[0],expected,rtol=1e-14)
    ensemble=np.mean([p[metrics.P_COLUMNS].to_numpy() for p,_ in entries],axis=0)
    assert not np.isclose(point[0],metrics.outcome_metrics(ensemble,y)['log_loss'])
    for i,count in enumerate(counts):
        ix=np.repeat(np.arange(8),count[rounds-1])
        scores=[metrics.outcome_metrics(p.iloc[ix][metrics.P_COLUMNS],y[ix]) for p,_ in entries]
        for j,name in enumerate([*metrics.ADDITIVE,'conditional_winner_log_loss','draw_auc','draw_ap']):
            np.testing.assert_allclose(samples[i,j],np.mean([s[name] for s in scores]),rtol=1e-13,atol=1e-13)


def test_fresh_worker_full_export_restore_and_manifest_rejection(tmp_path):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    run=tmp_path/'disposable_suite';data=run/'shared/data';data.mkdir(parents=True)
    store=sample()
    for name,value in [('history',store.history),('memberships',store.team_seasons),('schedule',store.schedule)]:value.to_parquet(data/f'{name}.parquet',index=False)
    populations={k:context(store,positions).metadata.reset_index(drop=True) for k,positions in dict(recent=[2,3],expanded=[1,2,3],inner=[6,7],outer=[8]).items()}
    for key,frame in populations.items():frame.to_parquet(data/f'fold_1_{key}.parquet',index=False)
    cfg=core.config(competitions=(10,));backend=_Backend(cfg,store,17,'cpu');backend.initialize(context(store,[2,3]))
    pre=run/'shared/preprocessing';pre.mkdir();write_json(pre/'fold_1.json',dict(state=backend.preprocessor.state()))
    folder=run/'shared/initialization/fold_1/snapshot/seed_17';folder.mkdir(parents=True)
    structured=core.config('snapshot','G_shared_gamma',(10,));model=core.initialize_state(structured,backend.preprocessor,17)
    state=model.state_dict()
    for name,value in backend.model.state_dict().items():
        if not name.startswith('goal_head.'):state[name]=value
    model.load_state_dict(state)
    model.goal_head.initialize_means(torch.tensor(populations['expanded'].competition_id.to_numpy(dtype=np.int64)),
        torch.tensor(populations['expanded'][['goals_for','goals_against']].to_numpy(float)))
    torch.save(model.state_dict(),folder/'G_shared_gamma.pt')
    job=dict(job_id='smoke',study=4,fold='fold_1',mode='snapshot',arm='G_shared_gamma',seed=17,path='04_shared_random_effects/smoke',status='planned',provenance='disposable')
    p=dict(disposable=True,provenance=store.provenance,store_version=store.version,competitions=[10],matrix=[job],
        resources=dict(device='cuda:0',threads=1,input_cache_bytes=32*2**20,headroom_bytes=2**30,rss_limit_gib=6.,cuda_fraction=.8))
    write_json(run/'MASTER_PROTOCOL.json',p);write_json(run/'SOURCE_MANIFEST.json',{})
    (run/'frozen_source.zip').write_bytes(b'disposable test identity only')
    write_json(run/'FREEZE.json',dict(master_sha256=digest(run/'MASTER_PROTOCOL.json'),shared_hashes={}))
    environment=os.environ.copy();environment['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    result=subprocess.run([sys.executable,str(core.ROOT/'examples/run_gat_suite.py'),'fit','--run',str(run),'--job','smoke'],
        text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=environment,timeout=120)
    assert result.returncode==0,result.stdout
    receipt,preds,curve=reporting.audited_model(run,job)
    assert len(preds['outer'])==1 and receipt['reload_exact']
    assert (run/job['path']/'centering.json').exists()
    from gat_suite.fitting import fit
    with pytest.raises(ValueError,match='absent'):fit(run,'undeclared')


def test_reports_import_and_render_tables_and_assets(tmp_path):
    source=reporting.table(pd.DataFrame([dict(a='line',b=.123456789)]))
    reporting.html('# A report\n\n'+source,tmp_path/'report.html')
    text=(tmp_path/'report.html').read_text()
    assert '<table>' in text and '0.123457' in text
