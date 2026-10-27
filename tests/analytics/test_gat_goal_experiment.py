"""Small frozen-worker and diagnostic checks before the seven real-data fits."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'examples'))
import compare_gat_goals as experiment
from test_graph_goal_auxiliary import backend
from test_temporal_graph import sample,config
from xdiyo_analytics.graphs import GoalAuxiliary


def test_full_population_gradient_diagnostics_are_read_only_and_handle_zero_norms():
    for aux in (GoalAuxiliary(),GoalAuxiliary(detach_encoder=True),GoalAuxiliary(beta=0.)):
        b=backend(aux)
        b.step()
        before=experiment.digest_state((b.model.state_dict(),b.optimizer.state_dict(),[p.grad for p in b.model.parameters()]))
        result=experiment.gradient_diagnostics(b)
        assert result['state_unchanged'] and result['outcome_norm']>0
        assert all(not n.startswith(('head.','goal_head.')) for n in result['parameters'])
        assert result['fixtures']==3 and not result['regularizer_included']
        if aux.detach_encoder:
            assert result['goal_norm']==0 and result['weighted_gradient_ratio']==0
            assert result['cosine_similarity'] is None and result['zero_norms']==['goal']
        elif aux.beta==0:
            assert result['goal_norm']>0 and result['weighted_gradient_ratio']==0
        else:
            assert result['weighted_gradient_ratio']>0 and -1<=result['cosine_similarity']<=1
        assert experiment.digest_state((b.model.state_dict(),b.optimizer.state_dict(),[p.grad for p in b.model.parameters()]))==before
    with torch.no_grad():
        for p in b.model.head.parameters(): p.zero_()
    result=experiment.gradient_diagnostics(b)
    assert result['outcome_norm']==0 and result['weighted_gradient_ratio'] is None
    assert result['cosine_similarity'] is None and 'outcome' in result['zero_norms']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_frozen_cuda_workers_copy_initial_state_restore_and_exclude_test(tmp_path):
    store=sample()
    frames={'train':store.schedule.iloc[[2,3,4]],'validation':store.schedule.iloc[[6,7]]}
    for name,frame in [('history',store.history),('memberships',store.team_seasons),('schedule',store.schedule),*frames.items()]:
        frame.to_parquet(tmp_path/f'{name}.parquet',index=True)
    cfg=config(deterministic=True)
    p=dict(source_hashes=experiment.source_hashes(),data_hashes={n:experiment.base.digest(tmp_path/n) for n in experiment.DATA},
        provenance=store.provenance,store_version=store.version,
        populations={name:dict(identity_sha256=experiment.base.frame_hash(frame)) for name,frame in frames.items()},
        config=asdict(cfg),arms={name:asdict(aux) for name,aux in experiment.ARMS.items()},
        matrix=[dict(arm=arm,seed=17) for arm in experiment.ARMS],control=asdict(experiment.control(17,2)),
        device='cuda:0',input_cache_bytes=16*2**20,input_cache_headroom_bytes=2**30,
        worker_rss_limit_gib=4.,cuda_allocator_config='garbage_collection_threshold:0.6',cuda_memory_fraction=.8)
    experiment.base.write_json(tmp_path/'protocol.json',p)
    env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    def run(command,arm=None):
        args=[sys.executable,str(ROOT/'examples/compare_gat_goals.py'),command,'--run',str(tmp_path),'--seed','17']
        if arm: args+=['--arm',arm]
        completed=subprocess.run(args,cwd=ROOT,env=env,text=True,capture_output=True,timeout=90)
        assert completed.returncode==0,completed.stdout+completed.stderr
    run('initialize')
    for arm in experiment.ARMS:
        run('fit',arm)
        r=json.loads((tmp_path/f'fit/{arm}-17/result.json').read_text())
        assert r['summary']['attempts'][0]['steps']==2
        assert r['reload_exact'] and r['public_boundary_enforced']
        assert r['initial']['state_unchanged'] and r['selected']['state_unchanged']
        assert r['validation']['decomposition_error']<1e-12
    for split in ('train','validation'):
        a=pd.read_parquet(tmp_path/f'fit/control-17/{split}_predictions.parquet')
        b=pd.read_parquet(tmp_path/f'fit/detached-17/{split}_predictions.parquet')
        np.testing.assert_allclose(a[['p_home','p_draw','p_away']],b[['p_home','p_draw','p_away']],rtol=1e-5,atol=1e-6)
    assert not list(tmp_path.rglob('*test*predictions*'))
    # Same-width configuration edits cannot silently alter a frozen worker.
    p['data_hashes']['train.parquet']='changed'
    experiment.base.write_json(tmp_path/'protocol.json',p)
    with pytest.raises(ValueError,match='Frozen data'): experiment.load(tmp_path)
