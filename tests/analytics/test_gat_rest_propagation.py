"""Frozen matrix, semantic initialization, selection and secondary RPS intervals."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'examples'))
from gat_rest_propagation import core,analysis,selection,metrics
from gat_rest_propagation.preparation import mapped_initial
from xdiyo_analytics.graphs import TemporalGraphConfig
from xdiyo_analytics.graphs.extensions import GoalAuxiliary
from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor,MaskedScale
from xdiyo_analytics.graphs.network import TemporalNetwork

def test_matrix_and_tie_rule_are_fixed_before_results():
    jobs=core.matrix();assert len(jobs)==12 and len({j['job_id'] for j in jobs})==12
    assert all(j['configuration']==core.FULL for j in jobs[:6])
    assert all(j['configuration']==core.PRIOR for j in jobs[6:])
    records=[dict(**j,inner_micro_ce=1.2) for j in jobs]
    for fold in core.FOLDS:assert selection.choose(records,fold)['configuration']==core.PRIOR
    # Fold-local scores cannot consult the other fold's outcome.
    records[0]['inner_micro_ce']=1.1
    assert selection.choose(records,'fold_1')['configuration']==core.FULL
    assert selection.choose(records,'fold_2')['configuration']==core.PRIOR
    with pytest.raises(ValueError,match='three matched'):selection.choose(records[1:],'fold_1')
    assert len(analysis.contrasts())==6

def test_semantic_mapping_copies_only_unchanged_and_equivalent_mask_columns():
    cfg=TemporalGraphConfig('Europe/Berlin',embedding_dim=4,heads=2,hidden_width=8,goal_auxiliary=GoalAuxiliary(beta=0))
    old=GraphPreprocessor(replace(cfg,fixture_gap_encoding='legacy_standardized_gap_v1'))
    new=GraphPreprocessor(cfg)
    for pre in (old,new):
        pre.leagues={1:1,2:2};pre.fixture_scale=MaskedScale().fit(np.arange(20).reshape(4,5));pre.edge_scale=MaskedScale().fit(np.ones((4,2)))
    original=TemporalNetwork(old.config,2,old.edge_dim,old.fixture_dim)
    mapped,rows=mapped_initial(original.state_dict(),old,new)
    target=TemporalNetwork(cfg,2,new.edge_dim,new.fixture_dim);target.load_state_dict(mapped)
    assert len(rows)==new.fixture_dim
    for row in rows:
        for head in ('head.0.weight','goal_head.weight'):
            if row['old_index'] is None:assert not mapped[head][:,row['new_index']].count_nonzero()
            else:torch.testing.assert_close(mapped[head][:,row['new_index']],original.state_dict()[head][:,row['old_index']],atol=0,rtol=0)
    assert sum(row['old_index'] is None for row in rows)==4
    assert {r['old'] for r in rows if r['new'].endswith('no_eligible_history')}=={'home_rest_days__missing','away_rest_days__missing'}

def test_rps_has_independent_ordered_oracle_and_endpoint_distance():
    p=np.array([[1.,0,0],[0,1,0],[0,0,1],[.2,.3,.5]])
    y=np.array([2,2,2,1])
    expected=.5*((p[:,0]-(y==0))**2+(p[:,2]-(y==2))**2)
    np.testing.assert_array_equal(metrics.rps(p,y),expected)
    np.testing.assert_array_equal(expected[:3],[1.,.5,0.])

def test_paired_intervals_preserve_mixture_and_cover_rps(tmp_path,monkeypatch):
    (tmp_path/'03_comparisons').mkdir()
    monkeypatch.setattr(core,'load',lambda *args,**kwargs:({'competitions':[1,2]},None))
    models={}
    for f,fold in enumerate(core.FOLDS):
        folder=tmp_path/'shared/bootstrap'/fold;folder.mkdir(parents=True)
        league=np.array([1,1,2,2] if f==0 else [1,2,2,2])
        np.save(folder/'fixture_week.npy',np.arange(4))
        for length in (2,4,8):np.save(folder/f'blocks_{length}.npy',np.array([[1,1,1,1],[2,1,2,1],[1,2,1,2]]))
        for i,name in enumerate(analysis.MODELS):
            frames=[]
            for seed in range(3 if name not in ('pooled_logistic','league_logistic','league_frequency') else 1):
                frame=pd.DataFrame({'competition_id':league})
                for view in ('raw','calibrated'):
                    for m,metric in enumerate(('log_loss','rps','draw_log_loss')):
                        frame[f'{view}_{metric}']=np.arange(4)*.01+(i+1)*.02+f*.1+seed*.003+m*.05
                frames.append(frame)
            models[fold,name]=frames
    result=analysis.paired_intervals(tmp_path,models)
    assert len(result)==(6*2+1)*2*3*3*2
    assert set(result[result.metric=='draw_log_loss'].contrast)=={analysis.contrasts()[0][0]}
    for row in result[result.fold=='combined'].itertuples():
        terms=dict(analysis.contrasts())[row.contrast];expected=0
        for name,sign in terms:
            frames=[]
            for fold in core.FOLDS:
                first=models[fold,name][0][['competition_id']].copy()
                first['value']=np.mean([f[f'{row.view}_{row.metric}'] for f in models[fold,name]],axis=0);frames.append(first)
            all_rows=pd.concat(frames)
            value=all_rows.value.mean() if row.aggregation=='micro' else all_rows.groupby('competition_id').value.mean().mean()
            expected+=sign*value
        assert row.difference==pytest.approx(expected,abs=1e-14)
