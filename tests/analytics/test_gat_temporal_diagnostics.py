"""Score identities and controlled interventions, independent of observed results."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'examples'))
import gat_temporal_diagnostics as diagnostic
from test_temporal_graph import sample,config


def test_binary_decomposition_and_uniform_reference():
    y=np.tile([1,0,-1],21)
    p=np.random.default_rng(18).dirichlet([2,2,2],len(y))
    values=diagnostic.score_rows(y,p)
    np.testing.assert_allclose(values['log_loss'],values['draw_log_loss']+values['winner_contribution'],atol=1e-13)
    uniform=diagnostic.prediction_summary(y,np.full((len(y),3),1/3))
    assert uniform['log_loss']==pytest.approx(np.log(3))
    assert uniform['brier']==pytest.approx(2/3)
    assert uniform['draw_brier']==pytest.approx(2/9)
    assert uniform['conditional_winner_log_loss']==pytest.approx(np.log(2))
    assert uniform['draw_auc']==.5
    assert uniform['draw_average_precision']==pytest.approx(1/3)


@pytest.mark.parametrize('bad', [[[.5,.5,.5]], [[.6,-.1,.5]], [[.5,np.nan,.5]]])
def test_invalid_probability_rejection(bad):
    with pytest.raises(ValueError,match='finite ordered'):
        diagnostic.probabilities(bad)


def test_score_cancellation_does_not_imply_equal_predictions():
    a=np.array([[.8,.1,.1],[.35,.2,.45]])
    b=np.array([[.6,.2,.2],[.2,.2,.6]])
    result=diagnostic.prediction_difference([1,-1],a,b)
    assert result['mean_log_loss_change']==pytest.approx(0,abs=1e-14)
    assert result['total_variation']['mean']>.1
    assert result['candidate_improves_count']==result['candidate_worsens_count']==1


def components(device, mode):
    import torch
    from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
    from xdiyo_analytics.graphs.network import TemporalNetwork
    store=sample()
    cfg=config(temporal_mode=mode,deterministic=True)
    snapshot=store.snapshot(pd.Timestamp('2020-02-01T12:00Z'),cfg,(1,2,3))
    pre=GraphPreprocessor(cfg).fit(store.history,(snapshot,),store.schedule,np.zeros((len(store.schedule),5)))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        model=TemporalNetwork(cfg,len(pre.leagues),pre.edge_dim,pre.fixture_dim).to(device)
    data=pre.native_tensors(snapshot,device,store)
    home=torch.tensor([0,1],device=device)
    away=torch.tensor([1,0],device=device)
    fixture=torch.zeros((2,pre.fixture_dim),device=device)
    return cfg,pre,model,data,home,away,fixture


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_matched_initial_function_gradients_and_immutable_control_inputs(device):
    torch=pytest.importorskip('torch')
    if device.startswith('cuda') and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from gat_temporal_mechanisms import copy_snapshot_function,mask_time_channels
    from xdiyo_analytics.graphs.execution import deterministic_execution
    cfg,pre,base,bdata,home,away,fixture=components(device,'snapshot')
    _,_,expanded,data,_,_,_=components(device,'time_features')
    copy_snapshot_function(expanded,base,pre.edge_dim)
    original=data['x'].clone()
    with deterministic_execution(cfg,device):
        ref=base(bdata,home,away,fixture)
        actual=expanded(data,home,away,fixture)
        torch.testing.assert_close(actual,ref,atol=1e-6,rtol=1e-5)
        loss=torch.nn.functional.cross_entropy(actual,torch.tensor([0,2],device=device))
        loss.backward()
        assert sum(float(p.weight.grad[:,pre.edge_dim:].abs().sum()) for l in expanded.layers for p in (l.e,l.u))>0
        expanded.zero_grad(set_to_none=True)
        handles=mask_time_channels(expanded,pre.edge_dim)
        masked=expanded(data,home,away,fixture)
        torch.testing.assert_close(masked,actual,atol=0,rtol=0)
        torch.nn.functional.cross_entropy(masked,torch.tensor([0,2],device=device)).backward()
        assert all(torch.count_nonzero(p.weight.grad[:,pre.edge_dim:])==0 for l in expanded.layers for p in (l.e,l.u))
        torch.testing.assert_close(original,data['x'],atol=0,rtol=0)
        for handle in handles: handle.remove()


@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('mode',['snapshot','time_features','historical'])
def test_projection_capture_preserves_native_logits_and_attention_bounds(device,mode):
    torch=pytest.importorskip('torch')
    if device.startswith('cuda') and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from gat_temporal_mechanisms import ProjectionCapture
    from xdiyo_analytics.graphs.execution import deterministic_execution
    cfg,pre,model,data,home,away,fixture=components(device,mode)
    with deterministic_execution(cfg,device),torch.no_grad():
        expected=model(data,home,away,fixture)
        capture=ProjectionCapture(model,pre.edge_dim-len(pre.temporal_columns))
        observed,_=capture.encode(data,home,away,fixture)
        capture.close()
        torch.testing.assert_close(observed,expected,atol=0,rtol=0)
        records=capture.result()['projections']
        for name,values in records.items():
            if name.endswith('attention.effective_fraction'):
                assert 0<values['minimum']<=values['maximum']<=1.00001
            if name.endswith('value.pre_tanh'):
                assert values['count']>0 and 0<=values['mean_tanh_derivative']<=1


def test_historical_sender_switch_can_matter_with_time_connections_zero():
    torch=pytest.importorskip('torch')
    from gat_temporal_mechanisms import copy_snapshot_function
    cfg,pre,baseline,*_=components('cpu','snapshot')
    _,_,model,data,home,away,fixture=components('cpu','historical')
    copy_snapshot_function(model,baseline,pre.edge_dim)
    with torch.no_grad():
        historical=model(data,home,away,fixture)
        model.temporal_mode='time_features'
        current=model(data,home,away,fixture)
    assert float((current-historical).abs().max())>1e-6


def test_bounded_quantiles_keep_exact_moments():
    torch=pytest.importorskip('torch')
    from gat_temporal_mechanisms import Moments
    values=torch.tensor([-4.,-2.,0.,1.,4.])
    accumulator=Moments(sample=3)
    accumulator.add(values)
    result=accumulator.result()
    assert result['count']==5 and result['quantile_sample_count']==3
    assert result['mean']==pytest.approx(-.2)
    assert result['rms']==pytest.approx(float(values.double().square().mean().sqrt()))
    assert result['fraction_abs_gt_3']==.4


def test_matched_workers_use_only_training_validation_and_native_patience(tmp_path):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from gat_temporal_mechanisms import matched
    from xdiyo_analytics.graphs.execution import deterministic_execution
    store=sample()
    cfg=config(deterministic=True)
    frames=dict(train=store.schedule.iloc[[1,2,3,4]],validation=store.schedule.iloc[[6,7,8]])
    protocol=dict(seed=17,max_steps=8,early_stopping=dict(patience=2,min_delta=10.),
        device='cuda:0',input_cache_bytes=16*2**20,input_cache_headroom_bytes=2**30)
    diagnostic.experiment.write_json(tmp_path/'protocol.json',protocol)
    with deterministic_execution(cfg,'cuda:0'):
        for arm in ('zero_time','learn_time'):
            matched(SimpleNamespace(run=tmp_path,arm=arm),protocol,dict(config=asdict(cfg)),store,frames)
            result=json.loads((tmp_path/'matched'/arm/'result.json').read_text())
            assert result['training_summary']['attempts'][0]['steps']==3
            assert result['training_summary']['termination_reason']=='early_stopping'
            assert set(result['selected'])=={'train','validation'}
    for split in ('train','validation'):
        a=pd.read_parquet(tmp_path/'matched/zero_time'/f'initial_{split}_predictions.parquet')
        b=pd.read_parquet(tmp_path/'matched/learn_time'/f'initial_{split}_predictions.parquet')
        np.testing.assert_array_equal(a,b)
