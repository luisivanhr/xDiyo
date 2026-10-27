"""Comparison receipts, paired metrics and a disposable native-worker smoke."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from test_temporal_graph import sample, config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'examples'))
import compare_gat_temporal as experiment


def test_ordered_metrics_and_paired_round_bootstrap():
    result = experiment.metrics([1,0,-1],np.eye(3))
    assert result['log_loss'] == 0 and result['brier'] == 0
    assert all(x['ece_10'] == 0 for x in result['class_calibration'].values())
    for difference in (0., .25, -.1):
        result = experiment.paired_blocks(np.full(17,difference),[1]*5+[2]*3+[3]*4+[4]*5,
                                         replicates=100,seed=7,consecutive_rounds=2)
        for key in ('mean','lower_95','upper_95'):
            assert result[key] == pytest.approx(difference)


@pytest.mark.parametrize('p', [[[.1,.1,.1]], [[1.,-1.,1.]], [[float('nan'),0.,1.]]])
def test_metrics_reject_invalid_probabilities(p):
    with pytest.raises(ValueError,match='Invalid'):
        experiment.metrics([1],p)


def test_rolling_features_use_released_past_only():
    store = sample()
    target = store.schedule.iloc[[4,5]]
    before = experiment.rolling_features(store,config(),target)
    h = store.history
    h.loc[h.event_id.ge(5),'goals_for'] = 999
    h.loc[h.event_id.ge(5),'goals_against'] = 999
    # The first target cannot see its own result. The second can see the first.
    from xdiyo_analytics.graphs import TemporalEventStore
    changed = TemporalEventStore(h,store.team_seasons)
    after = experiment.rolling_features(changed,config(),target)
    np.testing.assert_array_equal(before[0],after[0])
    assert not np.array_equal(before[1],after[1])


@pytest.mark.skipif(importlib.util.find_spec('torch') is None,reason='optional torch')
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('with_patience', [False, True])
def test_frozen_inputs_native_worker_and_exact_budget(tmp_path, device, with_patience):
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    store = sample()
    frames = dict(train=store.schedule.iloc[[1,2,3]],validation=store.schedule.iloc[[5,6]],test=store.schedule.iloc[[7,8]])
    for name,frame in frames.items():
        frame.to_parquet(tmp_path/f'{name}.parquet',index=True)
    for name,frame in [('history',store.history),('memberships',store.team_seasons),('schedule',store.schedule)]:
        frame.to_parquet(tmp_path/f'{name}.parquet',index=False)
    protocol = dict(source_hashes=experiment.source_hashes(),data_hashes={p.name:experiment.digest(p) for p in tmp_path.glob('*.parquet')},
        provenance=store.provenance,store_version=store.version,populations={k:{'identity_sha256':experiment.frame_hash(v)} for k,v in frames.items()},
        config=asdict(config(deterministic=True)),max_steps=20 if with_patience else 2,
        early_stopping=dict(patience=15,min_delta=10.) if with_patience else None,
        worker_rss_limit_gib=4.,seeds=[17],profile_steps=1,device=device,
        input_cache_bytes=16*2**20 if device.startswith('cuda') else 0,
        cuda_allocator_config='garbage_collection_threshold:0.6' if device.startswith('cuda') else None,
        cuda_memory_fraction=.8)
    experiment.write_json(tmp_path/'protocol.json',protocol)
    for phase in ('profile','fit'):
        run = subprocess.run([sys.executable,str(ROOT/'examples/compare_gat_temporal.py'),phase,'--run',str(tmp_path),
                              '--mode','historical','--seed','17'],capture_output=True,text=True)
        assert run.returncode == 0,run.stdout+run.stderr
    receipt = json.loads((tmp_path/'fit/historical-17/result.json').read_text())
    # A deliberately unreachable improvement threshold forces an exact 15-epoch
    # wait after epoch 1. It exercises native stopping, not just configuration.
    assert receipt['training_summary']['attempts'][0]['steps'] == (16 if with_patience else 2)
    assert receipt['training_summary']['termination_reason'] == ('early_stopping' if with_patience else 'max_steps')
    metadata = json.loads((tmp_path/'fit/historical-17/model/graph.json').read_text())
    assert metadata['control']['early_stopping'] == protocol['early_stopping']
    assert receipt['test']['count'] == 2
    assert receipt['memory']['sampled_peak_rss_bytes'] > 0
    if device.startswith('cuda'):
        assert receipt['memory']['peak_cuda_allocated_bytes'] > 0
        assert receipt['input_cache']['hits'] > 0
    protocol['data_hashes']['train.parquet'] = 'wrong'
    experiment.write_json(tmp_path/'protocol.json',protocol)
    with pytest.raises(ValueError,match='Prepared input changed'):
        experiment.load(tmp_path)


@pytest.mark.parametrize('losses,cap,reason', [
    ([2.,1.]+[1.]*15,200,'early_stopping'),
    ([2.]+[2.]*14+[1.]+[1.]*15,200,'early_stopping'),
    ([2.,1.9,1.8],3,'max_steps'),
    ([2.]+[2.]*15,16,'early_stopping'),
])
def test_patience_boundary_reset_and_cap(losses, cap, reason):
    progress = [dict(step=i+1,validation_loss=x) for i,x in enumerate(losses)]
    best = min(progress,key=lambda row:row['validation_loss'])
    summary = dict(termination_reason=reason,retained_step=best['step'],monitored_value=best['validation_loss'],
                   attempts=[dict(steps=len(progress))])
    protocol = dict(max_steps=cap,early_stopping=dict(patience=15,min_delta=0.))
    experiment.validate_training_progress(progress,summary,protocol)
    with pytest.raises(ValueError,match='declared cap/patience'):
        experiment.validate_training_progress(progress[:-1],summary,protocol)
    if reason == 'early_stopping':
        with pytest.raises(ValueError,match='declared cap/patience|exceeded'):
            experiment.validate_training_progress(progress+[dict(step=len(progress)+1,validation_loss=1.)],summary,protocol)


def test_extension_preserves_frozen_inputs_and_rejects_source_drift(tmp_path):
    parent, out = tmp_path/'parent', tmp_path/'extended'
    parent.mkdir()
    store = sample()
    frames = dict(train=store.schedule.iloc[[1,2,3]],validation=store.schedule.iloc[[5,6]],test=store.schedule.iloc[[7,8]])
    for name,frame in {**frames,'history':store.history,'memberships':store.team_seasons,'schedule':store.schedule}.items():
        frame.to_parquet(parent/f'{name}.parquet')
    sources = experiment.source_hashes()
    sources['examples/compare_gat_temporal.py'] = 'old-runner'
    protocol = dict(source_hashes=sources,data_hashes={p.name:experiment.digest(p) for p in parent.glob('*.parquet')},
        provenance=store.provenance,store_version=store.version,populations={k:{'identity_sha256':experiment.frame_hash(v)} for k,v in frames.items()},
        config=asdict(config(deterministic=True)),max_steps=20,modes=['snapshot'],seeds=[17],
        data_root=str(tmp_path/'source_data'),limitations=[],device='cuda:0')
    experiment.write_json(parent/'protocol.json',protocol)
    folder = parent/'fit/snapshot-17'
    (folder/'model').mkdir(parents=True)
    experiment.write_json(folder/'result.json',dict(protocol_sha256=experiment.digest(parent/'protocol.json')))
    experiment.write_json(folder/'progress.json',[])
    (folder/'model/weights.pt').write_bytes(b'fixture')
    (parent/'profile/snapshot').mkdir(parents=True)
    experiment.write_json(parent/'profile/snapshot/result.json',{})
    original = {str(p.relative_to(parent)):experiment.digest(p) for p in parent.rglob('*') if p.is_file()}
    args = SimpleNamespace(from_run=parent,run=out,max_steps=200,patience=15,min_delta=0.)
    experiment.prepare_from_run(args)
    extended,_,_ = experiment.load(out)
    assert extended['data_hashes'] == protocol['data_hashes']
    assert extended['config'] == json.loads(json.dumps(protocol['config'])) and extended['device'] == 'cuda:0'
    assert extended['max_steps'] == 200 and extended['early_stopping'] == dict(patience=15,min_delta=0.)
    assert original == {str(p.relative_to(parent)):experiment.digest(p) for p in parent.rglob('*') if p.is_file()}
    protocol['source_hashes']['src/xdiyo_analytics/graphs/network.py'] = 'changed'
    experiment.write_json(parent/'protocol.json',protocol)
    args.run = tmp_path/'rejected'
    with pytest.raises(ValueError,match='Model or shared experiment source changed'):
        experiment.prepare_from_run(args)
    assert not args.run.exists()
