"""Interaction gates for feature integration with native main performance/resume."""
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import json
import os
import subprocess
import sys
import zipfile
import io
import numpy as np
import pandas as pd
import pytest
import torch
from test_temporal_graph import sample, config, context
from test_graph_extensions import forecast, threads
from test_graph_preparation_memory import assert_exact
from xdiyo_analytics.graphs import (TemporalGraphAdapter, TemporalGraphSerializer,
    BoundaryMemory, StatisticsForecast, NumericStatisticTarget, TemporalEventStore)
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.graphs.memory import BoundaryReplay
from xdiyo_analytics.training import TrainingControl, CheckpointPolicy

MAIN = '861b00461943be7ff589c9bd7deace909e96c9c1'
FEATURE = '5ee87b5791f142f0c80432f1e0647d01f137a599'


def extension_fixture(kind='gat_combined', *, heatmaps=True, memory=True, route='combined'):
    from test_heatmap_cnn import graph_setup
    from xdiyo_analytics.neural import Reconstruction
    if kind == 'gat_combined' and heatmaps:
        spec, store = graph_setup(reconstruction=Reconstruction(loss='kl'))
        cfg = replace(spec.config, deterministic=True)
    else:
        store, cfg = sample(), config(deterministic=True)
    if kind != 'gat_statistics' and memory:
        cfg = replace(cfg, boundary_memory=BoundaryMemory('fixed_days', 7, '2020-01-01T00:00:00+01:00', .6))
    if kind != 'gat_memory':
        targets = (NumericStatisticTarget('signed', 'team::signed', 'normalized', 'signed_continuous'),
                   NumericStatisticTarget('bounded', 'team::bounded', 'percentage', 'bounded_continuous',
                                          transform='affine', lower=0., upper=100., side='away'))
        h = store.history
        h['team::signed'] = h.event_id.astype(float)-5
        h['team::bounded'] = h.event_id.astype(float)*4
        store = TemporalEventStore(h, store.team_seasons, provenance=store.provenance)
        cfg = replace(cfg, statistics_forecast=StatisticsForecast(targets, route=route))
    train = context(store, [1, 2, 3, 4]); train.validation = context(store, [5, 6])
    return TemporalGraphAdapter(cfg, store), train, context(store, [7, 8], fit=False)


def activate_cpu_cache(monkeypatch):
    from xdiyo_analytics.training import residency
    original = residency.InputResident
    def cached(device, policy=None):
        owner = original(device, policy)
        if policy and policy.max_bytes:
            owner.pool = residency._Pool('cpu')
            owner.pool.owners[owner] = policy
        return owner
    monkeypatch.setattr(residency, 'InputResident', cached)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (10**9, 10**9))


@pytest.mark.parametrize('heatmaps', [False, True])
@pytest.mark.parametrize('memory', [False, True])
@pytest.mark.parametrize('route', ['auxiliary', 'combined', 'bottleneck'])
def test_combined_resident_exact_updates_and_no_learned_cache(monkeypatch, heatmaps, memory, route, device='cpu'):
    if device == 'cpu':
        activate_cpu_cache(monkeypatch)
    else:
        from xdiyo_analytics.graphs.execution import prepare_determinism
        prepare_determinism(config(deterministic=True), device)
    a, train, query = extension_fixture(heatmaps=heatmaps, memory=memory, route=route)
    b, _, _ = extension_fixture(heatmaps=heatmaps, memory=memory, route=route)
    b.configure_input_cache(1_000_000, 0)
    x = _Backend(a.config, a.event_store, 3, device)
    y = _Backend(b.config, b.event_store, 3, device, b.input_cache_policy)
    x.initialize(train); y.initialize(train)
    for _ in range(2):
        assert_exact(x.checkpoint_metrics('train_objective', x.step()), y.checkpoint_metrics('train_objective', y.step()))
        assert_exact(x.model.state_dict(), y.model.state_dict())
        assert_exact(x.optimizer.state_dict(), y.optimizer.state_dict())
        assert_exact([p.grad for p in x.model.parameters()], [p.grad for p in y.model.parameters()])
    pool = y.train_groups._resident.pool
    assert pool.hits > 0 and pool.transient == 0
    assert pool.resident <= 1_000_000
    # Observe newly leased cached data before the replay can add learned state.
    with y.train_groups.segment_lease(0, None, ()) as (group, teams):
        assert 'initial_state' not in group[1]
    for name in x.predict(query):
        pd.testing.assert_frame_equal(x.predict(query)[name], y.predict(query)[name], check_exact=True)


@pytest.mark.parametrize('policy', ['season', 'fixed_days'])
@pytest.mark.parametrize('retention', [0., .4, 1.])
def test_memory_restore_queries_and_new_team(policy, retention):
    store = sample(); memberships = store.team_seasons
    row = memberships.iloc[[0]].copy()
    row['competition_id'] = 11; row['season_id'] = 901
    row['entry_at'] = pd.Timestamp('2020-01-08T00:00Z'); row['movement'] = 'promoted'
    store = TemporalEventStore(store.history, pd.concat([memberships,row],ignore_index=True))
    cfg = config(boundary_memory=BoundaryMemory(policy, 7, '2020-01-01T00:00:00+01:00', retention))
    a = TemporalGraphAdapter(cfg, store); a.fit_controlled(context(store,[1,2,3]), TrainingControl(max_steps=1))
    t = pd.Timestamp('2020-02-05T12:00Z')
    queries = pd.DataFrame({'team_id':[3,1,2,3], 'issuance_at':[t]*4}, index=[11,4,7,2])
    original = a.encode_teams(queries)
    pd.testing.assert_frame_equal(original, a.encode_teams(queries.iloc[::-1]).loc[queries.index], check_exact=True)
    state = a.backend_.snapshot(); a.backend_.step(); a.backend_.restore(state)
    pd.testing.assert_frame_equal(original, a.encode_teams(queries), check_exact=True)
    r = BoundaryReplay(a.backend_, t); snap, data = r.snapshot(t,[1,2,3])
    assert all(not v.requires_grad for v in r.state.values())
    assert snap.league[snap.teams.index(1)] == 11
    attention = a.attention_weights(queries.iloc[:2])
    if len(attention): assert set(attention.team_id) <= {3,1}


@pytest.mark.parametrize('kind', ['gat_memory','gat_statistics','gat_combined'])
def test_extension_finalization_reload_and_scaler_binding(tmp_path, kind):
    from xdiyo_analytics.training.native_resume import capture_backend, restore_backend
    a, train, query = extension_fixture(kind)
    a.configure_input_cache(1_000_000, 0)
    CheckpointPolicy(finalization='prediction_only').wrap(lambda:a, tmp_path/'runs', 'gates')().fit_controlled(train, TrainingControl(max_steps=2))
    for field in ('optimizer','train_groups','validation_groups','train_statistics','validation_statistics','_statistics_observations'):
        assert not hasattr(a.backend_, field)
    serializer = TemporalGraphSerializer(embed_event_store=True)
    serializer.save(a, tmp_path/'model'); b = serializer.load(tmp_path/'model')
    assert b.input_cache_policy == a.input_cache_policy
    for name, frame in a.predict(query).items(): pd.testing.assert_frame_equal(frame,b.predict(query)[name],check_exact=True)
    q = pd.DataFrame({'team_id':[3,2,1], 'issuance_at':[pd.Timestamp('2020-01-25T12:00Z')]*3})
    pd.testing.assert_frame_equal(a.encode_teams(q), b.encode_teams(q), check_exact=True)
    pd.testing.assert_frame_equal(a.attention_weights(q), b.attention_weights(q), check_exact=True)
    CheckpointPolicy().wrap(lambda:a,tmp_path/'runs','gates')().fit_controlled(train,TrainingControl(max_steps=2))
    if kind != 'gat_memory':
        assert a.backend_._statistics_observations == {}  # rebuilt lazily, never checkpointed
        state = capture_backend(a.backend_)
        state['statistics_scale']['scale'][0] *= 2
        with pytest.raises(ValueError,match='statistics scaling'): restore_backend(a.backend_,state)


@pytest.mark.parametrize('kind', ['gat_memory','gat_statistics','gat_combined'])
@pytest.mark.parametrize('boundary', ['', 'early_stopping'])
def test_extensions_fresh_process_continuation(tmp_path,kind,boundary,device='cpu'):
    worker = Path(__file__).with_name('native_resume_worker.py')
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='0' if device.startswith('cuda') else '-1',PYTHONDONTWRITEBYTECODE='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', XDIYO_RESUME_DEVICE=device,XDIYO_RESUME_CACHE='1',XDIYO_RESUME_BOUNDARY=boundary)
    for mode,code in [('reference',0),('interrupt',73),('resumed',0),('completed',0)]:
        run = subprocess.run([sys.executable,str(worker),str(tmp_path),kind,mode],env=env,capture_output=True,text=True)
        assert run.returncode==code, run.stdout+run.stderr
    load=lambda name:torch.load(tmp_path/name,map_location='cpu',weights_only=True)
    reference,resumed,completed=load('reference.pt'),load('resumed.pt'),load('completed.pt')
    assert_exact(reference['updates'],load('prefix.pt')+resumed['updates'])
    for key in ('backend','rng'): assert_exact(reference[key],resumed[key])
    assert_exact(reference['backend'],completed['backend']); assert not completed['updates']
    assert (tmp_path/'reference.json').read_bytes()==(tmp_path/'resumed.json').read_bytes()


@pytest.mark.parametrize('reference,kind',[(MAIN,'plain'),(MAIN,'cnn'),(FEATURE,'gat_memory'),(FEATURE,'gat_statistics'),(FEATURE,'gat_combined')])
def test_pinned_source_two_update_parity_and_legacy_reload(tmp_path,reference,kind,device='cpu'):
    # Full pinned Python package, independent imports/process and native config.
    root = Path(__file__).resolve().parents[2]
    archive=subprocess.check_output(['git','archive','--format=zip',reference,'src'],cwd=root)
    frozen=tmp_path/'frozen';frozen.mkdir()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:z.extractall(frozen)
    worker=Path(__file__).with_name('gat_integration_parity_worker.py')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if device.startswith('cuda') else '-1',PYTHONDONTWRITEBYTECODE='1',
             CUBLAS_WORKSPACE_CONFIG=':4096:8',XDIYO_PARITY_DEVICE=device)
    for name,source in [('reference',frozen),('candidate',root)]:
        run=subprocess.run([sys.executable,str(worker),str(source),str(tmp_path/name),kind],env=env,capture_output=True,text=True)
        assert run.returncode==0,run.stdout+run.stderr
    load=lambda name:torch.load(tmp_path/name/'trajectory.pt',map_location='cpu',weights_only=True)
    assert_exact(load('reference'),load('candidate'))
    assert (tmp_path/'reference/outputs.json').read_bytes()==(tmp_path/'candidate/outputs.json').read_bytes()
    serializer=TemporalGraphSerializer(device=device)
    old=serializer.load(tmp_path/'reference/model'); new=serializer.load(tmp_path/'candidate/model')
    from test_heatmap_cnn import graph_setup
    query=context(old.event_store,[7,8],fit=False)
    for key,value in old.predict(query).items():pd.testing.assert_frame_equal(value,new.predict(query)[key],check_exact=True)
    # The worker receipt is the pre-save output, not merely two reloaded models.
    from xdiyo_analytics.experiments.recovery import unpack
    before=unpack(json.loads((tmp_path/'candidate/outputs.json').read_text()))
    for key,value in before['predictions'].items():pd.testing.assert_frame_equal(value,new.predict(query)[key],check_exact=True)
    queries=pd.DataFrame({'team_id':[3,2,1], 'issuance_at':[pd.Timestamp('2020-02-02T12:00Z')]*3})
    pd.testing.assert_frame_equal(before['embeddings'],new.encode_teams(queries),check_exact=True)
    pd.testing.assert_frame_equal(before['attention'],new.attention_weights(queries),check_exact=True)
    if device.startswith('cuda'):
        cpu=TemporalGraphSerializer(device='cpu').load(tmp_path/'candidate/model')
        for key,value in before['predictions'].items():
            pd.testing.assert_frame_equal(value,cpu.predict(query)[key],check_exact=False,atol=1e-6,rtol=1e-5)
        pd.testing.assert_frame_equal(before['embeddings'],cpu.encode_teams(queries),check_exact=False,atol=1e-6,rtol=1e-5)
        pd.testing.assert_frame_equal(before['attention'],cpu.attention_weights(queries),check_exact=False,atol=1e-6,rtol=1e-5)


def test_ui_retains_both_sets_of_controls():
    from xdiyo_analytics.ui.recipe import catalog_for_ui,default_recipe,export_python
    from xdiyo_analytics.training import ExecutionPolicy
    c=catalog_for_ui(); a,_,_=extension_fixture()
    schema={s['id']:{f['name'] for f in s['fields']} for s in c.schema()}
    assert {'boundary_memory','statistics_forecast'} <= schema['graphs.TemporalGraphConfig']
    assert {'input_cache_bytes','input_cache_headroom_bytes'} <= schema['training.ExecutionPolicy']
    assert 'finalization' in schema['training.CheckpointPolicy']
    for value in (a.config,ExecutionPolicy(input_cache_bytes=1024),CheckpointPolicy(finalization='prediction_only')):
        assert c.build(json.loads(json.dumps(c.encode(value))))==value
    recipe=default_recipe('data'); recipe['execution']=c.encode(ExecutionPolicy(input_cache_bytes=1024))
    text=export_python(recipe)
    assert 'input_cache_bytes' in text
