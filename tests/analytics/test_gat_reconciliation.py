"""Boundaries specific to merging retained research into the native model."""
from pathlib import Path
from dataclasses import asdict, replace, fields
from hashlib import sha256
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
import torch
from test_temporal_graph import sample, config, context
from test_graph_phase_context import source, spec
from test_ui_workflow import ui_recipe
from xdiyo_analytics.graphs import (TemporalGraphConfig, TemporalGraphModel, TemporalGraphAdapter,
    TemporalGraphSerializer, GATResearchBNV1, GATResearchCE1V1, GATResearchBCDDWV1,
    GoalAuxiliary, RetainedGATTraining)
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor, MaskedScale
from xdiyo_analytics.graphs.compatibility import identity_config, MAIN_IDENTITY, EXPLICIT_IDENTITY
from xdiyo_analytics.graphs.compatibility_records import RETAINED_EXPORTS
from xdiyo_analytics.graphs.phase_context import PhaseRankFeatures, PhaseRankLookup, CORRECTED_POLICY
from xdiyo_analytics.graphs.research_training import MONITOR, RetainedGATBackend
from xdiyo_analytics.training import TrainingControl, EarlyStopping
from xdiyo_analytics.ui.recipe import catalog_for_ui


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_generic_defaults_and_explicit_research_identity():
    cfg=TemporalGraphConfig('UTC')
    assert cfg.temporal_mode=='snapshot' and cfg.fixture_gap_encoding=='legacy_standardized_gap_v1'
    assert cfg.statistic_availability_policy=='legacy_unmasked_v1'
    assert cfg.goal_auxiliary is None and cfg.phase_rank_features is None
    assert 'statistic_availability_policy' not in identity_config(cfg)
    explicit=replace(cfg,statistic_availability_policy='per_stat_release_v1')
    assert identity_config(explicit)['statistic_availability_policy']=='per_stat_release_v1'
    with pytest.raises(ValueError,match='main identity'):identity_config(explicit,MAIN_IDENTITY)
    with pytest.raises(ValueError,match='Unsupported'):replace(cfg,statistic_availability_policy='infer')


def test_release_policy_early_late_missing_and_reverse_query():
    from xdiyo_analytics.graphs import TemporalEventStore
    original=sample();h=original.history
    col='team::stat';other='opponent::stat';h[col]=3.;h[other]=3.
    h['available_at::'+col]=pd.Timestamp('2020-01-20T00:00Z')
    h.loc[h.index[:2],'available_at::'+col]=pd.NaT
    store=TemporalEventStore(h,original.team_seasons,schedule=original.schedule)
    def prep(policy):
        p=GraphPreprocessor(config(statistics=(col,),statistic_availability_policy=policy))
        p.edge_scale=MaskedScale().fit(p.numeric_edges(h));p.leagues={10:1}
        return p
    old,new=prep('legacy_unmasked_v1'),prep('per_stat_release_v1')
    frame=h.iloc[:6]
    early=pd.Timestamp('2020-01-10T00:00Z');late=pd.Timestamp('2020-01-25T00:00Z')
    a=new._encode_edges(frame,cutoff=early);b=new._encode_edges(frame,cutoff=late)
    # Three numeric columns: own goals, opponent goals, selected statistic.
    np.testing.assert_array_equal(a[:,5],1)
    np.testing.assert_array_equal(b[:,5],[1,1,0,0,0,0])
    np.testing.assert_array_equal(old._encode_edges(frame,cutoff=early)[:,5],0)
    np.testing.assert_array_equal(a,new._encode_edges(frame,cutoff=early))
    early_snap=store.snapshot(early,old.config);late_snap=store.snapshot(late,old.config)
    before=new.cached_edges(early_snap,store).copy();new.cached_edges(late_snap,store)
    np.testing.assert_array_equal(before,new.cached_edges(early_snap,store))


def test_retained_factories_are_fixed_expanded_configs_and_catalogue_roundtrips():
    key='a'*64
    configurations=[GATResearchBNV1(),GATResearchCE1V1(key),GATResearchBCDDWV1(key,10),GATResearchBCDDWV1(key,20)]
    c=catalog_for_ui()
    for cfg in configurations:
        assert type(cfg) is TemporalGraphConfig
        assert cfg.temporal_mode=='time_features' and cfg.statistic_availability_policy=='per_stat_release_v1'
        assert cfg.fixture_gap_encoding=='bounded_observed_gap_v1'
        assert cfg.goal_auxiliary is not None and cfg.goal_auxiliary.architecture=='linear'
        assert cfg.heatmaps is cfg.boundary_memory is cfg.statistics_forecast is None
        assert cfg.embedding_dim==32 and cfg.layers==2 and cfg.heads==2
        assert c.build(json.loads(json.dumps(c.encode(cfg))))==cfg
    assert configurations[2]!=configurations[3]
    assert c.build({'component':'graphs.GATResearchBNV1'})==configurations[0]
    assert c.build({'component':'graphs.GATResearchCE1V1','params':{'source_id':key}})==configurations[1]
    with pytest.raises(TypeError):GATResearchCE1V1()
    with pytest.raises(TypeError):GATResearchBCDDWV1(key)
    with pytest.raises(ValueError):GATResearchBCDDWV1(key,1)
    with pytest.raises(TypeError):GATResearchBNV1(embedding_dim=64)
    # A versioned factory must explicitly name every config constructor field.
    import inspect,ast
    from xdiyo_analytics.graphs.research_profiles import _config
    call=next(n for n in ast.walk(ast.parse(inspect.getsource(_config))) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='TemporalGraphConfig')
    assert {k.arg for k in call.keywords}=={f.name for f in fields(TemporalGraphConfig)}


def test_public_phase_path_and_failed_attachment_are_atomic(tmp_path):
    store=source();cfg=replace(config(),phase_rank_features=spec(store,'E1'))
    store.phase_context.save(tmp_path)
    model=TemporalGraphModel(cfg,membership_policy='explicit',phase_context_path=str(tmp_path))
    rebuilt=model.event_store(store.history,store.team_seasons)
    assert rebuilt.phase_context.source_id==store.phase_context.source_id
    assert model.build(rebuilt).config==cfg
    with pytest.raises(ValueError,match='requires enabled'):TemporalGraphModel(config(),phase_context_path=str(tmp_path)).event_store(store.history,store.team_seasons)
    with pytest.raises(ValueError,match='require phase_context_path'):TemporalGraphModel(cfg).event_store(store.history,store.team_seasons)
    a=TemporalGraphAdapter(cfg,store);a.fit_controlled(context(store,[2,3,4]),TrainingControl(max_steps=1))
    old_pre=a.backend_.preprocessor;old_state=deepcopy(old_pre.state())
    wrong=PhaseRankLookup(store.phase_context.frame,provenance=store.phase_context._provenance,policy=CORRECTED_POLICY)
    replacement=type(store)(store.history,store.team_seasons,schedule=store.schedule,phase_context=wrong)
    with pytest.raises(ValueError,match='source'):a.attach_event_store(replacement)
    assert a.event_store is store and a.backend_.store is store and a.backend_.preprocessor is old_pre
    assert old_pre.state()==old_state and replacement.phase_context is wrong


def test_compatibility_manifest_is_immutable_and_unambiguous():
    assert type(RETAINED_EXPORTS) is tuple and len(RETAINED_EXPORTS)==18
    for name in ('graph_sha256','weights_sha256','model_version'):
        assert len({getattr(r,name) for r in RETAINED_EXPORTS})==18
    assert {r.seed for r in RETAINED_EXPORTS}=={'17','42','73'}


def miniature():
    store=sample();cfg=config(goal_auxiliary=GoalAuxiliary(beta=.1),deterministic=True)
    ctx=context(store,[1,2,3]);ctx.validation=context(store,[5,6])
    training=RetainedGATTraining(competitions=(10,))
    control=TrainingControl(max_steps=2,early_stopping=EarlyStopping(15,0),monitor=MONITOR)
    return store,cfg,ctx,training,control


def test_backend_dispatch_validation_and_read_only_metrics():
    store,cfg,ctx,training,control=miniature()
    generic=TemporalGraphAdapter(cfg,store);assert type(generic._make_backend(17)) is _Backend
    model=TemporalGraphModel(cfg,control=control,research_training=training)
    a=model.build(store);assert type(a._make_backend(17)) is RetainedGATBackend
    a.fit(ctx);backend=a.backend_
    assert backend._validation_forward_calls>0
    saved={k:v.clone() for k,v in backend.model.state_dict().items()}
    value=backend._loss(backend.validation_groups,backend.validation_targets,backend.validation_weights)
    assert value==backend._inner_summary['micro']
    for k,v in backend.model.state_dict().items():torch.testing.assert_close(v,saved[k],rtol=0,atol=0)
    with pytest.raises(ValueError,match='raw validation CE'):
        TemporalGraphModel(cfg,research_training=training).build(store)
    with pytest.raises(ValueError,match='declared competitions'):
        RetainedGATBackend(cfg,store,17,'cpu',training=RetainedGATTraining(competitions=(10,11))).initialize(ctx)
    ctx.validation.sample_weight=np.array([1.,2.])
    with pytest.raises(ValueError,match='equal per-fixture'):
        RetainedGATBackend(cfg,store,17,'cpu',training=training).initialize(ctx)


def test_replay_hashes_and_contract_roundtrip(tmp_path):
    with pytest.raises(ValueError,match='both preprocessing'):
        RetainedGATTraining(preprocessing_path='x')
    store,cfg,ctx,training,control=miniature()
    b=RetainedGATBackend(cfg,store,17,'cpu',training=training);b.initialize(ctx)
    pre=tmp_path/'pre.json';weights=tmp_path/'initial.pt'
    pre.write_text(json.dumps({'state':b.preprocessor.state()}));torch.save(b.model.state_dict(),weights)
    spec=RetainedGATTraining(competitions=(10,),preprocessing_path=str(pre),preprocessing_sha256=sha256(pre.read_bytes()).hexdigest(),initial_state_path=str(weights),initial_state_sha256=sha256(weights.read_bytes()).hexdigest())
    c=catalog_for_ui();assert c.build(json.loads(json.dumps(c.encode(spec))))==spec
    replay=RetainedGATBackend(cfg,store,99,'cpu',training=spec);replay.initialize(ctx)
    assert replay.initial_digest==b.initial_digest
    pre.write_text('{}')
    with pytest.raises(ValueError,match='hash changed'):spec.inputs()


def test_known_export_hashes_and_ambiguous_research_policy_fail_closed(tmp_path,monkeypatch):
    import xdiyo_analytics.graphs.compatibility as c
    from xdiyo_analytics.graphs.compatibility_records import RetainedExport
    store=sample();meta={'config':asdict(config()),'model_version':'v','event_store_version':store.version,'preprocessing':{}}
    path=tmp_path/'model';path.mkdir();(path/'weights.pt').write_bytes(b'fixed-test-weight-payload')
    meta['config'].pop('statistic_availability_policy')
    def write(): (path/'graph.json').write_text(json.dumps(meta))
    write();hash_file=lambda n:sha256((path/n).read_bytes()).hexdigest()
    record=RetainedExport('time_B/N','fold_1','17',hash_file('graph.json'),hash_file('weights.pt'),'v',store.version,c._json_digest(meta['config']),c._json_digest({}))
    monkeypatch.setattr(c,'RETAINED_EXPORTS',(record,))
    resolved=c.resolve_graph_compatibility(meta,path,store)
    assert resolved.statistic_policy==c.RELEASE and resolved.identity_schema==c.RESEARCH_IDENTITY
    (path/'weights.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash or identity changed'):c.resolve_graph_compatibility(meta,path,store)
    (path/'weights.pt').write_bytes(b'fixed-test-weight-payload');meta['config']['hidden_width']+=1;write()
    with pytest.raises(ValueError,match='hash or identity changed'):c.resolve_graph_compatibility(meta,path,store)
    monkeypatch.setattr(c,'RETAINED_EXPORTS',())
    h=store.history;h['team::stat']=2.;h['opponent::stat']=2.;h['available_at::team::stat']=pd.Timestamp('2020-01-01T00:00Z')
    from xdiyo_analytics.graphs import TemporalEventStore
    released=TemporalEventStore(h,store.team_seasons,schedule=store.schedule)
    meta['config']['statistics']=['team::stat'];write()
    with pytest.raises(ValueError,match='frozen-source compatibility record'):c.resolve_graph_compatibility(meta,path,released)
    meta['config']['statistic_availability_policy']=c.RELEASE;write()
    with pytest.raises(ValueError,match='both be present'):c.resolve_graph_compatibility(meta,path,released)


def test_retained_execution_metadata_cannot_silently_change_resume(tmp_path):
    from xdiyo_analytics.training.native_resume import capture_backend, restore_backend, execution_profile
    store,cfg,ctx,spec,control=miniature()
    a=TemporalGraphAdapter(cfg,store,research_training=spec,control=control)
    b=a._make_backend(17);b.initialize(ctx);b.checkpoint_metrics(MONITOR,b.step())
    state=capture_backend(b)
    assert state['research_contract']==spec.identity()
    assert execution_profile(a)['research_training']==spec.identity()
    rebuilt=a._make_backend(17);rebuilt.initialize(ctx);restore_backend(rebuilt,state)
    assert rebuilt._validation_forward_calls==b._validation_forward_calls
    bad=deepcopy(state);bad['research_contract']['competitions']=(11,)
    with pytest.raises(ValueError,match='population or replay contract'):restore_backend(rebuilt,bad)
    bad=deepcopy(state);bad['research_execution']={'omit_diagnostic_replay':True}
    with pytest.raises(ValueError,match='execution or diagnostic policy'):restore_backend(rebuilt,bad)


def test_current_phase_producer_uses_native_oracle_and_original_cutoffs():
    from test_derived_standings import population, oracle
    from xdiyo_analytics.graphs import build_phase_context
    from xdiyo_analytics.graphs.phase_context import KEYS, WINDOWS
    h,m=population(4);h['source_league']='Premier_League';h['source_season']='20_21'
    q=h.loc[h.side.eq('home')].reset_index(drop=True)
    fixtures=q.rename(columns={'team_id':'home_id','opponent_id':'away_id'})
    cuts=q.kickoff_at-pd.Timedelta(days=1)
    source=build_phase_context(h,m,fixtures,cutoffs=cuts,provenance={'test':'native'},available_at='release',rank_timing='available_at_cutoff')
    assert source.policy==CORRECTED_POLICY
    values=source.frame.set_index(list(KEYS))
    for i,row in q.iterrows():
        for window in WINDOWS:
            _,ranks,available=oracle(h,m,row,cuts.iloc[i],window)
            for name,team in [('home',row.team_id),('away',row.opponent_id)]:
                assert values.loc[tuple(row[k] for k in KEYS),f'{name}_rank_{window}']==(4-ranks[team])/3
    changed=h.copy();last=changed.kickoff_at.max()
    changed.loc[changed.kickoff_at.eq(last)&changed.side.eq('home'),['goals_for','goals_against']]=[99,0]
    changed.loc[changed.kickoff_at.eq(last)&changed.side.eq('away'),['goals_for','goals_against']]=[0,99]
    later=build_phase_context(changed,m,fixtures,cutoffs=cuts,provenance={'test':'native'},available_at='release',rank_timing='available_at_cutoff')
    pd.testing.assert_frame_equal(source.frame,later.frame,check_exact=True)
    assert source.source_id==later.source_id
    with pytest.raises(ValueError,match='align'):
        build_phase_context(h,m,fixtures,cutoffs=cuts.iloc[::-1],provenance={})


@pytest.mark.parametrize('family,window',[('BN',None),('CE1',None),('BCDDW',10),('BCDDW',20)])
def test_retained_public_recipe_prepare_json_python_and_reuse(ui_recipe,tmp_path,family,window):
    from test_graph_integration import graph_recipe
    from xdiyo_analytics.ui.recipe import prepare_recipe,export_python,node,make_candidate
    from xdiyo_analytics.graphs.integration import fixture_cutoffs
    from xdiyo_analytics.graphs.phase_context import VALUES
    # Supply explicit native stage evidence in the synthetic source exports.
    from pathlib import Path
    for path in Path(ui_recipe['data']['data_root']).rglob('matches.parquet'):
        matches=pd.read_parquet(path);matches['tournament_id']=17
        matches.to_parquet(path,index=False)
        manifest_path=path.parent/'manifest.json';manifest=json.loads(manifest_path.read_text())
        manifest['tables']['matches'].update(bytes=path.stat().st_size,sha256=sha256(path.read_bytes()).hexdigest())
        stats_path=path.parent/'statistics.parquet';stats=pd.read_parquet(stats_path)
        shots=stats.copy();shots['key']='totalShotsOnGoal';shots['value']=12
        off=stats.copy();off['group_name']='Shots';off['key']='shotsOffGoal';off['value']=4
        stats=pd.concat([stats,shots,off],ignore_index=True);stats.to_parquet(stats_path,index=False)
        manifest['tables']['statistics'].update(rows=len(stats),bytes=stats_path.stat().st_size,sha256=sha256(stats_path.read_bytes()).hexdigest())
        manifest_path.write_text(json.dumps(manifest))
    recipe=graph_recipe(ui_recipe,tmp_path)
    recipe['model']['params']['round_stage_column']='tournament_id'
    # Build the permitted native history first, then supply an exact-query test
    # lookup. This is public integration coverage, not a standings reproduction.
    base=prepare_recipe(recipe)
    from xdiyo_analytics.graphs import TemporalEventStore
    store=TemporalEventStore(**base.outputs['graph_inputs'])
    frame=store.schedule[['competition_id','season_id','event_id','home_id','away_id','kickoff_at']].copy()
    frame['cutoff_at']=fixture_cutoffs(store,GATResearchBNV1(),store.schedule)
    for name in VALUES:frame[name]=.5
    lookup=PhaseRankLookup(frame,provenance={'miniature':True},policy=CORRECTED_POLICY)
    phase_path=tmp_path/'phase';lookup.save(phase_path)
    params={} if family=='BN' else dict(source_id=lookup.source_id)
    if window is not None:params['window']=window
    cfg=node('graphs.GATResearch'+family+'V1',**params)
    recipe['model']['params'].update(config=cfg,
        research_training=node('graphs.RetainedGATTraining',competitions=[17]),
        control=node('training.TrainingControl',max_steps=2,monitor=MONITOR),
        phase_context_path=None if family=='BN' else str(phase_path))
    prepared=prepare_recipe(json.loads(json.dumps(recipe)))
    from xdiyo_analytics.experiments.recovery import pack,unpack
    original_prepared=prepared
    prepared=unpack(pack(prepared))
    adapter=make_candidate(recipe,catalog_for_ui(),prepared.outputs).model_factory()
    assert adapter.config==catalog_for_ui().build(cfg)
    assert adapter.event_store.phase_context is None if family=='BN' else adapter.event_store.phase_context.source_id==lookup.source_id
    text=export_python(recipe);namespace={};exec(text.split('prepared =')[0],namespace)
    again=prepare_recipe(namespace['recipe'])
    pd.testing.assert_frame_equal(original_prepared.dataset.metadata,again.dataset.metadata)
    assert adapter.event_store.version==make_candidate(recipe,catalog_for_ui(),original_prepared.outputs).model_factory().event_store.version
    assert catalog_for_ui().build(namespace['recipe']['model']).research_training.competitions==(17,)
    from xdiyo_analytics.ui.recipe import run_recipe
    from xdiyo_analytics.training import save_model,load_model
    recipe['candidate']={'validation':node('training.ValidationTail',fraction=.4)}
    recipe['post_reporters']={'loss':node('reporting.GATLossReporter',type='overall')}
    result=run_recipe(recipe,prepared=prepared)
    assert result.training.folds[0].model.config==adapter.config
    restored=run_recipe(recipe,prepared=prepared)
    for left,right in zip(result.training.folds,restored.training.folds):
        pd.testing.assert_frame_equal(left.predictions['predict_proba'],right.predictions['predict_proba'])
    path=save_model(result.training,tmp_path/'public_model',fold_id=0)
    saved=load_model(path).model
    assert saved.research_training is None and saved.saved_research_training_['contract']=='retained_all13_raw_ce_v1'
    if family!='BN':
        bad=deepcopy(recipe);bad['model']['params']['config']['params']['source_id']='0'*64
        with pytest.raises(ValueError,match='source or policy'):prepare_recipe(bad)


def retained_fixture(family):
    """Miniature native resume fixture with explicitly declared one-league scope."""
    from xdiyo_analytics.graphs import TemporalEventStore
    from xdiyo_analytics.graphs.research_profiles import RETAINED_STATISTICS
    original=source();h=original.history;h['round_stage']='regular'
    for column in RETAINED_STATISTICS:
        h[column]=2.
        h[column.replace('team::','opponent::',1)]=2.
    schedule=original.schedule;schedule['round_stage']='regular'
    cfg=GATResearchBNV1();base=TemporalEventStore(h,original.team_seasons,schedule=schedule)
    phase=original.phase_context.frame
    from xdiyo_analytics.graphs.phase_context import KEYS
    cuts=base.schedule[list(KEYS)].copy()
    cuts['cutoff_at']=[cfg.history_cutoff(t) for t in base.issue_times(base.schedule,cfg)]
    keys=pd.MultiIndex.from_frame(phase[list(KEYS)])
    phase['cutoff_at']=cuts.set_index(list(KEYS)).loc[keys,'cutoff_at'].to_numpy()
    lookup=PhaseRankLookup(phase,provenance={'miniature':True},policy=CORRECTED_POLICY)
    store=TemporalEventStore(h,original.team_seasons,schedule=schedule,phase_context=lookup)
    cfg={'bn':lambda:GATResearchBNV1(),'ce1':lambda:GATResearchCE1V1(lookup.source_id),
         'cddw':lambda:GATResearchBCDDWV1(lookup.source_id,10)}[family]()
    train=context(store,[1,2,3,4]);train.validation=context(store,[5,6])
    return TemporalGraphAdapter(cfg,store,research_training=RetainedGATTraining(competitions=(10,))),train,context(store,[7,8],fit=False)


def test_native_checkpoint_rejects_changed_source_fingerprint(tmp_path,monkeypatch):
    from xdiyo_analytics.training import CheckpointPolicy
    import xdiyo_analytics.experiments.recovery as recovery
    store,cfg,ctx,spec,control=miniature()
    adapter=TemporalGraphAdapter(cfg,store,research_training=spec,control=control)
    CheckpointPolicy(every=1).wrap(lambda:adapter,tmp_path/'checkpoints','source-gate')().fit_controlled(ctx,control)
    checkpoint=Path(adapter.training_checkpoint_)
    original=recovery.training_source_files
    # Replacing one member of the numerical source list simulates another
    # runtime without editing the source checkout or rebasing the checkpoint.
    monkeypatch.setattr(recovery,'training_source_files',lambda root:list(original(root))[1:])
    with pytest.raises(ValueError,match='fingerprint'):
        adapter.fit_resumable(ctx,control=control,checkpoint=checkpoint,save_checkpoint=lambda writer:None)
