"""Focused cross-league population, semantic pairing, gradient and calendar risks."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'examples'))
from test_temporal_graph import sample
from gat_cross_league import core,bootstrap
from gat_cross_league.initialization import expand_state,feature_names,raw_predictions
from gat_cross_league.execution import calibration_by_league
from gat_cross_league.data import within_window
from gat_cross_league.components import ComponentStore
from gat_suite import metrics
from temporal_team_gat import make_context
from xdiyo_analytics.graphs import TemporalEventStore
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
from xdiyo_analytics.graphs.network import TemporalNetwork


def two_leagues():
    first=sample();h=first.history;h['is_awarded']=False
    foreign=h.copy();foreign['competition_id']=20
    foreign['team_id']+=10;foreign['opponent_id']+=10
    foreign['kickoff_at']+=pd.Timedelta(hours=2)
    m=first.team_seasons;other=m.copy();other['competition_id']=20;other['team_id']+=10
    return TemporalEventStore(pd.concat([h,foreign],ignore_index=True),pd.concat([m,other],ignore_index=True))


def backend(store,frame,name='time_C',state=None):
    cfg=replace(core.config(name),embedding_dim=4,hidden_width=8)
    model=_Backend(cfg,store,17,'cpu');model.initialize(make_context(frame.reset_index(drop=True)))
    if state is not None:model.model.load_state_dict(state)
    return model


def fitting(store):
    s=store.schedule
    return s[((s.competition_id==10)&s.event_id.isin([2,3,4]))|((s.competition_id==20)&s.event_id.isin([2,3]))].reset_index(drop=True)


def gradients(model):
    return {n:p.grad.clone() if p.grad is not None else None for n,p in model.named_parameters()}


def test_global_fixture_reduction_and_foreign_supervision():
    store=two_leagues();frame=fitting(store);b=backend(store,frame)
    b.optimizer.zero_grad(set_to_none=True)
    native=b._loss(b.train_groups,b.train_targets,b.train_weights,True)
    native_goal=b._goal_loss;g=gradients(b.model)
    b.optimizer.zero_grad(set_to_none=True);outcomes=[];goals=[]
    for i in range(len(b.train_groups)):
        pos,data,h,a,f=b.train_groups[i]
        logits,rates=b.model(data,h,a,f,return_goals=True)
        outcomes.append(torch.nn.functional.cross_entropy(logits,b.train_targets[pos],reduction='none'))
        goals.append((rates-b.train_goals[pos]*rates.log()).sum(1))
    outcome=torch.cat(outcomes).mean();goal=torch.cat(goals).mean()
    (outcome+.1*goal).backward()
    np.testing.assert_allclose([native,native_goal],[outcome.item(),goal.item()],rtol=1e-6)
    for name,grad in gradients(b.model).items():
        if grad is not None:torch.testing.assert_close(grad,g[name],atol=1e-6,rtol=1e-5)
    foreign=frame.competition_id.eq(20).to_numpy()
    b.train_targets[foreign]=(b.train_targets[foreign]+1)%3
    b.train_goals[foreign]=torch.tensor([3.,1.])
    b.optimizer.zero_grad(set_to_none=True)
    changed=b._loss(b.train_groups,b.train_targets,b.train_weights,True)
    assert abs(changed-native)>1e-5
    assert not torch.allclose(b.model.global_prior.grad,g['global_prior'])
    # Label perturbation only, with all context/input tensors held fixed.
    local=frame[~foreign].reset_index(drop=True);local_b=backend(store,local)
    local_before=local_b._loss(local_b.train_groups,local_b.train_targets,local_b.train_weights)
    unused_foreign_targets=frame[foreign].assign(goals_for=99,goals_against=0)
    assert len(unused_foreign_targets)==2
    assert local_b._loss(local_b.train_groups,local_b.train_targets,local_b.train_weights)==local_before
    assert len(frame)==5 and len(set(core.keys(frame)))==5 and frame.event_id.duplicated().any()


def test_auxiliary_routing_and_beta_zero_optimizer_isolation():
    store=two_leagues();frame=fitting(store);b=backend(store,frame)
    pos,data,h,a,f=b.train_groups[1]
    logits,rates=b.model(data,h,a,f,return_goals=True)
    ((rates-b.train_goals[pos]*rates.log()).sum(1).mean()).backward()
    assert b.model.global_prior.grad.abs().sum()>0
    assert b.model.goal_head.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in b.model.head.parameters())
    b.optimizer.zero_grad(set_to_none=True)
    logits,rates=b.model(data,h,a,f,return_goals=True)
    torch.nn.functional.cross_entropy(logits,b.train_targets[pos]).backward()
    assert all(p.grad is None for p in b.model.goal_head.parameters())
    control=backend(store,frame,'time_B');before={n:v.clone() for n,v in control.model.goal_head.state_dict().items()}
    reference=_Backend(replace(control.config,goal_auxiliary=None),store,17,'cpu')
    reference.initialize(make_context(frame))
    reference.model.load_state_dict({n:v for n,v in control.model.state_dict().items() if not n.startswith('goal_head.')})
    for _ in range(3):
        control.step();reference.step()
        for name,value in reference.model.state_dict().items():
            torch.testing.assert_close(value,control.model.state_dict()[name],atol=1e-6,rtol=1e-5)
    for name,value in control.model.goal_head.state_dict().items():torch.testing.assert_close(value,before[name],atol=0,rtol=0)


def test_semantic_expansion_not_leading_columns_and_rng_preserved():
    b=backend(two_leagues(),fitting(two_leagues()),'snapshot_B')
    old=b.preprocessor;state=old.state();state['leagues']=[[5,1],[10,2],[20,3],[35,4]]
    new=GraphPreprocessor.from_state(old.config,state)
    rng=torch.random.get_rng_state().clone()
    expanded=expand_state(b.model.state_dict(),old,new,123)
    torch.testing.assert_close(rng,torch.random.get_rng_state(),atol=0,rtol=0)
    old_names,new_names=feature_names(old),feature_names(new)
    for name in ('layers.0.e.weight','layers.0.u.weight','head.0.weight','goal_head.weight'):
        kind='edge' if name.startswith('layers') else 'joint'
        for i,key in enumerate(old_names[kind]):
            torch.testing.assert_close(expanded[name][:,new_names[kind].index(key)],b.model.state_dict()[name][:,i],atol=0,rtol=0)
        for key in ('competition:5','competition:35'):
            assert expanded[name][:,new_names[kind].index(key)].count_nonzero()==0
    assert expanded['league.weight'][0].count_nonzero()==0
    temporal=GraphPreprocessor.from_state(replace(old.config,temporal_mode='time_features'),state)
    ts=expand_state(expanded,new,temporal,456)
    for name,value in ts.items():
        if name.endswith(('.e.weight','.u.weight')):assert value[:,-14:].count_nonzero()==0


def test_foreign_future_perturbation_and_original_round_anchors():
    store=two_leagues();cfg=core.config('time_C');frame=fitting(store)
    b=backend(store,frame);query=store.schedule[(store.schedule.competition_id==20)&(store.schedule.event_id==3)].reset_index(drop=True)
    h=store.history;future=(h.competition_id==20)&h.event_id.eq(9)
    h.loc[future & h.side.eq('home'),['goals_for','goals_against']]=[99,1]
    h.loc[future & h.side.eq('away'),['goals_for','goals_against']]=[1,99]
    changed=TemporalEventStore(h,store.team_seasons,schedule=store.schedule)
    one=raw_predictions(store,query,b.config,b.preprocessor,b.model.state_dict())
    two=raw_predictions(changed,query,b.config,b.preprocessor,b.model.state_dict())
    for a,z in zip(one,two):np.testing.assert_array_equal(a,z)
    schedule=store.schedule;issue=store.issue_times(schedule,cfg)
    pd.testing.assert_series_equal(store.issue_times(schedule.iloc[::3],cfg),issue.iloc[::3])
    available=store.label_availability(schedule,cfg)
    take=within_window(store,schedule,issue.iloc[2],available.iloc[-3],cfg)
    assert (issue[take]>=issue.iloc[2]).all() and (available[take]<=available.iloc[-3]).all()


def test_per_league_calibration_isolation_and_rank_invariance():
    frame=fitting(two_leagues());p=np.random.default_rng(7).dirichlet([2,2,2],len(frame))
    first=calibration_by_league(frame,p)
    changed=frame.copy();changed.loc[changed.competition_id.eq(20),['goals_for','goals_against']]=[9,9]
    other=calibration_by_league(changed,p)
    assert first['10']==other['10'] and first['20']['intercept']!=other['20']['intercept']
    for league,ix in frame.groupby('competition_id').indices.items():
        q=metrics.calibrate(p[ix],first[str(league)]['intercept'])
        np.testing.assert_array_equal(np.argsort(q[:,1]),np.argsort(p[ix,1]))
        rows=metrics.outcome_rows(q,core.truth(frame.iloc[ix]))
        np.testing.assert_allclose(rows.log_loss,rows.draw_log_loss+rows.winner_contribution)


def test_calendar_groups_breaks_and_schedule_only_rejection():
    issue=pd.to_datetime(['2020-01-01T10:00Z','2020-01-01T10:00Z','2020-02-03T12:00Z'])
    weeks,ix=bootstrap.week_groups(issue)
    assert len(weeks)==6 and ix.tolist()==[0,0,5]
    counts=np.array([[1,0,0],[0,0,0],[0,0,0],[0,1,0]])
    samples,receipt=bootstrap.calendar_samples(counts,2,np.random.default_rng(7),replicates=100,cap=10000)
    assert (samples@counts[:,:2]>0).all() and receipt['unavailable_strata']==[2]
    assert receipt['rejected']>0
    failed,receipt=bootstrap.calendar_samples(counts,2,np.random.default_rng(7),replicates=100,cap=1)
    assert failed is None and not receipt['interval_available']


def test_macro_micro_original_fold_weights_and_absence():
    scores=np.array([[1.,2.],[3.,4.]]);counts=np.array([[1.,6.],[3.,1.]])
    result=bootstrap.fixed_mixture(scores,counts)
    np.testing.assert_allclose(result['per_league'],[2.5,16/7])
    np.testing.assert_allclose(result['macro'],(2.5+16/7)/2)
    np.testing.assert_allclose(result['micro'],26/11)
    absent=bootstrap.fixed_mixture(np.array([[1.,np.nan],[3.,np.nan]]),np.array([[1.,0.],[3.,0.]]))
    assert absent['macro'] is None and not absent['all_leagues_supported'] and absent['micro']==2.5


def test_feature_whitelist_and_fixed_matrix():
    assert len(core.matrix())==36
    for name in core.CONFIGS:
        cfg=core.config(name)
        assert cfg.layers==2 and cfg.statistics==() and cfg.fixture_features==()
        assert cfg.include_goal_features and cfg.heatmaps is None and cfg.statistics_forecast is None and cfg.boundary_memory is None
        assert cfg.goal_auxiliary.architecture=='linear'
        assert cfg.goal_auxiliary.beta==(.1 if name=='time_C' else 0)


def test_fresh_cuda_worker_selection_restore_and_exports(tmp_path):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    import json,subprocess,os
    from compare_gat_temporal import write_json,digest
    from gat_cross_league.reporting import audited_fit
    run=tmp_path/'disposable';data=run/'shared/data';data.mkdir(parents=True)
    original=sample();h=original.history;m=original.team_seasons
    h['competition_id']=35;m['competition_id']=35
    store=TemporalEventStore(h,m)
    for name,frame in [('history',store.history),('memberships',store.team_seasons),('schedule',store.schedule)]:frame.to_parquet(data/f'{name}.parquet',index=False)
    job=dict(job_id='disposable',configuration='time_C',fold='fold_1',seed=17,arm='pooled',path='02_all13_supervision/disposable')
    folder=run/'shared/populations';folder.mkdir()
    frames={phase:store.schedule.iloc[ix].reset_index(drop=True) for phase,ix in [('fit',[1,2,3]),('inner',[6,7]),('outer',[8])]}
    for arm in core.ARMS:
        for phase,frame in frames.items():frame.to_parquet(folder/f'fold_1_{arm}_{phase}.parquet',index=False)
    model=_Backend(core.config('time_C'),store,17,'cpu');model.initialize(make_context(frames['fit']))
    (run/'shared/preprocessing').mkdir();write_json(run/'shared/preprocessing/fold_1.json',dict(state=model.preprocessor.state()))
    path=run/'shared/initialization/fold_1/seed_17';path.mkdir(parents=True)
    torch.save(model.model.state_dict(),path/'time_features.pt')
    from dataclasses import asdict
    p=dict(provenance=store.provenance,store_version=store.version,matrix=[job],competitions=[35],
        configurations={'time_C':asdict(core.config('time_C'))},
        resources=dict(device='cuda:0',threads=1,input_cache_bytes=32*2**20,headroom_bytes=2**30,rss_limit_gib=8.,cuda_fraction=.8,complete_components=False))
    write_json(run/'PROTOCOL.json',p)
    write_json(run/'FREEZE.json',dict(protocol_sha256=digest(run/'PROTOCOL.json'),source_hashes={},input_hashes={}))
    env=os.environ.copy();env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1')
    result=subprocess.run([sys.executable,str(core.ROOT/'examples/run_gat_cross_league.py'),'fit','--run',str(run),'--job','disposable'],
        capture_output=True,text=True,env=env,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    receipt,predictions,curve=audited_fit(run,job)
    assert len(predictions['outer'])==1 and receipt['reload_exact']
    assert 'bundesliga_train_loss' in curve
    continuation=torch.load(run/job['path']/'attempt_01/last_training.pt',weights_only=True)
    assert continuation['active']['step']==receipt['summary']['attempts'][0]['steps']
    assert continuation['active']['best_step']==receipt['summary']['best_step']


def test_primary_bootstrap_scores_seeds_and_common_fixture_weights(tmp_path):
    from gat_cross_league.reporting import primary_samples,NAMES
    from compare_gat_temporal import write_json
    folder=tmp_path/'shared/bootstrap/bundesliga/fold_1';folder.mkdir(parents=True)
    counts=np.array([[1,1,1,1],[2,0,2,0],[0,1,2,1],[1,0,0,3]],dtype=np.int16)
    np.save(folder/'blocks_4.npy',counts);write_json(folder/'rounds.json',[1,2,3,4])
    (tmp_path/'04_comparisons').mkdir()
    y=np.array([0,1,0,1,2,1,2,1]);rounds=np.repeat([1,2,3,4],2);frames=[]
    for seed in core.SEEDS:
        p=np.random.default_rng(seed).dirichlet(np.ones(3),size=8)
        frame=pd.DataFrame(dict(y=y,round=rounds));frame[metrics.P_COLUMNS]=p
        for name,value in metrics.outcome_rows(p,y).items():frame[f'raw_{name}']=value
        frames.append(frame)
    point,samples,_,_=primary_samples(tmp_path,'fold_1','example','raw',frames,4)
    expected=np.mean([metrics.outcome_metrics(f[metrics.P_COLUMNS],y)['log_loss'] for f in frames])
    np.testing.assert_allclose(point[0],expected,atol=1e-14)
    ensemble=np.mean([f[metrics.P_COLUMNS] for f in frames],axis=0)
    assert abs(expected-metrics.outcome_metrics(ensemble,y)['log_loss'])>1e-3
    for i,count in enumerate(counts):
        indices=np.repeat(np.arange(8),count[rounds-1])
        independent=[metrics.outcome_metrics(f.iloc[indices][metrics.P_COLUMNS],y[indices]) for f in frames]
        for j,name in enumerate(NAMES):
            np.testing.assert_allclose(samples[i,j],np.mean([s[name] for s in independent]),atol=1e-13,rtol=1e-13)


def test_secondary_bootstrap_recomputes_stratum_denominators_with_fixed_mixture(tmp_path):
    from gat_cross_league.reporting import secondary
    p={'competitions':[10,20]};models={};manual=[];original=[]
    weights=np.array([[1,1],[2,0],[0,2],[1,1]],dtype=np.int16)
    for fold,multiplicity in [('fold_1',1),('fold_2',2)]:
        week=np.array([0,0,0,1,1,1,1]);league=np.array([10,10,20,10,20,20,20])
        frame=pd.DataFrame(dict(competition_id=league,season_id=900,event_id=np.arange(1,8),y=np.arange(7)%3))
        folder=tmp_path/'shared/populations';folder.mkdir(parents=True,exist_ok=True)
        for phase in ('fit','inner','outer'):frame.to_parquet(folder/f'{fold}_pooled_{phase}.parquet',index=False)
        base=tmp_path/'shared/bootstrap/calendar'/fold;base.mkdir(parents=True)
        counts=np.column_stack([np.bincount(week,weights=league==x,minlength=2) for x in p['competitions']])
        np.save(base/'fixture_week.npy',week);np.save(base/'counts_by_league.npy',counts)
        for length in (2,4,8):np.save(base/f'blocks_{length}.npy',weights)
        for key in [*core.CONFIGS,'league_logistic','pooled_logistic','league_frequency']:
            copies=[]
            for seed in ([1,2,3] if key in core.CONFIGS else [1]):
                f=frame.copy();value=np.arange(7)*.1+1+(seed*.07*multiplicity if key=='time_C' else 0)
                for view in ('raw','calibrated'):
                    for metric in metrics.ADDITIVE:f[f'{view}_{metric}']=value
                copies.append(f)
            models[fold,key]=copies
        difference=np.mean([f.raw_log_loss for f in models[fold,'time_C']],axis=0)-models[fold,'time_B'][0].raw_log_loss.to_numpy()
        samples=[]
        for row in weights:
            sampled=row[week]
            samples.append([np.average(difference[league==l],weights=sampled[league==l]) for l in p['competitions']])
        manual.append(np.array(samples));original.append(counts.sum(0))
    (tmp_path/'04_comparisons').mkdir()
    table=secondary(tmp_path,p,models)
    original=np.array(original);fixed=original/original.sum(0)
    expected=sum(values*fixed[i] for i,values in enumerate(manual)).mean(1)
    row=table[(table.contrast=='time_C - time_B')&(table.view=='raw')&(table.block_weeks==4)&(table.metric=='log_loss')&(table.aggregation=='macro')].iloc[0]
    np.testing.assert_allclose([row.lower,row.upper],np.quantile(expected,[.025,.975]),atol=1e-14)
