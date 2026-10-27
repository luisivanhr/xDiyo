"""Focused corrected-clock, causal-input and full-inner-monitor receipts."""
from pathlib import Path
import json,os,sys,subprocess,time
import numpy as np
import pandas as pd
import torch
from . import core,metrics
from .backend import AllLeagueBackend,context_for
from gat_cross_league.initialization import raw_predictions
from gat_cross_league import core as previous
from gat_suite.fitting import setup
from compare_gat_temporal import MemoryProbe,timed
from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
from xdiyo_analytics.graphs.streaming import GroupPlan
from temporal_team_gat import make_context

def checks(run):
    run=Path(run);p,store=core.load(run);old_p,old=previous.load(core.PARENT,frozen=True);setup(p)
    cfg=core.config('time_C');records=[];coverage=[]
    for event,anchor in [(11234494,'2023-04-30T16:30:00Z'),(13631351,'2025-04-23T18:30:00Z')]:
        query=store.schedule[store.schedule.event_id.eq(event)].reset_index(drop=True)
        assert len(query)==1
        issue=store.issue_times(query,cfg).iloc[0];assert issue==pd.Timestamp(anchor)
        old_query=old.schedule[old.schedule.event_id.eq(event)]
        old_issue=old.issue_times(old_query,previous.config()).iloc[0]
        teams=[int(query.home_id.iloc[0]),int(query.away_id.iloc[0])]
        snap=store.snapshot(issue,cfg,teams);old_snap=old.snapshot(old_issue,previous.config(),teams)
        late=snap.edges[snap.edges.competition_id.eq(38)&snap.edges.kickoff_at.ge(old_issue)]
        assert len(late)>0 and not snap.edges.event_id.eq(event).any()
        assert (snap.edges.kickoff_at<snap.cutoff).all()
        history=store.history;future=history.kickoff_at.ge(snap.cutoff)
        assert history.loc[future,'event_id'].eq(event).any()
        history.loc[future & history.side.eq('home'),['goals_for','goals_against']]=[99,1]
        history.loc[future & history.side.eq('away'),['goals_for','goals_against']]=[1,99]
        changed=core.ComponentStore(history,store.team_seasons,schedule=store.schedule,provenance=store.provenance)
        changed_snap=changed.snapshot(issue,cfg,teams)
        pd.testing.assert_frame_equal(snap.edges,changed_snap.edges)
        np.testing.assert_array_equal(snap.log_weight,changed_snap.log_weight)
        np.testing.assert_array_equal(snap.support,changed_snap.support)
        state=json.loads((run/'shared/preprocessing/fold_2.json').read_text())['state'];pre=GraphPreprocessor.from_state(cfg,state)
        initial=torch.load(core.initial_path(run,dict(configuration='time_C',fold='fold_2',seed=17)),map_location='cpu',weights_only=True)
        outputs=[raw_predictions(s,query,cfg,pre,initial,'cuda:0') for s in (store,changed)]
        for one,two in zip(*outputs):np.testing.assert_allclose(one,two,atol=1e-6,rtol=1e-5)
        records.append(dict(event=event,old_issue=old_issue,corrected_issue=issue,cutoff=snap.cutoff,
            old_edges=len(old_snap.edges),new_edges=len(snap.edges),eligible_late_belgian_edges=len(late),
            target_and_future_input_perturbation_exact=True,prediction_atol=1e-6,prediction_rtol=1e-5,
            max_output_error=max(float(np.max(np.abs(a-b))) for a,b in zip(*outputs))))
    for fold in core.FOLDS:
        for phase,frame in core.populations(run,fold).items():
            issue=store.issue_times(frame,cfg)
            pd.testing.assert_series_equal(store.issue_times(frame.iloc[::17],cfg),issue.iloc[::17])
            store.assert_targets_unobserved(frame,issue,cfg)
            numeric,_=GroupPlan(store,cfg,make_context(frame),issue).fitting_inputs()
            for league,indices in frame.groupby('competition_id').indices.items():
                row=dict(fold=fold,phase=phase,competition_id=int(league),count=len(indices))
                for name,column in zip(('neutral_venue','home_rest_days','away_rest_days','home_support','away_support'),numeric.T):
                    row[name+'_missing']=int(np.isnan(column[indices]).sum())
                coverage.append(row)
    pd.DataFrame(coverage).to_csv(run/'shared/checks/fixture_coverage.csv',index=False)
    assert store.version!=old.version and cfg.round_keys!=previous.config().round_keys
    # A newly constructed backend owns its own prepared groups; all immutable
    # context/version and resume identities include the corrected store/population.
    core.write_json(run/'shared/checks/correctness.json',dict(status='passed',real_examples=records,
        old_store=old.version,new_store=store.version,cache_policy='New store/version, stage round keys and object-scoped PreparedGroups',
        target_exclusion_all_six_populations=True,filtered_anchors_preserved=True,coverage_rows=len(coverage)))
    print(json.dumps(dict(status='causal_checks_passed',examples=len(records),coverage_rows=len(coverage))),flush=True)

def profile(run):
    run=Path(run);p,store=core.load(run);setup(p)
    job=next(j for j in p['matrix'] if j['configuration']=='time_C' and j['fold']=='fold_2' and j['seed']==17)
    context,pop=context_for(run,job);backend=AllLeagueBackend(run,p,store,job)
    print(json.dumps(dict(stage='profile_initialize',pid=os.getpid())),flush=True)
    with MemoryProbe(p['resources']['rss_limit_gib']) as probe:
        _,preparation=timed(lambda:backend.initialize(context),'cuda:0');updates=[]
        print(json.dumps(dict(stage='profile_initialized',seconds=preparation)),flush=True)
        for step in range(3):
            before=backend._validation_forward_calls
            values,seconds=timed(lambda:backend.checkpoint_metrics(core.MONITOR,backend.step()),'cuda:0');updates.append(seconds)
            assert backend._validation_forward_calls-before==len(backend.validation_groups)
            probability=metrics.base.normalized(backend._inner_probability);y=core.truth(pop['inner'])
            independent=-np.log(probability[np.arange(len(y)),y])
            np.testing.assert_allclose(values[core.MONITOR],independent.mean(),rtol=1e-6,atol=1e-7)
            summary=metrics.summarize(independent,pop['inner'].competition_id)
            assert len(summary['per_league'])==13
            np.testing.assert_allclose(summary['macro'],values['all13_inner_raw_hda_ce_macro'],rtol=1e-6,atol=1e-7)
            for league,value in summary['per_league'].items():np.testing.assert_allclose(value,values[f'inner_raw_ce_league_{league}'],rtol=1e-6,atol=1e-7)
            print(json.dumps(dict(stage='profile_update',step=step+1,seconds=seconds,all13_monitor_verified=True)),flush=True)
        result=dict(disposable=True,outer_predictions_produced=False,fit_fixtures=len(pop['fit']),inner_fixtures=len(pop['inner']),
            inner_leagues=13,validation_once_per_update=True,monitor=core.MONITOR,preparation_seconds=preparation,update_seconds=updates,
            memory=probe.result(),cuda_allocated_peak=torch.cuda.max_memory_allocated(),cuda_reserved_peak=torch.cuda.max_memory_reserved(),
            maximum_18_by_200_update_hours=float(np.mean(updates[1:])*18*200/3600))
    core.write_json(run/'shared/checks/profile.json',result);print(json.dumps(result),flush=True)

def freeze(run):
    run=Path(run);p,_=core.load(run)
    if (run/'FREEZE.json').exists():raise ValueError('Already frozen')
    for name in ('correctness.json','profile.json','focused_tests.txt','regression.txt'):
        path=run/'shared/checks'/name;assert path.exists(),name
        if name.endswith('.txt'):
            content=path.read_text();assert 'passed' in content and 'FAILED' not in content and 'ERROR' not in content,name
    from .analysis import contrasts
    p['contrasts']=contrasts()
    p['scoring']=dict(seed_reduction='Separately scored seeds, then arithmetic mean; no probability ensemble',
        combined_micro_weights=[4315/8590,4275/8590],macro='Original fixture-weighted folds within each league; equal 13-league mean',
        calibration='League draw intercept fit only to own corrected inner predictions after raw checkpoint and configuration selection',
        rps='Normalized by 2; home/draw/away order; secondary only',intervals='CE only; paired calendar weeks, fixed original league/fold weights')
    core.write_json(run/'PROTOCOL.json',p)
    sources=list((core.ROOT/'src/xdiyo_analytics').rglob('*.py'))
    for package in ('gat_all_league','gat_cross_league','gat_suite'):sources+=list((core.ROOT/'examples'/package).glob('*.py'))
    sources += [core.ROOT/f'examples/{name}.py' for name in ('run_gat_all_league','compare_gat_temporal','compare_gat_goals','temporal_team_gat')]
    sources += [core.ROOT/'tests/analytics/test_gat_all_league.py']
    core.write_json(run/'FREEZE.json',dict(protocol_sha256=core.digest(run/'PROTOCOL.json'),
        sources={f.relative_to(core.ROOT).as_posix():core.digest(f) for f in sources},
        inputs={f.relative_to(run).as_posix():core.digest(f) for base in ('data','populations','preprocessing','initialization','bootstrap')
            for f in (run/'shared'/base).rglob('*') if f.is_file()},frozen_at=pd.Timestamp.now(tz='UTC').isoformat()))
    import sklearn,psutil
    core.write_json(run/'ENVIRONMENT.json',dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,
        host_ram_gib=psutil.virtual_memory().total/2**30,deterministic=True,precision='native highest',cublas_workspace=':4096:8',
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=core.ROOT,text=True).strip()))
    core.status(run,stage='frozen')
