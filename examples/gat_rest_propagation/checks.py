"""Real archive provenance, matched native inputs, resource and freeze receipts."""
from pathlib import Path
import json,sys,subprocess,time
import numpy as np
import pandas as pd
import torch
from . import core,metrics
from .backend import AllLeagueBackend,context_for
from gat_suite.fitting import setup,predict
from compare_gat_temporal import MemoryProbe,timed
from temporal_team_gat import make_context
from xdiyo_analytics.graphs import TemporalGraphSerializer
from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
from xdiyo_analytics.graphs.streaming import GroupPlan,prepare_group
from xdiyo_analytics.graphs.observed_gap import encode_observed_gap

def checks(run):
    run=Path(run);p,store=core.load(run);setup(p);cfg=core.config();old_cfg=core.config('archived_time_B')
    distributions=[];examples=[];input_checks=[];legacy_checks=[]
    for fold in core.FOLDS:
        state=json.loads((run/'shared/preprocessing'/f'{fold}.json').read_text())['state']
        pre=GraphPreprocessor.from_state(cfg,state)
        old=GraphPreprocessor.from_state(old_cfg,json.loads((core.PARENT/'shared/preprocessing'/f'{fold}.json').read_text())['state'])
        assert pre.edge_scale.state()==old.edge_scale.state() and pre.fixture_scale.state()==old.fixture_scale.state()
        for phase,frame in core.populations(run,fold).items():
            issue=store.issue_times(frame,cfg);pd.testing.assert_series_equal(issue,store.issue_times(frame,old_cfg))
            store.assert_targets_unobserved(frame,issue,cfg)
            plan=GroupPlan(store,cfg,make_context(frame),issue)
            old_plan=GroupPlan(store,old_cfg,make_context(frame),issue)
            prior_plan=GroupPlan(store,core.config(core.PRIOR),make_context(frame),issue)
            records=[];seen=np.zeros(len(frame),int)
            for i in range(len(plan.descriptors)):
                raw=plan.raw(i);positions,snap,home,away,meta,numeric=raw
                old_raw=old_plan.raw(i)
                np.testing.assert_array_equal(numeric,old_raw[-1]);np.testing.assert_array_equal(positions,old_raw[0])
                assert snap.cutoff==old_raw[1].cutoff
                np.testing.assert_array_equal(snap.edges.index,old_raw[1].edges.index)
                assert not snap.edges.event_id.isin(meta.event_id).any()
                encoded=pre.fixture(meta,numeric);legacy_encoded=old.fixture(old_raw[4],numeric)
                assert encoded.shape==(len(positions),26) and np.isfinite(encoded).all()
                for j,name in enumerate(pre.fixture_encoded_columns):
                    if name in old.fixture_encoded_columns:
                        np.testing.assert_array_equal(encoded[:,j],legacy_encoded[:,old.fixture_encoded_columns.index(name)])
                if i in (0,len(plan.descriptors)//2,len(plan.descriptors)-1):
                    before=prepare_group(raw,pre,'cpu',store)
                    prior=GraphPreprocessor.from_state(core.config(core.PRIOR),state)
                    after=prepare_group(prior_plan.raw(i),prior,'cpu',store)
                    for name in ('x','receiver','sender','log_weight','gate','league','entry'):
                        torch.testing.assert_close(before[1][name],after[1][name],rtol=0,atol=0)
                    for a,b in zip(before[2:],after[2:]):torch.testing.assert_close(a,b,rtol=0,atol=0)
                table=meta[[*core.MATCH_KEYS,'home_id','away_id','kickoff_at']].copy()
                table['issuance_at']=issue.iloc[positions].to_numpy();table['cutoff_at']=snap.cutoff
                for side,col in (('home',1),('away',2)):
                    for suffix in ('observed_gap_days','latest_eligible_observed_at','has_eligible_observation'):
                        table[f'{side}_{suffix}']=meta[f'{side}_{suffix}'].to_numpy()
                    for suffix in ('observed_gap_r7','no_eligible_history','no_eligible_observation_30d'):
                        name=f'{side}_{suffix}';table[name]=encoded[:,pre.fixture_encoded_columns.index(name)]
                    table[f'{side}_legacy_standardized_gap']=legacy_encoded[:,col]
                if meta.event_id.eq(11425870).any():
                    row=table[table.event_id.eq(11425870)].iloc[0]
                    assert row.issuance_at==pd.Timestamp('2023-08-18T18:30Z')
                    assert row.cutoff_at==pd.Timestamp('2023-08-17T22:00Z')
                    assert row.away_latest_eligible_observed_at==pd.Timestamp('2021-05-10T12:00Z')
                    np.testing.assert_allclose(row.away_observed_gap_days,830.2708333333334,rtol=0,atol=1e-12)
                    np.testing.assert_array_equal([row.away_no_eligible_history,row.away_no_eligible_observation_30d],[0,1])
                    assert abs(row.away_observed_gap_r7-830.2708333333334/837.2708333333334)<1e-15
                    assert row.kickoff_at==pd.Timestamp('2023-08-20T18:30Z')
                    # Independently query eligible directed rows for this observed timestamp.
                    eligible=store.eligible(snap.cutoff)
                    assert eligible.loc[eligible.team_id.eq(row.away_id),'kickoff_at'].max()==row.away_latest_eligible_observed_at
                    examples.append(dict(fold=fold,phase=phase,**row.to_dict()))
                table['_position']=positions;records.append(table);seen[positions]+=1
            assert (seen==1).all()
            features=pd.concat(records).sort_values('_position').drop(columns='_position').reset_index(drop=True)
            assert core.same_keys(features,frame)
            folder=run/'01_rest_fix/feature_diagnostics';folder.mkdir(parents=True,exist_ok=True)
            features.to_parquet(folder/f'{fold}_{phase}.parquet',index=False)
            for league,part in features.groupby('competition_id'):
                for side in ('home','away'):
                    for name in ('observed_gap_days','observed_gap_r7','no_eligible_history','no_eligible_observation_30d','legacy_standardized_gap'):
                        x=part[f'{side}_{name}']
                        distributions.append(dict(fold=fold,phase=phase,competition_id=int(league),side=side,feature=name,
                            count=len(x),missing=int(x.isna().sum()),mean=float(x.mean()),
                            **{f'q{q:g}':float(x.quantile(q)) for q in (0,.01,.5,.9,.99,1)}))
            input_checks.append(dict(fold=fold,phase=phase,count=len(frame),issuance_groups=len(plan.descriptors),
                all_targets_once=True,raw_gap_arithmetic_exact=True,non_gap_inputs_exact=True,
                edge_indices_and_cutoffs_exact=True,matched_modes_input_probes=3))
            print(json.dumps(dict(stage='inputs_verified',fold=fold,phase=phase,count=len(frame))),flush=True)
        # Existing saved full-population model is loaded under explicit legacy semantics.
        folder=core.PARENT/'01_pooled/time_B'/fold/'seed_17'
        adapter=TemporalGraphSerializer(store,device='cuda:0').load(folder/'model')
        inner=core.populations(run,fold)['inner'];ctx=make_context(inner)
        got=predict(adapter.backend_,adapter.backend_._stream(ctx),inner)[0]
        expected=pd.read_parquet(folder/'inner_raw.parquet')
        assert core.same_keys(inner,expected)
        np.testing.assert_array_equal(metrics.base.normalized(got),expected[metrics.P_COLUMNS].to_numpy())
        legacy_checks.append(dict(fold=fold,seed=17,full_inner_prediction_exact=True,
            model_version=adapter.backend_.model_version,encoding=adapter.config.fixture_gap_encoding))
        del adapter
    pd.DataFrame(distributions).to_csv(run/'01_rest_fix/feature_diagnostics/distributions.csv',index=False)
    core.write_json(run/'01_rest_fix/feature_diagnostics/case_11425870.json',examples)
    # Baselines and archived time_B are reused only after exact IDs, labels and clock checks.
    reused=[]
    for fold in core.FOLDS:
        for phase,expected in core.populations(run,fold).items():
            paths=[core.PARENT/'01_pooled/time_B'/fold/f'seed_{seed}'/f'{phase}_predictions.parquet' for seed in core.SEEDS]
            root=core.PARENT/'02_baselines'/fold
            paths += [root/'logistic_pooled'/f'{phase}_predictions.parquet']
            paths += [root/f'{method}_{league}'/f'{phase}_predictions.parquet' for method in ('logistic','frequency') for league in p['competitions']]
            for path in paths:
                table=pd.read_parquet(path);subset=expected[expected.competition_id.isin(table.competition_id)]
                index=core.keys(table).get_indexer(core.keys(subset));assert (index>=0).all() and len(table)==len(subset)
                table=table.iloc[index].reset_index(drop=True);assert core.same_keys(table,subset)
                np.testing.assert_array_equal(table.y,core.truth(subset))
                np.testing.assert_array_equal(pd.to_datetime(table.issuance_at,utc=True),store.issue_times(subset,cfg))
                np.testing.assert_array_equal(pd.to_datetime(table.cutoff_at,utc=True),[cfg.history_cutoff(v) for v in store.issue_times(subset,cfg)])
                assert table[metrics.P_COLUMNS].notna().all().all()
                reused.append(dict(path=path.relative_to(core.PARENT).as_posix(),sha256=core.digest(path),rows=len(table)))
    core.write_json(run/'shared/checks/reused_predictions.json',reused)
    core.write_json(run/'shared/checks/correctness.json',dict(status='passed',inputs=input_checks,
        legacy_models=legacy_checks,real_examples=len(examples),reused_prediction_tables=len(reused),
        bootstrap='Parent bytes, target positions, local weeks, and per-league counts verified in preparation',
        schema=pre.fixture_schema,raw_gap_provenance_only=True,nominal_parameters=35993))
    baseline_reuse(run)

def baseline_reuse(run):
    """Compact fixed probes of the six unchanged rolling-goal inputs and design."""
    import joblib
    from gat_suite.baselines import rolling_features
    from gat_cross_league.baselines import design
    run=Path(run);p,store=core.load(run);rows=[]
    for fold in core.FOLDS:
        folder=core.PARENT/'02_baselines'/fold;transform=joblib.load(folder/'rolling_transform.joblib')
        for phase,frame in core.populations(run,fold).items():
            # First and last fixture of every league, determined without scores.
            positions=np.unique(np.concatenate([np.array([ix[0],ix[-1]]) for ix in frame.groupby('competition_id').indices.values()]))
            probe=frame.iloc[positions]
            a=rolling_features(store,core.config(),probe);b=rolling_features(store,core.config('archived_time_B'),probe)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(a,np.load(folder/f'{phase}_rolling.npy')[positions])
            np.testing.assert_array_equal(design(a,probe,transform,p['competitions']),np.load(folder/f'{phase}_design.npy')[positions])
            rows.append(dict(fold=fold,phase=phase,probes=len(probe),rolling_columns=6,exact=True))
    core.write_json(run/'shared/checks/baseline_input_reuse.json',dict(status='passed',probes=rows,
        claim='Six rolling-goal inputs and fixed fitted transform; no native fixture gap is used'))

def profile(run):
    run=Path(run);p,store=core.load(run);setup(p)
    job=next(j for j in p['matrix'] if j['configuration']==core.FULL and j['fold']=='fold_2' and j['seed']==17)
    context,pop=context_for(run,job);backend=AllLeagueBackend(run,p,store,job)
    with MemoryProbe(p['resources']['rss_limit_gib']) as probe:
        _,preparation=timed(lambda:backend.initialize(context),'cuda:0');updates=[]
        for step in range(3):
            before=backend._validation_forward_calls
            values,seconds=timed(lambda:backend.checkpoint_metrics(core.MONITOR,backend.step()),'cuda:0');updates.append(seconds)
            assert backend._validation_forward_calls-before==len(backend.validation_groups)
            probability=metrics.base.normalized(backend._inner_probability);y=core.truth(pop['inner'])
            independent=-np.log(probability[np.arange(len(y)),y])
            np.testing.assert_allclose(values[core.MONITOR],independent.mean(),rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(metrics.summarize(independent,pop['inner'].competition_id)['macro'],values['all13_inner_raw_hda_ce_macro'],rtol=1e-6,atol=1e-7)
            assert all(parameter.grad is None for parameter in backend.model.goal_head.parameters())
            print(json.dumps(dict(stage='profile_update',step=step+1,seconds=seconds,monitor_verified=True)),flush=True)
        result=dict(disposable=True,outer_predictions_produced=False,preparation_seconds=preparation,update_seconds=updates,
            memory=probe.result(),cuda_allocated_peak=torch.cuda.max_memory_allocated(),cuda_reserved_peak=torch.cuda.max_memory_reserved(),
            upper_12_by_200_update_hours=float(np.mean(updates[1:])*12*200/3600),validation_once_per_update=True,
            goal_head_gradient_none=True,monitor=core.MONITOR)
    core.write_json(run/'shared/checks/profile.json',result);print(json.dumps(result),flush=True)

def freeze(run):
    run=Path(run);p,_=core.load(run)
    if (run/'FREEZE.json').exists():raise ValueError('Already frozen')
    for name in ('correctness.json','profile.json','focused_tests.txt','regression.txt','baseline_input_reuse.json','additional_test_resolution.json'):
        path=run/'shared/checks'/name;assert path.exists(),name
        if name.endswith('.txt'):
            content=path.read_text();assert 'passed' in content and 'FAILED' not in content and 'ERROR' not in content,name
    from .analysis import contrasts
    p['contrasts']=contrasts()
    p['scoring']=dict(seed_reduction='Score seeds separately, then arithmetic mean; never a probability ensemble',
        micro_weights=[4315/8590,4275/8590],macro='Original within-league fold mixture then equal 13-league mean',
        calibration='Per-league draw intercept from own inner after raw selection',
        uncertainty='CE and RPS for six contrasts; draw CE for full-minus-prior; raw and calibrated; 4-week primary, 2/8 sensitivity')
    core.write_json(run/'PROTOCOL.json',p)
    sources=list((core.ROOT/'src/xdiyo_analytics').rglob('*.py'))
    sources.append(core.ROOT/'src/xdiyo_analytics/ui/inventory.json')
    for package in ('gat_rest_propagation','gat_all_league','gat_cross_league','gat_suite'):sources+=list((core.ROOT/'examples'/package).glob('*.py'))
    sources += [core.ROOT/f'examples/{name}.py' for name in ('run_gat_rest_propagation','compare_gat_temporal','compare_gat_goals','temporal_team_gat')]
    sources += [core.ROOT/f'tests/analytics/{name}.py' for name in ('test_graph_observed_gap','test_graph_sender_context','test_gat_rest_propagation')]
    reused=json.loads((run/'shared/checks/reused_predictions.json').read_text())
    parent_hashes={v['path']:v['sha256'] for v in reused}
    for path in (core.PARENT/'02_baselines').rglob('*'):
        if path.is_file() and path.suffix in ('.joblib','.npy','.json'):
            parent_hashes[path.relative_to(core.PARENT).as_posix()]=core.digest(path)
    for name in ('PROTOCOL.json','COMPLETE.json'):parent_hashes[name]=core.digest(core.PARENT/name)
    core.write_json(run/'FREEZE.json',dict(protocol_sha256=core.digest(run/'PROTOCOL.json'),
        sources={f.relative_to(core.ROOT).as_posix():core.digest(f) for f in sources},
        inputs={f.relative_to(run).as_posix():core.digest(f) for base in ('data','populations','preprocessing','initialization','bootstrap')
                for f in (run/'shared'/base).rglob('*') if f.is_file()},reused_parent=parent_hashes,
        frozen_at=pd.Timestamp.now(tz='UTC').isoformat()))
    import sklearn,psutil
    core.write_json(run/'ENVIRONMENT.json',dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,
        host_ram_gib=psutil.virtual_memory().total/2**30,deterministic=True,precision='native highest',cublas_workspace=':4096:8',
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=core.ROOT,text=True).strip()))
    core.status(run,stage='frozen')
