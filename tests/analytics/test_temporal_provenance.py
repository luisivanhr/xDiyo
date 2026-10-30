"""Temporal correctness, distinct from legacy numerical accounting."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.evaluation import (
    Parlay, MultiBet, BetSlip, compose_bets, FixedStake, FractionalKelly,
    StakeContext, FrozenTable, allocate_batch, replay_bankroll, settlements_from_legs,
)
from xdiyo_analytics.evaluation.temporal_provenance import (
    bind_forecasts, guard_candidates, validate_frame, require_verified_comparison,
    save_temporal_table, load_temporal_table, dependency_hash, CERT, POOL, STATUS, EVIDENCE,
)
from xdiyo_analytics.evaluation.ticket_allocation import ticket_batch, allocate_singles
from test_bet_tickets import ledger

T = '2024-12-31T00:00:00+00:00'
TRAIN = '2024-12-29T00:00:00+00:00'


def population(n=3):
    f = ledger(('win',)*n)
    f['decision_at'] = T
    f['forecast_as_of'] = T
    f['first_kickoff_at'] = f.kickoff_at
    f['feature_hash'] = [f'features-{i}' for i in range(n)]
    f['model_hash'], f['preprocessing_hash'] = 'model-v1', 'scaler-v1'
    f['quote_id'] = [f'quote-{i}' for i in range(n)]
    f['quote_at'], f['market'], f['selection'] = TRAIN, 'total_goals', 'over'
    f['quote_snapshot_hash'], f['quote_crosswalk_hash'] = 'prices-v1','mapping-v1'
    evidence = {}
    for i,row in f.iterrows():
        stages = [dict(kind=k,mode='not_used',reason='No such stage in this model')
                  for k in ('ratings','calibration','tuning')]
        for k,h in [('preprocessing','scaler-v1'),('fitting','model-v1')]:
            stages.append(dict(kind=k,mode='fitted',artifact_hash=h,training_cutoff=TRAIN,
                available_as_of=TRAIN,created_at=TRAIN,fit_rows_hash='exact-training-row-ids',
                fit_partition='train',inputs=[dict(role='training_labels',hash='labels-v1',available_as_of=TRAIN)]))
        evidence[i] = dict(generated_at=T,forecast_as_of=T,model_available_as_of=TRAIN,
            assumptions=[],stages=stages,dependencies=[dict(id=role,role=role,
                hash=row.feature_hash if role=='features' else f'{role}-v1',reference=f'artifacts/{role}',
                available_as_of=TRAIN,availability_kind='observed')
                for role in ('features','source','schedule','membership')])
        for dep in evidence[i]['dependencies']:
            dep['content'] = {'fixture_id':int(row.event_id),'data':dep['hash']}
            dep['hash'] = dependency_hash(dep['content'])
            if dep['role']=='features': f.loc[i,'feature_hash']=dep['hash']
        for stage in stages:
            if stage['mode']=='fitted':
                content={'model':stage['artifact_hash']}
                stage['artifact_hash']=dependency_hash(content)
                stage['artifact']={'hash':stage['artifact_hash'],'content':content}
                stage['fit_rows_hash']=dependency_hash([1,2])
                stage['fit_rows']={'hash':stage['fit_rows_hash'],'content':[1,2]}
                f.loc[i,'model_hash' if stage['kind']=='fitting' else 'preprocessing_hash']=stage['artifact_hash']
                for dep in stage['inputs']:
                    dep['rows_hash']=stage['fit_rows_hash']
                    dep['content']={'training_rows':[1,2], 'data':dep['hash']}
                    dep['hash']=dependency_hash(dep['content'])
    return f,evidence


def certified(n=3):
    f,e = population(n)
    return bind_forecasts(f,e)


def policy(**kwargs):
    return Parlay(size=2,probability_mode='independent',payoff='binary',**kwargs)


def test_verified_flat_and_exact_identity_persistence(tmp_path):
    f = certified()
    t,m,metrics = compose_bets(f,policy())
    assert t[STATUS].eq('verified').all()
    assert m[STATUS].eq('verified').all()
    assert metrics.comparison_eligible.all()
    require_verified_comparison(t,m)
    path = tmp_path/'tickets.parquet'
    save_temporal_table(t,path)
    restored = load_temporal_table(path)
    assert restored.ticket_id.tolist()==t.ticket_id.tolist()
    assert restored[STATUS].eq('verified').all()
    assert json.loads(t[CERT].iloc[0])['payload']['legs'][0]['event_id'] > 2**63
    from xdiyo_analytics.experiments.recovery import dump_bundle,load_bundle
    dump_bundle(tmp_path/'run.json',{'tickets':t,'members':m})
    again=load_bundle(tmp_path/'run.json')
    require_verified_comparison(again['tickets'],again['members'])
    # A false saved success label cannot turn legacy accounting into verification.
    old=ledger().assign(temporal_status='verified',comparison_eligible=True)
    dump_bundle(tmp_path/'legacy.json',old)
    assert load_bundle(tmp_path/'legacy.json')[STATUS].eq('unverified').all()


@pytest.mark.parametrize('field,value',[
    ('decision_at','2025-01-01T00:00:00Z'), # kickoff equality
    ('decision_at','2025-01-02T00:00:00Z'),
    ('forecast_as_of','2024-12-31T01:00:00Z'),
    ('quote_at','2024-12-31T01:00:00Z'),
    ('issued_at','2024-12-31T01:00:00Z'),
    ('model_available_as_of','2024-12-31T01:00:00Z'),
    ('features_available_as_of','2024-12-31T01:00:00Z'),
    ('calibration_available_as_of','2024-12-31T01:00:00Z'),
])
def test_discarded_candidate_contradiction_before_gate_free_selection(field,value):
    f,_=population()
    f.loc[0,'take']=False
    f.loc[0,field]=value
    with pytest.raises(ValueError,match='Temporal provenance'):
        compose_bets(f,policy())


def test_resumed_first_start_and_postponed_long_gap():
    f,_=population(2)
    f.loc[1,'kickoff_at']=pd.Timestamp('2025-06-01',tz='UTC')
    f.loc[1,'first_kickoff_at']=pd.Timestamp('2024-12-20',tz='UTC')
    with pytest.raises(ValueError,match='first_kickoff_at'):
        compose_bets(f,policy())
    f.loc[1,'first_kickoff_at']=f.loc[1,'kickoff_at']
    t,_,_=compose_bets(f,policy())
    assert len(t)==1 # Months apart with a common eligible cutoff is not leakage.


@pytest.mark.parametrize('field', ['forecast_as_of','feature_hash','model_hash','preprocessing_hash','event_id','p_win'])
def test_timestamp_relabel_or_dependency_change_requires_regeneration(field):
    f=certified()
    f.loc[0,field]={'event_id':77,'p_win':.55,'forecast_as_of':TRAIN}.get(field,'changed')
    with pytest.raises(ValueError,match='binding changed'):
        compose_bets(f,policy())


@pytest.mark.parametrize('stage', ['preprocessing','fitting','calibration','tuning','ratings'])
def test_delayed_training_inputs_fail_at_training_cutoff(stage):
    f,e=population()
    fitting=next(s for s in e[0]['stages'] if s['kind']=='fitting')
    s=next(s for s in e[0]['stages'] if s['kind']==stage)
    s.update(deepcopy(fitting),kind=stage)
    if stage=='preprocessing':
        s['artifact_hash']=f.loc[0,'preprocessing_hash']
        s['artifact']={'hash':s['artifact_hash'],'content':{'model':'scaler-v1'}}
    s['inputs'][0]['available_as_of']=T
    with pytest.raises(ValueError,match='training_input_available_as_of'):
        bind_forecasts(f,e)


def test_train_only_and_artifact_bindings():
    f,e=population()
    e[0]['stages'][-1]['fit_partition']='holdout'
    with pytest.raises(ValueError,match='train-only'): bind_forecasts(f,e)
    e[0]['stages'][-1]['fit_partition']='train'
    e[0]['stages'][-1]['artifact_hash']='other-model'
    with pytest.raises(ValueError,match='artifact hash'): bind_forecasts(f,e)


def test_retrospective_recomputation_preserves_physical_times():
    f,e=population()
    now='2026-10-30T00:00:00Z'
    for item in e.values():
        item['generated_at']=now
        for s in item['stages']:
            if s['mode']=='fitted': s['created_at']=now
        item['assumptions']=[dict(id='replay-v1',scope='retrospective_replay',rationale='Recompute from archived eligible inputs',
                                 reference='protocol/2024',available_as_of=T)]
    f=bind_forecasts(f,e)
    t,_,_=compose_bets(f,policy())
    assert t[STATUS].eq('conditional_research').all()
    assert json.loads(f[EVIDENCE].iloc[0])['generated_at']==now
    with pytest.raises(ValueError,match='conditional_research'): require_verified_comparison(t)
    for item in e.values(): item['assumptions']=[]
    assert compose_bets(bind_forecasts(population()[0],e),policy())[0][STATUS].eq('unverified').all()


@pytest.mark.parametrize('value', [None,pd.NaT,pd.NA,np.nan])
def test_missing_is_unverified_not_contradictory(value):
    f,_=population()
    f['forecast_as_of']=value
    t,_,_=compose_bets(f,policy())
    assert t[STATUS].eq('unverified').all()
    assert not t.comparison_eligible.any()


@pytest.mark.parametrize('value',[123,True,'',object()])
def test_invalid_timestamps_rejected(value):
    f,_=population(); f['decision_at']=value
    with pytest.raises((ValueError,TypeError)): compose_bets(f,policy())


def test_timezone_equivalence_and_naive_limitation():
    f=certified()
    f['decision_at']='2024-12-31T09:00:00+09:00'
    assert compose_bets(f,policy())[0][STATUS].eq('verified').all()
    raw,e=population(); raw['decision_at']='2024-12-31'
    assert compose_bets(bind_forecasts(raw,e),policy())[0][STATUS].eq('unverified').all()


def test_unknown_quotes_and_bounded_assumptions():
    f,e=population(); f['quote_at']=pd.NaT
    assert compose_bets(bind_forecasts(f,e),policy())[0][STATUS].eq('unverified').all()
    f['quote_availability_mode']='research_assumed'
    f['quote_assumption_id']='opening-v1'
    f['quote_assumption_rationale']='Bounded research assumption, not observed publication'
    f['quote_assumption_reference']='protocol/opening'
    f['assumed_available_at']=TRAIN
    assert compose_bets(bind_forecasts(f,e),policy())[0][STATUS].eq('conditional_research').all()


def test_pool_row_order_and_future_source_truncation():
    f=certified()
    a=guard_candidates(f,policy()); b=guard_candidates(f.iloc[::-1],policy())
    assert a[CERT].iloc[0]==b[CERT].iloc[0]
    # Hash only historically eligible source rows: perturb/remove future records.
    source=pd.DataFrame({'available_at':pd.to_datetime([TRAIN,'2025-02-01'],utc=True,format='mixed'),'v':[1,99]})
    def historic(data): return dependency_hash(data.loc[data.available_at<=pd.Timestamp(T)].to_dict('records'))
    h=historic(source); source.loc[1,'v']=999
    assert historic(source)==historic(source.iloc[:1])==h
    # A future dependency cannot be justified by merely retaining an old cutoff.
    raw,e=population(); e[0]['dependencies'][1]['available_as_of']='2025-02-01T00:00:00Z'
    with pytest.raises(ValueError,match='dependency_available_as_of'): bind_forecasts(raw,e)


@pytest.mark.parametrize('kind', ['policy','member','decision','certificate','pool'])
def test_tampering_rejected_on_direct_helpers_and_reload(kind,tmp_path):
    p=policy(); t,m,_=compose_bets(certified(),p)
    templates={'tickets':p}
    if kind=='policy': templates['tickets']=replace(p,stake=10)
    elif kind=='member': m.loc[0,'event_id']=77
    elif kind=='decision': t['decision_at']='2024-12-01T00:00:00Z'
    elif kind=='certificate': t.loc[0,CERT]=t.loc[0,CERT].replace('model-v1','model-v2')
    else: t.loc[0,POOL]='{}'
    with pytest.raises(ValueError): ticket_batch(t,m,templates,('event_id',))
    if kind in ('certificate','pool'):
        t.to_parquet(tmp_path/'tampered.parquet',compression='zstd')
        with pytest.raises(ValueError): load_temporal_table(tmp_path/'tampered.parquet')


@pytest.mark.parametrize('template',[Parlay(size=1),policy(),MultiBet(size=3,sizes=(1,2,3)),BetSlip({'a':policy(),'b':Parlay(size=1)})])
def test_all_shapes_mandatory_guard_and_accounting(template):
    t,_,_=compose_bets(certified(),template)
    assert t[STATUS].eq('verified').all()
    bad=certified(); bad['issued_at']='2025-02-01T00:00:00Z'
    with pytest.raises(ValueError): compose_bets(bad,template)


@pytest.mark.parametrize('allocation',[FixedStake(2,'u'),FractionalKelly('u',alpha=.1)])
def test_allocation_and_reserve_conservation(allocation):
    p=policy(); t,m,_=compose_bets(certified(2),p)
    batch=ticket_batch(t,m,{'tickets':p},('event_id',))
    context=StakeContext(100,100,'u',T)
    alloc=allocate_batch(FrozenTable(batch),allocation,context)
    assert alloc.amounts.ge(0).all()
    with pytest.raises(ValueError,match='Placement time'):
        allocate_batch(FrozenTable(batch),allocation,StakeContext(100,100,'u',TRAIN))
    m=m.assign(leg_id=['first','second'])
    outcomes=pd.DataFrame({'available_at':pd.to_datetime(['2025-01-01T04:00Z','2025-06-01T04:00Z']),
                           'return_multiplier':[0.,2.]},index=['first','second'])
    settlements=settlements_from_legs(m,outcomes)
    assert settlements.available_at.iloc[0]==outcomes.available_at.max()
    result=replay_bankroll(FrozenTable(batch),settlements,allocation,initial_capital=100,currency='u')
    assert result.ledger.kind.tolist()==['place','settle']
    assert result.ledger.time.iloc[-1]==outcomes.available_at.max()
    np.testing.assert_allclose(result.ledger.cash+result.ledger.reserve,result.ledger.wealth)
    assert result.summary['outstanding_reserve']==0
    altered=settlements.copy(); altered['available_at']=outcomes.available_at.min()
    with pytest.raises(ValueError,match='availability/payoff'):
        replay_bankroll(FrozenTable(batch),altered,allocation,initial_capital=100,currency='u')


def test_single_allocation_preflights_unselected_rows():
    f,_=population(); f.loc[0,'take']=False; f.loc[0,'forecast_as_of']='2025-01-01T00:00:00Z'
    with pytest.raises(ValueError): allocate_singles(f,FixedStake(1,'u'),StakeContext(100,100,'u',T))


def test_scalar_bulk_future_discard_and_membership(monkeypatch):
    from test_guarded_bulk import population as bulk_population
    from test_quote_availability import policy as bulk_policy
    from xdiyo_analytics.evaluation import bulk_tickets
    from xdiyo_analytics.evaluation.all_combinations import prepare_pools
    f=bulk_population(4); p=bulk_policy()
    actual=compose_bets(f,p)
    with monkeypatch.context() as scoped:
        scoped.setattr(bulk_tickets,'supports_bulk',lambda *a:False)
        expected=compose_bets(f,p)
    for a,b in zip(actual,expected):
        pd.testing.assert_frame_equal(a,b,check_exact=True)
    prepared={'tickets':prepare_pools(f,p,('event_id',),'tickets',outcome_free=True)}
    f.loc[0,'odds']=9
    with pytest.raises(ValueError,match='membership'):
        bulk_tickets.execute_bulk(f,prepared,p)


def test_discarded_forecast_valid_at_own_later_issue_cannot_influence_earlier_pairing():
    f,_=population()
    f.loc[2,'take']=False
    f.loc[2,['decision_at','forecast_as_of']]='2024-12-31T12:00:00Z'
    with pytest.raises(ValueError,match='candidate_group_decision_at'):
        compose_bets(f,policy())


def test_complete_candidate_quote_preflight_precedes_user_selector():
    from test_post_training_betting import bet_sample
    from xdiyo_analytics.evaluation import evaluate_bets
    ctx,bets,labels,_=bet_sample()
    ctx.metadata['decision_at']='2000-01-01T00:00:00Z'
    ctx.metadata['quote_at']='2000-01-02T00:00:00Z'
    def forbidden(context):
        pytest.fail('selector ran before full temporal preflight')
    bets['offer']=replace(bets['offer'],take=forbidden)
    with pytest.raises(ValueError,match='quote_at'):
        evaluate_bets(ctx,bets,labels=labels)


def test_callbacks_cannot_access_outcomes_or_history():
    from test_post_training_betting import bet_sample
    from xdiyo_analytics.evaluation import evaluate_bets
    ctx,bets,labels,_=bet_sample()
    bets['offer']=replace(bets['offer'],take=lambda safe:safe.y.iloc[:,0].gt(0))
    with pytest.raises(ValueError,match='Observed outcomes are unavailable'):
        evaluate_bets(ctx,bets,labels=labels)


def test_dependency_file_rechecked_on_reload_and_each_call(tmp_path):
    import hashlib
    f,e=population()
    path=tmp_path/'history.json'; path.write_bytes(b'eligible historical source')
    for item in e.values():
        dep=next(d for d in item['dependencies'] if d['role']=='source')
        dep.pop('content'); dep.update(path=str(path),hash=hashlib.sha256(path.read_bytes()).hexdigest())
    t,_,_=compose_bets(bind_forecasts(f,e),policy())
    save_temporal_table(t,tmp_path/'artifact.parquet')
    require_verified_comparison(t)
    path.write_bytes(b'future source silently substituted')
    with pytest.raises(ValueError,match='content hash changed'): validate_frame(t)
    with pytest.raises(ValueError,match='content hash changed'): load_temporal_table(tmp_path/'artifact.parquet')
    path.unlink()
    assert load_temporal_table(tmp_path/'artifact.parquet')[STATUS].eq('unverified').all()


def test_bare_hash_without_content_cannot_verify():
    f,e=population()
    for item in e.values(): item['dependencies'][0].pop('content')
    t,_,_=compose_bets(bind_forecasts(f,e),policy())
    assert t[STATUS].eq('unverified').all()


def test_publication_and_model_stage_equality_conventions():
    f,e=population()
    f['quote_at']=T
    t,_,_=compose_bets(bind_forecasts(f,e),policy())
    assert t[STATUS].eq('verified').all() # availability <= cutoff; placement < kickoff
    for item in e.values(): item['stages'][-1]['available_as_of']=T
    with pytest.raises(ValueError,match='model_available_as_of'): bind_forecasts(f,e)


def test_allocation_evidence_cannot_be_reused_with_changed_amounts():
    from xdiyo_analytics.evaluation import BankrollLedger
    p=policy(); t,m,_=compose_bets(certified(2),p)
    batch=FrozenTable(ticket_batch(t,m,{'tickets':p},('event_id',)))
    a=allocate_batch(batch,FixedStake(1,'u'),StakeContext(100,100,'u',T))
    a.amounts.iloc[0]=2
    with pytest.raises(ValueError,match='allocation evidence'):
        BankrollLedger(100,'u').place(batch,a.amounts,T)


def test_betting_rankings_require_derived_verified_evidence():
    from xdiyo_analytics.experiments.ranking import rank_runs
    _,_,metrics=compose_bets(certified(),policy())
    metric=metrics.loc[metrics.metric.eq('roi')].iloc[0].to_dict()
    metric['type']='overall'
    record=dict(run_id='run',name='run',config_hash='x',status='complete',metrics=[metric])
    assert rank_runs([record],{'roi':1}).status.tolist()==['ranked']
    missing=deepcopy(record)
    missing['metrics'][0].pop('temporal_comparison_evidence')
    assert rank_runs([missing],{'roi':1}).status.tolist()==['unranked_temporal_unverified']
    altered=deepcopy(record); altered['metrics'][0]['value']=999
    with pytest.raises(ValueError,match='metric binding'): rank_runs([altered],{'roi':1})


def test_empty_comparison_is_unverified():
    from xdiyo_analytics.evaluation.temporal_provenance import annotate_metrics,metric_classification
    result=annotate_metrics(pd.DataFrame([{'metric':'roi','value':None}]),pd.DataFrame())
    assert metric_classification(result.iloc[0])=='unverified'


def test_multiple_groups_and_simultaneous_tickets_deterministic():
    f=certified(4)
    f.loc[2:,'round']=11 # Binding prevents modifying a grouping dependency.
    with pytest.raises(ValueError,match='binding'): compose_bets(f,policy())
    f,e=population(4); f.loc[2:,'round']=11
    f=bind_forecasts(f,e)
    p=policy(); t,m,_=compose_bets(f,p)
    batch=ticket_batch(t,m,{'tickets':p},('event_id',))
    settlements=pd.DataFrame({'available_at':['2025-01-02T00:00:00Z']*2,'return_multiplier':[2.,0.]},index=batch.index)
    a=replay_bankroll(FrozenTable(batch),settlements,FixedStake(1,'u'),initial_capital=10,currency='u')
    b=replay_bankroll(FrozenTable(batch.iloc[::-1]),settlements.iloc[::-1],FixedStake(1,'u'),initial_capital=10,currency='u')
    pd.testing.assert_frame_equal(a.ledger,b.ledger)
    assert a.summary['temporal_status']=='unverified' # Legacy settlement has no complete-leg proof.


def test_report_export_revalidates_and_nonbetting_report_is_unchanged():
    from xdiyo_analytics.reporting.contracts import AnalysisReport,StudyRun,StudyResult,Artifact
    from xdiyo_analytics.evaluation.temporal_provenance import revalidate_report
    table=pd.DataFrame({'x':[1]})
    study=StudyRun('plain','overall','score',None,'match',np.array([0]),1,
                   StudyResult('Plain',tables={'plain':table},artifacts=[Artifact('table',table)]))
    plain=AnalysisReport([study])
    assert revalidate_report(plain) is plain
    t,_,metrics=compose_bets(certified(),policy())
    study.result=StudyResult('Betting',tables={'tickets':t,'metrics':metrics},artifacts=[Artifact('table',t)])
    assert revalidate_report(plain).studies[0].result.tables['tickets'][STATUS].eq('verified').all()
    t.loc[t.index[0],'odds']+=1
    with pytest.raises(ValueError,match='binding'): plain.to_html()


def test_ledger_reload_does_not_upgrade_missing_settlement_or_allocation():
    from xdiyo_analytics.evaluation import BankrollLedger
    from xdiyo_analytics.evaluation.temporal_provenance import revalidate_artifact
    p=policy();t,m,_=compose_bets(certified(2),p)
    batch=FrozenTable(ticket_batch(t,m,{'tickets':p},('event_id',)))
    ledger=BankrollLedger(100,'u')
    amounts=allocate_batch(batch,FixedStake(1,'u'),StakeContext(100,100,'u',T)).amounts
    ledger.place(batch,amounts,T)
    ledger.settle(batch.frame.index[0],gross_return=0.,time='2025-01-03T00:00:00Z',available_at='2025-01-03T00:00:00Z')
    assert revalidate_artifact(pd.DataFrame(ledger.events))[STATUS].eq('unverified').all()
    altered=pd.DataFrame(ledger.events); altered.loc[0,'amount']=99.
    with pytest.raises(ValueError,match='ledger event'): revalidate_artifact(altered)
    events=pd.DataFrame(ledger.events[:1]).drop(columns=['temporal_allocation_policy','temporal_allocation_policy_hash','temporal_event_hash'])
    assert revalidate_artifact(events)[STATUS].eq('unverified').all()


def test_unsupported_content_cannot_be_certified_from_truncated_repr():
    f,e=population()
    for item in e.values():
        dep=next(d for d in item['dependencies'] if d['role']=='source')
        dep['content']=np.arange(2000)
        dep['hash']=dependency_hash(dep['content'])
    assert bind_forecasts(f,e)[STATUS].eq('unverified').all()


@pytest.mark.parametrize('missing',['dependency_hash','model_hash','declared_cutoff','model_availability'])
def test_missing_proof_is_unverified_instead_of_an_invented_contradiction(missing):
    f,e=population()
    if missing=='model_hash':f=f.drop(columns='model_hash')
    for item in e.values():
        if missing=='dependency_hash':item['dependencies'][0].pop('hash')
        if missing=='declared_cutoff':item.pop('forecast_as_of')
        if missing=='model_availability':item.pop('model_available_as_of')
    assert bind_forecasts(f,e)[STATUS].eq('unverified').all()


def test_allocation_cannot_inspect_other_groups_via_complete_pool_audit():
    f,e=population(4);f.loc[2:,'round']=11
    t,m,_=compose_bets(bind_forecasts(f,e),policy())
    batch=FrozenTable(ticket_batch(t,m,{'tickets':policy()},('event_id',)))
    class Inspect:
        def allocate(self, offered, context):
            assert not set(offered.frame)&{CERT,POOL,EVIDENCE}
            assert not any(c.startswith('temporal_') for c in offered.frame)
            return FixedStake(1,'u').allocate(offered,context)
    result=allocate_batch(batch,Inspect(),StakeContext(100,100,'u',T))
    assert result.amounts.sum()==2
    assert result.audit[STATUS].eq('unverified').all()


def test_ticket_certificate_retains_explicit_composite_fixture_keys():
    from xdiyo_analytics.evaluation.temporal_provenance import inspect_certificate
    f,e=population();f['fixture_key']=['fixture-'+str(i) for i in range(len(f))]
    keys=('competition_id','fixture_key')
    f=bind_forecasts(f,e,match_columns=keys)
    t,m,_=compose_bets(f,policy(),match_columns=keys)
    assert inspect_certificate(t[CERT].iloc[0])['match_columns']==list(keys)
    assert t[STATUS].eq('verified').all()
    m.loc[m.index[0],'fixture_key']='substitute'
    from xdiyo_analytics.evaluation.ticket_allocation import ticket_batch
    with pytest.raises(ValueError,match='binding|membership'):
        ticket_batch(t,m,{'tickets':policy()},keys)


def test_new_unbound_probability_cannot_be_consumed_by_native_staking():
    from xdiyo_analytics.evaluation.stake_policy import ModelProbabilitySource
    t,m,_=compose_bets(certified(2),policy())
    batch=ticket_batch(t,m,{'tickets':policy()},('event_id',))
    batch['new_probability']=.999
    source=ModelProbabilitySource(column='new_probability',assume_available_at_decision=True)
    with pytest.raises(ValueError,match='probability output'):
        allocate_batch(FrozenTable(batch),FractionalKelly(currency='u',probability_source=source),StakeContext(100,100,'u',T))


def test_retrospective_assumption_cannot_backdate_forecast_before_its_model():
    f,e=population()
    for item in e.values():
        item['stages'][-1]['created_at']='2026-01-01T00:00:00Z'
        item['assumptions']=[dict(id='replay',scope='retrospective_replay',rationale='eligible historical replay',
                                  reference='protocol/replay',available_as_of=TRAIN)]
    with pytest.raises(ValueError,match='physical_artifact_created_at'):
        bind_forecasts(f,e)


def test_evidence_native_parquet_dtype_and_shared_buffers():
    from io import BytesIO
    from xdiyo_analytics.evaluation.temporal_provenance import _evidence_series
    for values in ([], ['large evidence ' * 1000] * 4):
        series = _evidence_series(values, pd.RangeIndex(len(values)))
        frame = series.to_frame('evidence')
        buffer = BytesIO(); frame.to_parquet(buffer); buffer.seek(0)
        pd.testing.assert_frame_equal(frame, pd.read_parquet(buffer))
        if len(values) and isinstance(series.dtype, pd.StringDtype) and series.dtype.storage == 'pyarrow':
            chunks = series.array._pa_array.chunks
            assert len({chunk.buffers()[2].address for chunk in chunks}) == 1
        if len(values):
            series.iloc[0] = 'changed'
            assert series.iloc[1] == values[1]
