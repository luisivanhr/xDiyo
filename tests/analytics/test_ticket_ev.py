"""Synthetic retained evidence only: never fit, calibrate, or run experiments."""
from dataclasses import replace
from io import BytesIO, StringIO
import json
from math import comb

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.evaluation import AllCombinations, BetSlip, Parlay, compose_bets, preview_combinations
from test_all_combinations import sample, binary_context, KEYS


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    from xdiyo_analytics.training import TrainingRunner, ProbabilityCalibrator
    from xdiyo_analytics.experiments import FootballExperiment
    from xdiyo_analytics.ui import recipe
    def forbidden(*a, **k):
        raise AssertionError('Ticket EV verification must not fit/calibrate/run a production experiment')
    for cls, method in [(TrainingRunner,'run'), (ProbabilityCalibrator,'fit'), (FootballExperiment,'run')]:
        monkeypatch.setattr(cls, method, forbidden)
    monkeypatch.setattr(recipe, 'run_recipe', forbidden)


def oracle():
    return sample().assign(p_win=[.75,.5,.25,.75], odds=[2.,2.,2.,4.])


def policy(**kwargs):
    return AllCombinations(probability_mode='independent', min_ev=0., **kwargs)


def audit(tickets):
    return pd.DataFrame(tickets.attrs['ticket_candidates'])


@pytest.mark.parametrize('k,count,payout,profit', [(2,4,28,24),(3,3,48,45),(4,1,32,31)])
@pytest.mark.parametrize('stake', [0.,.25,1.,2.])
def test_oracle_and_stake_units(k,count,payout,profit,stake):
    tickets,members,metrics=compose_bets(oracle(),policy(legs=k,stake=stake),match_columns=KEYS)
    assert len(tickets)==count and len(members)==k*count
    assert tickets.stake.sum()==count*stake
    assert tickets.payout.sum()==payout*stake and tickets.profit.sum()==profit*stake
    assert tickets['take'].all()
    evidence=audit(tickets)
    assert len(evidence)==comb(4,k)
    assert evidence['take'].equals(evidence.expected_profit.gt(0))
    assert not {'profit','payout','stake','settlement'} & set(evidence)
    summary=tickets.attrs['ticket_selection_summary'][0]
    assert summary['candidate_tickets']==count+summary['rejected_by_ev']+summary['rejected_missing_probability']
    assert summary['selected_stake']==count*stake
    assert summary['candidate_stake']==comb(4,k)*stake
    if k==2:
        assert sorted(tickets.expected_profit)==[.5,.5,2.,3.5]
    if stake==0:
        assert metrics.loc[metrics.metric.eq('roi'),'value'].isna().all()


@pytest.mark.parametrize('k',[1,2,3,4,5,6])
def test_general_k_strict_breakeven_and_empty(k):
    data=sample(6).assign(p_win=.5,odds=2.)
    tickets,members,metrics=compose_bets(data,policy(legs=k))
    assert tickets.empty and members.empty
    assert len(audit(tickets))==comb(6,k)
    assert audit(tickets).expected_profit.eq(0).all()
    assert metrics.loc[metrics.metric.isin(['profit','bets_placed','settled_stakes']),'value'].eq(0).all()
    assert metrics.loc[metrics.metric.eq('roi'),'value'].isna().all()
    assert compose_bets(data.iloc[:0],policy(legs=k))[0].empty
    assert compose_bets(data.iloc[:k-1],policy(legs=k))[0].empty


@pytest.mark.parametrize('threshold,expected',[(.5,2),(np.nextafter(.5,-np.inf),4),(np.nextafter(.5,np.inf),2),(-1.,6)])
def test_threshold_equality_and_neighbors(threshold,expected):
    tickets=compose_bets(oracle(),replace(policy(),min_ev=threshold))[0]
    assert len(tickets)==expected


@pytest.mark.parametrize('value',[True,False,'0',[],np.nan,np.inf,-np.inf,1j])
def test_invalid_threshold(value):
    with pytest.raises(ValueError,match='finite real'):
        AllCombinations(min_ev=value,probability_mode='independent')


def test_incompatible_configuration_including_empty_and_missing_column():
    with pytest.raises(ValueError,match='independent'):
        compose_bets(sample(0),AllCombinations(min_ev=0))
    with pytest.raises(ValueError,match='retained win probabilities'):
        compose_bets(sample().drop(columns='p_win',errors='ignore'),policy())


def test_legacy_none_and_missing_probabilities():
    data=oracle().assign(p_win=np.nan)
    for mode in ('none','independent'):
        a=compose_bets(data,AllCombinations(probability_mode=mode))
        b=compose_bets(data,AllCombinations(probability_mode=mode,min_ev=None))
        for x,y in zip(a,b):
            pd.testing.assert_frame_equal(x,y)
        assert len(a[0])==6 and a[0].probability.isna().all()
    data.loc[2,'p_win']=np.nan
    data.loc[[0,1,3],'p_win']=[.75,.5,.75]
    tickets=compose_bets(data,policy())[0]
    assert len(tickets)==3
    summary=tickets.attrs['ticket_selection_summary'][0]
    assert summary['rejected_missing_probability']==3 and summary['rejected_by_ev']==0
    tickets=compose_bets(data.assign(p_win=np.nan),policy())[0]
    assert tickets.empty and audit(tickets).rejection_reason.eq('missing_probability').all()


@pytest.mark.parametrize('bad',[-.1,1.1,np.inf,-np.inf,'bad','0.5',True])
@pytest.mark.parametrize('min_ev', [None, 0.])
def test_corrupt_probability_not_hidden_by_missing(bad, min_ev):
    data=oracle().astype({'p_win':object})
    data.loc[0,'p_win']=np.nan
    data.loc[1,'p_win']=bad
    with pytest.raises(ValueError,match='numeric, finite'):
        compose_bets(data,replace(policy(), min_ev=min_ev))


def test_zero_one_push_and_extreme_products():
    assert len(compose_bets(oracle().assign(p_win=[0.,1.,1.,1.]),policy())[0])==3
    for settlement in ('win','loss','push','void','missing'):
        with pytest.raises(ValueError,match='nonzero p_push'):
            compose_bets(oracle().assign(p_push=.01,settlement=settlement),policy())
    with pytest.raises(ValueError,match='underflow'):
        compose_bets(oracle().assign(p_win=1e-200),policy())
    with pytest.raises(ValueError,match='Combined odds overflow'):
        compose_bets(oracle().assign(odds=1e200),policy())
    with pytest.raises(ValueError,match='payout overflow'):
        compose_bets(oracle().assign(odds=1e150,p_win=1.),policy(stake=1e100))
    with pytest.raises(ValueError,match='accounting total overflow'):
        compose_bets(oracle().assign(settlement='loss'),policy(stake=1e308))


def test_price_eligibility_and_candidate_guard(monkeypatch):
    data=oracle()
    for price in (np.nan,np.inf,1.,0.,-2.):
        data.loc[2,'odds']=price
        preview=preview_combinations(data,policy())
        assert preview.attrs['total_tickets']==3
        assert preview.missing_price_events.tolist()==[1]
    with pytest.raises(ValueError):
        compose_bets(oracle().assign(odds='bad'),policy())
    import xdiyo_analytics.evaluation.tickets as engine
    def forbidden(*a,**k): raise AssertionError('Expanded before complete preflight')
    monkeypatch.setattr(engine,'expand_pools',forbidden)
    slip=BetSlip({'singles':Parlay(size=1),'small':policy(legs=4),'too_big':replace(policy(),max_tickets=5,min_ev=1e50)})
    with pytest.raises(ValueError,match='requested 6 tickets'):
        compose_bets(oracle(),slip)
    assert preview_combinations(oracle(),slip).attrs['total_tickets']==7


@pytest.mark.parametrize('state',['win','loss','push','void','missing'])
@pytest.mark.parametrize('rule',['remove','refund','loss'])
def test_outcome_independence_and_settlement(state,rule):
    config=policy(on_push=rule,on_void=rule)
    data=oracle()
    original=compose_bets(data,config)[0]
    data=data.assign(settlement=state,profit=123.,payout=-99.,actual_goals=999,target=-1)
    tickets,members,metrics=compose_bets(data,config)
    pd.testing.assert_frame_equal(audit(original),audit(tickets))
    legacy,_,_=compose_bets(data,replace(config,min_ev=None))
    cols=['ticket_id','payout','profit','settlement','accounting_status']
    pd.testing.assert_frame_equal(tickets[cols].reset_index(drop=True),legacy.loc[legacy.ticket_id.isin(tickets.ticket_id),cols].reset_index(drop=True))
    if state=='missing':
        assert tickets.profit.isna().all()
        assert metrics.loc[metrics.metric.eq('settled_stakes'),'value'].item()==0


def test_ids_duplicates_shuffle_and_threshold_stability():
    data=oracle().assign(event_id=pd.Series([2**63+i for i in range(4)],dtype='uint64'))
    duplicate=pd.concat([data,data.iloc[[0]].assign(row_position=100)])
    a=compose_bets(data,policy())
    b=compose_bets(duplicate.sample(frac=1,random_state=7),policy())
    assert a[0].ticket_id.tolist()==b[0].ticket_id.tolist()
    pd.testing.assert_frame_equal(audit(a[0]),audit(b[0]))
    assert set(b[1].event_id)==set(data.event_id)
    higher=compose_bets(data,replace(policy(),min_ev=.5))[0]
    assert set(higher.ticket_id)<=set(a[0].ticket_id)
    with pytest.raises(ValueError,match='Conflicting selections'):
        compose_bets(pd.concat([data,data.iloc[[0]].assign(p_win=.9)]),policy())


@pytest.mark.parametrize('group',['fold_id','competition_id','season_id','tournament_id','round'])
def test_group_isolation(group):
    a=oracle(); b=oracle()
    b[group]+=1
    b['event_id']+=100
    tickets,members,_=compose_bets(pd.concat([a,b]),policy(),match_columns=KEYS)
    assert len(tickets)==8
    assert members.groupby('ticket_id')[group].nunique().eq(1).all()


@pytest.mark.parametrize('threshold',[.8,.9])
@pytest.mark.parametrize('min_ev',[0.,100.])
def test_report_reuse_export_and_binary_equality(threshold,min_ev,monkeypatch):
    from xdiyo_analytics.evaluation import BetOffer, BinaryDrawThreshold
    from xdiyo_analytics.reporting import BetOutcomeReporter, BetPerformanceReporter
    from xdiyo_analytics.reporting.tickets import ticket_html
    ctx,option,labels=binary_context()
    ctx.predictions['predict_proba'].iloc[0]=[threshold,1-threshold]
    offers={'draw':BetOffer(option,pd.Series([2.,3.,4.,5.,np.nan],index=ctx.y.index))}
    result=BetOutcomeReporter(type='overall',partition='test',offers=offers,labels=labels,
        policy=BinaryDrawThreshold(threshold),composition=replace(policy(),min_ev=min_ev),show_badges=False).run(ctx)
    assert result.tables['alternatives']['take'].iloc[0]
    assert result.tables['alternatives'].p_win.iloc[0]==1-threshold
    ctx.previous_results={'source':result}
    import xdiyo_analytics.reporting.tickets as rendering
    def forbidden(*a,**k): raise AssertionError('Source reuse recomposed tickets')
    monkeypatch.setattr(rendering,'compose_bets',forbidden)
    reused=BetPerformanceReporter(type='overall',partition='test',source='source').run(ctx)
    for key in ('tickets','ticket_legs','leg_ledger','alternatives','combination_preview','ticket_candidates','ticket_selection_summary'):
        pd.testing.assert_frame_equal(result.tables[key],reused.tables[key])
        restored=pd.read_csv(StringIO(result.tables[key].to_csv(index=False)))
        assert len(restored)==len(result.tables[key])
    if min_ev==100:
        assert 'EV filter' in ticket_html(result.tables['tickets'],result.tables['ticket_legs'],{})
    with pytest.raises(ValueError,match='only one reporter'):
        BetPerformanceReporter(type='overall',partition='test',source='source',composition=policy()).run(ctx)


def test_manual_report_clear_error():
    from xdiyo_analytics.evaluation import BetOffer, BinaryDrawThreshold, prepare_bets
    from xdiyo_analytics.reporting import BetPerformanceReporter
    ctx,option,labels=binary_context()
    ctx.metadata=ctx.metadata.drop(columns=['p_win'])
    specs,_=prepare_bets(ctx,{'draw':BetOffer(option,2.)},BinaryDrawThreshold())
    with pytest.raises(ValueError,match='retained win probabilities'):
        BetPerformanceReporter(type='overall',partition='test',bets=specs,labels=labels,composition=policy()).run(ctx)


def test_catalog_mixed_templates_and_old_recipe():
    from xdiyo_analytics.ui import catalog_for_ui
    catalog=catalog_for_ui()
    obj=BetSlip({'filtered':policy(legs=3),'legacy':AllCombinations(legs=4),'single':Parlay(size=1)})
    assert catalog.build(json.loads(json.dumps(catalog.encode(obj))))==obj
    assert catalog.build({'component':'evaluation.AllCombinations','params':{'legs':2}}).min_ev is None


def test_empty_report_audits_and_html_exports():
    from xdiyo_analytics.reporting.contracts import StudyResult, StudyRun, AnalysisReport
    from xdiyo_analytics.reporting.tickets import add_tickets
    ctx,_,_=binary_context()
    for data in (oracle().iloc[:0], oracle().assign(p_win=np.nan)):
        result=add_tickets(StudyResult('Synthetic',tables={'ledger':data}),policy(),ctx)
        for key in ('ticket_candidates','ticket_selection_summary'):
            assert len(pd.read_csv(StringIO(result.tables[key].to_csv(index=False))))==len(result.tables[key])
        report=AnalysisReport([StudyRun('Tickets','overall','test',None,'match',np.array([],dtype=int),0,result)])
        html=report.to_html()
        assert 'No tickets placed' in html and 'ticket EV filter' in html
        assert 'ticket_candidates' in html and 'ticket_selection_summary' in html
        assert 'data:text/csv' in html


@pytest.mark.parametrize('min_ev', [None, 0., 100.])
@pytest.mark.parametrize('missing', [False, True])
def test_ticket_parquet_roundtrip_preserves_audit_and_exact_ids(min_ev, missing):
    data = oracle().assign(competition_id=np.uint64(2**63 + 17))
    if missing:
        data.loc[0, 'p_win'] = np.nan
    tickets, members, metrics = compose_bets(data, replace(policy(), min_ev=min_ev))
    # Native JSON scalars, exact large integers, and null missing evidence.
    serialized = json.dumps(tickets.attrs, allow_nan=False)
    assert json.loads(serialized) == tickets.attrs
    assert tickets.attrs['ticket_candidates'][0]['competition_id'] == 2**63 + 17
    for table in (tickets, members, metrics):
        buffer = BytesIO()
        table.to_parquet(buffer)
        buffer.seek(0)
        restored = pd.read_parquet(buffer)
        pd.testing.assert_frame_equal(table, restored)
        assert table.attrs == restored.attrs
    assert len(audit(tickets)) == 6
    assert audit(tickets)['take'].sum() == len(tickets)


@pytest.mark.parametrize('min_ev', [None, 0.])
@pytest.mark.parametrize('dtype', ['float64', 'Float64', 'object'])
@pytest.mark.parametrize('missing', [False, True])
def test_numeric_probability_dtypes_have_identical_decisions(min_ev, dtype, missing):
    data = oracle()
    if missing:
        data.loc[0, 'p_win'] = np.nan
    config = replace(policy(), min_ev=min_ev)
    expected = compose_bets(data, config)[0]
    actual = compose_bets(data.astype({'p_win': dtype}), config)[0]
    pd.testing.assert_frame_equal(expected, actual)
    assert expected.attrs == actual.attrs
    if min_ev == 0 and not missing:
        assert len(actual) == 4


@pytest.mark.parametrize('kind', ['enabled', 'disabled', 'mixed', 'parlay'])
def test_combined_slip_accounting_overflow_is_rejected(kind):
    data = oracle().iloc[:2].assign(settlement='loss')
    large = policy(stake=1e308)
    legacy = replace(large, min_ev=None)
    parlay = Parlay(stake=1e308)
    templates = {'enabled': (large, large), 'disabled': (legacy, legacy),
                 'mixed': (large, parlay), 'parlay': (parlay, parlay)}[kind]
    for template in templates:
        tickets, _, _ = compose_bets(data, template)
        assert len(tickets) == 1 and tickets.stake.iloc[0] == 1e308
    with pytest.raises(ValueError, match='accounting total overflow'):
        compose_bets(data, BetSlip(dict(zip(['first', 'second'], templates))))


@pytest.mark.parametrize('min_ev', [None, 0., 100.])
def test_saved_report_tables_reuse_without_recomposition(tmp_path, monkeypatch, min_ev):
    from xdiyo_analytics.evaluation import BetOffer, BinaryDrawThreshold
    from xdiyo_analytics.reporting import BetOutcomeReporter, BetPerformanceReporter
    from xdiyo_analytics.reporting.contracts import StudyResult
    ctx, option, labels = binary_context()
    offers = {'draw': BetOffer(option, pd.Series([2., 3., 4., 5., np.nan], index=ctx.y.index))}
    original = BetOutcomeReporter(type='overall', partition='test', offers=offers, labels=labels,
        policy=BinaryDrawThreshold(.8), composition=replace(policy(), min_ev=min_ev), show_badges=False).run(ctx)
    restored = {}
    for name, table in original.tables.items():
        path = tmp_path / f'{name}.parquet'
        table.to_parquet(path)
        restored[name] = pd.read_parquet(path)
        pd.testing.assert_frame_equal(table, restored[name])
        assert table.attrs == restored[name].attrs
    ctx.previous_results = {'saved': StudyResult('Restored', tables=restored)}
    import xdiyo_analytics.reporting.tickets as rendering
    def forbidden(*args, **kwargs):
        raise AssertionError('Saved result reuse must not recompose tickets')
    monkeypatch.setattr(rendering, 'compose_bets', forbidden)
    reused = BetPerformanceReporter(type='overall', partition='test', source='saved').run(ctx)
    for name in ('tickets', 'ticket_legs', 'ticket_candidates', 'ticket_selection_summary'):
        pd.testing.assert_frame_equal(original.tables[name], reused.tables[name])
