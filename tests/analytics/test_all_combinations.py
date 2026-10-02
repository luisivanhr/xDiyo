"""Recorded/synthetic legs and probabilities only; no fitting or backtests."""
from copy import deepcopy
from dataclasses import replace
import json
from math import comb

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.evaluation import (AllCombinations, BetSlip, Parlay, BinaryDrawThreshold,
                                      BetOffer, prepare_bets, compose_bets, preview_combinations)
from xdiyo_analytics.labels import BetOption, Outcome, LabelData
from xdiyo_analytics.reporting import BetOutcomeReporter, BetPerformanceReporter
from xdiyo_analytics.ui import catalog_for_ui
from post_training_samples import context
from test_bet_tickets import ledger

KEYS = ('competition_id', 'season_id', 'event_id')


def sample(n=4):
    return ledger(('win',)*n).assign(tournament_id=50, odds=np.arange(2, n+2, dtype=float))


@pytest.mark.parametrize('n,k', [(n,k) for n in (0,1,4,5) for k in (1,2,3,4,5)])
def test_exact_counts_stakes_and_unique_memberships(n,k):
    data = sample(n)
    policy = AllCombinations(legs=k)
    preview = preview_combinations(data, policy, match_columns=KEYS)
    tickets, members, _ = compose_bets(data, policy, match_columns=KEYS)
    expected = comb(n,k) if n >= k else 0
    assert len(tickets) == expected == preview.attrs['total_tickets']
    assert tickets.stake.sum() == expected == preview.attrs['total_stake']
    assert len(members) == k*expected
    assert not members.duplicated(['ticket_id', *KEYS]).any()
    memberships = [tuple(sorted(g.event_id.tolist())) for _,g in members.groupby('ticket_id')]
    assert len(set(memberships)) == expected


@pytest.mark.parametrize('k', [True,False,0,-1,1.5,'2',None])
def test_invalid_legs(k):
    with pytest.raises(ValueError, match='positive integer'):
        compose_bets(sample(), AllCombinations(legs=k))


@pytest.mark.parametrize('k,payout,profit', [(2,71,65),(3,154,150),(4,120,119)])
def test_known_accounting(k,payout,profit):
    tickets, _, metrics = compose_bets(sample(), AllCombinations(legs=k))
    assert tickets.payout.sum() == payout and tickets.profit.sum() == profit
    assert metrics.set_index('metric').loc['profit','value'] == profit
    assert tickets.probability.isna().all()
    assert (tickets.last_kickoff_at >= tickets.kickoff_at).all()


@pytest.mark.parametrize('column', ['competition_id','season_id','tournament_id','round','fold_id'])
def test_no_cross_group_pairing(column):
    a, b = sample(2), sample(2)
    b[column] += 1
    if column not in KEYS and column != 'fold_id':
        b['event_id'] += 100
    data = pd.concat([a,b],ignore_index=True)
    tickets, members, _ = compose_bets(data, AllCombinations(), match_columns=KEYS)
    assert len(tickets) == 2
    assert members.groupby('ticket_id')[column].nunique().eq(1).all()


def test_filter_quotes_duplicates_conflicts_and_shuffle():
    data = sample()
    data.loc[3,'take'] = False
    for k,expected in [(2,3),(3,1),(4,0)]:
        assert len(compose_bets(data, AllCombinations(legs=k))[0]) == expected
    data.loc[2,'odds'] = np.nan
    preview = preview_combinations(data, AllCombinations())
    assert preview.eligible_events.tolist() == [2]
    assert preview.missing_price_events.tolist() == [1]
    duplicate = pd.concat([sample(), sample().iloc[[0]].assign(row_position=100)])
    first = compose_bets(duplicate, AllCombinations())
    second = compose_bets(duplicate.sample(frac=1, random_state=42), AllCombinations())
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    assert len(first[0]) == 6
    conflict = pd.concat([sample(), sample().iloc[[0]].assign(bet='alternative')])
    with pytest.raises(ValueError, match='Conflicting selections'):
        compose_bets(conflict, AllCombinations())
    with pytest.raises(ValueError, match='stage'):
        compose_bets(sample().drop(columns='tournament_id'), AllCombinations())
    assert len(compose_bets(sample().drop(columns='tournament_id'), AllCombinations(stage_column=None))[0]) == 6
    data = sample().assign(settlement='loss', payout=9999., profit=9999.)
    changed = compose_bets(data, AllCombinations())
    assert first[0].ticket_id.tolist() == changed[0].ticket_id.tolist()
    pd.testing.assert_frame_equal(first[1][['ticket_id',*KEYS]],changed[1][['ticket_id',*KEYS]])


def test_preflight_all_groups_before_any_expansion(monkeypatch):
    import xdiyo_analytics.evaluation.tickets as module
    data = sample(4)
    more = sample(4).assign(round=11,event_id=lambda x:x.event_id+100)
    data = pd.concat([data,more])
    policy = AllCombinations(max_tickets=10)
    preview = preview_combinations(data, policy)
    assert preview.attrs['total_tickets'] == 12
    def forbidden(*args):
        raise AssertionError('Expansion started before preflight completed')
    monkeypatch.setattr(module,'expand_pools',forbidden)
    with pytest.raises(ValueError,match='requested 12.*max_tickets=10'):
        compose_bets(data, BetSlip({'small': AllCombinations(legs=4), 'large':policy}))


@pytest.mark.parametrize('states,state,payout', [
    (('win','loss'),'loss',0), (('win','push'),'win',2),
    (('win','void'),'win',2), (('win','missing'),'missing',np.nan),
    (('loss','missing'),'loss',0)])
def test_native_settlement(states,state,payout):
    data=sample(2).assign(settlement=states)
    result=compose_bets(data,AllCombinations())[0].iloc[0]
    assert result.settlement==state
    assert pd.isna(result.payout) if pd.isna(payout) else result.payout==payout
    if state=='missing':assert pd.isna(result.profit)


def binary_context():
    data=sample(5).assign(home_id=101,away_id=202).drop(columns=['fold_id','row_position'])
    option=BetOption(Outcome(perspective='home'),selection='draw')
    probabilities=pd.DataFrame([[.2,.8],[.5,.5],[.6,.4],[np.nan,np.nan],[.1,.9]],
                               columns=pd.MultiIndex.from_product([['draw'],[0,1]]))
    ctx=context(pd.DataFrame({'draw':[1,1,0,1,1]}),{'predict_proba':probabilities},data,identity_columns=KEYS)
    ctx.definitions={'label':option}
    meta=ctx.metadata.reset_index(drop=True)
    labels={'draw':LabelData(pd.DataFrame({'draw':[1,1,0,1,1]}),meta,'match','home',KEYS,option,
                            pd.DataFrame({'draw':['win','win','loss','win','win']}))}
    return ctx,option,labels


def test_binary_threshold_baseline_equality_unknowns_and_report_totals():
    ctx,option,labels=binary_context()
    quote=pd.Series([2.,3.,4.,5.,np.nan],index=ctx.y.index)
    offers={'draw':BetOffer(option,quote)}
    baseline,base=prepare_bets(ctx,offers,BinaryDrawThreshold(None))
    filtered,table=prepare_bets(ctx,offers,BinaryDrawThreshold(.5))
    assert base['take'].tolist()==[True,True,True,True,False]
    assert table['take'].tolist()==[True,True,False,False,False]
    assert table.p_win.iloc[0]==.8 and table.p_loss.iloc[0]==.2
    assert table.reason.iloc[3]=='Probability unavailable'
    for specs,n in [(baseline,4),(filtered,2)]:
        reporter=BetPerformanceReporter(type='overall',partition='test',bets=specs,labels=labels,
            composition=AllCombinations())
        result=reporter.run(ctx)
        assert len(result.tables['tickets'])==comb(n,2)
        assert result.tables['ledger'].cumulative_known_profit.iloc[-1]==result.tables['tickets'].profit.sum()
    report=BetOutcomeReporter(type='overall',partition='test',offers=offers,labels=labels,
        policy=BinaryDrawThreshold(.5),composition=AllCombinations(),show_badges=False).run(ctx)
    assert report.tables['combination_preview'].missing_probability_events.tolist()==[1]
    ctx.y.iloc[:]=-999
    _,after=prepare_bets(ctx,offers,BinaryDrawThreshold(.5))
    pd.testing.assert_frame_equal(table,after)
    ctx.definitions={'label':option.source}
    with pytest.raises(ValueError,match='predicted label'):
        prepare_bets(ctx,offers,BinaryDrawThreshold(.5))


def test_catalog_policies_and_slip_roundtrip():
    catalog=catalog_for_ui()
    policies=[AllCombinations(legs=k) for k in (2,3,4)]
    for value in [*policies, BetSlip({'pairs':policies[0],'singles':Parlay(size=1)}),BinaryDrawThreshold(.5)]:
        assert catalog.build(json.loads(json.dumps(catalog.encode(value))))==value
    schema={s['id']:s for s in catalog.schema()}
    assert 'evaluation.AllCombinations' in schema
    choices=next(f for f in schema['reporting.BetOutcomeReporter']['fields'] if f['name']=='composition')['components']
    assert 'evaluation.AllCombinations' in choices


@pytest.mark.parametrize('rule,expected,payout', [('remove','win',2.),('refund','push',1.),('loss','loss',0.)])
def test_push_override_and_explicit_probability(rule,expected,payout):
    result=compose_bets(sample(2).assign(settlement=['win','push']),
                        AllCombinations(on_push=rule, probability_mode='independent'))[0].iloc[0]
    assert result.settlement==expected and result.payout==payout
    assert result.probability==pytest.approx(.8*.8)
    assert result.probability_assumption=='independent'


def test_large_counts_are_exact_and_grouping_cannot_use_outcomes():
    count=preview_combinations(sample(100),AllCombinations(legs=50)).attrs['total_tickets']
    assert count==comb(100,50) and count>2**64
    with pytest.raises(ValueError,match='stage_column'):
        compose_bets(sample(),AllCombinations(stage_column='profit'))


def test_stage_identity_survives_label_and_dataset_assembly():
    from label_samples import label_history
    from xdiyo_analytics.labels import create_labels
    from xdiyo_analytics.datasets import assemble_dataset
    history=label_history().assign(tournament_id=99,stage_id=123)
    label=create_labels(history,{'draw':BetOption(Outcome(perspective='home'),selection='draw')})['draw']
    features=history[[*label.identity_columns,'team_id','side']].assign(known=1.)
    dataset=assemble_dataset(features,label,layout='match')
    assert dataset.metadata.tournament_id.eq(99).all()
    assert dataset.metadata.stage_id.eq(123).all()


@pytest.mark.parametrize('threshold',[-.1,1.1,float('nan'),True])
def test_binary_threshold_is_frozen_and_validated(threshold):
    ctx,option,_=binary_context()
    with pytest.raises(ValueError,match='max_non_draw_probability'):
        prepare_bets(ctx,{'draw':BetOffer(option,2.)},BinaryDrawThreshold(threshold))


def test_three_way_probabilities_cannot_be_used_as_binary_draw():
    ctx,option,_=binary_context()
    ctx.predictions['predict_proba'].columns=pd.MultiIndex.from_product([['draw'],[-1,0]])
    with pytest.raises(ValueError,match='0=non-draw'):
        prepare_bets(ctx,{'draw':BetOffer(option,2.)},BinaryDrawThreshold(.5))


def test_native_opening_odds_binary_recipe_and_performance_source(tmp_path):
    import hashlib
    from xdiyo_analytics.odds import OddsSeries, save_crosswalk
    from xdiyo_analytics.odds.crosswalk import MATCH_KEYS
    from xdiyo_analytics.ui.recipe import node
    from xdiyo_analytics.analysis import PostTrainingAnalysis
    from test_post_training_betting import as_training
    ctx,option,labels=binary_context()
    ctx.metadata['source_league']='Synthetic'
    ctx.metadata['source_season']='24_25'
    ctx.identity_columns=ctx.match_columns=MATCH_KEYS
    labels['draw'].metadata=ctx.metadata.reset_index(drop=True)
    labels['draw'].identity_columns=MATCH_KEYS
    # A synthetic native odds partition, containing quotes only.
    source=tmp_path/'database'
    path=source/'league=Synthetic/season=24_25/Odds/quotes-0000.parquet'
    path.parent.mkdir(parents=True)
    records=[]
    for timing in ['opening','closing']:
        for i in range(5):
            price=(float(i+2) if i<4 else np.nan) if timing=='opening' else 99.
            records.append(dict(match_id=str(i),market='1x2',selection='draw',line=None,quote_type=timing,
                                decimal_odds=price,status='valid' if pd.notna(price) else 'missing',
                                source_column='draw_'+timing+'_odds'))
    pd.DataFrame(records).to_parquet(path,index=False)
    identity=dict(adapter_version='1',source_files=[],league_seasons=[['Synthetic','24_25']],league_aliases={})
    snapshot=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    manifest=dict(identity,snapshot=snapshot,files=[dict(path=path.relative_to(source).as_posix(),
                                                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
    (source/'manifest.json').write_text(json.dumps(manifest))
    mapping=ctx.metadata[list(MATCH_KEYS)].copy().reset_index(drop=True)
    mapping['vendor_match_id']=[str(i) for i in range(5)]
    mapping['snapshot']=snapshot
    mapping['status']='matched'
    mapping['reason']='synthetic identity'
    mapping['vendor_scheduled_at']=ctx.metadata.kickoff_at.to_numpy()
    saved=save_crosswalk(mapping,snapshot=source,output_root=tmp_path/'mappings')
    odds=OddsSeries(str(source),str(saved),('24_25',),'1x2','draw',quote_type='opening',settlement_confirmed=True)
    catalog=catalog_for_ui()
    for cutoff,expected in [(None,6),(.5,1)]:
        reporter=node('reporting.BetOutcomeReporter',type='overall',partition='test',
            offers={'draw':node('evaluation.BetOffer',option=catalog.encode(option),odds=catalog.encode(odds))},
            policy=node('evaluation.BinaryDrawThreshold',max_non_draw_probability=cutoff),
            labels={'ref':'settlements'},show_badges=False,
            composition=node('evaluation.AllCombinations',legs=2))
        rebuilt=catalog.build(json.loads(json.dumps(reporter)),{'settlements':labels})
        training=as_training(ctx)
        training.definitions=deepcopy(ctx.definitions)
        report=PostTrainingAnalysis({'decisions':rebuilt,
            'performance':BetPerformanceReporter(type='overall',partition='test',source='decisions')}).run(training)
        before,after=[s.result for s in report.studies]
        assert len(after.tables['tickets'])==expected
        assert after.tables['ticket_legs'].odds.max()<99
        assert after.tables['odds_provenance'].quote_type.eq('opening').all()
        assert after.tables['combination_preview'].missing_price_events.tolist()==[1]
        assert after.tables['ledger'].cumulative_known_profit.iloc[-1]==before.tables['tickets'].profit.sum()
        assert before.tables['ticket_legs'].quote_snapshot_hash.eq(odds.snapshot_hash).all()
