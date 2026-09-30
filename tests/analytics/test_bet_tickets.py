"""Composite stake accounting, outcome-independent construction and UI recipes."""
from dataclasses import replace
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.evaluation import Parlay, MultiBet, BetSlip, compose_bets, BetOffer, TightestLine
from xdiyo_analytics.reporting import BetOutcomeReporter, BetPerformanceReporter
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.ui import catalog_for_ui
from test_bet_outcomes import reporting_context
from test_post_training_betting import as_training
from test_ui_workflow import ui_recipe


def ledger(states=('win','win','loss','win','push','void')):
    n = len(states)
    return pd.DataFrame(dict(fold_id=[4]*n, row_position=range(n),
        event_id=[2**63+100+i for i in range(n)], competition_id=[1]*n,
        source_season=['24_25']*n, season_id=[100]*n, round=[10]*n,
        kickoff_at=pd.date_range('2025-01-01', periods=n, freq='h', tz='UTC'),
        bet=['over']*n, take=[True]*n, odds=[2.]*n, stake=[20.]*n,
        settlement=states, p_win=[.8]*n, p_push=[0.]*n, p_loss=[.2]*n))


@pytest.mark.parametrize('states,expected,payout', [
    (('win','win'), 'win', 8.), (('win','push'), 'win', 4.),
    (('win','void'), 'win', 4.), (('push','void'), 'push', 2.),
    (('void','void'), 'void', 2.), (('loss','missing'), 'loss', 0.),
    (('win','missing'), 'missing', np.nan),
])
def test_default_settlement(states, expected, payout):
    tickets, members, metrics = compose_bets(ledger(states), Parlay(stake=2))
    assert tickets.settlement.tolist() == [expected]
    np.testing.assert_allclose(tickets.payout, [payout], equal_nan=True)
    assert tickets.stake.iloc[0] == 2  # never sums the leg stakes
    assert members.event_id.tolist() == [2**63+100, 2**63+101]
    assert tickets.probability.isna().all()


@pytest.mark.parametrize('rule,states,expected,payout', [
    ('refund', ('win','push'), 'push', 1.),
    ('refund', ('missing','push'), 'push', 1.),
    ('refund', ('loss','push'), 'loss', 0.),
    ('loss', ('win','push'), 'loss', 0.),
])
def test_explicit_push_rules(rule, states, expected, payout):
    tickets, _, _ = compose_bets(ledger(states), Parlay(on_push=rule))
    assert tickets.settlement.iloc[0] == expected
    assert tickets.payout.iloc[0] == payout


def test_system_stakes_and_slip_accounting():
    data = ledger(('win','win','loss'))
    policy = MultiBet(size=3, sizes=(2,3), stake=8, stake_mode='total')
    tickets, membership, metrics = compose_bets(data, policy)
    assert tickets.n_legs.tolist() == [2,2,2,3]
    assert tickets.stake.sum() == 8
    assert tickets.profit.tolist() == [6,-2,-2,-2]
    assert len(membership) == 9
    assert metrics.set_index('metric').loc['profit','value'] == 0
    slip = BetSlip({'singles':Parlay(size=1, stake=1), 'system':policy})
    tickets, _, metrics = compose_bets(data, slip)
    assert tickets.stake.sum() == 11
    assert tickets.profit.sum() == 1
    assert set(metrics.target) == {'singles','system'}


def test_groupings_rounds_seasons_days_and_folds():
    data = ledger(('win',)*6)
    data['competition_id'] = [1,2,1,2,1,2]
    data['season_id'] = [100,200,100,200,100,200]
    data['round'] = [10,10,11,11,10,10]
    data.loc[4:,'source_season'] = '25_26'
    assert compose_bets(data, Parlay(grouping='league_round'))[0].shape[0] == 2
    assert compose_bets(data, Parlay(grouping='round'))[0].shape[0] == 3
    assert compose_bets(data, Parlay(grouping='day'))[0].shape[0] == 3
    data['fold_id'] = range(6)
    assert compose_bets(data, Parlay(grouping='day'))[0].empty
    with pytest.raises(ValueError, match='source_season'):
        compose_bets(data.drop(columns='source_season'), Parlay(grouping='round'))


def test_membership_never_uses_outcomes_and_keeps_one_leg_per_fixture():
    data = ledger()
    data['p_win'] = [.7,.9,.6,.8,.5,.95]
    policy = Parlay(order_by='probability', probability_mode='independent')
    first = compose_bets(data, policy)
    data['settlement'] = 'loss'
    second = compose_bets(data, policy)
    pd.testing.assert_frame_equal(first[1][['ticket_id','event_id']], second[1][['ticket_id','event_id']])
    assert first[0].probability.iloc[0] == pytest.approx(.95*.9)
    duplicate = data.iloc[[0]].assign(bet='other', row_position=20)
    tickets, members, _ = compose_bets(pd.concat([data,duplicate]), policy)
    assert len(tickets) == 3
    assert not members.duplicated(['ticket_id','event_id']).any()
    data.loc[5,'take'] = False
    assert len(compose_bets(data, policy)[0]) == 2  # leftover leg not padded


def test_missing_quotes_and_unresolved_legs():
    data = ledger(('win','win'))
    data.loc[0,'odds'] = np.nan
    result = compose_bets(data, Parlay())[0]
    assert result.profit.isna().all() and result.odds.isna().all()
    data.loc[0,'settlement'] = 'void'
    result = compose_bets(data, Parlay())[0]
    assert result.profit.iloc[0] == 1.  # no quote needed for a removed leg
    data.loc[1,'settlement'] = 'missing'
    assert compose_bets(data, Parlay())[2].set_index('metric').loc['profit','status'] == 'partial'


def test_no_ticket_explosion_or_invalid_sizes():
    with pytest.raises(ValueError, match='max_tickets'):
        compose_bets(ledger(), MultiBet(size=6, sizes=(2,3), max_tickets=10))
    with pytest.raises(ValueError, match='System sizes'):
        compose_bets(ledger(), MultiBet(size=2, sizes=(3,)))
    with pytest.raises(ValueError, match='p_win'):
        compose_bets(ledger().drop(columns='p_win'), Parlay(order_by='probability'))


def test_reporters_reuse_ticket_ledger_and_render_team_details():
    ctx, specs, labels, _ = reporting_context()
    reporter = BetOutcomeReporter(type='overall', partition='test', labels=labels,
        offers={'offer':BetOffer(specs['offer'].option, 2.)}, policy=TightestLine(min_probability=.7),
        composition=BetSlip({'doubles':Parlay(grouping='day')}))
    # The sample fixtures are separated in time; use a single day for grouping.
    ctx.metadata['kickoff_at'] = pd.date_range('2025-01-01', periods=len(ctx.y), freq='h', tz='UTC')
    report = PostTrainingAnalysis({'decisions':reporter, 'profit':BetPerformanceReporter(
        type='overall', partition='test', source='decisions')}).run(as_training(ctx))
    decisions, performance = [s.result for s in report.studies]
    assert len(decisions.tables['ledger']) == 3
    pd.testing.assert_frame_equal(decisions.tables['ticket_legs'], performance.tables['ticket_legs'])
    pd.testing.assert_frame_equal(decisions.tables['bet_metrics'], performance.tables['metrics'])
    assert 'Bet tickets' in report.to_html()
    assert 'Selected bet' in decisions.artifacts[0].data
    assert 'Net profit' in decisions.artifacts[0].data
    assert len(decisions.tables['leg_ledger']) == len(ctx.y)
    assert decisions.artifacts[1].options['profit'] is False


def test_manual_performance_and_source_composition():
    ctx, specs, labels, _ = reporting_context()
    ctx.metadata['kickoff_at'] = pd.date_range('2025-01-01', periods=len(ctx.y), freq='h', tz='UTC')
    reporter = BetPerformanceReporter(type='overall', partition='test', bets=specs,
                                      labels=labels, composition=MultiBet(grouping='day'))
    result = reporter.run(ctx)
    assert len(result.tables['tickets']) == 8
    assert result.tables['metrics'].set_index('metric').loc['bets_placed','value'] == 8


def test_ui_catalog_roundtrip_and_discoverable_fields():
    catalog = catalog_for_ui()
    policy = BetSlip({'singles':Parlay(size=1, grouping='round'),
                     'system':MultiBet(size=4, sizes=(2,3), stake_mode='total')})
    decoded = catalog.build(catalog.encode(policy), {})
    assert tuple(decoded.tickets['system'].sizes) == (2,3)
    schema = {c['id']:c for c in catalog.schema()}
    fields = {f['name']:f for f in schema['evaluation.Parlay']['fields']}
    assert [choice['value'] for choice in fields['grouping']['choices']] == ['league_round','round','day']
    assert all(f['help'] for f in fields.values())
    for key in ('reporting.BetOutcomeReporter','reporting.BetPerformanceReporter'):
        field = next(f for f in schema[key]['fields'] if f['name']=='composition')
        assert 'evaluation.BetSlip' in field['components']


def test_empty_tickets_report_zero_stakes_and_undefined_roi():
    tickets, _, metrics = compose_bets(ledger(('win',)), Parlay())
    assert tickets.empty
    values = metrics.set_index('metric')
    assert values.loc['bets_placed', 'value'] == 0
    assert values.loc['profit', 'value'] == 0
    assert values.loc['roi', 'status'] == 'undefined'


def test_no_second_composition_of_source_tickets():
    ctx, specs, labels, _ = reporting_context()
    ctx.metadata['kickoff_at'] = pd.Timestamp('2025-01-01', tz='UTC')
    reporter = BetOutcomeReporter(type='overall', partition='test', labels=labels,
        offers={'offer':BetOffer(specs['offer'].option, 2.)}, policy=TightestLine(min_probability=.7),
        composition=Parlay(grouping='day'))
    with pytest.raises(ValueError, match='already contains tickets'):
        PostTrainingAnalysis({'decisions':reporter, 'profit':BetPerformanceReporter(
            type='overall', partition='test', source='decisions', composition=Parlay(grouping='day'))}).run(as_training(ctx))


def test_source_singles_can_be_composed_by_performance():
    ctx, specs, labels, _ = reporting_context()
    ctx.metadata['kickoff_at'] = pd.Timestamp('2025-01-01', tz='UTC')
    reporter = BetOutcomeReporter(type='overall', partition='test', labels=labels,
        offers={'offer':BetOffer(specs['offer'].option, 2.)}, policy=TightestLine(min_probability=.7))
    report = PostTrainingAnalysis({'decisions':reporter, 'profit':BetPerformanceReporter(
        type='overall', partition='test', source='decisions', composition=Parlay(grouping='day'))}).run(as_training(ctx))
    assert len(report.studies[1].result.tables['tickets']) == 3


def test_ui_ticket_changes_refresh_saved_run_without_fitting(ui_recipe, monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from xdiyo_analytics.ui import prepare_recipe, run_recipe
    from xdiyo_analytics.ui.recipe import node
    recipe = deepcopy(ui_recipe)
    recipe['model'] = node('sklearn.linear_model.LogisticRegression', max_iter=200)
    recipe['post_reporters'] = {}
    prepared = prepare_recipe(recipe)
    # Shared synthetic day/round so this small fixture has complete tickets.
    prepared.dataset.metadata['round'] = 1
    original = run_recipe(recipe, prepared=prepared)
    def forbidden(*args, **kwargs):
        raise AssertionError('Changing bet composition must not fit or predict again')
    for method in ('fit','predict','predict_proba'):
        monkeypatch.setattr(LogisticRegression, method, forbidden)
    recipe['post_reporters'] = {'tickets': node('reporting.BetOutcomeReporter', type='overall', partition='score',
        offers={'under':node('evaluation.BetOffer', option=node('labels.BetOption', source=recipe['labels']['corners'],
                    selection='under', line=100.5), odds=2.)}, policy=node('evaluation.TightestLine', min_probability=0.),
        composition=node('evaluation.MultiBet', size=3, sizes=[2,3], grouping='league_round'))}
    refreshed = run_recipe(recipe, prepared=prepared)
    assert refreshed.reused and refreshed.record['run_id'] == original.record['run_id']
    tables = refreshed.post_report.studies[0].result.tables
    assert len(tables['tickets']) > 0
    cached = run_recipe(recipe, prepared=prepared)
    pd.testing.assert_frame_equal(cached.post_report.studies[0].result.tables['tickets'], tables['tickets'])


def test_ticket_details_escape_names_and_embed_catalog_badges(tmp_path):
    from xdiyo_analytics.reporting import TeamCatalog
    ctx, specs, labels, _ = reporting_context()
    ctx.metadata['kickoff_at'] = pd.date_range('2025-01-01', periods=len(ctx.y), freq='h', tz='UTC')
    reporter = BetOutcomeReporter(type='overall', partition='test', labels=labels,
        offers={'offer':BetOffer(specs['offer'].option, 2.)}, policy=TightestLine(min_probability=.7),
        composition=Parlay(grouping='day'))
    from xdiyo_analytics.reporting.match_results import fixture_rows
    _, teams, _ = fixture_rows(ctx, reporter)
    badge = tmp_path/'badge.svg'
    badge.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><circle cx="10" cy="10" r="9" fill="blue"/></svg>')
    reporter.catalog = TeamCatalog({key:{'name':'<Example team>', 'badge_path':str(badge)} for key in teams})
    html = reporter.run(ctx).artifacts[0].data
    assert 'data:image/svg+xml;base64,' in html
    assert '&lt;Example team&gt;' in html and '<Example team>' not in html
    assert 'Settled multiplier 2' in html
