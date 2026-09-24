"""Identity joins and explicit win/loss/push/void accounting, without a strategy."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from label_samples import STAT, ALL_PERIODS, label_history
from post_training_samples import context, RecordedModel
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.evaluation import BetSpec, evaluate_bets
from xdiyo_analytics.labels import BetOption, MatchTotal, TeamValue, create_labels
from xdiyo_analytics.reporting import BetPerformanceReporter
from xdiyo_analytics.training import FoldResult, TrainingResult


def bet_sample(*, team=False):
    history = label_history()
    option = BetOption(TeamValue(STAT) if team else MatchTotal(STAT), selection='over', line=11,
                       void_statuses=('cancelled',))
    labels = create_labels(history, {'offer': option})
    label = labels['offer']
    order = [2, 0, 4, 1, 5, 3] if not team else list(range(len(label.y)))
    metadata = label.metadata.iloc[order].copy()
    truth = pd.DataFrame({'count': np.arange(len(metadata), dtype=float)})
    ctx = context(truth, {'predict': truth*0}, metadata, layout=label.unit, identity_columns=label.identity_columns)
    ctx.match_columns = tuple(name for name in label.identity_columns if name != 'team_id')
    return ctx, {'offer': BetSpec(option, 2.5, True, stake=2.)}, labels, history


def as_training(ctx, *, repeat=False):
    folds = []
    for fold_id in ([4, 9] if repeat else [4]):
        y, metadata = ctx.y.copy(), ctx.metadata.copy()
        y.index = metadata.index = pd.Index(np.arange(len(y)), name='row_position')
        predictions = {name: frame.set_axis(y.index) for name, frame in ctx.predictions.items()}
        positions = np.arange(len(y))
        folds.append(FoldResult(fold_id, RecordedModel(), np.array([], dtype=int), positions, positions,
                                ('synthetic',), tuple(y.columns), predictions, y, metadata))
    return TrainingResult(folds, ctx.layout, ctx.identity_columns, ctx.match_columns, 'total')


def test_profit_and_roi_count_push_void_stakes_and_keep_unresolved():
    ctx, bets, labels, _ = bet_sample()
    before = deepcopy(labels)
    ledger, metrics = evaluate_bets(ctx, bets, labels=labels)
    assert ledger.settlement.tolist() == ['win', 'push', 'void', 'loss', 'missing', 'missing']
    np.testing.assert_allclose(ledger.payout, [5., 2., 2., 0., np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(ledger.profit, [3., 0., 0., -2., np.nan, np.nan], equal_nan=True)
    assert ledger.event_id.tolist() == ctx.metadata.event_id.tolist()
    assert min(ledger.event_id) > 2**63
    rows = metrics.set_index('metric')
    expected = {'profit': 1., 'roi': 1/8, 'bets_placed': 6, 'settled_bets': 4,
                'unresolved_bets': 2, 'settled_stakes': 8., 'payout': 9.,
                'win_count': 1, 'loss_count': 1, 'push_count': 1, 'void_count': 1}
    assert rows.value.to_dict() == expected
    assert rows.loc['profit', 'status'] == rows.loc['roi', 'status'] == 'partial'
    assert rows.n.eq(4).all() and rows.n_missing.eq(2).all()
    assert_frame_equal(labels['offer'].metadata, before['offer'].metadata)
    assert_frame_equal(labels['offer'].settlement, before['offer'].settlement)


def test_identity_indexed_quotes_and_occurrence_stakes_are_reordered_exactly():
    ctx, bets, labels, _ = bet_sample()
    keys = labels['offer'].identity_columns
    index = pd.MultiIndex.from_frame(labels['offer'].metadata[list(keys)])
    quotes = pd.Series([1.5, 2., 2.5, 3., 3.5, 4.], index=index).iloc[::-1]
    stakes = pd.Series([2., 3., 4., 5., 6., 7.], index=ctx.y.index)
    spec = replace(bets['offer'], odds=quotes, stake=stakes)
    ledger, metrics = evaluate_bets(ctx, {'offer': spec}, labels=labels)
    assert ledger.odds.tolist() == [2.5, 1.5, 3.5, 2., 4., 3.]
    np.testing.assert_allclose(ledger.profit, [3., 0., 0., -5., np.nan, np.nan], equal_nan=True)
    rows = metrics.set_index('metric')
    assert rows.loc['settled_stakes', 'value'] == 14.
    assert rows.loc['roi', 'value'] == pytest.approx(-2/14)


def test_supplied_labels_and_history_paths_agree_and_dict_specs_work():
    ctx, bets, labels, history = bet_sample()
    expected = evaluate_bets(ctx, bets, labels=labels)
    spec = bets['offer']
    actual = evaluate_bets(ctx, {'offer': dict(option=spec.option, odds=spec.odds, take=spec.take, stake=spec.stake)}, history=history)
    for left, right in zip(actual, expected):
        assert_frame_equal(left, right)


def test_nonselected_rows_are_zero_and_do_not_create_unresolved_bets():
    ctx, bets, labels, _ = bet_sample()
    take = pd.Series([True, True, True, True, False, False], index=ctx.y.index)
    ledger, metrics = evaluate_bets(ctx, {'offer': replace(bets['offer'], take=lambda supplied: take.reindex(supplied.y.index))}, labels=labels)
    assert ledger.profit.tolist() == [3., 0., 0., -2., 0., 0.]
    assert ledger.accounting_status.tolist() == ['settled']*4+['not_placed']*2
    assert metrics.set_index('metric').loc['profit', 'status'] == 'ok'
    empty, rows = evaluate_bets(ctx, {'offer': replace(bets['offer'], take=False, odds=np.nan, stake=np.nan)}, labels=labels)
    assert empty.profit.eq(0).all() and empty.stake.eq(0).all()
    rows = rows.set_index('metric')
    assert rows.loc['profit', 'value'] == 0 and rows.loc['profit', 'status'] == 'ok'
    assert rows.loc['roi', 'status'] == 'undefined' and rows.loc['bets_placed', 'value'] == 0


def test_missing_quotes_stakes_and_settlement_coverage_stay_unresolved():
    ctx, bets, labels, _ = bet_sample()
    odds = pd.Series(2.5, index=ctx.y.index)
    stakes = pd.Series(2., index=ctx.y.index)
    odds.iloc[0] = np.nan
    stakes.iloc[1] = np.nan
    ledger, metrics = evaluate_bets(ctx, {'offer': replace(bets['offer'], odds=odds, stake=stakes)}, labels=labels)
    assert ledger.accounting_status.tolist() == ['unresolved', 'unresolved', 'settled', 'settled', 'unresolved', 'unresolved']
    assert metrics.set_index('metric').loc['unresolved_bets', 'value'] == 4
    shorter = deepcopy(labels)
    for field in ('y', 'metadata', 'settlement'):
        setattr(shorter['offer'], field, getattr(shorter['offer'], field).iloc[1:])
    ledger, _ = evaluate_bets(ctx, bets, labels=shorter)
    assert ledger.settlement.iloc[1] == 'missing'


def test_bet_comparison_binding_excludes_decisions_and_tracks_quotes_and_policy():
    ctx, bets, labels, _ = bet_sample()
    def metric(spec):
        return evaluate_bets(ctx, {'offer': spec}, labels=labels)[1].iloc[0]
    base = metric(bets['offer'])
    different_take = metric(replace(bets['offer'], take=False))
    different_quotes = metric(replace(bets['offer'], odds=3.))
    different_policy = metric(replace(bets['offer'], policy='different_rule'))
    different_stake = metric(replace(bets['offer'], stake=3.))
    assert base.sample_hash == different_take.sample_hash and base.parameters == different_take.parameters
    assert base.sample_hash != different_quotes.sample_hash
    assert base.parameters != different_policy.parameters
    assert base.parameters != different_stake.parameters


def test_bet_reporter_orders_known_profit_and_retains_occurrence_scope():
    ctx, bets, labels, _ = bet_sample()
    reporter = BetPerformanceReporter(type='overall', partition='test', bets=bets, labels=labels)
    report = PostTrainingAnalysis({'bets': reporter}).run(as_training(ctx))
    study = report.studies[0]
    ledger = study.result.tables['ledger']
    assert ledger.kickoff_at.is_monotonic_increasing
    assert ledger.cumulative_known_profit.tolist() == [0., -2., 1., 1., 1., 1.]
    assert ledger.profit.isna().sum() == 2
    assert study.scope.fold_id.tolist() == [4]*6
    assert len(study.result.artifacts) == 2


def test_repeated_bets_need_per_fold_or_unique_pooling():
    ctx, bets, labels, _ = bet_sample()
    training = as_training(ctx, repeat=True)
    per_fold = BetPerformanceReporter(type='per_fold', partition='test', bets=bets, labels=labels)
    assert len(PostTrainingAnalysis({'bets': per_fold}).run(training).studies) == 2
    repeated = BetPerformanceReporter(type='overall', partition='test', pooling='occurrences', bets=bets, labels=labels)
    with pytest.raises(ValueError, match='unique observations'):
        PostTrainingAnalysis({'bets': repeated}).run(training)
    unique = replace(repeated, pooling='first')
    result = PostTrainingAnalysis({'bets': unique}).run(training)
    assert len(result.studies[0].result.tables['ledger']) == 6


@pytest.mark.parametrize('field,value', [('odds', 1.), ('odds', -2.), ('odds', np.inf),
                                       ('stake', -1.), ('stake', np.inf), ('take', 1), ('take', np.nan),
                                       ('take', [True]*6), ('odds', [2.]*6)])
def test_invalid_placed_bet_inputs_raise(field, value):
    ctx, bets, labels, _ = bet_sample()
    with pytest.raises((ValueError, TypeError)):
        evaluate_bets(ctx, {'offer': replace(bets['offer'], **{field: value})}, labels=labels)


def test_invalid_or_ambiguous_settlement_sources_raise():
    ctx, bets, labels, history = bet_sample()
    with pytest.raises(ValueError, match='exactly one'):
        evaluate_bets(ctx, bets)
    with pytest.raises(ValueError, match='exactly one'):
        evaluate_bets(ctx, bets, labels=labels, history=history)
    with pytest.raises(ValueError, match='option'):
        evaluate_bets(ctx, {'offer': replace(bets['offer'], option=replace(bets['offer'].option, line=10))}, labels=labels)
    for corruption in ('reversed_index', 'duplicate_identity', 'unknown_category', 'no_settlement'):
        changed = deepcopy(labels)
        label = changed['offer']
        if corruption == 'reversed_index':
            label.settlement = label.settlement.iloc[::-1]
        elif corruption == 'duplicate_identity':
            label.metadata.iloc[1, label.metadata.columns.get_loc('event_id')] = label.metadata.event_id.iloc[0]
        elif corruption == 'unknown_category':
            label.settlement = label.settlement.astype(object)
            label.settlement.iloc[0, 0] = 'guessed'
        else:
            label.settlement = None
        with pytest.raises(ValueError):
            evaluate_bets(ctx, bets, labels=changed)
    option = BetOption(MatchTotal(ALL_PERIODS), 'over', line=11)
    with pytest.raises(ValueError, match='one settlement'):
        evaluate_bets(ctx, {'offer': BetSpec(option, 2., True)}, history=history)


def test_bet_series_must_match_named_identity_or_occurrences():
    ctx, bets, labels, _ = bet_sample()
    for series in (pd.Series(2., index=range(6)), pd.Series(2., index=ctx.y.index[::-1]),
                       pd.Series(2., index=pd.MultiIndex.from_frame(ctx.metadata[list(ctx.identity_columns)]).set_names([f'wrong_{i}' for i in range(len(ctx.identity_columns))]))):
        with pytest.raises(ValueError, match='Series must match'):
            evaluate_bets(ctx, {'offer': replace(bets['offer'], odds=series)}, labels=labels)


def test_team_match_accounting_joins_each_team_without_collapsing_matches():
    ctx, bets, labels, _ = bet_sample(team=True)
    for field in ('y', 'metadata', 'settlement'):
        setattr(labels['offer'], field, getattr(labels['offer'], field).iloc[::-1])
    ledger, metrics = evaluate_bets(ctx, bets, labels=labels)
    assert ledger.settlement.tolist() == ['loss', 'loss', 'loss', 'loss', 'push', 'loss',
                                           'missing', 'missing', 'void', 'void', 'missing', 'loss']
    assert ledger.team_id.tolist() == [2**63+31, 2**63+32]*6
    assert ledger.event_id.nunique() == 6 and len(ledger) == 12
    totals = metrics.set_index('metric')
    assert totals.loc['profit', 'value'] == -12
    assert totals.loc['settled_stakes', 'value'] == 18
    assert totals.loc['roi', 'value'] == pytest.approx(-2/3)
    assert totals.loc['unresolved_bets', 'value'] == 3


@pytest.mark.parametrize('team', [False, True])
def test_bet_accounting_rejects_mismatched_prediction_and_label_layout(team):
    ctx, bets, labels, _ = bet_sample(team=team)
    ctx.layout = 'match' if team else 'team_match'
    with pytest.raises(ValueError, match='layout'):
        evaluate_bets(ctx, bets, labels=labels)
