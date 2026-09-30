"""Prediction-only decisions, shared settlement and persisted UI bet reports."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom

from label_samples import STAT
from post_training_samples import context
from test_post_training_betting import bet_sample, as_training
from test_ui_workflow import ui_recipe
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.evaluation import BetOffer, TightestLine, HighestExpectedProfit, prepare_bets
from xdiyo_analytics.labels import BetOption, MatchTotal
from xdiyo_analytics.reporting import BetOutcomeReporter, BetPerformanceReporter
from xdiyo_analytics.ui import prepare_recipe, run_recipe, catalog_for_ui
from xdiyo_analytics.ui.recipe import node


def decision_context(values=(.15, .25, .6), classes=(7, 8, 12)):
    return context(pd.DataFrame({'count': [99.]}), {'predict_proba': pd.DataFrame(
        [values], columns=pd.MultiIndex.from_product([['count'], classes]))})


def offer(side, line, odds=None):
    return BetOffer(BetOption(MatchTotal(STAT), selection=side, line=line), odds)


@pytest.mark.parametrize('side,expected', [('under', 'middle'), ('over', 'middle')])
def test_tightest_line_uses_exact_half_lines_and_probability_loss(side, expected):
    offers = {name: offer(side, line) for name, line in [('low', 7.5), ('middle', 8.5), ('high', 12.5)]}
    if side == 'under':
        ctx = decision_context((.8, .1, .1))
    else:
        ctx = decision_context((.1, .1, .8))
    _, alternatives = prepare_bets(ctx, offers, TightestLine(min_probability=.75, max_probability_loss=.11))
    assert alternatives.loc[alternatives['take'], 'bet'].tolist() == [expected]
    assert alternatives.p_push.eq(0).all()
    np.testing.assert_allclose(alternatives.p_win, [.8, .9, 1.] if side == 'under' else [.9, .8, 0.])


def test_ev_accounts_for_push_and_outcomes_cannot_change_policy():
    ctx = decision_context((.2, .3, .5), (6, 7, 8))
    offers = {'under': offer('under', 7, 3.), 'over': offer('over', 7, 2.)}
    specs, before = prepare_bets(ctx, offers, HighestExpectedProfit())
    np.testing.assert_allclose(before.expected_profit, [-.1, .3])
    assert before.loc[before['take'], 'bet'].tolist() == ['over']
    ctx.y.iloc[:] = -1000
    _, after = prepare_bets(ctx, offers, HighestExpectedProfit())
    pd.testing.assert_frame_equal(before, after)
    assert specs['over'].take.all() and not specs['under'].take.any()
    _, skipped = prepare_bets(ctx, offers, HighestExpectedProfit(min_ev=.4))
    assert not skipped['take'].any()


def test_different_fold_supports_do_not_treat_real_missing_probabilities_as_zero():
    ctx = decision_context()
    index = pd.MultiIndex.from_tuples([(4, 0), (9, 1), (9, 2)], names=['fold_id', 'row_position'])
    a = pd.DataFrame([[.2, .8]], columns=pd.MultiIndex.from_product([['count'], [7, 8]]), index=index[:1])
    b = pd.DataFrame([[.6, .4], [np.nan, .4]], columns=pd.MultiIndex.from_product([['count'], [8, 12]]), index=index[1:])
    ctx.y = pd.DataFrame({'count': [0, 0, 0]}, index=index)
    ctx.predictions = {'predict_proba': pd.concat([a, b])}
    ctx.fold_results = {4: SimpleNamespace(predictions={'predict_proba': a}), 9: SimpleNamespace(predictions={'predict_proba': b})}
    _, alternatives = prepare_bets(ctx, {'under': offer('under', 8.5)}, TightestLine(min_probability=.5))
    np.testing.assert_allclose(alternatives.p_win, [1., .6, np.nan], equal_nan=True)
    assert alternatives['take'].tolist() == [True, True, False]
    assert alternatives.reason.iloc[-1] == 'Probability unavailable'


def reporting_context(*, team=False):
    ctx, specs, labels, history = bet_sample(team=team)
    ctx.predictions['predict_proba'] = pd.DataFrame(np.tile([.05, .15, .8], (len(ctx.y), 1)),
        index=ctx.y.index, columns=pd.MultiIndex.from_product([['count'], [7, 11, 12]]))
    ctx.predictions['predict_proba_raw'] = pd.DataFrame(np.tile([.8, .15, .05], (len(ctx.y), 1)),
        index=ctx.y.index, columns=ctx.predictions['predict_proba'].columns)
    return ctx, specs, labels, history


@pytest.mark.parametrize('team', [False, True])
def test_reporter_uses_calibrated_output_and_reuses_exact_settlement(team, monkeypatch):
    ctx, specs, labels, history = reporting_context(team=team)
    spec = specs['offer']
    reporter = BetOutcomeReporter(type='overall', partition='test', offers={'offer': BetOffer(spec.option, 2.5, 2.)},
                                  policy=TightestLine(min_probability=.7), labels=labels)
    def forbidden(*args, **kwargs):
        raise AssertionError('Performance must reuse the exact earlier ledger, not settle again')
    monkeypatch.setattr('xdiyo_analytics.reporting.betting.evaluate_bets', forbidden)
    report = PostTrainingAnalysis({'decisions': reporter,
        'performance': BetPerformanceReporter(type='overall', partition='test', source='decisions')}).run(as_training(ctx))
    decisions, performance = [s.result for s in report.studies]
    ledger = decisions.tables['ledger']
    assert 'metrics' not in decisions.tables
    assert 'metrics' in performance.tables
    pd.testing.assert_frame_equal(decisions.tables['bet_metrics'], performance.tables['metrics'])
    pd.testing.assert_frame_equal(ledger.sort_values('row_position').reset_index(drop=True),
        performance.tables['ledger'][ledger.columns].sort_values('row_position').reset_index(drop=True))
    assert decisions.tables['bets'].bet.eq('Over 11').all()
    assert decisions.tables['bets'].probability.eq(.8).all()
    if not team:
        assert ledger.settlement.tolist() == ['win', 'push', 'void', 'loss', 'missing', 'missing']
        np.testing.assert_allclose(ledger.profit, [3., 0., 0., -2., np.nan, np.nan], equal_nan=True)
    raw = replace(reporter, output='predict_proba_raw').run(ctx)
    assert raw.tables['bets'].bet.eq('No bet').all()
    assert raw.tables['ledger'].profit.eq(0).all()


def test_missing_odds_display_is_explicit_and_performance_requires_quotes():
    ctx, specs, labels, _ = reporting_context()
    reporter = BetOutcomeReporter(type='overall', partition='test', labels=labels,
        offers={'offer': BetOffer(specs['offer'].option)}, policy=TightestLine(min_probability=.7))
    result = reporter.run(ctx)
    assert result.tables['bets'].probability.eq(.8).all()
    assert result.tables['bets'].profit.isna().all()
    assert result.artifacts[0].options['odds'] is False
    with pytest.raises(ValueError, match='odds'):
        PostTrainingAnalysis({'decisions': reporter, 'performance': BetPerformanceReporter(
            type='overall', partition='test', source='decisions')}).run(as_training(ctx))
    fallback = replace(reporter, default_odds=2.5).run(ctx)
    assert fallback.tables['ledger'].odds.eq(2.5).all()
    _, alternatives = prepare_bets(ctx, reporter.offers, HighestExpectedProfit())
    assert not alternatives['take'].any()
    assert alternatives.reason.str.contains('Missing odds').all()


def test_source_requires_matching_scope_and_prior_report():
    ctx, specs, labels, _ = reporting_context()
    reporter = BetOutcomeReporter(type='per_fold', partition='test', labels=labels,
        offers={'offer': BetOffer(specs['offer'].option, 2.)}, policy=TightestLine(min_probability=.7))
    with pytest.raises(ValueError, match='earlier'):
        PostTrainingAnalysis({'decisions': reporter, 'performance': BetPerformanceReporter(
            type='overall', partition='test', pooling='first', source='decisions')}).run(as_training(ctx, repeat=True))


def test_nb_pipeline_retains_mean_when_point_prediction_is_mode_and_recipe_reuses(ui_recipe, monkeypatch):
    from xdiyo_analytics.training import NegativeBinomialRegressor
    recipe = deepcopy(ui_recipe)
    recipe['model'] = node('training.NegativeBinomialRegressor', dispersion=.4, prediction='mode', max_iter=200)
    source = deepcopy(recipe['labels']['corners'])
    recipe['post_reporters'] = {
        'decisions': node('reporting.BetOutcomeReporter', type='overall', partition='score',
            offers={'under': node('evaluation.BetOffer', option=node('labels.BetOption', source=source, selection='under', line=7.5))},
            policy=node('evaluation.TightestLine', min_probability=0.), default_odds=2.),
        'performance': node('reporting.BetPerformanceReporter', type='overall', partition='score', source='decisions')}
    catalog = catalog_for_ui()
    built = catalog.build(recipe['post_reporters']['decisions'])
    assert isinstance(built, BetOutcomeReporter) and isinstance(built.offers['under'], BetOffer)
    assert isinstance(catalog.build(node('evaluation.HighestExpectedProfit')), HighestExpectedProfit)
    prepared = prepare_recipe(recipe)
    result = run_recipe(recipe, prepared=prepared)
    fold = result.training.folds[0]
    retained = fold.predictions['count_distribution']['corners']
    assert (retained['mean'] > fold.predictions['predict']['corners']).all()
    np.testing.assert_allclose(retained.dispersion, .4)
    decisions = result.post_report.studies[0].result
    np.testing.assert_allclose(decisions.tables['bets'].probability,
        nbinom.cdf(7, 1/.4, 1/(1+.4*retained['mean'])))
    assert 'Alpha Home' in result.to_html()
    def forbidden(*args, **kwargs):
        raise AssertionError('Cached reports must not fit or predict')
    monkeypatch.setattr(NegativeBinomialRegressor, 'fit', forbidden)
    monkeypatch.setattr(NegativeBinomialRegressor, 'predict', forbidden)
    monkeypatch.setattr(NegativeBinomialRegressor, 'predict_mean', forbidden)
    cached = run_recipe(recipe, prepared=prepared)
    assert cached.reused and cached.record['run_id'] == result.record['run_id']
    pd.testing.assert_frame_equal(cached.post_report.studies[0].result.tables['ledger'], decisions.tables['ledger'], check_dtype=False)


def test_ui_classifier_retains_probabilities_before_reporter_is_added(ui_recipe, monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from xdiyo_analytics.ui.inventory import inventory
    recipe = deepcopy(ui_recipe)
    recipe['model'] = node('sklearn.linear_model.LogisticRegression', max_iter=200)
    recipe['post_reporters'] = {}
    assert recipe['prediction_methods'] == ['predict']
    prepared = prepare_recipe(recipe)
    original = run_recipe(recipe, prepared=prepared)
    assert 'predict_proba' in original.training.folds[0].predictions
    source = deepcopy(recipe['labels']['corners'])
    recipe['post_reporters'] = {'bets': node('reporting.BetOutcomeReporter', type='overall', partition='score',
        offers={'under': node('evaluation.BetOffer', option=node('labels.BetOption', source=source, selection='under', line=7.5), odds=2.)},
        policy=node('evaluation.HighestExpectedProfit', min_ev=-1.))}
    def forbidden(*args, **kwargs):
        raise AssertionError('Adding a report must not fit or predict again')
    monkeypatch.setattr(LogisticRegression, 'fit', forbidden)
    monkeypatch.setattr(LogisticRegression, 'predict', forbidden)
    monkeypatch.setattr(LogisticRegression, 'predict_proba', forbidden)
    cached = run_recipe(recipe, prepared=prepare_recipe(recipe))
    assert cached.reused and cached.record['run_id'] == original.record['run_id']
    probabilities = original.training.folds[0].predictions['predict_proba']['corners']
    expected = probabilities.loc[:, probabilities.columns <= 7].sum(axis=1)
    np.testing.assert_allclose(cached.post_report.studies[0].result.tables['bets'].probability, expected)
    components = inventory()['components']
    policy = next(f for f in components['reporting.BetOutcomeReporter']['fields'] if f['name'] == 'policy')
    for name in ('evaluation.TightestLine', 'evaluation.HighestExpectedProfit'):
        assert name in components
        assert components[name]['category'] in policy['categories']


@pytest.mark.parametrize('quote', [0., 1., np.inf, -2.])
def test_invalid_quotes_are_rejected_before_decisions(quote):
    with pytest.raises(ValueError, match='Decimal odds'):
        prepare_bets(decision_context(), {'under': offer('under', 7.5, quote)}, HighestExpectedProfit())


@pytest.mark.parametrize('partition', ['test', 'score'])
def test_nb_mean_pooling_averages_fold_probabilities_not_parameters(partition):
    ctx, specs, labels, history = reporting_context()
    ctx.predictions = {'count_distribution': pd.DataFrame(
        np.tile([11., .4], (len(ctx.y), 1)), index=ctx.y.index,
        columns=pd.MultiIndex.from_product([['count'], ['mean', 'dispersion']]))}
    training = as_training(ctx, repeat=True)
    for fold, mean in zip(training.folds, [2., 20.]):
        fold.predictions['count_distribution'][('count', 'mean')] = mean
    if partition == 'score':
        training.folds[1].score_positions = training.folds[1].score_positions[1:]
    reporter = BetOutcomeReporter(type='overall', partition=partition, pooling='mean', history=history,
        offers={'under': offer('under', 7.5, 2.)}, policy=TightestLine(min_probability=0.))
    result = PostTrainingAnalysis({'decisions': reporter}).run(training).studies[0].result
    expected = (nbinom.cdf(7, 2.5, 1/1.8) + nbinom.cdf(7, 2.5, 1/9.))/2
    assert expected == pytest.approx(.5715701065176235)
    expected_rows = np.full(len(ctx.y), expected)
    if partition == 'score':
        expected_rows[0] = nbinom.cdf(7, 2.5, 1/1.8)
    np.testing.assert_allclose(result.tables['bets'].probability, expected_rows)
    assert not np.isclose(expected, nbinom.cdf(7, 2.5, 1/5.4))


def test_nb_mean_pooling_without_original_distributions_is_explicit_error():
    ctx = decision_context()
    ctx.y.index = pd.MultiIndex.from_tuples([(-1, 0)], names=['fold_id', 'row_position'])
    ctx.pooling = 'mean'
    ctx.predictions = {'count_distribution': pd.DataFrame([[11., .4]], index=ctx.y.index,
        columns=pd.MultiIndex.from_product([['count'], ['mean', 'dispersion']]))}
    with pytest.raises(ValueError, match='(?i)(original|origin|retained|fold|distribution)'):
        prepare_bets(ctx, {'under': offer('under', 7.5, 2.)}, TightestLine(min_probability=0.), output='count_distribution')


