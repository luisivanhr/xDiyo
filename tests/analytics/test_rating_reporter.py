"""Held-out rating diagnostics, retained producers, recovery and UI contract."""
from copy import deepcopy
from dataclasses import replace
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.features import BayesianFixture, BayesianRating, MatchResultGlicko, evaluate_features
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.ratings import build_ratings, build_bayesian_ratings
from xdiyo_analytics.reporting import RatingReporter, TeamCatalog
from xdiyo_analytics.training import TrainingResult, FoldResult
from test_match_score import score_data, A, B
from test_ui_workflow import ui_recipe


def sample():
    data = score_data()
    data.matches['season_id'] = 1
    history = build_team_history(data)
    history['team_name'] = history.team_id.map({A: 'Chelsea', B: 'Arsenal'})
    history['source_league'] = 'Premier League'
    cutoffs = history.kickoff_at - pd.Timedelta(hours=6)
    retained = {}
    evaluate_features(history, {'glicko': MatchResultGlicko(), 'bayes': BayesianRating(),
                               'fixture': BayesianFixture()}, cutoffs=cutoffs, rating_runs=retained)
    metadata = data.matches.copy()
    metadata['kickoff_at'] = pd.to_datetime(metadata.kickoff_utc, unit='s', utc=True)
    metadata['home_name'], metadata['away_name'] = 'Chelsea', 'Arsenal'
    metadata['source_league'] = 'Premier League'
    y = pd.DataFrame({'goals': np.arange(len(metadata), dtype=float)})
    folds = []
    for fid, test, score in [(2, [3, 4], [4]), (8, [4, 5], [5])]:
        frame = y.iloc[test].copy()
        folds.append(FoldResult(fid, object(), np.array([0, 1]), np.array(test), np.array(score),
                               ('x',), ('goals',), {'predict': frame.copy()}, frame,
                               metadata.iloc[test].copy()))
    keys = ('competition_id', 'season_id', 'event_id')
    training = TrainingResult(folds, 'match', keys, keys, 'total')
    resources = {'history': history, 'rating_cutoffs': cutoffs, 'ratings': retained,
                 'team_catalog': TeamCatalog({A: 'Chelsea', B: 'Arsenal'})}
    return training, resources


def report(training, resources, **kwargs):
    return PostTrainingAnalysis({'ratings': RatingReporter(type='overall', **kwargs)}).run(
        training, resources=resources)


def test_inline_producers_retained_once_and_do_not_change_features():
    history = build_team_history(score_data())
    definitions = {'team': BayesianRating(), 'fixture': BayesianFixture(), 'glicko': MatchResultGlicko()}
    sink = {}
    before = evaluate_features(history, definitions)
    after = evaluate_features(history, definitions, rating_runs=sink)
    pd.testing.assert_frame_equal(before, after)
    assert set(sink) == {'feature:team', 'feature:glicko'}


def test_test_scope_cutoffs_intervals_and_future_invariance():
    training, resources = sample()
    result = report(training, resources)
    table = result.studies[0].result.tables['rating_states']
    assert set(table.event) == {str(2**63+101+i) for i in (3, 4, 5)}
    assert set(table.fold) == {'2', '8'}
    assert set(table.panel) == {'Rating', 'Attack', 'Defense · vulnerability', 'Home advantage'}
    assert (pd.to_datetime(table.kickoff)-pd.to_datetime(table.cutoff)).eq(pd.Timedelta(hours=6)).all()
    assert (table.lower <= table['mean']).all() and (table['mean'] <= table.upper).all()
    rating = table.loc[table.panel.eq('Rating')].iloc[0]
    h = resources['history']
    q = h.loc[h.event_id.astype(str).eq(rating.event) & h.team_id.astype(str).eq(rating.team)]
    value = resources['ratings']['feature:glicko'].features(q, side='for', cutoffs=resources['rating_cutoffs'].loc[q.index])
    rd = value.iloc[0]['result::team::rd']
    assert rating.upper-rating['mean'] == pytest.approx(NormalDist().inv_cdf(.975)*rd)
    future = deepcopy(resources)
    for run in future['ratings'].values():
        extra = run.snapshots.iloc[-1:].copy()
        extra['recorded_at'] = pd.Timestamp('2030-01-01', tz='UTC')
        extra['latest_kickoff_at'] = pd.Timestamp('2029-12-31', tz='UTC')
        extra['rating'] = 999999
        run.snapshots = pd.concat([run.snapshots, extra], ignore_index=True)
    pd.testing.assert_frame_equal(table, report(training, future).studies[0].result.tables['rating_states'])


def test_score_and_fold_selection_and_gamma_intervals():
    from scipy.stats import gamma
    training, resources = sample()
    result = PostTrainingAnalysis({'r': RatingReporter(type='per_fold', partition='score')}, fold_ids=[8]).run(
        training, resources=resources)
    table = result.studies[0].result.tables['rating_states']
    assert set(table.event) == {str(2**63+106)}
    from xdiyo_analytics.reporting.ratings import _bounds
    assert _bounds(2., 1., .95, gamma=True) == pytest.approx(gamma.ppf([.025, .975], a=4, scale=.5))
    assert _bounds(2., None, .95) == (None, None)


def test_bayesian_bands_default_to_tenth_ninetieth_percentiles_only():
    from scipy.stats import gamma
    training, resources = sample()
    result = report(training, resources).studies[0].result
    table = result.tables['rating_states']
    for panel, mean_field, sd_field in (
        ('Attack', 'attack_mean', 'attack_sd'),
        ('Defense · vulnerability', 'defence_vulnerability_mean', 'defence_vulnerability_sd'),
        ('Home advantage', 'home_advantage_mean', 'home_advantage_sd'),
    ):
        row = table.loc[table.panel.eq(panel)].iloc[0]
        run = resources['ratings'][row.source]
        h = resources['history']
        q = h.loc[h.event_id.astype(str).eq(row.event)]
        q = q.loc[q.side.eq('home')] if row.team is None else q.loc[q.team_id.astype(str).eq(row.team)]
        cutoffs = resources['rating_cutoffs'].loc[q.index]
        if row.team is None:
            values = run.fixture_features(q, cutoffs=cutoffs, fields=(mean_field, sd_field)).iloc[0]
            mean, sd = values[mean_field], values[sd_field]
        else:
            values = run.features(q, cutoffs=cutoffs, side='for').iloc[0]
            mean, sd = values[f'score::team::{mean_field}'], values[f'score::team::{sd_field}']
        assert [row.lower, row.upper] == pytest.approx(gamma.ppf([.1, .9], a=(mean/sd)**2, scale=sd**2/mean))
    wider = report(training, resources, bayesian_interval=.95).studies[0].result.tables['rating_states']
    pd.testing.assert_frame_equal(table.loc[table.panel.eq('Rating')], wider.loc[wider.panel.eq('Rating')])
    assert (wider.loc[wider.panel.ne('Rating'), 'upper'] > table.loc[table.panel.ne('Rating'), 'upper']).all()
    assert result.artifacts[0].data['bayesian_interval'] == .8
    with pytest.raises(ValueError, match='bayesian_interval'):
        report(training, resources, bayesian_interval=1.)


def test_recovery_payload_and_badges_and_safe_html(tmp_path):
    from xdiyo_analytics.experiments.recovery import dump_bundle, load_bundle
    training, resources = sample()
    badge = tmp_path/'badge.svg'
    badge.write_text('<svg xmlns="http://www.w3.org/2000/svg"><circle cx="10" cy="10" r="9"/></svg>')
    resources['team_catalog'] = TeamCatalog({A: {'name': 'Chelsea </script>', 'badge_path': str(badge)}, B: 'Arsenal'})
    path = tmp_path/'bundle.json'
    dump_bundle(path, resources)
    restored = load_bundle(path)
    result = report(training, restored)
    html = result.to_html()
    assert 'initializeRatings' in html and 'data:image/svg+xml;base64,' in html
    assert 'Chelsea </script>' not in html
    assert 'Chelsea \\u003c/script>' in html
    assert 'All teams' in html and 'Home advantage' in html


def test_ui_inventory_and_recipe_roundtrip():
    from xdiyo_analytics.ui.recipe import catalog_for_ui, node, validate_recipe, default_recipe
    catalog = catalog_for_ui()
    entry = next(e for e in catalog.schema() if e['id']=='reporting.RatingReporter')
    assert entry['category'] == 'post_reporter'
    fields = {f['name']: f for f in entry['fields']}
    assert fields['interval']['kind'] == 'number'
    assert fields['catalog']['hidden'] and fields['ratings']['hidden']
    recipe = default_recipe()
    recipe['post_reporters'] = {'Ratings': node('reporting.RatingReporter', type='overall')}
    validate_recipe(recipe)
    reporter = catalog.build(recipe['post_reporters']['Ratings'], {'team_catalog': TeamCatalog({})})
    assert isinstance(reporter, RatingReporter) and reporter.partition == 'test'


def test_missing_resources_and_invalid_config():
    training, resources = sample()
    assert 'No retained ratings' in report(training, {}).studies[0].result.notes[-1]
    with pytest.raises(ValueError, match='interval'):
        report(training, resources, interval=1)
    with pytest.raises(ValueError, match='distinct retained'):
        report(training, resources, ratings=['nope'])
    with pytest.raises(ValueError, match='cannot average'):
        report(training, resources, pooling='mean')


def test_custom_numeric_rating_and_fitted_model_discovery():
    from types import SimpleNamespace
    training, resources = sample()
    for fold in training.folds:
        fold.model = SimpleNamespace(rating_report_run_=resources['ratings']['feature:bayes'])
    result = report(training, {}, ratings=['model:fold:2'])
    table = result.studies[0].result.tables['rating_states']
    assert set(table.fold) == {'2'}
    assert pd.to_datetime(table.kickoff).equals(pd.to_datetime(table.cutoff))
    run = resources['ratings']['feature:glicko']
    custom = replace(run, initial_state={'points': 0.}, snapshots=run.snapshots.assign(points=12.))
    result = report(training, {**resources, 'ratings': {'points': custom}}, ratings=['points'])
    table = result.studies[0].result.tables['rating_states']
    assert set(table.panel)=={'points'} and table.lower.isna().all()


def test_recipe_preparation_export_and_report_only_reuse(ui_recipe, tmp_path, monkeypatch):
    from xdiyo_analytics.ui import prepare_recipe, run_recipe
    from xdiyo_analytics.ui.recipe import node, export_python
    from sklearn.linear_model import Ridge
    ui_recipe['features']['glicko'] = node('features.MatchResultGlicko')
    # Named saved runs are retained even when the predictor only reads means.
    history = prepare_recipe(ui_recipe).outputs['history']
    saved = build_bayesian_ratings(history)
    saved.save(tmp_path/'bayesian')
    ui_recipe['feature_options'] = {'ratings': {'saved': node('input.BayesianRatingRun', path=str(tmp_path/'bayesian'))}}
    ui_recipe['features']['attack'] = node('features.Rating', name='saved', side='for', fields=['attack_mean'])
    ui_recipe['post_reporters'] = {'Ratings': node('reporting.RatingReporter', type='overall')}
    prepared = prepare_recipe(ui_recipe)
    assert set(prepared.outputs['ratings']) == {'feature:glicko', 'saved'}
    source = export_python(ui_recipe)
    compile(source, '<exported recipe>', 'exec')
    first = run_recipe(ui_recipe, prepared=prepared)
    assert first.post_report.studies[0].result.tables['rating_states'].panel.eq('Attack').any()
    monkeypatch.setattr(Ridge, 'fit', lambda *args, **kwargs: pytest.fail('Reporting change must not fit again'))
    ui_recipe['post_reporters']['Ratings']['params']['interval'] = .8
    second = run_recipe(ui_recipe)
    assert second.reused
    assert second.training.folds[0].model is not None
    assert second.post_report.studies[0].result.artifacts[0].data['interval'] == .8


def test_rendered_source_team_band_controls_and_mobile_layout(tmp_path):
    api = pytest.importorskip('playwright.sync_api')
    training, resources = sample()
    path = tmp_path/'ratings.html'
    report(training, resources).to_html(path)
    errors = []
    with api.sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(path.as_uri())
        page.locator('.rating-chart .main-svg').first.wait_for()
        page.get_by_label('Rating source', exact=True).select_option(label='feature:glicko · result')
        assert page.locator('.rating-chart').count() == 1
        page.get_by_label('Rating source', exact=True).select_option(label='feature:bayes · score')
        assert page.locator('.rating-chart').count() == 3
        page.get_by_label('Teams', exact=True).select_option(label='Chelsea')
        names = page.locator('.rating-chart').first.evaluate('(p)=>p.data.filter(t=>t.name).map(t=>t.name)')
        assert names and all('Chelsea' in name for name in names)
        page.get_by_label('Teams', exact=True).select_option(label='All teams')
        page.get_by_role('button', name='Chelsea', exact=True).click()
        assert page.get_by_role('button', name='Chelsea', exact=True).get_attribute('aria-pressed') == 'false'
        names = page.locator('.rating-chart').first.evaluate('(p)=>p.data.filter(t=>t.name).map(t=>t.name)')
        assert names and all('Chelsea' not in name for name in names)
        page.get_by_role('button', name='Show all lines').click()
        opacity = page.locator('.rating-chart .js-fill').evaluate_all('(nodes)=>nodes.map(p=>getComputedStyle(p).fillOpacity)')
        assert opacity and all(float(v) == pytest.approx(.13) for v in opacity)
        page.get_by_label('Uncertainty bands', exact=True).uncheck()
        assert page.locator('.rating-chart').first.evaluate('(p)=>p.data.every(t=>t.name)')
        page.set_viewport_size({'width': 390, 'height': 844})
        page.wait_for_function('document.documentElement.scrollWidth <= innerWidth')
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        assert not errors
        browser.close()
