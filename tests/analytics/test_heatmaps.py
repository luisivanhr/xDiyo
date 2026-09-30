"""Independent spatial arithmetic, past-only alignment, and report/UI integration."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

from test_ui_workflow import ui_recipe
from xdiyo_analytics.features import Heatmap, Lag, RollingMean, EMA, ForAgainst, evaluate_features
from xdiyo_analytics.features.spatial import heatmap_grid, heatmap_values, heatmap_columns
from xdiyo_analytics.reporting import HeatmapReporter, TeamCatalog
from xdiyo_analytics.ui import prepare_recipe, run_recipe, catalog_for_ui
from xdiyo_analytics.ui.recipe import node


def points(xy, weights=None):
    frame = pd.DataFrame(xy, columns=['x', 'y'])
    frame['kind'] = 'touch'
    if weights is not None:
        frame['weight'] = weights
    return frame


@pytest.mark.parametrize('normalization,scale', [('count', 1.), ('mass', 1/5), ('density', 1/12500)])
def test_grid_boundary_bins_orientation_and_normalization(normalization, scale):
    # 50 enters the upper bin; 100 remains in the final bin. Grid order is [y,x].
    frame = points([(0, 0), (49.999, 0), (50, 0), (0, 50), (100, 100)])
    expected = np.array([[2., 1.], [1., 1.]]) * scale
    actual = heatmap_grid(frame, Heatmap(grid_size=2, normalization=normalization))
    np.testing.assert_allclose(actual, expected)
    if normalization == 'density':
        assert actual.sum() * 2500 == pytest.approx(1.)


def test_weights_and_kind_selection_use_only_selected_point_mass():
    frame = points([(0, 0), (100, 0), (0, 100), (100, 100)], [2., None, 0., 100.])
    frame.loc[3, 'kind'] = 'ignore'
    weighted = heatmap_grid(frame, Heatmap(grid_size=2, kinds='touch', use_weights=True))
    np.testing.assert_allclose(weighted, [[2/3, 1/3], [0., 0.]])
    unweighted = heatmap_grid(frame, Heatmap(grid_size=2, kinds=('touch',)))
    np.testing.assert_allclose(unweighted, [[1/3, 1/3], [1/3, 0.]])


@pytest.mark.parametrize('normalization', ['count', 'mass', 'density'])
def test_gaussian_matches_independent_fine_grid_scipy_reference(normalization):
    frame = points([(0, 0), (100, 100), (25, 60)], [2., 3., 4.])
    fine = np.zeros((12, 12))
    fine[0, 0], fine[11, 11], fine[7, 3] = 2., 3., 4.
    smooth = gaussian_filter(fine, sigma=1.3, mode='reflect')
    expected = np.array([[smooth[y:y+4, x:x+4].sum() for x in (0, 4, 8)] for y in (0, 4, 8)])
    if normalization != 'count':
        expected /= 9.
    if normalization == 'density':
        expected /= (100/3)**2
    actual = heatmap_grid(frame, Heatmap(grid_size=3, resolution=12, sigma=1.3,
        method='gaussian', normalization=normalization, use_weights=True))
    np.testing.assert_allclose(actual, expected, rtol=1e-13)
    assert actual.sum() == pytest.approx(9. if normalization == 'count' else (1. if normalization == 'mass' else 9/10000))


@pytest.mark.parametrize('case', ['empty', 'zero_weight', 'no_matching_kind'])
def test_unavailable_map_stays_missing_instead_of_zero(case):
    frame = points([] if case == 'empty' else [(5, 5)], [] if case == 'empty' else [0.])
    spec = Heatmap(grid_size=2, use_weights=case == 'zero_weight', kinds=('absent',) if case == 'no_matching_kind' else None)
    assert np.isnan(heatmap_grid(frame, spec)).all()
    # A real observed map can have zero cells, which remain usable observations.
    np.testing.assert_array_equal(heatmap_grid(points([(5, 5)]), Heatmap(grid_size=2)), [[1, 0], [0, 0]])


@pytest.mark.parametrize('coordinate', [-.01, 100.01, np.nan, np.inf])
def test_invalid_coordinates_do_not_silently_clip(coordinate):
    with pytest.raises(ValueError, match='coordinates'):
        heatmap_grid(points([(coordinate, 50)]))


@pytest.mark.parametrize('weight', [-1, np.inf])
def test_invalid_weights_raise(weight):
    with pytest.raises(ValueError, match='weights'):
        heatmap_grid(points([(1, 1)], [weight]), Heatmap(use_weights=True))


@pytest.mark.parametrize('options', [dict(grid_size=1), dict(grid_size=True), dict(method='invalid'),
    dict(normalization='invalid'), dict(side='invalid'), dict(method='gaussian', resolution=13),
    dict(method='gaussian', sigma=0)])
def test_invalid_spatial_configuration(options):
    with pytest.raises(ValueError):
        Heatmap(**options)


@pytest.fixture
def spatial_history():
    a, b, c, d = [2**63 + i for i in (101, 102, 103, 104)]
    rows, observations = [], []
    own = [(0, 0), (100, 0), None, (100, 100), (100, 100)]
    opponents = [(0, 100), (100, 100), (100, 0), (0, 0), (0, 0)]
    for i, opponent in enumerate((b, c, d, b, c)):
        event = 2**63 + 501 + i
        for team, other, side, xy in [(a, opponent, 'home' if i % 2 == 0 else 'away', own[i]),
                                     (opponent, a, 'away' if i % 2 == 0 else 'home', opponents[i])]:
            common = dict(event_id=event, source_league='Alpha', source_season='24_25')
            rows.append(dict(**common, team_id=team, opponent_id=other, side=side,
                competition_id=17, season_id=202425, kickoff_at=pd.Timestamp('2024-09-01', tz='UTC') + pd.Timedelta(days=i),
                status='notstarted' if i == 4 else 'finished'))
            if xy is not None:
                observations.append(dict(**common, team_id=team, x=xy[0], y=xy[1], kind='touch'))
    history, maps = pd.DataFrame(rows), pd.DataFrame(observations)
    for frame in (history, maps):
        for key in ('event_id', 'team_id', 'opponent_id'):
            if key in frame:
                frame[key] = pd.array(frame[key], dtype='uint64[pyarrow]')
    history.index = pd.Index([91, 13, 85, 7, 60, 4, 12, 25, 89, 33], name='stored_row')
    return history, maps


def test_alignment_preserves_large_identifiers_row_order_and_source_scope(spatial_history):
    history, maps = spatial_history
    # Same exact event/team IDs in another source must not contaminate Alpha.
    duplicate = maps.copy()
    duplicate['source_league'], duplicate['x'], duplicate['y'] = 'Other', 50., 50.
    maps = pd.concat([duplicate, maps.sample(frac=1, random_state=42)], ignore_index=True)
    shuffled = history.sample(frac=1, random_state=11)
    actual = heatmap_values(shuffled, maps, Heatmap(grid_size=2, side='both'))
    assert actual.index.equals(shuffled.index)
    # Focal team's first map differs from its opponent by a full y-bin.
    first = actual.loc[91].to_numpy()
    np.testing.assert_array_equal(first, [1, 0, 0, 0, 0, 1, 0, 0])
    assert actual.loc[60].iloc[:4].isna().all()
    np.testing.assert_array_equal(actual.loc[60].iloc[4:].to_numpy(), [0, 0, 1, 0])


def test_temporal_heatmaps_keep_missing_matches_and_exclude_current_future(spatial_history):
    history, maps = spatial_history
    spec = Heatmap(grid_size=2, orientation='team')
    features = {'lag': Lag(spec), 'mean': RollingMean(spec, 2), 'ema': EMA(spec, span=3),
                'opponents': RollingMean(ForAgainst(spec, side='against'), 2)}
    actual = evaluate_features(history, features, heatmaps=maps)
    assert actual.loc[91].isna().all()
    assert actual.loc[12].filter(like='lag::').isna().all()  # previous match has no own map
    np.testing.assert_allclose(actual.loc[12].filter(like='mean::'), [0, 1, 0, 0])
    np.testing.assert_allclose(actual.loc[12].filter(like='ema::'), [.5, .5, 0, 0])
    np.testing.assert_allclose(actual.loc[12].filter(like='opponents::'), [.5, 0, .5, 0])
    # Editing current/future observations cannot alter a prediction at row12.
    contaminated = maps.copy()
    mask = contaminated.event_id.astype(object).ge(2**63 + 504)
    contaminated.loc[mask, ['x', 'y']] = 20.
    updated = evaluate_features(history, features, heatmaps=contaminated)
    pd.testing.assert_series_equal(actual.loc[12], updated.loc[12])
    # Later upcoming fixture uses only the four finished matches, not its own map.
    np.testing.assert_allclose(actual.loc[89].filter(like='mean::'), [0, 0, 0, 1])


def test_explicit_cutoff_availability_and_equal_kickoffs_apply_to_heatmaps(spatial_history):
    history, maps = spatial_history
    features = {'map': RollingMean(Heatmap(grid_size=2, orientation='team'), 2)}
    cutoff = history.kickoff_at - pd.Timedelta(days=1)
    actual = evaluate_features(history, features, heatmaps=maps, cutoffs=cutoff)
    np.testing.assert_allclose(actual.loc[12], [.5, .5, 0, 0])
    available = history.kickoff_at.copy()
    available.loc[85] = pd.Timestamp('2024-10-01', tz='UTC')
    actual = evaluate_features(history, features, heatmaps=maps, available_at=available)
    np.testing.assert_allclose(actual.loc[12], [1, 0, 0, 0])
    tied = history.copy()
    tied.loc[[85, 7], 'kickoff_at'] = tied.loc[12, 'kickoff_at']
    actual = evaluate_features(tied, features, heatmaps=maps)
    np.testing.assert_allclose(actual.loc[12], [1, 0, 0, 0])


def test_raw_heatmap_is_rejected_as_prediction_feature_and_missing_input_is_explicit(spatial_history):
    history, maps = spatial_history
    with pytest.raises(ValueError, match='rolling operator'):
        evaluate_features(history, {'raw': Heatmap(grid_size=2)}, heatmaps=maps)
    with pytest.raises(ValueError, match='Load heatmap_points'):
        evaluate_features(history, {'map': Lag(Heatmap(grid_size=2))})


def test_reporter_uses_historical_opponents_names_venue_badges_and_complete_rows(spatial_history, tmp_path):
    history, maps = spatial_history
    values = evaluate_features(history, {'opponents': RollingMean(Heatmap(grid_size=2, side='against'), 2)}, heatmaps=maps)
    # Row12 is A away to B, but its eligible historical opponents are C and D.
    selected = [91, 12]
    badge = tmp_path/'badge.svg'
    badge.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>')
    catalog = TeamCatalog({2**63+101: {'name': 'Focal A', 'badge_path': str(badge)}, 2**63+102: 'Today B'})
    context = SimpleNamespace(X=values.loc[selected], metadata=history.loc[selected], layout='team_match',
        match_columns=('source_league', 'source_season', 'event_id'),
        definitions={'spatial_features':values.attrs['spatial_features']})
    result = HeatmapReporter(type='overall', partition='train', catalog=catalog).run(context)
    store = result.artifacts[0].data
    assert store.teams[str(2**63+101)]['name'] == 'Focal A'
    assert store.teams[str(2**63+101)]['badge'].startswith('data:image/svg+xml;base64,')
    assert store.panel_pair('0', 'opponents')['panels'][0]['missing'] is True
    panel = store.panel_pair('1', 'opponents')['panels'][1]
    assert panel['side'] == 'away' and panel['team_id'] == str(2**63+101)
    assert panel['spec']['side'] == 'against'
    np.testing.assert_allclose(panel['values'], [0, .5, 0, .5])
    without_badges = HeatmapReporter(type='overall', partition='train', catalog=catalog, show_badges=False).run(context)
    assert without_badges.artifacts[0].data.teams[str(2**63+101)]['badge'] is None


def test_reporter_match_prefix_filtering_common_scale_and_incomplete_grid(spatial_history):
    history, maps = spatial_history
    values = evaluate_features(history, {'map': RollingMean(Heatmap(grid_size=2), 2)}, heatmaps=maps)
    columns = {'home::'+name: values.loc[[85, 12], name].to_numpy() for name in values}
    metadata = pd.DataFrame({'home_id': pd.array([2**63+101, 2**63+102], dtype='uint64'),
                             'away_id': pd.array([2**63+102, 2**63+101], dtype='uint64')})
    metadata['event_id'] = pd.array([2**63+1, 2**63+2], dtype='uint64')
    specs = deepcopy(values.attrs['spatial_features'])
    for spec in specs.values():
        spec['columns'] = ['home::'+column for column in spec['columns']]
        spec['fixture_side'] = 'home'
    context = SimpleNamespace(X=pd.DataFrame(columns), metadata=metadata, layout='match',
        match_columns=('event_id',), definitions={'spatial_features':specs})
    result = HeatmapReporter(type='per_fold', partition='train').run(context)
    store = result.artifacts[0].data
    assert store.panel_pair('0', 'map')['scale'] == [0., 1.]
    assert store.panel_pair('1', 'map')['scale'] == [0., 1.]
    assert store.panel_pair('0', 'map')['panels'][1]['missing'] is True
    filtered = HeatmapReporter(type='overall', partition='train', teams=[2**63+999]).run(context)
    assert filtered.artifacts[0].data.manifest()['total_fixtures'] == 0
    context.X = context.X.iloc[:, :-1]
    with pytest.raises(ValueError, match='incomplete'):
        HeatmapReporter(type='overall', partition='train').run(context)


def test_reporter_keeps_each_historical_grid_without_averaging_or_zero_fill(spatial_history):
    history, maps = spatial_history
    values = evaluate_features(history, {'map': Lag(Heatmap(grid_size=2))}, heatmaps=maps)
    # A's rows: first unavailable, then [1,0,0,0], then [0,1,0,0].
    selected = [91, 85, 60]
    metadata = history.loc[selected].copy()
    metadata['side'] = 'home'
    context = SimpleNamespace(X=values.loc[selected], metadata=metadata, layout='team_match',
        match_columns=('event_id',), definitions={'spatial_features':values.attrs['spatial_features']})
    result = HeatmapReporter(type='overall', partition='train').run(context)
    store = result.artifacts[0].data
    assert store.manifest()['total_fixtures'] == 3
    assert store.panel_pair('0', 'map')['panels'][0]['missing'] is True
    for fixture, row in [('1',85), ('2',60)]:
        np.testing.assert_array_equal(store.panel_pair(fixture, 'map')['panels'][0]['values'], values.loc[row])


def test_ui_recipe_autoloads_points_prepares_grids_and_renders_reports(ui_recipe, monkeypatch):
    from xdiyo_analytics.data import load_seasons
    original = load_seasons(**ui_recipe['data'])
    maps = original.statistics[['event_id', 'team_id', 'source_league', 'source_season']].drop_duplicates().copy()
    maps['x'], maps['y'], maps['kind'] = np.arange(len(maps)) % 100, 25., 'touch'
    original.tables['heatmap_points'] = maps
    calls = []
    def load(**options):
        calls.append(options)
        return deepcopy(original)
    monkeypatch.setattr('xdiyo_analytics.data.load_seasons', load)
    recipe = deepcopy(ui_recipe)
    recipe['features'] = {'spatial': node('features.RollingMean', source=node('features.Heatmap',
        grid_size=2, side='both'), window=3)}
    recipe['pre_reporters'] = {'maps': node('reporting.HeatmapReporter', type='overall', partition='train')}
    prepared = prepare_recipe(recipe)
    assert 'heatmap_points' in calls[0]['tables']
    assert len(prepared.dataset.X.columns) == 16  # two fixture sides x for/against x four cells
    assert len(heatmap_columns(prepared.dataset.X.columns)) == 4
    result = run_recipe(recipe, prepared=prepared)
    study = next(study for study in result.report.studies if study.name.endswith('maps'))
    assert len(study.result.artifacts) == 1
    store = study.result.artifacts[0].data
    assert len(store.maps) == 2  # for/against families, with paired home/away panels
    assert {team['name'] for team in store.teams.values()} == {'Alpha Home', 'Alpha Away'}
    assert {panel['side'] for panel in store.panel_pair('0', next(iter(store.maps)))['panels']} == {'home', 'away'}
    built = catalog_for_ui().build(recipe['features']['spatial'])
    assert isinstance(built.source, Heatmap)
    schema = next(entry for entry in catalog_for_ui().schema() if entry['id'] == 'reporting.HeatmapReporter')
    assert schema['category'] == 'pre_reporter'
    from xdiyo_analytics.experiments import FootballExperiment
    from xdiyo_analytics.reporting.heatmaps import SpatialFixtureData
    from xdiyo_analytics.reporting.spatial_live import SpatialStore
    from xdiyo_analytics.training import EstimatorAdapter
    experiment = FootballExperiment(recipe['name'], output_dir=recipe['output_dir'])
    restored = experiment.load(result.record['run_id'], load_models=False)
    restored_store = next(s for s in restored.report.studies if s.name.endswith('maps')).result.artifacts[0].data
    assert isinstance(restored_store, SpatialFixtureData)
    pd.testing.assert_frame_equal(restored_store.values, store.values)
    assert restored_store.manifest() == store.manifest()
    transport = SpatialStore()
    request = {'id': transport.register(restored_store), 'fixture': '1', 'map': next(iter(store.maps))}
    assert transport.fetch(request) == store.panel_pair(request['fixture'], request['map'])
    def forbidden(*args, **kwargs):
        raise AssertionError('Reopening a spatial report must not refit')
    monkeypatch.setattr(EstimatorAdapter, 'fit', forbidden)
    reused = run_recipe(recipe, prepared=prepared)
    assert reused.reused and reused.record['run_id'] == result.record['run_id']
