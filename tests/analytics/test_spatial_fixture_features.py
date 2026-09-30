"""Fixture-frame spatial reductions and optional same-venue historical windows."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from test_ui_workflow import ui_recipe
from xdiyo_analytics.features import (
    Heatmap, RegionMass, Lag, RollingMean, RollingStd, RollingZScore, EMA,
    ForAgainst, H2H, evaluate_features,
)
from xdiyo_analytics.features.spatial import heatmap_values, region_values


@pytest.fixture
def venue_history():
    values = [1., 10., 3., 20., 5., 30., 7., 40.]
    a, b, c, d = [2**63+i for i in (11, 12, 13, 14)]
    opponents = [b, c, b, d, b, c, b, d]
    history = pd.DataFrame({
        'event_id': pd.array([2**63+201+i for i in range(8)], dtype='uint64[pyarrow]'),
        'team_id': pd.array([a]*8, dtype='uint64[pyarrow]'),
        'opponent_id': pd.array(opponents, dtype='uint64[pyarrow]'),
        'competition_id': 17, 'season_id': 202425,
        'source_league': 'Alpha', 'source_season': '24_25',
        'side': ['home', 'away']*4,
        'kickoff_at': pd.date_range('2024-09-01', periods=8, tz='UTC'),
        'status': ['finished']*7+['notstarted']})
    records = []
    for row, value in zip(history.itertuples(), values):
        for team, x, y, weight in [(a, 10, 10, value), (row.opponent_id, 80, 20, 100+value)]:
            records.append(dict(event_id=row.event_id, team_id=team, source_league='Alpha', source_season='24_25',
                                x=x, y=y, weight=weight, kind='touch'))
    points = pd.DataFrame(records)
    for key in ('event_id', 'team_id'):
        points[key] = pd.array(points[key], dtype='uint64[pyarrow]')
    return history, points


def vector(frame, row, name):
    return frame.loc[row, [c for c in frame if c.startswith(name+'::')]].to_numpy(dtype=float)


def test_rotation_is_180_degrees_on_both_axes_after_historical_reduction(venue_history):
    history, points = venue_history
    event = int(history.event_id.iloc[0])
    a = int(history.team_id.iloc[0])
    first = points.loc[points.event_id.astype(object).eq(event) & points.team_id.astype(object).eq(a)].iloc[0]
    extra = pd.DataFrame([{**first.to_dict(), 'x': x, 'y': y, 'weight': w}
                          for x, y, w in [(10, 10, 1), (80, 10, 2), (10, 80, 3), (80, 80, 4)]])
    retained = points.loc[~(points.event_id.astype(object).eq(event) & points.team_id.astype(object).eq(a))]
    points = pd.concat([retained, extra], ignore_index=True)
    spec = dict(grid_size=2, normalization='count', use_weights=True)
    output = evaluate_features(history, {'home': Lag(Heatmap(**spec)),
        'team': Lag(Heatmap(**spec, orientation='team'))}, heatmaps=points)
    # Away target: y and x reverse together, not a horizontal mirror.
    np.testing.assert_array_equal(vector(output, 1, 'team'), [1, 2, 3, 4])
    np.testing.assert_array_equal(vector(output, 1, 'home'), [4, 3, 2, 1])
    # Home target uses its previous away observation in the original team frame.
    np.testing.assert_array_equal(vector(output, 2, 'home'), [10, 0, 0, 0])


def test_mixed_venue_mean_is_reduced_in_team_frame_then_rotated_once(venue_history):
    history, points = venue_history
    source = Heatmap(grid_size=2, normalization='count', use_weights=True)
    output = evaluate_features(history, {'mean': RollingMean(source, 2),
        'nested': Lag(RollingMean(source, 2))}, heatmaps=points)
    # First own observations are home=1 and away=10 at the same team-coordinate cell.
    np.testing.assert_allclose(vector(output, 2, 'mean'), [5.5, 0, 0, 0])
    # Current away row uses previous 10 and3, then rotates their average.
    np.testing.assert_allclose(vector(output, 3, 'mean'), [0, 0, 0, 6.5])
    np.testing.assert_allclose(vector(output, 3, 'nested'), [0, 0, 0, 5.5])


def test_against_maps_rotate_opponent_coordinates_before_target_fixture_frame(venue_history):
    history, points = venue_history
    spec = dict(grid_size=2, normalization='count', use_weights=True, side='against')
    output = evaluate_features(history, {'team': Lag(Heatmap(**spec, orientation='team')),
                                         'home': Lag(Heatmap(**spec))}, heatmaps=points)
    # Opponent point [x80,y20] becomes focal [x20,y80]; away fixture then rotates again.
    np.testing.assert_array_equal(vector(output, 1, 'team'), [0, 0, 101, 0])
    np.testing.assert_array_equal(vector(output, 1, 'home'), [0, 101, 0, 0])
    np.testing.assert_array_equal(vector(output, 2, 'home'), [0, 0, 110, 0])
    wrapped = evaluate_features(history, {'home': Lag(ForAgainst(Heatmap(grid_size=2,
        normalization='count', use_weights=True), side='against'))}, heatmaps=points)
    pd.testing.assert_frame_equal(wrapped, output.filter(like='home::'), check_names=True)


@pytest.mark.parametrize('normalization,total', [('mass', 1.), ('density', 1.), ('count', 12.)])
@pytest.mark.parametrize('orientation', ['home', 'team'])
def test_odd_grid_regions_apportion_middle_cell_and_integrate_density(venue_history, normalization, total, orientation):
    history, points = venue_history
    first = points.iloc[0].to_dict()
    maps = pd.DataFrame([{**first, 'x': x, 'y': 10, 'weight': w} for x, w in [(10, 2), (50, 4), (90, 6)]])
    source = Heatmap(grid_size=3, normalization=normalization, use_weights=True, orientation=orientation)
    result = evaluate_features(history, {'own': Lag(RegionMass(source, 'own_half')),
        'opponent': Lag(RegionMass(source, 'opponent_half')),
        'after': RegionMass(Lag(source), 'own_half')}, heatmaps=maps)
    assert result.loc[1, 'own'] == pytest.approx(total/3)
    assert result.loc[1, 'opponent'] == pytest.approx(2*total/3)
    assert result.loc[1, 'own']+result.loc[1, 'opponent'] == pytest.approx(total)
    assert result.loc[1, 'after'] == pytest.approx(result.loc[1, 'own'])
    assert result.loc[0].isna().all() and result.loc[2].isna().all()


def test_region_against_uses_focal_own_half_and_missing_contributing_cells(venue_history):
    history, points = venue_history
    source = Heatmap(grid_size=3, side='against', orientation='team')
    # Opponent x80 maps to focal x20: all its mass is in our own half.
    result = evaluate_features(history, {'own': Lag(RegionMass(source)),
        'opponent': Lag(RegionMass(source, 'opponent_half'))}, heatmaps=points)
    assert result.loc[1, 'own'] == pytest.approx(1)
    assert result.loc[1, 'opponent'] == pytest.approx(0)
    raw = heatmap_values(history, points, source)
    raw.iloc[0, 2] = np.nan  # Far-right cell has zero contribution to own half.
    assert region_values(raw, 'own_half').iloc[0, 0] == pytest.approx(1)
    assert np.isnan(region_values(raw, 'opponent_half').iloc[0, 0])
    raw.iloc[0, 1] = np.nan  # Boundary cell contributes to both halves.
    assert region_values(raw, 'own_half').iloc[0].isna().all()


@pytest.mark.parametrize('operator,kwargs,expected_home,expected_away', [
    (Lag, {}, 5., 30.), (Lag, {'periods': 2}, 3., 20.),
    (RollingMean, {'window': 2}, 4., 25.),
    (RollingStd, {'window': 2}, np.sqrt(2), np.sqrt(50)),
    (RollingZScore, {'window': 2}, 1/np.sqrt(2), 1/np.sqrt(2)),
    (EMA, {'span': 3}, 3.5, 22.5),
])
def test_same_venue_filters_before_each_temporal_window(venue_history, operator, kwargs, expected_home, expected_away):
    history, points = venue_history
    source = RegionMass(Heatmap(grid_size=2, normalization='count', use_weights=True))
    output = evaluate_features(history, {'feature': operator(source, venue='same', **kwargs)}, heatmaps=points)
    assert output.loc[6, 'feature'] == pytest.approx(expected_home)
    assert output.loc[7, 'feature'] == pytest.approx(expected_away)
    assert output.loc[0:1].isna().all().all()
    metadata = output.attrs['spatial_features']['feature']
    assert metadata['kind'] == 'region' and metadata['region'] == 'own_half'
    assert metadata['operators'][-1]['venue'] == 'same'


def test_same_venue_missing_match_is_not_refilled_and_h2h_respects_cutoff(venue_history):
    history, points = venue_history
    missing = points.loc[~points.event_id.astype(object).eq(int(history.event_id.iloc[4]))]
    source = RegionMass(Heatmap(grid_size=2, normalization='count', use_weights=True))
    result = evaluate_features(history, {'lag': Lag(source, venue='same'),
        'mean': RollingMean(source, window=2, venue='same')}, heatmaps=missing)
    assert np.isnan(result.loc[6, 'lag'])
    assert result.loc[6, 'mean'] == 3.
    result = evaluate_features(history, {'h2h': H2H(RollingMean(source, window=2, venue='same'))},
        heatmaps=points, cutoffs=history.kickoff_at-pd.Timedelta(days=2))
    assert result.loc[6, 'h2h'] == 2.  # only matches0,2; match4 exactly cutoff is excluded
    assert result.loc[7, 'h2h'] == 20.  # prior away match with this same opponent


@pytest.mark.parametrize('operator', [Lag, RollingMean, RollingStd, RollingZScore, EMA])
def test_invalid_venue_is_rejected_by_all_temporal_operators(venue_history, operator):
    history, points = venue_history
    with pytest.raises(ValueError, match='venue'):
        evaluate_features(history, {'bad': operator(Heatmap(grid_size=2), venue='neutral')}, heatmaps=points)


def test_region_prediction_safety_and_nonspatial_source(venue_history):
    history, points = venue_history
    with pytest.raises(ValueError, match='Observed Stat'):
        evaluate_features(history, {'raw': RegionMass(Heatmap(grid_size=2))}, heatmaps=points)
    with pytest.raises(ValueError, match='orientation'):
        Heatmap(orientation='away')
    with pytest.raises(ValueError, match='half'):
        RegionMass(Heatmap(), 'left_touchline')
    from xdiyo_analytics.features import IsHome
    with pytest.raises(TypeError, match='spatial grid'):
        evaluate_features(history, {'bad': RegionMass(IsHome())}, heatmaps=points)


@pytest.mark.parametrize('disabled', ['none', 'zero_rounds'])
def test_disabled_warm_start_retains_spatial_frame_and_metadata(venue_history, disabled):
    from xdiyo_analytics.features import WarmStart, SeededEMA, Hard
    history, points = venue_history
    history['round'] = np.arange(1, len(history)+1)
    source = RollingMean(Heatmap(grid_size=2, normalization='count', use_weights=True), window=2, venue='same')
    ordinary = evaluate_features(history, {'grid': source}, heatmaps=points)
    policy = None if disabled == 'none' else SeededEMA(handoff=Hard(0))
    warmed = evaluate_features(history, {'grid': WarmStart(source, policy)}, heatmaps=points)
    # A disabled warmup wrapper leaves values, final orientation and discovery intact.
    pd.testing.assert_frame_equal(warmed, ordinary)
    assert warmed.attrs['spatial_features'] == ordinary.attrs['spatial_features']


def test_spatial_metadata_survives_keyed_match_assembly_for_grids_and_regions(ui_recipe, monkeypatch):
    from xdiyo_analytics.data import load_seasons
    from xdiyo_analytics.ui import prepare_recipe
    from xdiyo_analytics.ui.recipe import node
    data = load_seasons(**ui_recipe['data'])
    points = data.statistics[['event_id', 'team_id', 'source_league', 'source_season']].drop_duplicates().copy()
    points['x'], points['y'], points['kind'] = 15., 25., 'touch'
    data.tables['heatmap_points'] = points
    monkeypatch.setattr('xdiyo_analytics.data.load_seasons', lambda **kwargs: deepcopy(data))
    source = node('features.Heatmap', grid_size=3, side='both')
    recipe = deepcopy(ui_recipe)
    recipe['features'] = {'spatial': node('features.RollingMean', source=source, window=2, venue='same'),
        'half': node('features.Lag', source=node('features.RegionMass', source=source, region='own_half'))}
    prepared = prepare_recipe(recipe)
    specs = prepared.dataset.definitions['spatial_features']
    assert len(specs) == 8  # 2 fixture sides x 2 perspectives x grid/region
    assert {spec['kind'] for spec in specs.values()} == {'grid', 'region'}
    assert {spec['fixture_side'] for spec in specs.values()} == {'home', 'away'}
    all_columns = []
    for key, spec in specs.items():
        assert key.startswith(spec['fixture_side']+'::')
        assert spec['grid_size'] == 3 and spec['orientation'] == 'home'
        assert spec['side'] in ('for', 'against') and spec['family']
        assert all(c in prepared.dataset.X for c in spec['columns'])
        assert len(spec['columns']) == (9 if spec['kind'] == 'grid' else 1)
        all_columns.extend(spec['columns'])
    assert set(all_columns) == set(prepared.dataset.X)
    assert len(all_columns) == len(set(all_columns))


@pytest.fixture
def fixture_report_context():
    columns = ['h0', 'h1', 'h2', 'h3', 'a0', 'a1', 'a2', 'a3', 'hhalf', 'ahalf']
    values = pd.DataFrame([[1., 2., 3., 4., 5., 6., 7., 8., 4., 12.],
                           [11., 12., 13., 14., 15., 16., 17., 18., 24., 32.]], columns=columns)
    metadata = pd.DataFrame({'event_id': pd.array([2**63+77, 2**63+78], dtype='uint64[pyarrow]'),
        'home_id': pd.array([2**63+11, 2**63+12], dtype='uint64[pyarrow]'),
        'away_id': pd.array([2**63+12, 2**63+11], dtype='uint64[pyarrow]'),
        'source_league': 'Alpha', 'source_season': '24_25', 'round': [1, 2]})
    specs = {}
    for side, prefix in [('home', 'h'), ('away', 'a')]:
        base = dict(side='against', orientation='home', normalization='count', grid_size=2,
                    fixture_side=side, operators=[dict(operator='RollingMean', window=5, venue='same')])
        specs[side+'::grid'] = dict(**base, kind='grid', columns=[prefix+str(i) for i in range(4)], family='grid')
        specs[side+'::region'] = dict(**base, kind='region', region='own_half', columns=[prefix+'half'], family='region')
    return SimpleNamespace(X=values, metadata=metadata, layout='match',
        match_columns=('source_league', 'source_season', 'event_id'), definitions={'spatial_features': specs})


def test_fixture_panels_use_exact_declared_grid_region_values_ids_and_badges(fixture_report_context, tmp_path):
    from xdiyo_analytics.reporting import HeatmapReporter, TeamCatalog
    from xdiyo_analytics.reporting.heatmaps import SpatialFixtureData
    context = fixture_report_context
    badge = tmp_path/'fixture-badge.svg'
    badge.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>')
    catalog = TeamCatalog({2**63+11: {'name': 'Alpha A', 'badge_path': str(badge)}, 2**63+12: 'Alpha B'})
    result = HeatmapReporter(type='overall', partition='train', catalog=catalog, cache_size=3).run(context)
    store = result.artifacts[0].data
    assert isinstance(store, SpatialFixtureData)
    manifest = store.manifest()
    assert manifest['cache_size'] == 3 and manifest['total_fixtures'] == 2
    assert {entry['kind'] for entry in manifest['maps']} == {'grid', 'region'}
    assert manifest['teams'][str(2**63+11)]['name'] == 'Alpha A'
    assert manifest['teams'][str(2**63+11)]['badge'].startswith('data:image/svg+xml;base64,')
    assert manifest['fixtures'][0]['home'] == str(2**63+11)
    assert str(2**63+77) not in manifest['fixtures'][0]['label']
    assert manifest['fixtures'][0]['label'].startswith('Alpha A vs Alpha B')
    assert all('rows' not in entry for entry in manifest['fixtures'])
    for fixture, expected in [('0', context.X.iloc[0]), ('1', context.X.iloc[1])]:
        grids = store.panel_pair(fixture, 'grid')['panels']
        regions = store.panel_pair(fixture, 'region')['panels']
        for side_index, prefix in enumerate(('h', 'a')):
            np.testing.assert_array_equal(grids[side_index]['values'], expected[[prefix+str(i) for i in range(4)]])
            assert regions[side_index]['values'] == [expected[prefix+'half']]
            assert grids[side_index]['spec']['operators'][0]['venue'] == 'same'
            assert regions[side_index]['spec']['region'] == 'own_half'
            assert 'columns' not in grids[side_index]['spec']
    assert store.panel_pair('0', 'grid')['scale'] == [0., 18.]
    # Reading another fixture cannot mutate or average the retained first pair.
    assert store.panel_pair('0', 'grid')['panels'][0]['values'] == [1., 2., 3., 4.]
    offline = store.offline(limit=1)
    assert offline['manifest']['included_fixtures'] == 1
    assert set(offline['pairs']) == {'0'} and set(offline['pairs']['0']) == {'grid', 'region'}


def test_team_match_missing_opposite_row_stays_missing_with_duplicate_dataframe_index(venue_history):
    from xdiyo_analytics.reporting import HeatmapReporter
    history, points = venue_history
    features = evaluate_features(history, {'grid': Lag(Heatmap(grid_size=2))}, heatmaps=points)
    selected = [6, 7]
    values, metadata = features.loc[selected].copy(), history.loc[selected].copy()
    values.index = metadata.index = pd.Index([5, 5])
    context = SimpleNamespace(X=values, metadata=metadata, layout='team_match',
        match_columns=('source_league', 'source_season', 'event_id'),
        definitions={'spatial_features': features.attrs['spatial_features']})
    store = HeatmapReporter(type='overall', partition='train').run(context).artifacts[0].data
    first, second = store.panel_pair('0', 'grid')['panels'], store.panel_pair('1', 'grid')['panels']
    assert first[0]['missing'] is False and first[1]['missing'] is True
    assert second[0]['missing'] is True and second[1]['missing'] is False
    np.testing.assert_array_equal(first[0]['values'], values.iloc[0])
    np.testing.assert_array_equal(second[1]['values'], values.iloc[1])
    assert first[0]['team_id'] == second[1]['team_id'] == str(2**63+11)


def test_fixture_view_preserves_partial_missingness_caps_population_and_validates_choices(fixture_report_context):
    from xdiyo_analytics.reporting import HeatmapReporter
    context = fixture_report_context
    context.X.loc[0, 'h0'] = np.nan
    context.X.loc[0, ['a0', 'a1', 'a2', 'a3']] = np.nan
    store = HeatmapReporter(type='overall', partition='train', maps=['grid'], max_fixtures=1).run(context).artifacts[0].data
    panels = store.panel_pair('0', 'grid')['panels']
    assert panels[0]['values'] == [None, 2., 3., 4.] and panels[0]['missing'] is False
    assert panels[1]['values'] == [None]*4 and panels[1]['missing'] is True
    assert len(store.values) == 1 and store.manifest()['total_fixtures'] == 1
    for fixture in ('-1', '1', '01'):
        with pytest.raises(KeyError, match='fixture'):
            store.panel_pair(fixture, 'grid')
    with pytest.raises(KeyError):
        store.panel_pair('0', 'unknown')
    with pytest.raises(ValueError, match='positive integer'):
        HeatmapReporter(type='overall', partition='train', cache_size=0).run(context)
    with pytest.raises(ValueError, match='incomplete'):
        HeatmapReporter(type='overall', partition='train').run(SimpleNamespace(**{**vars(context), 'X':context.X.drop(columns='h0')}))


def test_live_store_sends_manifest_then_fetches_only_requested_fixture_pair(fixture_report_context, monkeypatch):
    import json
    import re
    from xdiyo_analytics.reporting import HeatmapReporter
    from xdiyo_analytics.reporting.spatial_live import SpatialStore
    from xdiyo_analytics.reporting.spatial_view import render_spatial
    data = HeatmapReporter(type='overall', partition='train', cache_size=2).run(fixture_report_context).artifacts[0].data
    store = SpatialStore()
    original = data.panel_pair
    calls = []
    def pair(fixture, feature):
        calls.append((fixture, feature))
        return original(fixture, feature)
    monkeypatch.setattr(data, 'panel_pair', pair)
    document = render_spatial(data, transport={'register': store.register, 'url': '/spatial', 'token': 'test-token'})
    payload = json.loads(re.search(r'class="spatial-spec">(.*?)</script>', document).group(1))
    assert set(payload) == {'manifest', 'remote'}
    assert 'pairs' not in payload and not calls
    assert payload['manifest']['cache_size'] == 2
    key = payload['remote']['id']
    assert store.register(data) == key and len(store.data) == len(store.ids) == 1
    actual = store.fetch({'id': key, 'fixture': '1', 'map': 'region'})
    assert calls == [('1', 'region')]
    assert actual['panels'][0]['values'] == [24.]
    assert actual['panels'][1]['values'] == [32.]
    for fixture in ('0', '1', '0'):
        store.fetch({'id': key, 'fixture': fixture, 'map': 'grid'})
    assert len(store.data) == 1  # The server retains a single numerical store, not per-fetch copies.
    assert len(calls) == 4
    with pytest.raises(KeyError):
        store.fetch({'id': 'unknown', 'fixture': '0', 'map': 'grid'})


def test_live_report_http_requires_token_and_keeps_grids_out_of_initial_html(fixture_report_context):
    import json
    import re
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    from xdiyo_analytics.reporting import HeatmapReporter
    from xdiyo_analytics.reporting.contracts import AnalysisReport, StudyRun
    context = fixture_report_context
    context.X.loc[0, 'h0'] = np.nan
    result = HeatmapReporter(type='overall', partition='train').run(context)
    report = AnalysisReport([StudyRun(name='maps', type='overall', partition='train', fold_id=None,
        layout='match', row_positions=np.arange(2), n_matches=2, result=result)])
    handle = report.live()
    try:
        assert report.live() is handle
        with urlopen(handle.url, timeout=5) as response:
            assert response.status == 200
            document = response.read().decode()
        payload = json.loads(re.search(r'class="spatial-spec">(.*?)</script>', document).group(1))
        assert set(payload) == {'manifest', 'remote'} and 'pairs' not in payload
        assert payload['manifest']['total_fixtures'] == 2
        request = dict(id=payload['remote']['id'], fixture='0', map='grid')
        def fetch(token, **headers):
            return urlopen(Request(handle.url+'spatial', json.dumps(request).encode(), method='POST',
                headers={'Content-Type':'application/json', 'X-Builder-Token':token, **headers}), timeout=5)
        with pytest.raises(HTTPError) as error:
            fetch('wrong-token')
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            fetch(handle.token, Origin='https://unrelated.example')
        assert error.value.code == 403
        with fetch(handle.token) as response:
            assert response.headers['Cache-Control'] == 'no-store'
            fetched = json.load(response)
        assert fetched['panels'][0]['values'] == [None, 2., 3., 4.]
        assert fetched['panels'][1]['values'] == [5., 6., 7., 8.]
        request['id'] = 'not-a-registered-report'
        with pytest.raises(HTTPError) as error:
            fetch(handle.token)
        assert error.value.code == 400
        # Offline exports are bounded explicitly and embed only the chosen subset.
        offline = report.to_html(spatial_limit=1)
        exported = json.loads(re.search(r'class="spatial-spec">(.*?)</script>', offline).group(1))
        assert set(exported['pairs']) == {'0'}
        assert exported['manifest']['included_fixtures'] == 1
        assert exported['manifest']['total_fixtures'] == 2
    finally:
        report.close_live()
    assert handle.closed and handle.document == ''
    assert not handle.store.data and not handle.store.ids
