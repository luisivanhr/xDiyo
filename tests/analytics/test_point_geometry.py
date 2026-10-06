"""Exact point estimators, historical causality, native recipe/report integration."""
from copy import deepcopy
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from transition_samples import history_from_games, A, B, STAT
from test_ui_workflow import ui_recipe
from xdiyo_analytics.features import (SpatialPointSummary, RollingMean, RollingStd,
    RollingZScore, EMA, Lag, H2H, Product, Constant, ForAgainst, evaluate_features)
from xdiyo_analytics.features.point_geometry import point_summary_source, point_summary_values
from xdiyo_analytics.datasets import assemble_dataset
from xdiyo_analytics.labels import create_labels, MatchTotal
from xdiyo_analytics.reporting import HeatmapReporter
from xdiyo_analytics.ui import prepare_recipe, catalog_for_ui
from xdiyo_analytics.ui.recipe import node, export_python


def maps_for(history, xy=((0., 0.), (100., 100.))):
    records = []
    keys = [c for c in ('source_league', 'source_season', 'event_id', 'team_id') if c in history]
    for identity in history[keys].to_dict('records'):
        for order, (x, y) in enumerate(xy):
            records.append({**identity, 'kind':'player', 'x':x, 'y':y,
                            'point_order':order, 'raw_hash':'v1', 'weight':999.})
        records.append({**identity, 'kind':'goalkeeper', 'x':1., 'y':50.,
                        'point_order':len(xy), 'raw_hash':'v1', 'weight':0.})
    return pd.DataFrame(records)


def observed(history, points, **kwargs):
    spec = SpatialPointSummary(**kwargs)
    return point_summary_values(history, point_summary_source(history, points, spec.kinds, spec.min_points), spec)


@pytest.mark.parametrize('xy,expected', [([(20,30)], [20,0,0]),
    ([(0,0),(100,100)], [50,50,50]), ([(0,10),(0,10),(90,70)], [30,np.sqrt(1800),np.sqrt(800)])])
def test_exact_population_moments_preserve_repeated_observations(xy, expected):
    h = history_from_games([{}]); p = maps_for(h, xy)
    for field, value in zip(('mean_x','sd_x','sd_y'), expected):
        np.testing.assert_allclose(observed(h,p,field=field).to_numpy(), value)


def test_translation_scaling_rotation_and_combined_kinds():
    h = history_from_games([{}]); p = maps_for(h, [(20,30),(40,50)])
    base = [observed(h,p,field=f).iloc[0,0] for f in ('mean_x','sd_x','sd_y')]
    p.loc[p.kind == 'player',['x','y']] = p.loc[p.kind == 'player',['x','y']]*1.5 + 5
    got = [observed(h,p,field=f).iloc[0,0] for f in ('mean_x','sd_x','sd_y')]
    np.testing.assert_allclose(got, [base[0]*1.5+5,base[1]*1.5,base[2]*1.5])
    both = observed(h,p,field='mean_x',side='both')
    np.testing.assert_allclose(both.sum(axis=1),100)
    np.testing.assert_allclose(observed(h,p,field='sd_y',side='both'),base[2]*1.5)
    expected = p[p.team_id == A].x.mean()
    assert observed(h,p,kinds=('player','goalkeeper')).iloc[0,0] == expected


@pytest.mark.parametrize('kwargs', [dict(field='unknown'),dict(kinds=('players',)),dict(kinds=()),
    dict(kinds=('player','player')),dict(side='home'),dict(min_points=0),dict(min_points=True)])
def test_invalid_spec(kwargs):
    with pytest.raises(ValueError): SpatialPointSummary(**kwargs)


def test_identity_coordinate_and_source_validation():
    h = history_from_games([{}]); p = maps_for(h)
    with pytest.raises(ValueError, match='table'): observed(h,None)
    with pytest.raises(ValueError, match='both history'): observed(h,p.drop(columns='source_season'))
    with pytest.raises(ValueError, match='point_order'): observed(h,pd.concat([p,p]))
    p.loc[0,'raw_hash']='other'
    with pytest.raises(ValueError, match='source version'): observed(h,p)
    p = maps_for(h)
    for bad in (np.nan,np.inf,-1,101):
        p.loc[0,'x']=bad
        with pytest.raises(ValueError, match='coordinates for map'): observed(h,p)
    # Unselected kinds are filtered before coordinate validation.
    p = maps_for(h); p.loc[p.kind=='goalkeeper','x']=np.inf
    assert observed(h,p).notna().all().all()
    with pytest.raises(ValueError, match='coordinates'): observed(h,p,kinds=('goalkeeper',))


def test_source_identity_collisions_and_point_order_permutation():
    h = history_from_games([{},dict(season=2)])
    h.loc[2:,'event_id'] = h.event_id.iloc[0]
    p = maps_for(h)
    p.loc[p.source_season=='24_25','x']=20
    got=observed(h,p)
    np.testing.assert_allclose(got.iloc[:,0],[50,50,20,20])
    np.testing.assert_allclose(observed(h,p.sample(frac=1,random_state=7)),got)


def test_empty_table_is_missing_and_bad_strings_have_map_identity():
    h=history_from_games([{},{}]);p=maps_for(h)
    assert observed(h,p.iloc[:0]).isna().all().all()
    p['x']=p.x.astype(object);p.loc[0,'x']='bad'
    with pytest.raises(ValueError,match='coordinates for map'):
        observed(h,p)


def test_known_reference_rejects_observed_summary():
    h=history_from_games([{},{}])
    with pytest.raises(ValueError,match='reference must be historical'):
        evaluate_features(h,{'z':RollingZScore(SpatialPointSummary(),2,
            reference=Product(SpatialPointSummary(),Constant(2)))},heatmaps=maps_for(h))


def test_missing_and_low_count_audits_and_window_does_not_skip_missing():
    h = history_from_games([{}, {}, {}, {}]); p=maps_for(h,[(20,30)])
    p = p[~((p.event_id==h.event_id.iloc[2]) & (p.kind=='player'))]
    p = p[p.event_id != h.event_id.iloc[4]]
    raw=observed(h,p)
    assert {x['status'] for x in raw.attrs['point_map_coverage']} == {'ok','empty_selected_kind','missing_team_map'}
    assert observed(h,p,min_points=2).isna().all().all()
    got=evaluate_features(h,{'r':RollingMean(SpatialPointSummary(),2), 'lag':Lag(SpatialPointSummary())},heatmaps=p)
    assert got.iloc[-1].isna().all()  # Last two eligible maps are missing; never reach behind them.
    audit=got.attrs['spatial_point_audit']['features']['r']['history'][-1]
    assert (audit['eligible_matches'],audit['window_matches'],audit['usable_maps'])==(3,2,0)


@pytest.mark.parametrize('source', [SpatialPointSummary(), Product(SpatialPointSummary(),Constant(2)),
    H2H(SpatialPointSummary()), ForAgainst(SpatialPointSummary(),side='against')])
def test_recursive_observed_safety(source):
    h=history_from_games([{},{}])
    with pytest.raises(ValueError,match='before prediction'):
        evaluate_features(h,{'unsafe':source},heatmaps=maps_for(h))


def test_cutoffs_late_release_simultaneity_future_venue_and_seasons():
    h=history_from_games([dict(day=0),dict(day=2),dict(day=4,season=2,home=B,away=A),
                          dict(day=4,season=2),dict(day=6,season=2,status='notstarted')])
    p=maps_for(h,[(20,30)])
    p.loc[p.event_id==h.event_id.iloc[2],'x']=80
    available=h.kickoff_at+pd.Timedelta(hours=3)
    available.iloc[2:4]=h.kickoff_at.iloc[-1]+pd.Timedelta(days=1)
    options=dict(cutoffs=h.kickoff_at-pd.Timedelta(hours=1),available_at=available)
    specs={'all':RollingMean(SpatialPointSummary(),20), 'same':RollingMean(SpatialPointSummary(),20,venue='same')}
    got=evaluate_features(h,specs,heatmaps=p,**options)
    np.testing.assert_allclose(got['all'].iloc[4:],20)
    assert np.isnan(got['same'].iloc[4])  # Prior away venues of focal B do not seed its first home.
    # Target and same-cutoff maps are inaccessible, even if their coordinates change.
    q=p.copy();q.loc[q.event_id.isin(h.event_id.iloc[4:]),'x']=99
    changed=evaluate_features(h,specs,heatmaps=q,**options)
    np.testing.assert_allclose(changed.iloc[:8],got.iloc[:8],equal_nan=True)
    assert got.attrs['spatial_point_audit']['timing_sha256'] == changed.attrs['spatial_point_audit']['timing_sha256']


def test_shared_summary_cache_rebuild_and_six_column_assembly(monkeypatch):
    import xdiyo_analytics.features.evaluation as module
    original=module.point_summary_source; calls=[]
    def count(*args): calls.append(1);return original(*args)
    monkeypatch.setattr(module,'point_summary_source',count)
    h=history_from_games([{}, {}, dict(home=B,away=A)])
    p=maps_for(h,[(20,30),(40,70)])
    specs={f'activity_{f}_r20':RollingMean(SpatialPointSummary(f),20) for f in ('mean_x','sd_x','sd_y')}
    got=evaluate_features(h,specs,heatmaps=p,keyed=True)
    assert len(calls)==1
    label=create_labels(h,{'corners':MatchTotal(STAT)})['corners']
    ds=assemble_dataset(got.iloc[::-1],label,layout='match')
    assert list(ds.X)==[f'{side}::{name}' for side in ('home','away') for name in specs]
    assert len(ds.definitions['spatial_features'])==6
    assert all(v['orientation']=='team' for v in ds.definitions['spatial_features'].values())
    # Away mean stays at 30, not rotated to 70 at assembly.
    assert ds.X['away::activity_mean_x_r20'].iloc[-1]==30
    p.loc[p.kind=='player','x']+=10
    new=evaluate_features(h,specs,heatmaps=p,keyed=True)
    assert len(calls)==2 and new.iloc[-1,0]==40
    context=SimpleNamespace(X=ds.X.iloc[1:],metadata=ds.metadata.iloc[1:],layout='match',
        match_columns=ds.match_columns,definitions=ds.definitions)
    report=HeatmapReporter(type='overall',partition='train').run(context)
    assert len(report.artifacts[0].data.maps)==3
    assert len(report.tables['point_map_coverage'])==12
    assert report.tables['point_fixture_coverage'].both_teams_usable.sum()==6


def test_temporal_population_distinction_and_for_against():
    h=history_from_games([{}, {}, {}]);p=maps_for(h,[(0,20)])
    p.loc[p.event_id==h.event_id.iloc[2],'x']=100
    values=evaluate_features(h,{'s':RollingStd(SpatialPointSummary(),2),
        'against':Lag(ForAgainst(SpatialPointSummary(),side='against')),
        'ema':EMA(SpatialPointSummary(),span=3)},heatmaps=p)
    assert values.s.iloc[-1]==pytest.approx(np.sqrt(5000))
    assert values.against.iloc[-1]==0
    assert values.ema.iloc[-1]==50


def test_ui_auto_load_export_and_recipe_reproduction(ui_recipe,monkeypatch):
    from xdiyo_analytics.data import load_seasons
    from xdiyo_analytics.histories import build_team_history
    loaded=load_seasons(**ui_recipe['data'])
    loaded.tables['heatmap_points']=maps_for(build_team_history(loaded),[(20,30),(40,70)])
    calls=[]
    def load(**kwargs):calls.append(kwargs);return deepcopy(loaded)
    monkeypatch.setattr('xdiyo_analytics.data.load_seasons',load)
    recipe=deepcopy(ui_recipe)
    recipe['features']={f'activity_{f}_r20':node('features.RollingMean',source=node('features.SpatialPointSummary',field=f,kinds=['player']),window=20,min_periods=1,venue='all') for f in ('mean_x','sd_x','sd_y')}
    recipe=json.loads(json.dumps(recipe))
    prepared=prepare_recipe(recipe)
    assert 'heatmap_points' in calls[0]['tables']
    assert prepared.dataset.X.shape[1]==6
    scope={}
    exec(export_python(recipe).split('result = run_recipe')[0],scope)
    pd.testing.assert_frame_equal(scope['prepared'].dataset.X,prepared.dataset.X)
    assert isinstance(catalog_for_ui().build(recipe['features']['activity_sd_x_r20']).source,SpatialPointSummary)
    assert prepared.dataset.definitions['spatial_point_audit']['sources'][0]['provenance']==loaded.provenance


def test_prepared_feature_parquet_metadata_roundtrip(tmp_path):
    h=history_from_games([{},{}]);p=maps_for(h)
    values=evaluate_features(h,{'p':RollingMean(SpatialPointSummary(),2)},heatmaps=p,keyed=True)
    path=tmp_path/'features.parquet'
    values.to_parquet(path,compression='zstd')
    restored=pd.read_parquet(path)
    np.testing.assert_allclose(values,restored,equal_nan=True)
    assert restored.attrs['spatial_features']==values.attrs['spatial_features']
    assert restored.attrs['spatial_point_audit']['sources']==values.attrs['spatial_point_audit']['sources']
