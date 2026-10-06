"""Scalar spatial lineage, coverage, causal composition and native persistence."""
from copy import deepcopy
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from transition_samples import history_from_games, STAT
from test_point_geometry import maps_for
from test_ui_workflow import ui_recipe
from xdiyo_analytics.features import (SpatialPointSummary as Point, RollingMean, Lag, EMA,
    Constant, Product, Sum, Difference, Ratio, evaluate_features)
from xdiyo_analytics.labels import MatchTotal, create_labels
from xdiyo_analytics.datasets import assemble_dataset
from xdiyo_analytics.reporting import HeatmapReporter
from xdiyo_analytics.ui import prepare_recipe, catalog_for_ui
from xdiyo_analytics.ui.recipe import export_python


def evaluate(expr, h=None, p=None):
    h = history_from_games([{}, {}, {}, {}]) if h is None else h
    p = maps_for(h) if p is None else p
    return evaluate_features(h, {'derived':expr}, heatmaps=p, keyed=True)


def report(values, h, start=0):
    label = create_labels(h, {'corners':MatchTotal(STAT)})['corners']
    ds = assemble_dataset(values.iloc[::-1], label, layout='match')
    context = SimpleNamespace(X=ds.X.iloc[start:], metadata=ds.metadata.iloc[start:],
        layout=ds.layout, match_columns=ds.match_columns, definitions=ds.definitions)
    return HeatmapReporter(type='overall', partition='train').run(context)


@pytest.mark.parametrize('inside', [True, False])
def test_exact_reported_reproductions(inside):
    h = history_from_games([{}, {}, {}])
    expr = RollingMean(Product(Point(), Constant(2)), 20) if inside else Product(RollingMean(Point(),20), Constant(2))
    frame = evaluate(expr, h)
    np.testing.assert_allclose(frame.iloc[:,0], [np.nan,np.nan,100,100,100,100], equal_nan=True)
    spec = frame.attrs['spatial_features']['derived']
    assert spec['derived'] and spec['field_label'] == 'Derived spatial value'
    assert spec['orientation'] == 'derived' and spec['side'] is None
    assert spec['calculation']['operator'] == ('RollingMean' if inside else 'Product')
    tables = report(frame, h).tables
    assert len(tables['point_map_coverage']) == 6
    assert tables['point_fixture_coverage'].both_teams_usable.tolist() == [3]
    assert tables['point_output_coverage'].usable_output.tolist() == [False,False,True,True,True,True]


@pytest.mark.parametrize('op,expected', [(Sum,52), (Difference,48), (Product,100), (Ratio,25)])
@pytest.mark.parametrize('inside', [False, True])
@pytest.mark.parametrize('spatial_right', [False, True])
def test_operations_constants_or_spatial_operands(op, expected, inside, spatial_right):
    # goalkeeper mean_x is 1; scaling by 2 gives a second spatial operand of 2.
    right = Product(Point(kinds=('goalkeeper',)),Constant(2)) if spatial_right else Constant(2)
    expr = RollingMean(op(Point(),right),2) if inside else op(RollingMean(Point(),2),RollingMean(right,2))
    frame = evaluate(expr)
    np.testing.assert_allclose(frame.iloc[2:,0], expected)
    audit = frame.attrs['spatial_point_audit']['features']['derived']
    assert len(audit['maps']) == 8*(2 if spatial_right else 1)
    assert len({r['source_id'] for r in audit['maps']}) == (2 if spatial_right else 1)
    assert all(r['usable_output'] for r in audit['outputs'][2:])


def test_shared_sources_deduplicated_and_windows_kept_separate():
    h = history_from_games([{}, {}, {}, {}])
    a, b = RollingMean(Point(),2), RollingMean(Point('sd_x'),3)
    frame = evaluate(Sum(Product(a,a),b),h)
    audit = frame.attrs['spatial_point_audit']['features']['derived']
    assert len(audit['maps']) == 8
    assert len(audit['history']) == 16
    assert len({r['history_id'] for r in audit['history']}) == 2
    assert sorted(r['window_matches'] for r in audit['history'] if r['event_id']==h.event_id.iloc[-1]) == [2,2,3,3]
    tables = report(frame,h,start=2).tables
    assert len(tables['point_map_coverage']) == 4
    assert len(tables['point_history_coverage']) == 8
    assert tables['point_fixture_coverage'].both_teams_usable.tolist() == [2]
    assert set(tables['point_map_coverage'].event_id) == set(h.event_id.iloc[4:])


@pytest.mark.parametrize('right', [Point(side='against'),Point(min_points=3),Point(kinds=('goalkeeper',))])
def test_selection_identity_is_not_merged(right):
    frame = evaluate(RollingMean(Sum(Point(),right),2))
    maps = frame.attrs['spatial_point_audit']['features']['derived']['maps']
    assert len(maps)==16 and len({r['source_id'] for r in maps})==2


@pytest.mark.parametrize('op', [Difference,Ratio])
def test_order_and_noncommuting_history(op):
    h = history_from_games([{}, {}, {}, {}]); p = maps_for(h,[(10,20)])
    p.loc[p.event_id==h.event_id.iloc[2],'x'] = 40
    left,right = Point(),Constant(2)
    a=evaluate(RollingMean(op(left,right),2),h,p)
    b=evaluate(RollingMean(op(right,left),2),h,p)
    assert a.iloc[-1,0] != b.iloc[-1,0]
    assert a.attrs['spatial_features'] != b.attrs['spatial_features']
    # E[1/X] != 1/E[X]; preserve both values and tree order.
    inside=evaluate(RollingMean(Ratio(Constant(1),Point()),2),h,p)
    outside=evaluate(Ratio(Constant(1),RollingMean(Point(),2)),h,p)
    assert inside.iloc[-1,0] != outside.iloc[-1,0]


@pytest.mark.parametrize('expr,finite', [
    (Ratio(Point(),Constant(0)),False),
    (Ratio(Point(),Constant(0),zero_value=7),True),
    (Product(Point(),Constant(1e308)),False)])
def test_valid_maps_and_nonfinite_arithmetic_are_distinct(expr,finite):
    h=history_from_games([{}, {}, {}]); frame=evaluate(Lag(expr),h)
    audit=frame.attrs['spatial_point_audit']['features']['derived']
    assert all(r['valid_map'] for r in audit['maps'])
    assert all(r['usable_output']==finite for r in audit['outputs'][2:])
    assert all(r['usable_values']==int(finite) for r in audit['history'][2:])
    assert all('usable_maps' not in r for r in audit['history'])
    report(frame,h)


def test_missing_empty_low_count_maps_do_not_use_fallback():
    h=history_from_games([{}, {}, {}, {}]); p=maps_for(h)
    ids=h.event_id.unique()
    p=p[p.event_id != ids[0]]
    p=p[~((p.event_id==ids[1]) & (p.kind=='player'))]
    p=p[~((p.event_id==ids[2]) & (p.point_order==1))]
    frame=evaluate(Lag(Ratio(Point(min_points=2),Constant(0),zero_value=7)),h,p)
    audit=frame.attrs['spatial_point_audit']['features']['derived']
    assert {r['status'] for r in audit['maps']} == {'missing_team_map','empty_selected_kind','low_point_count','ok'}
    assert frame.isna().all().all()
    assert not any(r['usable_output'] for r in audit['outputs'])


def test_cache_order_independence_and_parquet(tmp_path):
    from xdiyo_analytics.experiments.recovery import dump_bundle, load_bundle, signature
    h=history_from_games([{}, {}, {}]); p=maps_for(h)
    base=RollingMean(Point(),2)
    features={'base':base,'sum':Sum(base,base),'derived':Product(base,Constant(2))}
    a=evaluate_features(h,features,heatmaps=p,keyed=True)
    b=evaluate_features(h,dict(reversed(list(features.items()))),heatmaps=p,keyed=True)
    assert a.attrs['spatial_features']==b.attrs['spatial_features']
    assert a.attrs['spatial_point_audit']==b.attrs['spatial_point_audit']
    assert a.attrs['spatial_features']['base']['source']=='SpatialPointSummary'
    a.to_parquet(tmp_path/'prepared.parquet',compression='zstd')
    restored=pd.read_parquet(tmp_path/'prepared.parquet')
    pd.testing.assert_frame_equal(restored,a,check_index_type=False)
    # Parquet's JSON metadata normalizes existing tuple-valued identity/grouping
    # keys to lists. Values, keys and all spatial records must remain identical.
    assert restored.attrs==json.loads(json.dumps(a.attrs))
    dump_bundle(tmp_path/'recovery.json',a)
    recovered=load_bundle(tmp_path/'recovery.json')
    pd.testing.assert_frame_equal(recovered,a)
    assert recovered.attrs==a.attrs and signature(recovered)==signature(a)
    a.attrs['spatial_features']['derived']['sources'][0]['field']='mutated'
    assert a.attrs['spatial_features']['base']['field']=='mean_x'


def test_native_recipe_json_python_and_report(ui_recipe,monkeypatch):
    from xdiyo_analytics.data import load_seasons
    from xdiyo_analytics.histories import build_team_history
    loaded=load_seasons(**ui_recipe['data'])
    loaded.tables['heatmap_points']=maps_for(build_team_history(loaded))
    monkeypatch.setattr('xdiyo_analytics.data.load_seasons',lambda **kwargs:deepcopy(loaded))
    recipe=deepcopy(ui_recipe)
    expr=Product(RollingMean(Point('depth_80'),2),Ratio(Lag(Point('sd_y')),Constant(2)))
    recipe['features']={'derived':catalog_for_ui().encode(expr)}
    recipe=json.loads(json.dumps(recipe))
    prepared=prepare_recipe(recipe)
    scope={};exec(export_python(recipe).split('result = run_recipe')[0],scope)
    pd.testing.assert_frame_equal(scope['prepared'].dataset.X,prepared.dataset.X)
    assert scope['prepared'].dataset.definitions['spatial_point_audit']==prepared.dataset.definitions['spatial_point_audit']
    ds=prepared.dataset
    HeatmapReporter(type='overall',partition='train').run(SimpleNamespace(
        X=ds.X,metadata=ds.metadata,layout=ds.layout,match_columns=ds.match_columns,definitions=ds.definitions))


@pytest.mark.parametrize('wrap', [lambda x:Lag(x),lambda x:EMA(x,span=3),lambda x:RollingMean(x,2)])
def test_temporal_stages_keep_branch_windows_and_counts(wrap):
    child=Sum(RollingMean(Point(),2),RollingMean(Point('sd_x'),3))
    frame=evaluate(wrap(child))
    rows=frame.attrs['spatial_point_audit']['features']['derived']['history']
    assert len(rows)==24
    assert len({r['history_id'] for r in rows})==3
    assert sorted({r['window_matches'] for r in rows if r['event_id']==rows[-1]['event_id']})


def test_causal_arithmetic_and_source_fingerprint():
    from xdiyo_analytics.features import H2H
    h=history_from_games([dict(day=0),dict(day=2),dict(day=4),dict(day=4),dict(day=6)])
    p=maps_for(h,[(20,30)])
    options=dict(cutoffs=h.kickoff_at-pd.Timedelta(hours=1),available_at=h.kickoff_at+pd.Timedelta(hours=3))
    features={'p':H2H(RollingMean(Product(Point(),Constant(2)),3))}
    a=evaluate_features(h,features,heatmaps=p,**options)
    p.loc[p.event_id.isin(h.event_id.iloc[4:]),'x']=90
    b=evaluate_features(h,features,heatmaps=p,**options)
    np.testing.assert_allclose(a.iloc[:8],b.iloc[:8],equal_nan=True)
    assert a.attrs['spatial_point_audit']['timing_sha256']==b.attrs['spatial_point_audit']['timing_sha256']
    assert a.attrs['spatial_point_audit']['features']['p']['maps'][0]['source_id']!=b.attrs['spatial_point_audit']['features']['p']['maps'][0]['source_id']
    with pytest.raises(ValueError,match='same H2H'):
        evaluate(Sum(H2H(Lag(Point())),Lag(Point())))
    with pytest.raises(ValueError,match='one output column'):
        evaluate(Product(Lag(Point(side='both')),Constant(2)))
