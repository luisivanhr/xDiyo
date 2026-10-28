"""Venue views retain spatial lineage, orientation and reporter labels."""
import numpy as np
import pytest

from test_spatial_fixture_features import venue_history, fixture_report_context
from xdiyo_analytics.features import Heatmap, RegionMass, RollingMean, evaluate_features
from xdiyo_analytics.reporting import HeatmapReporter


@pytest.mark.parametrize('venue,expected', [('home',6),('away',25)])
def test_fixed_venue_spatial_history(venue_history,venue,expected):
    h,points=venue_history
    expr=RollingMean(RegionMass(Heatmap(grid_size=2,normalization='count',use_weights=True)),2,venue=venue)
    result=evaluate_features(h,{'region':expr},heatmaps=points)
    assert result.region.iloc[-1]==expected
    spec=result.attrs['spatial_features']['region']
    assert spec['operators'][-1]['venue']==venue
    assert spec['region']=='own_half'
    # Neither changed values nor coordinates in the current target can enter.
    altered=points.copy()
    altered.loc[altered.event_id.astype(object).eq(h.event_id.iloc[-1]),['x','y','weight']]=[99,99,999]
    np.testing.assert_array_equal(evaluate_features(h,{'region':expr},heatmaps=altered),result)


@pytest.mark.parametrize('venue',['home','away'])
def test_reporter_labels_fixed_venues(fixture_report_context,venue):
    context=fixture_report_context
    for spec in context.definitions['spatial_features'].values():
        spec['operators'][0]['venue']=venue
    report=HeatmapReporter(type='overall',partition='train').run(context)
    store=report.artifacts[0].data
    for item in store.manifest()['maps']:
        assert venue+' matches only' in item['label']
    assert store.panel_pair('0','grid')['panels'][0]['spec']['operators'][0]['venue']==venue
