"""Optional geometry: exact moments, robust widths, invariant axes and UI."""
import numpy as np
import pytest
from transition_samples import history_from_games
from test_point_geometry import maps_for, observed
from xdiyo_analytics.features import SpatialPointSummary, Lag, evaluate_features
from xdiyo_analytics.features.point_geometry import POINT_FIELDS
from xdiyo_analytics.ui import catalog_for_ui


@pytest.mark.parametrize('xy', [ [(0,0),(100,100)], [(20,80),(40,30),(40,30),(60,10)],
                              [(0,0),(100,0),(100,100),(0,100)], [(20,40)] ])
def test_exact_extended_geometry(xy):
    h=history_from_games([{}]); p=maps_for(h,xy); a=np.array(xy,float)
    centered=a-a.mean(axis=0); cov=centered.T@centered/len(a)
    minor,major=np.maximum(np.linalg.eigvalsh(cov),0)
    expected=dict(mean_y=a[:,1].mean(), depth_80=np.diff(np.quantile(a[:,0],[.1,.9],method='linear'))[0],
        width_80=np.diff(np.quantile(a[:,1],[.1,.9],method='linear'))[0],cov_xy=cov[0,1],
        corr_xy=cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]) if cov[0,0]*cov[1,1]>0 else np.nan,
        major_variance=major,minor_variance=minor,major_spread=np.sqrt(major),minor_spread=np.sqrt(minor),
        anisotropy=(major-minor)/(major+minor) if major+minor else np.nan,
        axis_angle=(.5*np.arctan2(2*cov[0,1],cov[0,0]-cov[1,1]))%np.pi if major>minor else np.nan)
    for field,value in expected.items():
        np.testing.assert_allclose(observed(h,p,field=field),value,atol=1e-12,equal_nan=True)


@pytest.mark.parametrize('field', list(POINT_FIELDS))
def test_opponent_rotation_and_target_venue_invariance(field):
    h=history_from_games([{}, {}]);p=maps_for(h,[(10,20),(20,70),(60,80)])
    both=observed(h,p,field=field,side='both')
    a,b=both.iloc[0]
    assert b==pytest.approx(100-a if field in ('mean_x','mean_y') else a)
    lag=evaluate_features(h,{'geometry':Lag(SpatialPointSummary(field,side='both'))},heatmaps=p)
    np.testing.assert_allclose(lag.iloc[2:].to_numpy(),both.iloc[2:].to_numpy())


def test_rank_one_zero_spread_isotropy_and_lateral_axis():
    h=history_from_games([{}])
    for xy,angle in [([(10,20),(10,80)],np.pi/2), ([(10,40),(90,40)],0)]:
        p=maps_for(h,xy)
        assert observed(h,p,field='axis_angle').iloc[0,0]==pytest.approx(angle)
        assert observed(h,p,field='corr_xy').isna().all().all()
        assert observed(h,p,field='anisotropy').iloc[0,0]==1
    p=maps_for(h,[(20,20)])
    assert observed(h,p,field='axis_angle').isna().all().all()
    assert observed(h,p,field='anisotropy').isna().all().all()
    p=maps_for(h,[(0,0),(100,0),(100,100),(0,100)])
    assert observed(h,p,field='axis_angle').isna().all().all()
    assert observed(h,p,field='anisotropy').iloc[0,0]==0


def test_ui_fields_and_roundtrip():
    catalog=catalog_for_ui()
    fields=next(c['fields'] for c in catalog.schema() if c['id']=='features.SpatialPointSummary')
    assert {v['value'] for v in next(f['choices'] for f in fields if f['name']=='field')}==set(POINT_FIELDS)
    for field in POINT_FIELDS:
        value=SpatialPointSummary(field)
        assert catalog.build(catalog.encode(value))==value


def test_repeated_decimal_points_have_unidentifiable_axes():
    h=history_from_games([{}]); p=maps_for(h,[(.1,3.4)]*7)
    for field in ('major_variance','minor_variance','major_spread','minor_spread','cov_xy'):
        assert observed(h,p,field=field).iloc[0,0]==0
    for field in ('axis_angle','corr_xy','anisotropy'):
        assert observed(h,p,field=field).isna().all().all()
