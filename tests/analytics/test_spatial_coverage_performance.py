"""Differential coverage oracle and guard against per-cell DataFrame iloc."""
import inspect
import types
import numpy as np
import pandas as pd
from pandas.core.indexing import _iLocIndexer

from transition_samples import history_from_games
from test_point_geometry import maps_for
from xdiyo_analytics.features import (SpatialPointSummary as Point, RollingMean, Product, Constant, Ratio, evaluate_features)
import xdiyo_analytics.features.evaluation as module
from xdiyo_analytics.experiments.recovery import pack


def scalar_reference_evaluator():
    """Restore only the reviewed scalar coverage read, leaving all else equal."""
    code=inspect.getsource(module)
    old="'usable_output':bool(usable_output[row, col])"
    assert code.count(old)==1
    code=code.replace(old,"'usable_output':bool(np.isfinite(frame.iloc[row, col]))")
    code=code.replace('usable_output = np.isfinite(frame.to_numpy(dtype=float, na_value=np.nan))','pass')
    reference=types.ModuleType('xdiyo_analytics.features.scalar_coverage_reference')
    reference.__package__='xdiyo_analytics.features'
    exec(compile(code,'scalar_coverage_reference.py','exec'),reference.__dict__)
    return reference.evaluate_features


def test_no_spatial_scalar_reads_complete_output_and_metadata_equal(monkeypatch):
    h=history_from_games([{}, {}, {}, {}, {}]);p=maps_for(h,[(10,30),(60,70)])
    events=h.event_id.unique()
    p=p[p.event_id!=events[1]]
    p.loc[p.event_id==events[2],['x','y']]=20 # Undefined axis, valid raw map.
    p=p[~((p.event_id==events[3]) & (p.kind=='player'))] # Empty selected kind.
    features={
        'c':RollingMean(Point('axis_cos2'),2),'s':RollingMean(Point('axis_sin2'),2),
        'both':RollingMean(Point(side='both'),2),
        'zero':Ratio(RollingMean(Point(),2),Constant(0)),
        'fallback':Ratio(RollingMean(Point(),2),Constant(0),zero_value=7),
        'overflow':Product(RollingMean(Point(),2),Constant(1e308)),
    }
    expected=scalar_reference_evaluator()(h,features,heatmaps=p,keyed=True)
    original=_iLocIndexer.__getitem__
    def guard(self,key):
        if (isinstance(self.obj,pd.DataFrame) and 'point_map_coverage' in self.obj.attrs
            and isinstance(key,tuple) and len(key)==2
            and all(isinstance(k,(int,np.integer)) for k in key)):
            raise AssertionError('Per-cell DataFrame scalar access with spatial metadata')
        return original(self,key)
    monkeypatch.setattr(_iLocIndexer,'__getitem__',guard)
    actual=evaluate_features(h,features,heatmaps=p,keyed=True)
    pd.testing.assert_frame_equal(actual,expected)
    assert pack(actual.attrs)==pack(expected.attrs) # Includes NaN audit values and every record/order.
