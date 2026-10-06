"""Synthetic 200-team-row / 13-field coverage profile. No datasets or models.

PYTHONPATH=src python examples/profile_spatial_coverage.py --output profile.json
The scalar reference restores only the reviewed per-cell coverage read.
"""
import argparse
import cProfile
import inspect
import json
from pathlib import Path
import platform
import time
import types

import numpy as np
import pandas as pd
from pandas.core.indexing import _iLocIndexer
from xdiyo_analytics.features import SpatialPointSummary, RollingMean, evaluate_features
from xdiyo_analytics.features.point_geometry import POINT_FIELDS
from xdiyo_analytics.experiments.recovery import pack
import xdiyo_analytics.features.evaluation as evaluation


def population():
    rows=[];points=[]
    for game in range(100):
        for team,opponent,side in [(1,2,'home'),(2,1,'away')]:
            row=dict(event_id=1000+game,team_id=team,opponent_id=opponent,side=side,
                source_league='Synthetic',source_season='24_25',competition_id=1,season_id=1,
                status='finished',kickoff_at=pd.Timestamp('2024-01-01',tz='UTC')+pd.Timedelta(days=game))
            rows.append(row)
            for order in range(32):
                angle=2*np.pi*order/32
                points.append({**{k:row[k] for k in ('event_id','team_id','source_league','source_season')},
                    'kind':'player','x':50+30*np.cos(angle),'y':50+10*np.sin(angle),
                    'point_order':order,'raw_hash':'synthetic'})
    return pd.DataFrame(rows),pd.DataFrame(points)


def scalar_reference():
    code=inspect.getsource(evaluation)
    needle="'usable_output':bool(usable_output[row, col])"
    if code.count(needle)!=1:raise RuntimeError('Coverage code changed; review the benchmark reference.')
    code=code.replace(needle,"'usable_output':bool(np.isfinite(frame.iloc[row, col]))")
    code=code.replace('usable_output = np.isfinite(frame.to_numpy(dtype=float, na_value=np.nan))','pass')
    module=types.ModuleType('xdiyo_analytics.features.scalar_reference');module.__package__='xdiyo_analytics.features'
    exec(compile(code,'scalar_coverage_reference.py','exec'),module.__dict__)
    return module.evaluate_features


def measure(fn,h,p,features):
    scalar_reads=0
    original=_iLocIndexer.__getitem__
    def tracked(self,key):
        nonlocal scalar_reads
        if (isinstance(self.obj,pd.DataFrame) and 'point_map_coverage' in self.obj.attrs
            and isinstance(key,tuple) and len(key)==2 and all(isinstance(k,(int,np.integer)) for k in key)):
            scalar_reads+=1
        return original(self,key)
    profile=cProfile.Profile()
    _iLocIndexer.__getitem__=tracked
    try:
        start=time.perf_counter();profile.enable()
        result=fn(h,features,heatmaps=p,keyed=True)
        profile.disable();elapsed=time.perf_counter()-start
    finally:
        profile.disable();_iLocIndexer.__getitem__=original
    profile.create_stats()
    hot=[dict(file=key[0],line=key[1],function=key[2],calls=value[1],total_seconds=value[2],cumulative_seconds=value[3])
         for key,value in sorted(profile.stats.items(),key=lambda item:item[1][3],reverse=True)[:12]]
    return result,dict(elapsed_seconds=elapsed,spatial_scalar_reads=scalar_reads,hotspots=hot)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
    h,p=population()
    names=list(POINT_FIELDS)[:13]
    features={f:RollingMean(SpatialPointSummary(f),20) for f in names}
    before,reference=measure(scalar_reference(),h,p,features)
    after,optimized=measure(evaluate_features,h,p,features)
    pd.testing.assert_frame_equal(before,after)
    assert pack(before.attrs)==pack(after.attrs)
    result=dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,
        rows=len(h),fields=names,reference=reference,optimized=optimized,
        speedup=reference['elapsed_seconds']/optimized['elapsed_seconds'],complete_output_and_metadata_equal=True)
    if args.output:args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('reference','optimized')},indent=2))
    print('Reference:',reference['elapsed_seconds'],'seconds;',reference['spatial_scalar_reads'],'scalar reads')
    print('Optimized:',optimized['elapsed_seconds'],'seconds;',optimized['spatial_scalar_reads'],'scalar reads')


if __name__=='__main__':main()
