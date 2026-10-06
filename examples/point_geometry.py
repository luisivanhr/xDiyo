"""Feature-only Phase A reproduction and partition benchmark; never fits a model.

PYTHONPATH=src python examples/point_geometry.py --data-root data/xDiyo_data
Run twice in fresh processes; compare feature_sha256 and timing_sha256.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import pandas as pd

from xdiyo_analytics.data import load_seasons
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import SpatialPointSummary, RollingMean, Stat, evaluate_features
from xdiyo_analytics.labels import MatchTotal, create_labels
from xdiyo_analytics.datasets import assemble_dataset
from xdiyo_analytics.ui.recipe import node


def feature_specs():
    return {f'activity_{field}_r20':RollingMean(
        SpatialPointSummary(field=field, kinds=('player',), side='for'),
        window=20, min_periods=1, venue='all') for field in ('mean_x','sd_x','sd_y')}


def recipe_features():
    """Assign this mapping to an existing UI recipe's features (or merge it in)."""
    return {f'activity_{field}_r20':node('features.RollingMean',
        source=node('features.SpatialPointSummary',field=field,kinds=['player'],side='for',min_points=1),
        window=20,min_periods=1,venue='all') for field in ('mean_x','sd_x','sd_y')}


def peak_memory_bytes():
    if os.name == 'nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)] + [
                (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
                    'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        api=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
        api.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if not api(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    import resource
    import sys
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',default='data/xDiyo_data')
    parser.add_argument('--league',default='Premier_League')
    parser.add_argument('--season',default='23_24')
    parser.add_argument('--output',type=Path,help='Optional benchmark JSON output')
    args=parser.parse_args()
    start=time.perf_counter()
    data=load_seasons(args.data_root,[args.season],leagues=[args.league],
        tables=('matches','statistics','heatmap_points'),verify_hashes=True)
    points=data['heatmap_points'].copy(deep=False)
    points.attrs['source']=data.provenance
    history=build_team_history(data)
    loaded=time.perf_counter()
    # Explicit retrospective research assumptions, not verified publication times.
    cutoffs=history.groupby(['competition_id','season_id','round'],dropna=False).kickoff_at.transform('min')-pd.Timedelta(hours=1)
    available=history.kickoff_at+pd.Timedelta(hours=3)
    features=evaluate_features(history,feature_specs(),heatmaps=points,
        cutoffs=cutoffs,available_at=available,keyed=True)
    label=create_labels(history,{'corners':MatchTotal(Stat('ALL','Match overview','cornerKicks'))})['corners']
    dataset=assemble_dataset(features,label,layout='match',drop_missing_targets=False)
    finished=time.perf_counter()
    audit=dataset.definitions['spatial_point_audit']
    maps=pd.DataFrame(audit['features']['activity_mean_x_r20']['maps'])
    paired=maps.groupby(['source_league','source_season','event_id']).valid_map.sum()
    payload=dict(league=args.league,season=args.season,points=len(points),team_rows=len(history),
        matches=len(dataset.X),columns=list(dataset.X),
        feature_sha256=hashlib.sha256(pd.util.hash_pandas_object(dataset.X,index=True).to_numpy().tobytes()).hexdigest(),
        timing_sha256=audit['timing_sha256'],source_sha256=audit['sources'][0]['source_sha256'],
        load_seconds=loaded-start,prepare_seconds=finished-loaded,
        peak_process_working_set_bytes=peak_memory_bytes(),
        both_team_maps=int((paired==2).sum()),map_status=maps.status.value_counts().to_dict(),
        missing_feature_cells=int(dataset.X.isna().sum().sum()),source=data.provenance,
        recipe_features=recipe_features())
    text=json.dumps(payload,indent=2,default=str)
    if args.output:
        args.output.write_text(text+'\n',encoding='utf8')
    print(text)


if __name__ == '__main__':
    main()
