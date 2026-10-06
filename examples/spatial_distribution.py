"""Feature-only Phase B preparation; no model fitting or parameter selection.

PYTHONPATH=src python examples/spatial_distribution.py --output phase_b.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from point_geometry import peak_memory_bytes
from xdiyo_analytics.data import load_seasons
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import Heatmap, RegionMass, SpatialEntropy, RollingMean, Stat, evaluate_features
from xdiyo_analytics.datasets import assemble_dataset
from xdiyo_analytics.labels import MatchTotal, create_labels
from xdiyo_analytics.ui.recipe import node


def feature_specs():
    source=Heatmap(grid_size=6,method='grid',normalization='mass',kinds=('player',),orientation='team')
    return dict(activity_attacking_third_r20=RollingMean(RegionMass(source,'attacking_third'),20),
                activity_central_channel_r20=RollingMean(RegionMass(source,'central_channel'),20),
                activity_entropy_r20=RollingMean(SpatialEntropy(source),20))


def recipe_features():
    source=node('features.Heatmap',grid_size=6,method='grid',normalization='mass',
                kinds=['player'],orientation='team')
    reducers={
        'activity_attacking_third_r20':node('features.RegionMass',source=source,region='attacking_third'),
        'activity_central_channel_r20':node('features.RegionMass',source=source,region='central_channel'),
        'activity_entropy_r20':node('features.SpatialEntropy',source=source,normalized=True),
    }
    return {name:node('features.RollingMean',source=reducer,window=20,min_periods=1,venue='all')
            for name,reducer in reducers.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',default='data/xDiyo_data')
    parser.add_argument('--league',default='Premier_League')
    parser.add_argument('--season',default='23_24')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    start=time.perf_counter()
    data=load_seasons(args.data_root,[args.season],leagues=[args.league],
                     tables=('matches','statistics','heatmap_points'),verify_hashes=True)
    points=data['heatmap_points'].copy(deep=False);points.attrs['source']=data.provenance
    history=build_team_history(data)
    loaded=time.perf_counter()
    # Retrospective research assumptions, not measured source publication times.
    cutoffs=history.groupby(['competition_id','season_id','round'],dropna=False).kickoff_at.transform('min')-pd.Timedelta(hours=1)
    available=history.kickoff_at+pd.Timedelta(hours=3)
    values=evaluate_features(history,feature_specs(),heatmaps=points,cutoffs=cutoffs,available_at=available,keyed=True)
    label=create_labels(history,{'corners':MatchTotal(Stat('ALL','Match overview','cornerKicks'))})['corners']
    dataset=assemble_dataset(values,label,layout='match',drop_missing_targets=False)
    prepared=time.perf_counter()
    audit=dataset.definitions['spatial_distribution_audit']
    result=dict(league=args.league,season=args.season,points=len(points),matches=len(dataset.X),columns=list(dataset.X),
        feature_sha256=hashlib.sha256(pd.util.hash_pandas_object(dataset.X,index=True).to_numpy().tobytes()).hexdigest(),
        timing_sha256=audit['timing_sha256'],sources=audit['sources'],
        load_seconds=loaded-start,prepare_seconds=prepared-loaded,
        peak_process_working_set_bytes=peak_memory_bytes(),missing_feature_cells=int(dataset.X.isna().sum().sum()),
        recipe_features=recipe_features())
    text=json.dumps(result,indent=2,default=str)
    if args.output:args.output.write_text(text+'\n',encoding='utf8')
    print(text)


if __name__=='__main__':
    main()
