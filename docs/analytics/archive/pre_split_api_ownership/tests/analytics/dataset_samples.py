"""Identity-bearing feature values independent of label quantities."""
from copy import deepcopy
from dataclasses import replace
import pandas as pd
from label_samples import label_history

KEYS=['source_league','source_season','competition_id','season_id','event_id','team_id','side']

def inputs():
    history=label_history()
    features=pd.DataFrame({
        'form':pd.array([100+i for i in range(len(history))],dtype='Int64'),
        'conceded':pd.array([200+i for i in range(len(history))],dtype='int64[pyarrow]'),
    },index=history.index)
    features.index=pd.MultiIndex.from_frame(history[KEYS])
    features.attrs={'features':{'form':'synthetic form','conceded':'synthetic opponent statistic'},'nested':{'keep':[1]}}
    return history,features

def reorder_label(label,positions):
    return replace(label,y=label.y.iloc[positions].copy(),metadata=label.metadata.iloc[positions].copy(),
                   settlement=None if label.settlement is None else label.settlement.iloc[positions].copy(),
                   definition=deepcopy(label.definition))
