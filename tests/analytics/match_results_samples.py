"""Retained synthetic observations; reporting must never fit or predict."""
from copy import deepcopy
import numpy as np
import pandas as pd
from xdiyo_analytics.reporting import PostTrainingContext
from xdiyo_analytics.training import FoldResult, TrainingResult
from post_training_samples import RecordedModel

HOME=2**63+11
AWAY=2**63+13
MATCH=('competition_id','season_id','event_id')

def fixture_context(layout='match', n=4):
    cases=np.repeat(np.arange(n),2) if layout=='team_match' else np.arange(n)
    rows=len(cases);index=pd.MultiIndex.from_arrays([[4]*rows,np.arange(rows)],names=['fold_id','row_position'])
    meta=pd.DataFrame({'competition_id':17,'season_id':2025,
        'event_id':pd.array([2**63+101+int(i) for i in cases],dtype='uint64[pyarrow]'),
        'round':[float(i//2+1) if i<n-1 else np.nan for i in cases],
        'stage':'Regular','home_name':'Home club','away_name':'Away club'},index=index)
    if layout=='match':
        meta['home_id']=pd.array([HOME]*rows,dtype='uint64[pyarrow]')
        meta['away_id']=pd.array([AWAY]*rows,dtype='uint64[pyarrow]')
        identity=MATCH
    else:
        meta['side']=['home','away']*n
        meta['team_id']=pd.array([HOME,AWAY]*n,dtype='uint64[pyarrow]')
        meta['opponent_id']=pd.array([AWAY,HOME]*n,dtype='uint64[pyarrow]')
        meta['team_name']=['Home club','Away club']*n
        meta['opponent_name']=['Away club','Home club']*n
        identity=(*MATCH,'team_id')
    truth=pd.DataFrame({'count':cases.astype(float)+1},index=index)
    prediction=truth.copy();prediction['count']+=np.resize([0.,.25,-.5,.1],rows)
    return PostTrainingContext(truth,{'predict':prediction},meta,layout,identity,MATCH,'overall','test',None,'occurrences',definitions={})

def result_from_context(context, *, repeat=False, score=None):
    folds=[]
    for fold_id in ([4,9] if repeat else [4]):
        y=context.y.copy();meta=context.metadata.copy()
        y.index=meta.index=pd.Index(context.y.index.get_level_values('row_position'),name='row_position')
        outputs={}
        for name,value in context.predictions.items():
            value=value.copy();value.index=y.index;outputs[name]=value
        positions=y.index.to_numpy()
        folds.append(FoldResult(fold_id,RecordedModel(),np.array([],dtype=int),positions.copy(),
            positions.copy() if score is None else np.asarray(score,dtype=int),('synthetic',),tuple(y),outputs,y,meta))
    return TrainingResult(folds,context.layout,context.identity_columns,context.match_columns,'total',deepcopy(context.definitions))
