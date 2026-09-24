"""Synthetic identity fixtures and a framework-independent mean adapter."""
from copy import deepcopy
import numpy as np
import pandas as pd
from split_samples import dataset, rows_for
from xdiyo_analytics.splits import Fold, SplitPlan


def sample(*, layout='match', shuffle=False):
    data=dataset(12,layout=layout,shuffle=shuffle)
    case=data.metadata.case.to_numpy(dtype=float)
    data.X=pd.DataFrame({'signal':case,'curve':case**2,'unused':-case},index=data.X.index)
    data.y=pd.DataFrame({'first':2*case+3,'second':10-case},index=data.y.index)
    data.metadata['settlement::first']='observed'
    data.definitions={'features':{'signal':{'note':['synthetic']}},'label':{'kind':'synthetic'}}
    return data


def plan(data):
    first=Fold(rows_for(data,[0,1,2,3])[::-1],rows_for(data,[6,7,8])[::-1],rows_for(data,[7]),{'nested':['zero']})
    second=Fold(rows_for(data,[0,1,2,3,4,5,6]),rows_for(data,[8,9,10]),np.array([],dtype=int),{'nested':['one']})
    return SplitPlan([first,second],len(data.X),np.arange(len(data.X)))


class MeanAdapter:
    def __init__(self, *, mutate=False):
        self.calls=[]
        self.mutate=mutate

    def fit(self, context):
        self.calls.append('fit')
        self.fit_context=deepcopy(context)
        self.means=context.y.mean()
        if self.mutate:
            context.X.iloc[0,0]=99999
            context.y.iloc[0,0]=-99999
            context.metadata.iloc[0,context.metadata.columns.get_loc('case')]=-1
            context.definitions['features']['signal']['note'].append('fit mutation')
            context.fold_metadata['nested'].append('fit mutation')

    def predict(self, context):
        self.calls.append('predict')
        self.predict_context=deepcopy(context)
        assert not hasattr(context,'y')
        self.returned={'predict':pd.DataFrame(np.tile(self.means.to_numpy(),(len(context.X),1)),
                                             index=context.X.index,columns=self.means.index)}
        if self.mutate:
            context.X.iloc[0,0]=-88888
            context.metadata.iloc[0,context.metadata.columns.get_loc('case')]=-2
            context.definitions['features']['signal']['note'].append('predict mutation')
            context.fold_metadata['nested'].append('predict mutation')
        return self.returned
