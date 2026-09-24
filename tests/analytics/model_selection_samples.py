"""Small independent populations and adapters for selection boundary checks."""
from copy import deepcopy
import numpy as np
import pandas as pd
from split_samples import dataset,rows_for
from xdiyo_analytics.splits import Fold,SplitPlan
from xdiyo_analytics.selection import Candidate,TrialResult

def sample(*,layout='match',shuffle=False):
    data=dataset(16,layout=layout,shuffle=shuffle)
    x=data.metadata.case.to_numpy(dtype=float)
    data.X=pd.DataFrame({'signal':x,'wave':np.sin(x),'constant':1.},index=data.X.index)
    data.y=pd.DataFrame({'target':2*x+1+.2*np.cos(x)},index=data.X.index)
    return data

def plan(data,*,one=False):
    folds=[Fold(rows_for(data,[0,1,2,3,4])[::-1],rows_for(data,[6,7])[::-1],rows_for(data,[7]),{'part':'first'})]
    if not one:folds.append(Fold(rows_for(data,range(8)),rows_for(data,[10,11]),rows_for(data,[10,11]),{'part':'second'}))
    return SplitPlan(folds,len(data.X),np.arange(len(data.X))[::-1])

def development(data):return rows_for(data,range(12))[::-1]

def outer(data):return Fold(development(data),rows_for(data,range(12,16))[::-1],rows_for(data,[14,15]))

class FixedAdapter:
    def __init__(self,value=0.,*,slope=0.,fail=False,preprocessor=None):
        self.value=value;self.slope=slope;self.fail=fail;self.preprocessor=preprocessor
    def fit(self,context):
        if self.fail:raise RuntimeError('deliberate candidate fit failure')
        self.context=deepcopy(context);self.targets=list(context.y)
        if self.preprocessor is not None:self.preprocessor.fit(context.X)
        return self
    def fit_controlled(self,context,control):self.control=control;return self.fit(context)
    def predict(self,context):
        assert not hasattr(context,'y')
        self.prediction_context=deepcopy(context)
        values=self.value+self.slope*context.X['signal'].to_numpy()
        return {'predict':pd.DataFrame({target:values for target in self.targets},index=context.X.index)}

def candidate(name,value=1.,slope=2.,**kwargs):
    return Candidate(name,lambda:FixedAdapter(value,slope=slope),config={'value':value,'slope':slope},**kwargs)

def trial(name,metrics,*,complexity=None,status='complete',sample_hash='same-population'):
    records=[dict(metric=key,calculation=key,target='target',output='predict',value=value,
        direction='maximize' if key=='accuracy' else 'minimize',n=8,n_total=8,n_missing=0,status='ok',
        parameters='{}',sample_hash=sample_hash,type='overall',partition='score',fold_id=None,
        layout='match',scope_label='pooled') for key,value in metrics.items()]
    record=dict(run_id=name,name=name,config_hash='config-'+name,status=status,metrics=records)
    return TrialResult(name,candidate(name),record,complexity=complexity or {})
