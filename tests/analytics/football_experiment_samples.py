from copy import deepcopy
from functools import partial
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from model_selection_samples import sample,plan,outer
from xdiyo_analytics.experiments import PreparedExperiment
from xdiyo_analytics.selection import Candidate
from xdiyo_analytics.splits import SplitPlan
from xdiyo_analytics.training import EstimatorAdapter
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.reporting import PerformanceReporter

def prepared(*,layout='match',holdout=False):
    data=sample(layout=layout,shuffle=True)
    splits=SplitPlan([outer(data)],len(data.X),np.arange(len(data.X))) if holdout else plan(data)
    return PreparedExperiment(data,splits,outputs={'observed_history':data.metadata.copy()},config={'synthetic':True})

def ridge_factory(alpha=.1):
    return EstimatorAdapter(make_pipeline(StandardScaler(),Ridge(alpha=alpha)))

def ridge(alpha=.1):
    return Candidate('Ridge',partial(ridge_factory,alpha),config={'model':'Ridge','alpha':alpha,'scaler':'StandardScaler'})

def post():
    return PostTrainingAnalysis({'Errors':PerformanceReporter(type='overall',partition='score',metrics=['mse','mae'])})
