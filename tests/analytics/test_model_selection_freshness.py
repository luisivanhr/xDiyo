import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from model_selection_samples import sample,plan,development,outer,FixedAdapter
from training_controls_samples import ScriptBackend
from xdiyo_analytics.selection import Candidate,ModelSelection
from xdiyo_analytics.training import EstimatorAdapter,IterativeAdapter,TrainingControl

@pytest.mark.parametrize('reuse',['adapter','estimator','preprocessor','pipeline_preprocessor','nested_pipeline','iterative_backend','iterative_estimator','iterative_preprocessor'])
def test_shared_fitted_state_is_rejected_across_candidates(reuse):
    class CountingScaler(StandardScaler):
        def fit(self,X,y=None,**kwargs):
            self.fit_calls=getattr(self,'fit_calls',0)+1
            return super().fit(X,y,**kwargs)
    data=sample();shared_model=FixedAdapter();shared_estimator=Ridge();shared_preprocessor=CountingScaler()
    shared_backend=ScriptBackend([1.])
    def make():
        if reuse=='adapter':return shared_model
        if reuse=='estimator':return EstimatorAdapter(shared_estimator)
        if reuse=='preprocessor':return FixedAdapter(preprocessor=shared_preprocessor)
        if reuse=='pipeline_preprocessor':return EstimatorAdapter(Pipeline([('scale',shared_preprocessor),('model',Ridge())]))
        if reuse=='nested_pipeline':return EstimatorAdapter(Pipeline([('prep',Pipeline([('scale',shared_preprocessor)])),('model',Ridge())]))
        def backend(seed):
            item=shared_backend if reuse=='iterative_backend' else ScriptBackend([1.])
            if reuse=='iterative_estimator':item.estimator=shared_estimator
            if reuse=='iterative_preprocessor':item.preprocessor=shared_preprocessor
            return item
        return IterativeAdapter(backend)
    control=TrainingControl(max_steps=1,restore_best=False) if reuse.startswith('iterative') else None
    candidates=[Candidate(name,make,control=control) for name in ('first','second')]
    with pytest.raises(ValueError,match='fresh|reuse'):
        ModelSelection(candidates,metrics='mse').run(data,plan(data,one=True),development_positions=development(data))
    if reuse in ('preprocessor','pipeline_preprocessor','nested_pipeline'):
        assert shared_preprocessor.fit_calls==1

@pytest.mark.parametrize('reuse',['adapter','preprocessor','pipeline_preprocessor'])
def test_outer_evaluation_cannot_reuse_inner_fitted_state(reuse):
    data=sample();shared=FixedAdapter();scaler=StandardScaler()
    def make():
        if reuse=='adapter':return shared
        if reuse=='preprocessor':return FixedAdapter(preprocessor=scaler)
        return EstimatorAdapter(Pipeline([('scale',scaler),('model',Ridge())]))
    result=ModelSelection([Candidate('x',make)],metrics='mse').run(data,plan(data,one=True),development_positions=development(data))
    with pytest.raises(ValueError,match='fresh|reuse'):result.evaluate(data,outer(data))


def test_freshness_walk_handles_explicit_self_reference_without_recursing():
    data=sample()
    class CycleAdapter(FixedAdapter):
        def __init__(self):super().__init__();self.preprocessor=self
        def fit(self,context):self.targets=list(context.y);return self
    result=ModelSelection([Candidate('cyclic owner reference',CycleAdapter)],metrics='mse').run(data,plan(data),development_positions=development(data))
    assert result.winner.record['status']=='complete'
