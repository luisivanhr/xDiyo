"""Independent output and lifetime checks for the library performance branch."""
import gc
import json
import weakref
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics._identity import IdentityRegistry
from xdiyo_analytics.training import TrainingControl, run_iterations
from training_controls_samples import ScriptBackend, fit_context


def test_identity_registry_weak_unhashable_and_strong_fallback():
    class Object:
        __hash__ = None
        def __eq__(self, other):
            raise AssertionError('Identity tracking must never use equality')
    registry = IdentityRegistry()
    assert None not in registry
    obj = Object(); registry.add(obj); reference = weakref.ref(obj)
    assert obj in registry
    del obj; gc.collect()
    assert reference() is None and not registry._weak
    values = []; registry.add(values)
    assert values in registry and [] not in registry
    # High turnover must not confuse recycled IDs with a previous object.
    for _ in range(2000):
        obj = Object()
        assert obj not in registry
        registry.add(obj)
        del obj
    assert not registry._weak


def test_losing_restart_backends_collect_before_next_attempt():
    references = []
    def factory(seed):
        gc.collect()
        assert sum(ref() is not None for ref in references) <= 1
        backend = ScriptBackend([float(seed+1)])
        references.append(weakref.ref(backend))
        return backend
    winner, _, summary = run_iterations(factory, fit_context(), TrainingControl(max_steps=1, restarts=8))
    gc.collect()
    assert summary['selected_attempt'] == 0 and references[0]() is winner
    assert all(ref() is None for ref in references[1:])


def test_classifier_extracts_once_per_source_fold(monkeypatch):
    import xdiyo_analytics.reporting.classifier_diagnostics as module
    n=100; columns=pd.MultiIndex.from_tuples([('count',0),('count',1)])
    index=pd.MultiIndex.from_tuples([(0,i) for i in range(n)],names=['fold_id','row_position'])
    source=pd.DataFrame(np.tile([.2,.8],(n,1)),columns=columns)
    probability=source.copy();probability.index=index
    context=SimpleNamespace(y=pd.DataFrame({'count':np.ones(n,dtype=int)},index=index),
        predictions={'predict':pd.DataFrame({'count':np.ones(n,dtype=int)},index=index),'predict_proba':probability},
        fold_results={0:SimpleNamespace(predictions={'predict_proba':source},training_summary={})},pooling='occurrences')
    original=module._probabilities;calls=[]
    def count(*args):calls.append(len(args[0]));return original(*args)
    monkeypatch.setattr(module,'_probabilities',count)
    result=module.CountClassificationReporter(type='overall',partition='test').run(context)
    assert calls == [n,n]
    assert result.tables['metrics'].iloc[0].probability_rows==n


@pytest.mark.parametrize('mode', ['mean','mode'])
def test_count_response_reuses_mean_and_custom_transform_falls_back(mode):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xdiyo_analytics.training.estimators import EstimatorAdapter
    from xdiyo_analytics.training.counts import NegativeBinomialRegressor
    X=pd.DataFrame({'x':np.linspace(-1,1,30)})
    y=pd.Series(np.arange(30)%5)
    pipeline=Pipeline([('scale',StandardScaler()),('model',NegativeBinomialRegressor(prediction=mode))]).fit(X,y)
    adapter=EstimatorAdapter(pipeline);adapter.target_columns_=('count',)
    point=pipeline.predict(X);mean=pipeline[-1].predict_mean(pipeline[:-1].transform(X))
    result=adapter.predict(SimpleNamespace(X=X))
    np.testing.assert_array_equal(result['predict']['count'],point)
    np.testing.assert_array_equal(result['count_distribution'][('count','mean')],mean)
    class Stateful(StandardScaler):
        def transform(self,X,*args,**kwargs):
            self.calls=getattr(self,'calls',0)+1
            return super().transform(X,*args,**kwargs)
    custom=Pipeline([('scale',Stateful()),('model',NegativeBinomialRegressor(prediction=mode))]).fit(X,y)
    custom[0].calls=0;adapter.estimator=custom;adapter.predict(SimpleNamespace(X=X))
    assert custom[0].calls==2


def test_selection_guard_does_not_own_discarded_models():
    from xdiyo_analytics.selection.core import _FreshModels
    class Model:pass
    guard=_FreshModels();factory=guard.factory(SimpleNamespace(model_factory=Model))
    model=factory();reference=weakref.ref(model);del model;gc.collect()
    assert reference() is None
    live=Model();reuse=guard.factory(SimpleNamespace(model_factory=lambda:live));reuse()
    with pytest.raises(ValueError,match='fresh'):reuse()
