"""Explicit child estimator and fold-local preparation for portable compositions."""
from dataclasses import dataclass


@dataclass
class Estimator:
    estimator: object
    preprocessors: tuple = ()
    prediction_methods: tuple = ('predict',)
    target_transformer: object = None
    supports_sample_weight = True

    def build(self):
        from sklearn.base import clone
        from sklearn.pipeline import Pipeline
        from ..training import EstimatorAdapter, TargetTransformAdapter
        model=clone(self.estimator)
        if self.preprocessors:
            model=Pipeline([(f'prepare_{i}',clone(p)) for i,p in enumerate(self.preprocessors)]+[('model',model)])
        adapter=EstimatorAdapter(model,self.prediction_methods)
        if self.target_transformer is not None:
            adapter=TargetTransformAdapter(adapter,clone(self.target_transformer))
        return adapter

    def fit(self,context):
        self.adapter_=self.build()
        self.adapter_.fit(context)
        self.training_summary_=getattr(self.adapter_,'training_summary_',{})

    def predict(self,context):
        return self.adapter_.predict(context)
