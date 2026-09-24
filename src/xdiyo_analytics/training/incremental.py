"""Concrete iterative backend for sklearn-style partial_fit estimators."""

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..evaluation import Metric, evaluate_metrics
from .estimators import EstimatorAdapter
from .inspection import estimator_coefficients


@dataclass
class PartialFitBackend:
    """One partial_fit call per step, with train-only fitted preprocessing.

    Use inside IterativeAdapter(lambda seed: PartialFitBackend(...)). Construct a
    fresh estimator/preprocessor in that factory; pass seed to its random_state.
    loss is a Metric or registered name, required to produce one scalar for the
    selected target. This initial backend accepts one target; custom backends may
    support multiple outputs. It records train_loss and optional validation_loss.

    For classification, partial_fit_kwargs can explicitly supply classes;
    otherwise class labels are derived from fitting y only. Scheduling supports
    estimators configured with learning_rate='constant' and eta0, with no implicit
    override of another estimator's internal schedule. No batching is fabricated:
    each step passes the whole fitting population in the supplied row order.
    """
    estimator: object
    loss: object = "mse"
    preprocessor: object = None
    partial_fit_kwargs: object = None
    prediction_methods: tuple = ("predict",)

    def _transform(self, X, *, fit=False, y=None):
        if self.preprocessor is None:
            return X.copy()
        values = self.preprocessor.fit_transform(X, y) if fit else self.preprocessor.transform(X)
        if isinstance(values, pd.DataFrame):
            if not values.index.equals(X.index):
                raise ValueError("Preprocessing must preserve input row order/index.")
            return values.copy()
        if fit:
            if hasattr(self.preprocessor, "get_feature_names_out"):
                self.transformed_names_ = list(self.preprocessor.get_feature_names_out(X.columns))
            else:
                self.transformed_names_ = [f"transformed_{i}" for i in range(values.shape[1])]
        if hasattr(values, "toarray"):
            return pd.DataFrame.sparse.from_spmatrix(values, index=X.index, columns=self.transformed_names_)
        return pd.DataFrame(values, index=X.index, columns=self.transformed_names_)

    def initialize(self, context):
        from sklearn.base import is_classifier
        if not callable(getattr(self.estimator, "partial_fit", None)):
            raise TypeError("PartialFitBackend requires a partial_fit estimator.")
        if len(context.y.columns) != 1:
            raise ValueError("PartialFitBackend currently requires one selected target.")
        self.input_features_ = tuple(context.X.columns)
        self.target_columns_ = tuple(context.y.columns)
        self.context_ = replace(context, X=self._transform(context.X, fit=True, y=context.y.iloc[:, 0]))
        self.validation_ = (None if context.validation is None else
                            replace(context.validation, X=self._transform(context.validation.X)))
        self.kwargs_ = dict(self.partial_fit_kwargs or {})
        if is_classifier(self.estimator):
            self.kwargs_.setdefault("classes", np.unique(context.y.iloc[:, 0].to_numpy()))
        self.request_ = Metric(self.loss) if isinstance(self.loss, str) else self.loss
        if self.request_.target is not None and self.request_.target != self.target_columns_[0]:
            raise ValueError("The loss target must match the fitting target.")
        self.steps_ = 0

    def _outputs(self, context):
        # Loss evaluation may require probabilities even when final output only
        # requests labels. The loss registry resolves its named prediction output.
        from ..evaluation.metrics import METRICS
        needed = self.request_.output or ("predict_proba" if METRICS[self.request_.name].kind in {"probability", "uncertainty"} else "predict")
        adapter = EstimatorAdapter(self.estimator, tuple(dict.fromkeys((*self.prediction_methods, needed))))
        adapter.target_columns_ = self.target_columns_
        return adapter.predict(context)

    def step(self):
        self.estimator.partial_fit(self.context_.X, self.context_.y.iloc[:, 0], **self.kwargs_)
        self.steps_ += 1
        metrics = {}
        for name, context in (("train_loss", self.context_), ("validation_loss", self.validation_)):
            if context is not None:
                table = evaluate_metrics(context.y, self._outputs(context), [self.request_])
                if len(table) != 1:
                    raise ValueError("The monitored loss must produce one scalar.")
                metrics[name] = float(table.iloc[0].value)
        return metrics

    def predict(self, context):
        adapter = EstimatorAdapter(self.estimator, self.prediction_methods)
        adapter.target_columns_ = self.target_columns_
        return adapter.predict(replace(context, X=self._transform(context.X)))

    def snapshot(self):
        return deepcopy(self.estimator)

    def restore(self, state):
        self.estimator = deepcopy(state)

    def get_learning_rate(self):
        params = self.estimator.get_params()
        if params.get("learning_rate") != "constant" or "eta0" not in params:
            raise ValueError("Plateau scheduling requires an explicit constant learning_rate and eta0.")
        return float(params["eta0"])

    def set_learning_rate(self, value):
        self.get_learning_rate()
        self.estimator.set_params(eta0=value)

    def coefficient_table(self):
        return estimator_coefficients(self.estimator, self.context_.X.columns, self.target_columns_)
