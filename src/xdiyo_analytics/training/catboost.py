"""Optional CatBoost estimators with sklearn's public fresh-clone protocol.

Native CatBoost get_params deep-copies mutable parameters, which fails sklearn's
constructor identity check. This hook reconstructs an unfitted native estimator
from public parameters. No fitting, prediction or model mathematics is changed.
Import this module explicitly; importing training does not require CatBoost.
"""
from catboost import CatBoostClassifier as _Classifier
from catboost import CatBoostRegressor as _Regressor
from catboost import CatBoostRanker as _Ranker


class _FreshClone:
    def __sklearn_clone__(self):
        return type(self)(**self.get_params(deep=False))


class CatBoostClassifier(_FreshClone, _Classifier):
    __doc__ = _Classifier.__doc__


class CatBoostRegressor(_FreshClone, _Regressor):
    __doc__ = _Regressor.__doc__


class CatBoostRanker(_FreshClone, _Ranker):
    __doc__ = _Ranker.__doc__
