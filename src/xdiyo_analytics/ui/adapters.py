"""Concrete UI adapters for installed boosting estimators and their rankers."""

from dataclasses import dataclass, field, replace
import json

import numpy as np
import pandas as pd

from ..training.estimators import EstimatorAdapter


@dataclass
class BoostingAdapter(EstimatorAdapter):
    """LightGBM/XGBoost fit options, validation and whole-match ranking groups.

    Fit kwargs are forwarded to the native fit call. Validation is already split
    from training by the runner; preprocessing sees fitting rows only. CUDA
    requires a capable installed native build. auto conservatively uses CPU for
    LightGBM; XGBoost auto uses CUDA only when its build and optional Torch probe
    both confirm support. Explicit native failures are never hidden by fallback.
    """
    fit_kwargs: dict = field(default_factory=dict)
    sample_weight_column: str | None = None

    def _native(self):
        return self.estimator.steps[-1][1] if hasattr(self.estimator, 'steps') else self.estimator

    def configure_device(self, requested):
        model = self._native()
        family = type(model).__module__.split('.')[0]
        if family not in ('lightgbm', 'xgboost'):
            raise TypeError('BoostingAdapter requires a LightGBM or XGBoost estimator.')
        device = requested
        if requested == 'auto':
            device = 'cpu'
            if family == 'xgboost':
                import xgboost
                if xgboost.build_info().get('USE_CUDA'):
                    try:
                        from ..training import torch_devices
                        devices = torch_devices()
                        device = next((d for d in devices if d.startswith('cuda')), 'cpu')
                    except ImportError:
                        pass
        if device == 'cuda':
            device = 'cuda:0'
        if family == 'xgboost':
            model.set_params(device=device)
        else:
            model.set_params(device_type='cuda' if device.startswith('cuda') else 'cpu')
            if device.startswith('cuda'):
                model.set_params(gpu_device_id=int(device.split(':')[1]))
        self.device_ = device
        return device

    @staticmethod
    def _matrix(values, index):
        # Native libraries reject punctuation present in match-layout names.
        if isinstance(values, pd.DataFrame):
            values = values.copy()
            values.columns = [f'f{i}' for i in range(values.shape[1])]
            values.index = index
            return values
        if hasattr(values, 'toarray'):
            return values  # sparse matrices already carry no column names
        return pd.DataFrame(values, index=index, columns=[f'f{i}' for i in range(values.shape[1])])

    def fit(self, context):
        model = self._native()
        family = type(model).__module__.split('.')[0]
        if len(context.y.columns) != 1:
            raise ValueError('Native boosting adapters currently need one target.')
        if not self.prediction_methods or set(self.prediction_methods) - {'predict', 'predict_proba'}:
            raise ValueError('Choose predict and/or predict_proba.')
        self.target_columns_, self.feature_columns_ = tuple(context.y.columns), tuple(context.X.columns)
        self.native_model_ = model
        self.preprocessing_ = self.estimator[:-1] if hasattr(self.estimator, 'steps') and len(self.estimator.steps) > 1 else None
        y = context.y.iloc[:, 0]
        self.label_encoder_ = None
        if family == 'xgboost' and type(model).__name__.endswith('Classifier'):
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder_ = LabelEncoder().fit(y)
            y = pd.Series(self.label_encoder_.transform(y), index=y.index, name=y.name)
        X = self.preprocessing_.fit_transform(context.X, y) if self.preprocessing_ is not None else context.X
        X = self._matrix(X, context.X.index)
        kwargs = dict(self.fit_kwargs)
        if self.sample_weight_column is not None:
            kwargs['sample_weight'] = context.metadata[self.sample_weight_column].to_numpy()
        ranker = type(model).__name__.endswith('Ranker')
        def groups(meta):
            codes, _ = pd.factorize(pd.MultiIndex.from_frame(meta[list(context.match_columns)]), sort=False)
            order = np.argsort(codes, kind='stable')
            return order, np.bincount(codes)
        def take(values, order):
            return values.iloc[order] if hasattr(values, 'iloc') else values[order]
        if ranker:
            if context.layout != 'team_match':
                raise ValueError('A match ranker needs the team_match row layout.')
            order, sizes = groups(context.metadata)
            X, y = take(X, order), y.iloc[order]
            kwargs['group'] = sizes
            if 'sample_weight' in kwargs:
                if family == 'xgboost' and self.sample_weight_column is not None:
                    raise ValueError('XGBoost ranker weights are per group; supply native group weights explicitly.')
                if family == 'lightgbm':
                    kwargs['sample_weight'] = np.asarray(kwargs['sample_weight'])[order]
        if context.validation is not None:
            validation = context.validation
            VX = self.preprocessing_.transform(validation.X) if self.preprocessing_ is not None else validation.X
            VX = self._matrix(VX, validation.X.index)
            vy = validation.y.iloc[:, 0]
            if self.label_encoder_ is not None:
                vy = pd.Series(self.label_encoder_.transform(vy), index=vy.index, name=vy.name)
            if ranker:
                order, sizes = groups(validation.metadata)
                VX, vy = take(VX, order), vy.iloc[order]
                kwargs['eval_group'] = [sizes]
            kwargs['eval_set'] = [(VX, vy)]
        model.fit(X, y, **kwargs)
        if family == 'xgboost' and getattr(self, 'device_', 'cpu').startswith('cuda'):
            actual = json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
            if not actual.startswith('cuda'):
                raise RuntimeError('XGBoost fell back to CPU; the requested CUDA fit was not performed.')
        from ..training.inspection import estimator_history
        self.training_history_, self.training_summary_ = estimator_history(model, context.fold_id)

    def fit_controlled(self, context, control):
        if control is not None:
            raise ValueError('BoostingAdapter uses native estimator/fit stopping settings. Common TrainingControl requires IterativeAdapter or a custom controlled adapter.')
        self.fit(context)

    def predict(self, context):
        X = self.preprocessing_.transform(context.X) if self.preprocessing_ is not None else context.X
        X = self._matrix(X, context.X.index)
        # The public output contract is indexed even when preprocessing is sparse.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame.sparse.from_spmatrix(X, index=context.X.index,
                                                columns=[f'f{i}' for i in range(X.shape[1])])
        adapter = EstimatorAdapter(self.native_model_, self.prediction_methods)
        adapter.target_columns_ = self.target_columns_
        outputs = adapter.predict(replace(context, X=X))
        if self.label_encoder_ is not None:
            if 'predict' in outputs:
                frame = outputs['predict']
                outputs['predict'] = pd.DataFrame(self.label_encoder_.inverse_transform(frame.iloc[:, 0].to_numpy(dtype=int)),
                                                  index=frame.index, columns=frame.columns)
            if 'predict_proba' in outputs:
                frame = outputs['predict_proba']
                frame.columns = pd.MultiIndex.from_tuples([(target, self.label_encoder_.classes_[int(label)])
                                                          for target, label in frame.columns], names=frame.columns.names)
        return outputs
