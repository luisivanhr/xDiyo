"""Concrete UI adapters for installed boosting estimators and their rankers."""

from dataclasses import dataclass, field, replace
import json

import numpy as np
import pandas as pd

from ..training.estimators import EstimatorAdapter


@dataclass
class BoostingAdapter(EstimatorAdapter):
    """LightGBM/XGBoost/CatBoost fit options, validation and match ranking groups.

    Fit kwargs are forwarded to the native fit call. Validation is already split
    from training by the runner; preprocessing sees fitting rows only. CUDA
    requires a capable installed native build. auto conservatively uses CPU for
    LightGBM and CatBoost; XGBoost auto uses CUDA only when its build and optional Torch probe
    both confirm support. Explicit native failures are never hidden by fallback.
    """
    fit_kwargs: dict = field(default_factory=dict)
    sample_weight_column: str | None = None
    exact_counts: bool = False

    def _native(self):
        return self.estimator.steps[-1][1] if hasattr(self.estimator, 'steps') else self.estimator

    def _family(self):
        # Also recognize the native clone-compatible CatBoost subclasses.
        for cls in type(self._native()).__mro__:
            family = cls.__module__.split('.')[0]
            if family in ('catboost', 'lightgbm', 'xgboost'):
                return family
        return None

    def configure_device(self, requested):
        model = self._native()
        family = self._family()
        if family not in ('lightgbm', 'xgboost', 'catboost'):
            raise TypeError('BoostingAdapter requires a LightGBM, XGBoost or CatBoost estimator.')
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
        elif family == 'catboost':
            model.set_params(task_type='GPU' if device.startswith('cuda') else 'CPU')
            if device.startswith('cuda'):
                model.set_params(devices=device.split(':')[1])
        else:
            model.set_params(device_type='cuda' if device.startswith('cuda') else 'cpu')
            if device.startswith('cuda'):
                model.set_params(gpu_device_id=int(device.split(':')[1]))
        self.device_ = device
        return device

    def configure_threads(self, threads):
        if self._family() == 'catboost':
            # CatBoost get_params only reports explicitly supplied parameters.
            self._native().set_params(thread_count=threads)
        else:
            super().configure_threads(threads)

    @staticmethod
    def _matrix(values, index, *, preserve_names=False):
        # Native libraries reject punctuation present in match-layout names.
        if isinstance(values, pd.DataFrame):
            values = values.copy()
            if not (preserve_names and values.columns.is_unique and
                    all(isinstance(c, str) for c in values.columns)):
                values.columns = [f'f{i}' for i in range(values.shape[1])]
            values.index = index
            return values
        if hasattr(values, 'toarray'):
            return values  # sparse matrices already carry no column names
        return pd.DataFrame(values, index=index, columns=[f'f{i}' for i in range(values.shape[1])])

    def fit(self, context):
        model = self._native()
        family = self._family()
        if len(context.y.columns) != 1:
            raise ValueError('Native boosting adapters currently need one target.')
        if not self.prediction_methods or set(self.prediction_methods) - {'predict', 'predict_proba'}:
            raise ValueError('Choose predict and/or predict_proba.')
        self.target_columns_, self.feature_columns_ = tuple(context.y.columns), tuple(context.X.columns)
        self.native_model_ = model
        self.preprocessing_ = self.estimator[:-1] if hasattr(self.estimator, 'steps') and len(self.estimator.steps) > 1 else None
        y = context.y.iloc[:, 0]
        if self.exact_counts:
            if not type(model).__name__.endswith('Classifier'):
                raise ValueError('exact_counts requires a classifier.')
            values = pd.to_numeric(y, errors='raise').to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
                raise ValueError('Exact count classification requires finite nonnegative integer labels.')
            if family == 'xgboost':
                model.set_params(objective='multi:softprob', num_class=int(y.nunique()))
        self.label_encoder_ = None
        if family == 'xgboost' and type(model).__name__.endswith('Classifier'):
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder_ = LabelEncoder().fit(y)
            y = pd.Series(self.label_encoder_.transform(y), index=y.index, name=y.name)
        X = self.preprocessing_.fit_transform(context.X, y) if self.preprocessing_ is not None else context.X
        X = self._matrix(X, context.X.index, preserve_names=family == 'catboost')
        kwargs = dict(self.fit_kwargs)
        if family == 'catboost':
            # Native defaults otherwise create shared catboost_info directories
            # across fold workers. An explicit user setting remains respected.
            if 'allow_writing_files' not in model.get_params():
                model.set_params(allow_writing_files=False)
            reserved = {'group', 'group_id', 'eval_group', 'eval_set'} & kwargs.keys()
            if reserved:
                raise ValueError(f'CatBoost grouping/validation is owned by the runner: {sorted(reserved)}.')
        if context.sample_weight is not None:
            from ..weighting import estimator_weight_kwargs
            if 'sample_weight' in kwargs or self.sample_weight_column is not None:
                raise ValueError('Choose common class weights or native/metadata sample weights, not both.')
            kwargs.update(estimator_weight_kwargs(model, context, pipeline=False))
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
            if family == 'catboost':
                if {'pairs', 'pairs_weight', 'graph'} & kwargs.keys():
                    raise ValueError('Explicit CatBoost pairs/graph require a custom adapter; match grouping reorders rows.')
                kwargs['group_id'] = np.repeat(np.arange(len(sizes)), sizes)
                for name in ('baseline', 'subgroup_id', 'group_weight'):
                    if name in kwargs:
                        kwargs[name] = np.asarray(kwargs[name])[order]
            else:
                kwargs['group'] = sizes
            if 'sample_weight' in kwargs:
                if family == 'xgboost' and self.sample_weight_column is not None:
                    raise ValueError('XGBoost ranker weights are per group; supply native group weights explicitly.')
                if family in ('lightgbm', 'catboost'):
                    kwargs['sample_weight'] = np.asarray(kwargs['sample_weight'])[order]
        if context.validation is not None:
            validation = context.validation
            VX = self.preprocessing_.transform(validation.X) if self.preprocessing_ is not None else validation.X
            VX = self._matrix(VX, validation.X.index, preserve_names=family == 'catboost')
            vy = validation.y.iloc[:, 0]
            if self.label_encoder_ is not None:
                vy = pd.Series(self.label_encoder_.transform(vy), index=vy.index, name=vy.name)
            if ranker:
                order, sizes = groups(validation.metadata)
                VX, vy = take(VX, order), vy.iloc[order]
                if family != 'catboost':
                    kwargs['eval_group'] = [sizes]
            if family == 'catboost':
                from catboost import Pool
                from ..weighting import estimator_weight_kwargs
                weights = estimator_weight_kwargs(model, validation, pipeline=False).get('sample_weight')
                if self.sample_weight_column is not None:
                    weights = validation.metadata[self.sample_weight_column].to_numpy()
                if weights is not None and ranker:
                    weights = np.asarray(weights)[order]
                feature_options = {name: kwargs.get(name, model.get_params().get(name))
                                   for name in ('cat_features', 'text_features', 'embedding_features')}
                kwargs['eval_set'] = Pool(VX, vy, weight=weights,
                    group_id=np.repeat(np.arange(len(sizes)), sizes) if ranker else None,
                    **feature_options)
            else:
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
        X = self._matrix(X, context.X.index,
                         preserve_names=self._family() == 'catboost')
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
