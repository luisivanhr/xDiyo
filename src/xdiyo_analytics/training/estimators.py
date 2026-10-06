"""Adapter for estimators and pipelines exposing fit/predict methods."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EstimatorAdapter:
    """Wrap one fresh fit(X, y) estimator, including a preprocessing pipeline.

    The default output is predict. Request ('predict', 'predict_proba') for a
    classifier's labels and class probabilities, or only ('predict_proba',).
    Single-target y becomes a Series; multiple targets remain a DataFrame.
    Probability columns are a (target, class) MultiIndex using fitted classes_.
    Multi-output classifiers may return a list of probability arrays/classes.

    Pass all fitted preprocessing inside the estimator/pipeline so its fit sees
    only training rows. No cloning, imputation, scaling, label encoding, metric
    calculation or optional dependency import occurs here. The runner's factory
    creates this adapter and its fresh estimator for every fold. Custom fitting
    arguments, ranker grouping and other output methods belong in custom adapters.
    """

    estimator: object
    prediction_methods: tuple[str, ...] = ("predict",)
    supports_sample_weight = True

    def configure_device(self, device):
        if device not in {"cpu", "auto"}:
            raise ValueError("EstimatorAdapter declares CPU support; wrap a CUDA-capable estimator with DeviceAdapter and its native configuration.")
        return "cpu"

    def configure_threads(self, threads):
        if callable(getattr(self.estimator, "get_params", None)) and callable(getattr(self.estimator, "set_params", None)):
            names = [name for name in self.estimator.get_params(deep=True) if name.split("__")[-1] == "n_jobs"]
            if names:
                self.estimator.set_params(**{name: threads for name in names})

    def fit(self, context):
        methods = tuple(self.prediction_methods)
        if not methods or len(set(methods)) != len(methods) or set(methods) - {"predict", "predict_proba", "decision_function"}:
            raise ValueError("prediction_methods must select predict, predict_proba and/or decision_function.")
        if 'decision_function' in methods and (context.y.shape[1] != 1 or context.y.iloc[:, 0].nunique() != 2):
            raise ValueError('prediction_methods: decision_function currently requires one binary target with both classes.')
        for method in methods:
            if not callable(getattr(self.estimator, method, None)):
                raise TypeError(f"Estimator does not expose {method}().")
        self.target_columns_ = tuple(context.y.columns)
        self.feature_columns_ = tuple(context.X.columns)
        y = context.y.iloc[:, 0] if len(self.target_columns_) == 1 else context.y
        from ..weighting import estimator_weight_kwargs
        self.estimator.fit(context.X, y, **estimator_weight_kwargs(self.estimator, context))
        from .inspection import estimator_history
        self.training_history_, self.training_summary_ = estimator_history(self.estimator, context.fold_id)

    def coefficient_table(self):
        """Return named fitted coefficients/intercepts when the estimator supports it."""
        from .inspection import estimator_coefficients
        return estimator_coefficients(self.estimator, self.feature_columns_, self.target_columns_)

    def predict(self, context):
        result = {}
        for method in self.prediction_methods:
            values = getattr(self.estimator, method)(context.X)
            if isinstance(values, (pd.Series, pd.DataFrame)) and not values.index.equals(context.X.index):
                raise ValueError("Pandas estimator outputs must retain the prediction input index/order.")
            if method == "predict":
                if isinstance(values, pd.DataFrame) and list(values.columns) != list(self.target_columns_):
                    raise ValueError("DataFrame predictions must retain the selected target column order.")
                array = np.asarray(values)
                if array.ndim == 1:
                    array = array[:, None]
                if array.shape != (len(context.X), len(self.target_columns_)):
                    raise ValueError("predict must return one value per row and selected target.")
                result[method] = pd.DataFrame(array, index=context.X.index,
                                              columns=list(self.target_columns_))
            elif method == 'decision_function':
                classes = getattr(self.estimator, 'classes_', None)
                array = np.asarray(values, dtype=float)
                if classes is None or len(classes) != 2 or array.shape != (len(context.X),):
                    raise ValueError('decision_function requires a binary fitted class order and one margin per row; multiclass/OVO is unsupported.')
                target = self.target_columns_[0]
                frame = pd.DataFrame({target: array}, index=context.X.index)
                frame.attrs['response_schema'] = {target: dict(classes=list(classes), positive_class=classes[1])}
                from .margins import validate_margins
                validate_margins(frame)
                result[method] = frame
            else:
                classes = getattr(self.estimator, "classes_", None)
                if classes is None:
                    raise ValueError("predict_proba requires fitted classes_ for column identities.")
                if len(self.target_columns_) == 1:
                    arrays, labels = [values], [classes]
                else:
                    if not isinstance(values, (list, tuple)) or not isinstance(classes, (list, tuple)):
                        raise ValueError("Multi-target probabilities require one array/classes_ entry per target.")
                    arrays, labels = values, classes
                if len(arrays) != len(self.target_columns_) or len(labels) != len(arrays):
                    raise ValueError("Probability outputs must match the selected target count.")
                frames = []
                for target, array, label in zip(self.target_columns_, arrays, labels):
                    if isinstance(array, pd.DataFrame):
                        if not array.index.equals(context.X.index) or list(array.columns) != list(label):
                            raise ValueError("Probability DataFrames must retain row order and fitted class order.")
                    array = np.asarray(array)
                    if array.shape != (len(context.X), len(label)):
                        raise ValueError("Probability dimensions must match rows and fitted classes_.")
                    columns = pd.MultiIndex.from_tuples([(target, item) for item in label],
                                                        names=["target", "class"])
                    frames.append(pd.DataFrame(array, index=context.X.index, columns=columns))
                result[method] = pd.concat(frames, axis=1)
        # Retain a native count distribution for downstream O/U decisions. A
        # mode prediction is never mistaken for its conditional mean.
        from .counts import NegativeBinomialRegressor
        from sklearn.linear_model import PoissonRegressor
        estimator = self.estimator
        inputs = context.X
        if hasattr(estimator, 'steps') and isinstance(estimator.steps[-1][1], (NegativeBinomialRegressor, PoissonRegressor)):
            if len(estimator.steps) > 1:
                inputs = estimator[:-1].transform(inputs)
            estimator = estimator.steps[-1][1]
        if isinstance(estimator, (NegativeBinomialRegressor, PoissonRegressor)):
            mean = estimator.predict_mean(inputs) if isinstance(estimator, NegativeBinomialRegressor) else estimator.predict(inputs)
            dispersion = estimator.dispersion_ if isinstance(estimator, NegativeBinomialRegressor) else 0.
            columns = pd.MultiIndex.from_product([self.target_columns_, ['mean', 'dispersion']],
                                                names=['target', 'parameter'])
            result['count_distribution'] = pd.DataFrame(
                np.column_stack([mean, np.full(len(mean), dispersion)]),
                index=context.X.index, columns=columns)
        return result
