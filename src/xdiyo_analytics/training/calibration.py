"""Reusable probability calibration, fitted independently of the base model."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, softmax


def _probabilities(frame):
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels != 2:
        raise ValueError("Probabilities need a DataFrame with (target, class) columns.")
    if not frame.columns.is_unique or not len(frame.columns):
        raise ValueError("Probability columns must be distinct and nonempty.")
    for target in frame.columns.get_level_values(0).unique():
        values = frame.xs(target, axis=1, level=0).to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Class probabilities must be finite and between zero and one.")
        if not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0):
            raise ValueError("Class probabilities must sum to one for each target and row.")
    return frame


@dataclass
class ProbabilityCalibrator:
    """Calibrate complete class distributions using disjoint held-out predictions.

    method is temperature (one temperature per target), sigmoid (one-vs-rest
    logistic calibration), or isotonic (one-vs-rest monotone calibration).
    fit/transform also accept independently prepared out-of-fold predictions.
    The pipeline reserves the latest fraction of training kickoff batches;
    it never uses test labels or the base model's early-stopping population.
    Class weighting is not applied to the calibration labels.

    response_method='decision_function' instead fits one Platt-smoothed sigmoid
    to binary SVC margins. It requires explicit label availability (column or
    declared kickoff-plus-delay proxy), preserves prediction groups, and purges
    labels not known at calibration/issue boundaries. See temporal_svc_calibration.md.
    Existing serialized instances without response_method use predict_proba.
    For predict_proba, reviewed completion bounds also activate chronological
    label purging. With no explicit availability policy, its legacy kickoff
    proxy and zero prediction lead are retained. Direct partition/refit then
    needs issue_at (or fold fit_at); the runner supplies the outer boundary.
    """

    method: str = "temperature"
    fraction: float = 0.2
    time_column: str = "kickoff_at"
    update_predict: bool = True
    response_method: str = 'predict_proba'
    availability_column: object = None
    availability_delay: object = None
    cutoff_column: object = None
    prediction_group_by: tuple = ()
    prediction_lead: str = '1h'
    issue_at: object = None
    min_calibration_rows: int = 10
    min_calibration_per_class: int = 2

    def __post_init__(self):
        if self.response_method not in {'predict_proba', 'decision_function'}:
            raise ValueError('Calibration response_method must be predict_proba or decision_function.')
        if self.response_method == 'decision_function' and self.method != 'sigmoid':
            raise ValueError('Decision-margin calibration supports only binary sigmoid calibration.')
        if self.availability_column is not None and self.availability_delay is not None:
            raise ValueError('Choose availability_column or availability_delay, not both.')
        for name in ('min_calibration_rows', 'min_calibration_per_class'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if self.method not in {"temperature", "sigmoid", "isotonic"}:
            raise ValueError("Calibration method must be temperature, sigmoid or isotonic.")
        if isinstance(self.fraction, bool) or not 0 < self.fraction < 1:
            raise ValueError("Calibration fraction must lie strictly between zero and one.")

    def select(self, dataset, train_positions):
        from .calibration_split import guarded_policy
        if guarded_policy(self, dataset) is not None:
            return self.partition(dataset, train_positions)[1]
        from .control import ValidationTail
        return ValidationTail(self.fraction, self.time_column).select(dataset, train_positions)

    def partition(self, dataset, train_positions, fold_metadata=None):
        from .calibration_split import guarded_policy, temporal_partition
        policy = guarded_policy(self, dataset)
        if policy is not None:
            return temporal_partition(policy, dataset, train_positions, fold_metadata)
        reserved = self.select(dataset, train_positions)
        train = np.asarray(train_positions)
        return train[~np.isin(train, reserved)], reserved, {}

    def validate_model(self, model, fitting_labels, calibration_labels):
        """Fail incompatible margin configurations before preprocessing/model fitting."""
        if self.response_method != 'decision_function':
            return
        from sklearn.svm import SVC
        from .estimators import EstimatorAdapter
        if not isinstance(model, EstimatorAdapter):
            raise ValueError('Margin calibration requires an EstimatorAdapter with binary SVC.')
        estimator = model.estimator
        native = estimator.steps[-1][1] if hasattr(estimator, 'steps') else estimator
        if not isinstance(native, SVC) or native.probability not in (False, 'deprecated'):
            raise ValueError('Margin calibration requires SVC(probability=False); no internal probability CV is allowed.')
        if 'decision_function' not in model.prediction_methods or 'predict_proba' in model.prediction_methods:
            raise ValueError('Margin calibration requires decision_function, without base predict_proba.')
        if len(fitting_labels.columns) != 1 or fitting_labels.iloc[:, 0].nunique() != 2:
            raise ValueError('Margin calibration requires one binary target with both classes in base fitting rows.')
        self._check_margin_labels(calibration_labels, fitting_labels.columns[0],
                                  pd.Index(fitting_labels.iloc[:, 0].unique()))

    def _check_margin_labels(self, y, target, classes):
        if target not in y or y[target].isna().any() or not y[target].isin(classes).all():
            raise ValueError('Margin calibration labels must be nonmissing known fitted classes.')
        counts = y[target].value_counts().reindex(classes, fill_value=0)
        if len(y) < self.min_calibration_rows or (counts < self.min_calibration_per_class).any():
            raise ValueError('Insufficient calibration rows/per-class observations; both fitted classes are required.')

    def _fit_margins(self, margins, y):
        from .margins import validate_margins, fit_sigmoid
        schema = validate_margins(margins)
        if not isinstance(y, pd.DataFrame) or not margins.index.equals(y.index) or not len(y):
            raise ValueError('Calibration labels must align exactly with nonempty margin rows.')
        self.response_schema_ = deepcopy(schema)
        self.models_, self.diagnostics_ = {}, {}
        self.columns_ = pd.MultiIndex.from_tuples([(t, c) for t in margins for c in schema[t]['classes']], names=['target', 'class'])
        for target in margins:
            classes = pd.Index(schema[target]['classes'])
            self._check_margin_labels(y, target, classes)
            self.models_[target], self.diagnostics_[target] = fit_sigmoid(
                margins[target].to_numpy(dtype=float), (y[target] == classes[1]).to_numpy(dtype=float))
        self.n_samples_ = len(y)
        return self

    def fit(self, probabilities, y):
        """Fit on held-out/OOF probabilities with exactly aligned observed labels."""
        if self.response_method == 'decision_function':
            return self._fit_margins(probabilities, y)
        probabilities = _probabilities(probabilities)
        if not isinstance(y, pd.DataFrame) or not y.index.equals(probabilities.index) or not len(y):
            raise ValueError("Calibration labels need the same nonempty row index as probabilities.")
        self.columns_ = probabilities.columns.copy()
        self.models_ = {}
        for target in self.columns_.get_level_values(0).unique():
            frame = probabilities.xs(target, axis=1, level=0)
            if target not in y or y[target].isna().any():
                raise ValueError(f"Calibration needs nonmissing labels for {target!r}.")
            codes = frame.columns.get_indexer(y[target])
            if (codes < 0).any():
                raise ValueError(f"Calibration target {target!r} contains classes absent from the model's probability output.")
            p = np.clip(frame.to_numpy(dtype=float), 1e-12, 1-1e-12)
            if self.method == "temperature":
                logits = np.log(p)
                def loss(log_temperature):
                    q = softmax(logits / np.exp(log_temperature), axis=1)
                    return -np.log(np.clip(q[np.arange(len(q)), codes], 1e-300, 1)).mean()
                result = minimize_scalar(loss, bounds=(-5., 5.), method="bounded")
                if not result.success:
                    raise RuntimeError("Temperature calibration did not converge.")
                self.models_[target] = float(np.exp(result.x))
            else:
                models = []
                for i in range(p.shape[1]):
                    observed = (codes == i).astype(float)
                    if self.method == "isotonic":
                        from sklearn.isotonic import IsotonicRegression
                        model = IsotonicRegression(out_of_bounds="clip").fit(p[:, i], observed)
                    elif observed.min() == observed.max():
                        # A supported class can be absent from the calibration tail.
                        model = float((observed.sum()+1) / (len(observed)+2))
                    else:
                        x = np.log(p[:, i]) - np.log1p(-p[:, i])
                        def objective(theta):
                            z = theta[0]*x + theta[1]
                            return np.mean(np.logaddexp(0, z) - observed*z)
                        fitted = minimize(objective, [1., 0.], method="L-BFGS-B", bounds=[(-100, 100), (-100, 100)])
                        if not fitted.success:
                            raise RuntimeError("Sigmoid calibration did not converge.")
                        model = tuple(fitted.x)
                    models.append(model)
                self.models_[target] = models
        self.n_samples_ = len(y)
        return self

    def transform(self, probabilities):
        if self.response_method == 'decision_function':
            from .margins import validate_margins
            schema = validate_margins(probabilities)
            if not hasattr(self, 'models_'):
                raise RuntimeError('Fit the margin calibrator before prediction.')
            if schema != self.response_schema_ or list(probabilities) != list(self.models_):
                raise ValueError('Margin target/class schema and order must match fitted calibration.')
            frames = []
            for target, (a, b, scale) in self.models_.items():
                with np.errstate(over='ignore'):
                    p = expit((a / scale) * probabilities[target].to_numpy(dtype=float) + b)
                columns = pd.MultiIndex.from_product([[target], schema[target]['classes']], names=['target', 'class'])
                frames.append(pd.DataFrame(np.column_stack([1-p, p]), index=probabilities.index, columns=columns))
            return pd.concat(frames, axis=1)
        probabilities = _probabilities(probabilities)
        if not hasattr(self, "models_"):
            raise RuntimeError("Fit the probability calibrator before transforming predictions.")
        if not probabilities.columns.equals(self.columns_):
            raise ValueError("Probability target/class columns and order must match calibration.")
        result = probabilities.astype(float).copy()
        for target, models in self.models_.items():
            columns = self.columns_.get_level_values(0) == target
            raw = probabilities.loc[:, columns].to_numpy(dtype=float)
            p = np.clip(raw, 1e-12, 1-1e-12)
            if self.method == "temperature":
                q = softmax(np.log(p)/models, axis=1)
            else:
                parts = []
                for i, model in enumerate(models):
                    if self.method == "isotonic":
                        values = model.predict(p[:, i])
                    elif isinstance(model, tuple):
                        values = expit(model[0]*(np.log(p[:, i])-np.log1p(-p[:, i])) + model[1])
                    else:
                        values = np.full(len(p), model)
                    parts.append(values)
                q = np.column_stack(parts)
                total = q.sum(axis=1)
                # Isotonic steps can map every class to zero for a novel row.
                q[total == 0] = raw[total == 0]
                q /= q.sum(axis=1, keepdims=True)
            result.loc[:, columns] = q
        return result


@dataclass
class CalibratedAdapter:
    """Fitted model plus fitted calibrator, retained together for future prediction."""
    estimator: object
    calibrator: ProbabilityCalibrator
    calibration_positions_: object

    def __getattr__(self, name):
        estimator = self.__dict__.get("estimator")
        if estimator is None:
            raise AttributeError(name)
        return getattr(estimator, name)

    def predict(self, context):
        outputs = dict(self.estimator.predict(context))
        response = self.calibrator.response_method
        raw = outputs.get(response)
        if raw is None:
            raise ValueError(f'Calibration requires the model\'s {response} output.')
        calibrated = self.calibrator.transform(raw)
        if response == 'predict_proba':
            outputs["predict_proba_raw"] = raw.copy(deep=True)
        outputs["predict_proba"] = calibrated
        if self.calibrator.update_predict:
            labels = {}
            for target in calibrated.columns.get_level_values(0).unique():
                frame = calibrated.xs(target, axis=1, level=0)
                labels[target] = frame.columns.to_numpy()[frame.to_numpy().argmax(axis=1)]
            outputs["predict"] = pd.DataFrame(labels, index=calibrated.index)
        return outputs

    @property
    def training_history_(self):
        return getattr(self.estimator, "training_history_", pd.DataFrame())

    @property
    def training_summary_(self):
        return {**deepcopy(getattr(self.estimator, "training_summary_", {})),
                "calibration": {"method": self.calibrator.method, "n_rows": self.calibrator.n_samples_,
                                "fraction": self.calibrator.fraction, "time_column": self.calibrator.time_column,
                                "response_method": self.calibrator.response_method,
                                "min_calibration_rows": self.calibrator.min_calibration_rows,
                                "min_calibration_per_class": self.calibrator.min_calibration_per_class,
                                "response_schema": deepcopy(getattr(self.calibrator, 'response_schema_', None)),
                                "sigmoid": deepcopy(getattr(self.calibrator, 'diagnostics_', None)),
                                "temporal_split": deepcopy(getattr(self.calibrator, 'split_audit_', None))}}

    def coefficient_table(self):
        return self.estimator.coefficient_table()
