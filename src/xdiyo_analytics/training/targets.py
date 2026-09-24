"""Optional regression target scaling owned by each fresh fitting adapter."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .estimators import EstimatorAdapter


@dataclass
class TargetTransformAdapter:
    """Fit a numeric target transformer on fitting labels and invert predict once.

    Supports EstimatorAdapter regression models and its native boosting subclass.
    The wrapped adapter still owns X preprocessing, validation, devices and fit
    controls. Classifiers/rankers and other custom adapters need their own target
    contract and are rejected here. Checkpoint resumption is not forwarded.

    Validation labels use the SAME fitted target transformer. Retained y_true,
    scores and public predictions remain in original units. Native training-loss
    curves are in transformed-target units and are labelled accordingly.
    """

    estimator: object
    transformer: object

    def __post_init__(self):
        from sklearn.base import is_regressor
        if not isinstance(self.estimator, EstimatorAdapter):
            raise TypeError('Target scaling currently supports EstimatorAdapter regression models and native boosting regressors.')
        native = self.estimator.estimator
        if hasattr(native, 'steps'):
            native = native.steps[-1][1]
        if getattr(native, 'requires_untransformed_target', False):
            raise ValueError('Negative Binomial regression requires raw count-scale labels. Disable Scale the target.')
        if not is_regressor(native) or type(native).__name__.endswith('Ranker'):
            raise TypeError('Target scaling is for numeric regression, not classifiers or rankers.')
        if tuple(self.estimator.prediction_methods) != ('predict',):
            raise ValueError('Target scaling currently supports only the numeric predict output.')
        if not all(callable(getattr(self.transformer, name, None))
                   for name in ('fit', 'transform', 'inverse_transform')):
            raise TypeError('A target transformer must expose fit, transform and inverse_transform.')

    def configure_device(self, device):
        return self.estimator.configure_device(device)

    def configure_threads(self, threads):
        return self.estimator.configure_threads(threads)

    def _scaled_context(self, context):
        from sklearn.base import clone
        self.transformer_ = clone(self.transformer)
        self.target_columns_ = tuple(context.y.columns)
        self.transformer_.fit(context.y)

        def transform(frame):
            values = np.asarray(self.transformer_.transform(frame.copy(deep=True)))
            if values.shape != frame.shape:
                raise ValueError('Target transforms must preserve the number and order of target columns.')
            return pd.DataFrame(values, index=frame.index, columns=frame.columns)

        validation = context.validation
        if validation is not None:
            validation = replace(validation, y=transform(validation.y))
        return replace(context, y=transform(context.y), validation=validation)

    def _retain_diagnostics(self):
        self.training_history_ = getattr(self.estimator, 'training_history_', pd.DataFrame()).copy(deep=True)
        if 'metric' in self.training_history_:
            self.training_history_['metric'] = self.training_history_.metric.astype(str) + ' [transformed target]'
        self.training_summary_ = dict(getattr(self.estimator, 'training_summary_', {}))
        self.training_summary_.update(target_transformer=type(self.transformer_).__name__,
                                      training_loss_units='transformed_target',
                                      prediction_units='original_target')

    def fit(self, context):
        self.estimator.fit(self._scaled_context(context))
        self._retain_diagnostics()

    def fit_controlled(self, context, control):
        method = getattr(self.estimator, 'fit_controlled', None)
        if not callable(method):
            raise TypeError('This wrapped estimator does not support validation/training controls.')
        method(self._scaled_context(context), control)
        self._retain_diagnostics()

    def predict(self, context):
        outputs = self.estimator.predict(context)
        if set(outputs) != {'predict'}:
            raise ValueError('Target scaling expects exactly one numeric predict output.')
        frame = outputs['predict']
        if tuple(frame.columns) != self.target_columns_:
            raise ValueError('Target prediction columns must match the fitted target transform.')
        values = np.asarray(self.transformer_.inverse_transform(frame.copy(deep=True)))
        if values.shape != frame.shape:
            raise ValueError('Inverse target transform must preserve the prediction shape.')
        if np.isfinite(frame.to_numpy()).all() and not np.isfinite(values).all():
            raise ValueError('Inverse target transform produced non-finite predictions. '
                             'Power transforms can have a bounded inverse domain; choose a compatible transform or model.')
        return {'predict': pd.DataFrame(values, index=frame.index, columns=frame.columns)}

    def coefficient_table(self):
        """Invert affine target units; label nonlinear model coefficients explicitly."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
        table = self.estimator.coefficient_table().copy(deep=True)
        table['target_transformer'] = type(self.transformer_).__name__
        if not isinstance(self.transformer_, (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler)):
            # No single original-unit slope exists after a nonlinear inverse.
            table['coefficient_units'] = 'transformed_target'
            return table
        table['coefficient_units'] = 'original_target'
        shape = (1, len(self.target_columns_))
        means = np.asarray(self.transformer_.inverse_transform(np.zeros(shape)))[0]
        scales = np.asarray(self.transformer_.inverse_transform(np.ones(shape)))[0] - means
        for target, scale, mean in zip(self.target_columns_, scales, means):
            rows = table.target.eq(target)
            table.loc[rows, 'coefficient'] *= scale
            table.loc[rows & table.term.eq('intercept'), 'coefficient'] += mean
        return table
