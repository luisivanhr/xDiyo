"""Extract available estimator diagnostics without refitting a model."""

import numpy as np
import pandas as pd


def estimator_coefficients(estimator, feature_names, target_names):
    """Lasso/ElasticNet and other coef_ estimators, in their fitted feature space.

    Pipelines must expose transformed feature names; names are never guessed from
    the original width after a transformation. Binary classifier coefficients
    correspond to classes_[1]; multiclass rows retain their class labels.
    Regression coefficients retain target names. Intercepts are separate terms.
    """
    features = list(feature_names)
    if hasattr(estimator, "steps"):
        if len(estimator.steps) > 1:
            features = list(estimator[:-1].get_feature_names_out(features))
        estimator = estimator.steps[-1][1]
    if not hasattr(estimator, "coef_"):
        raise TypeError("The fitted estimator does not expose coefficients.")
    coef = estimator.coef_
    values = np.asarray(coef.toarray() if hasattr(coef, "toarray") else coef)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != len(features) or len(set(features)) != len(features):
        raise ValueError("Coefficient dimensions must match distinct fitted feature names.")
    targets = list(target_names)
    classes = getattr(estimator, "classes_", None)
    if classes is not None:
        if len(targets) != 1:
            raise ValueError("Classifier coefficient inspection requires one target.")
        labels = [classes[1]] if values.shape[0] == 1 and len(classes) == 2 else list(classes)
        if len(labels) != values.shape[0]:
            raise ValueError("Coefficient rows do not match fitted classes.")
        outputs = [(targets[0], label) for label in labels]
    else:
        if values.shape[0] != len(targets):
            raise ValueError("Coefficient rows do not match selected targets.")
        outputs = [(target, None) for target in targets]
    intercept = np.asarray(getattr(estimator, "intercept_", np.zeros(len(outputs)))).reshape(-1)
    if len(intercept) == 1:
        intercept = np.repeat(intercept, len(outputs))
    if len(intercept) != len(outputs):
        raise ValueError("Intercept dimensions do not match coefficient outputs.")
    rows = []
    for i, (target, label) in enumerate(outputs):
        for feature, value in zip(features, values[i]):
            rows.append(dict(target=target, label=label, feature=feature, coefficient=float(value), term="feature"))
        rows.append(dict(target=target, label=label, feature="intercept", coefficient=float(intercept[i]), term="intercept"))
    table = pd.DataFrame(rows)
    if hasattr(estimator, 'coefficient_units_'):
        table['coefficient_units'] = estimator.coefficient_units_
    if hasattr(estimator, 'prediction_') and getattr(estimator, 'coefficient_units_', None) == 'log_mean':
        table['prediction_statistic'] = estimator.prediction_
    return table


def estimator_history(estimator, fold_id):
    """Read native histories when available; never fabricate an iteration curve."""
    if hasattr(estimator, "steps"):
        estimator = estimator.steps[-1][1]
    rows = []
    if hasattr(estimator, "loss_curve_"):
        for step, value in enumerate(estimator.loss_curve_, 1):
            rows.append(dict(fold_id=fold_id, attempt=0, step=step, metric="train_loss", value=float(value), learning_rate=None))
    for partition, metrics in getattr(estimator, "evals_result_", {}).items():
        for name, values in metrics.items():
            for step, value in enumerate(values, 1):
                rows.append(dict(fold_id=fold_id, attempt=0, step=step, metric=f"{partition}.{name}", value=float(value), learning_rate=None))
    summary = {"history_source": "native_estimator" if rows else "unavailable",
               "termination_reason": "estimator_managed"}
    for attribute in ("n_iter_", "best_iteration_", "converged_", "dispersion_", "prediction_"):
        if hasattr(estimator, attribute):
            value = np.asarray(getattr(estimator, attribute))
            summary[attribute.rstrip("_")] = value.item() if value.ndim == 0 else value.tolist()
    return pd.DataFrame(rows, columns=["fold_id", "attempt", "step", "metric", "value", "learning_rate"]), summary
