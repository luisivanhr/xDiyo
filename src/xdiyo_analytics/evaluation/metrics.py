"""Reusable scalar metrics; no plotting, fitting or search dependency."""

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricDefinition:
    """function(y, prediction, **parameters) -> scalar; kind chooses its inputs.

    kind is numeric, label, probability or uncertainty. Probability/uncertainty
    functions receive a DataFrame with class labels as columns. Uncertainty does
    not require observed y. direction is minimize/maximize/None; None explicitly
    means descriptive rather than a universal optimization objective.
    """
    function: object
    kind: str
    direction: str | None


@dataclass(frozen=True)
class Metric:
    """One metric request. name identifies a registered calculation.

    key optionally names this configured metric (e.g. f1_macro vs f1_home).
    target=None expands over selected targets. output=None chooses predict or
    predict_proba according to kind. parameters are passed to the calculation.
    direction optionally supplies an explicit optimization preference.
    """
    name: str
    target: str | None = None
    output: str | None = None
    parameters: dict = field(default_factory=dict)
    key: str | None = None
    direction: str | None = None


METRICS = {}


def register_metric(name, function, *, kind, direction=None, replace=False):
    """Register a custom scalar metric; existing definitions require replace=True."""
    if not isinstance(name, str) or not name or not callable(function):
        raise ValueError("Metric registration needs a name and callable.")
    if kind not in {"numeric", "label", "probability", "uncertainty"}:
        raise ValueError("Unknown metric input kind.")
    if direction not in {None, "minimize", "maximize"}:
        raise ValueError("direction must be minimize, maximize or None.")
    if name in METRICS and not replace:
        raise ValueError(f"Metric {name!r} already exists.")
    METRICS[name] = MetricDefinition(function, kind, direction)


def list_metrics():
    """A discoverable table of built-in and explicitly registered custom metrics."""
    return pd.DataFrame([{"name": name, "kind": item.kind, "direction": item.direction}
                         for name, item in METRICS.items()])


def _sklearn(name, y, p, **parameters):
    from sklearn import metrics
    # Nullable/object numeric label Series need ordinary numpy scalar dtypes.
    return getattr(metrics, name)(np.asarray(y.tolist()), np.asarray(p.tolist()), **parameters)


def _classification(name):
    def calculate(y, p, **parameters):
        if name in {"precision_score", "recall_score", "f1_score"}:
            parameters.setdefault("average", "macro")
            parameters.setdefault("zero_division", 0)
        return _sklearn(name, y, p, **parameters)
    return calculate


def _probabilities(p):
    values = p.to_numpy(dtype=float)
    if values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Probabilities need at least two classes and finite values.")
    if ((values < 0) | (values > 1)).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Class probabilities must lie in [0,1] and sum to one per row.")
    return values


def _log_loss(y, p, *, binary=False, eps=1e-15):
    values = _probabilities(p)
    if not np.isfinite(eps) or not 0 < eps < 0.5:
        raise ValueError("eps must lie between zero and 0.5.")
    if binary and len(p.columns) != 2:
        raise ValueError("binary_cross_entropy requires exactly two classes.")
    positions = p.columns.get_indexer(y)
    if (positions < 0).any():
        raise ValueError("Observed classes are absent from the probability outputs.")
    return -np.log(np.clip(values[np.arange(len(y)), positions], eps, 1)).mean()


def _entropy(y, p):
    values = _probabilities(p)
    if values.shape[1] != 2:
        raise ValueError("binary_entropy requires exactly two classes.")
    logs = np.zeros_like(values)
    np.log(values, out=logs, where=values > 0)
    return -(values * logs).sum(axis=1).mean()


def _brier(y, p, *, positive_label=1):
    values = _probabilities(p)
    if values.shape[1] != 2 or positive_label not in p.columns or not y.isin(p.columns).all():
        raise ValueError("Binary Brier score needs two known classes and positive_label.")
    return np.mean((p[positive_label].to_numpy() - y.eq(positive_label).to_numpy()) ** 2)


def _auc(y, p, *, positive_label=1, multi_class="ovr", average="macro"):
    from sklearn.metrics import roc_auc_score
    _probabilities(p)
    if not y.isin(p.columns).all():
        raise ValueError("Observed classes are absent from probability outputs.")
    if len(p.columns) == 2:
        if positive_label not in p.columns:
            raise ValueError("positive_label is absent from probability outputs.")
        return roc_auc_score(y.eq(positive_label), p[positive_label])
    # Integer encoding preserves the adapter's declared class order.
    return roc_auc_score(p.columns.get_indexer(y), p.to_numpy(),
                         labels=np.arange(len(p.columns)), multi_class=multi_class, average=average)


for _name, _fn in {
    "mse": lambda y, p: np.mean((y - p) ** 2),
    "mae": lambda y, p: np.mean(np.abs(y - p)),
    "rmse": lambda y, p: np.sqrt(np.mean((y - p) ** 2)),
}.items():
    register_metric(_name, _fn, kind="numeric", direction="minimize")
register_metric("r2", lambda y, p, **kw: _sklearn("r2_score", y, p, **kw), kind="numeric", direction="maximize")
for _name, _function in {"accuracy": "accuracy_score", "balanced_accuracy": "balanced_accuracy_score",
                         "precision": "precision_score", "recall": "recall_score", "f1_score": "f1_score",
                         "mcc": "matthews_corrcoef"}.items():
    register_metric(_name, _classification(_function), kind="label", direction="maximize")
for _name in ("log_loss", "cross_entropy"):
    register_metric(_name, _log_loss, kind="probability", direction="minimize")
register_metric("binary_cross_entropy", lambda y, p, **kw: _log_loss(y, p, binary=True, **kw),
                kind="probability", direction="minimize")
register_metric("binary_entropy", _entropy, kind="uncertainty")
register_metric("brier_score", _brier, kind="probability", direction="minimize")
register_metric("roc_auc", _auc, kind="probability", direction="maximize")
for _alias, _original in {"f1": "f1_score", "mean_squared_error": "mse", "mean_absolute_error": "mae",
                         "accuracy_score": "accuracy", "precision_score": "precision",
                         "recall_score": "recall", "brier_score_loss": "brier_score"}.items():
    METRICS[_alias] = METRICS[_original]


def metric_inputs(y, predictions, request):
    """Resolve one target's aligned inputs and complete-case mask, without fitting."""
    definition = METRICS[request.name]
    output = request.output or ("predict_proba" if definition.kind in {"probability", "uncertainty"} else "predict")
    frame = predictions[output]
    if not frame.index.equals(y.index):
        raise ValueError("Prediction outputs and targets must retain the same index/order.")
    target = request.target
    truth = y[target]
    if definition.kind in {"probability", "uncertainty"}:
        if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels != 2:
            raise ValueError("Probability columns must be a (target, class) MultiIndex.")
        prediction = frame.xs(target, level=0, axis=1)
        if not prediction.columns.is_unique:
            raise ValueError("Probability classes must be distinct.")
        prediction = prediction.apply(pd.to_numeric, errors="raise")
        mask = pd.Series(np.isfinite(prediction.to_numpy(dtype=float, na_value=np.nan)).all(axis=1), index=y.index)
    elif definition.kind == "numeric":
        truth = pd.to_numeric(truth, errors="raise").astype(float)
        prediction = pd.to_numeric(frame[target], errors="raise").astype(float)
        mask = np.isfinite(truth) & np.isfinite(prediction)
    else:
        prediction = frame[target]
        mask = prediction.notna()
    if definition.kind != "uncertainty":
        mask = mask & truth.notna()
    return truth, prediction, mask, output


def _sample_hash(y, metadata, mask):
    """Hash actual evaluated identities/outcomes, not model predictions."""
    frame = metadata.loc[mask].copy()
    frame["__observed_target__"] = y.loc[mask]
    digest = hashlib.sha256()
    digest.update(repr([(str(c), str(t)) for c, t in zip(frame.columns, frame.dtypes)]).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def evaluate_metrics(y, predictions, metrics, *, metadata=None):
    """Compute requested scalar metrics, with explicit coverage and status.

    A string is shorthand for Metric(name). Missing/nonfinite pairs are excluded
    and counted. Undefined built-in metrics report value=NaN and their reason;
    malformed inputs/configurations raise. All losses use their ordinary positive
    value, not sklearn's negated scorer convention. Cross-entropy/entropy use nats.
    Precision/recall/F1 default to macro; pass average='binary', pos_label=... for
    a particular positive class. Custom functions must return a scalar.
    """
    from dataclasses import replace
    metadata = pd.DataFrame(index=y.index) if metadata is None else metadata
    if not metadata.index.equals(y.index):
        raise ValueError("Metric metadata must align with targets.")
    rows = []
    for spec in metrics:
        spec = Metric(spec) if isinstance(spec, str) else spec
        definition = METRICS[spec.name]
        if spec.direction not in {None, "minimize", "maximize"}:
            raise ValueError("Metric direction must be minimize/maximize/None.")
        for target in (list(y.columns) if spec.target is None else [spec.target]):
            request = replace(spec, target=target)
            truth, prediction, mask, output = metric_inputs(y, predictions, request)
            n, status, value = int(mask.sum()), "ok", np.nan
            if n == 0:
                status = "no_valid_observations"
            elif spec.name == "r2" and (n < 2 or truth[mask].nunique() < 2):
                status = "undefined_constant_or_small_target"
            elif spec.name == "roc_auc" and truth[mask].nunique() < 2:
                status = "undefined_single_observed_class"
            else:
                value = definition.function(truth.loc[mask], prediction.loc[mask], **spec.parameters)
                if not np.isscalar(value):
                    raise ValueError("Metrics must return scalar values; configure averaging explicitly.")
                value = float(value)
                if not np.isfinite(value):
                    value, status = np.nan, "undefined"
            observed = truth if definition.kind != "uncertainty" else pd.Series(0, index=truth.index)
            rows.append(dict(metric=spec.key or spec.name, calculation=spec.name, target=target, output=output,
                             value=value, direction=spec.direction or definition.direction, n=n,
                             n_total=len(y), n_missing=len(y)-n, status=status,
                             parameters=json.dumps(spec.parameters, sort_keys=True, allow_nan=False),
                             sample_hash=_sample_hash(observed, metadata, mask)))
    result = pd.DataFrame(rows, columns=["metric", "calculation", "target", "output", "value", "direction",
                                         "n", "n_total", "n_missing", "status", "parameters", "sample_hash"])
    if result.duplicated(["metric", "target", "output"]).any():
        raise ValueError("Configured metrics need distinct keys per target/output.")
    return result
