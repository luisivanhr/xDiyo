"""Fit-local classification weights, shared by reports and model adapters."""

from dataclasses import dataclass
from collections.abc import Mapping
import inspect

import numpy as np
import pandas as pd


@dataclass
class ObservationWeights:
    """Original-label weights indexed by exact fitting row identity."""
    target: str
    values: pd.Series
    classes: pd.DataFrame

    def aligned(self, index, targets):
        if list(targets) != [self.target]:
            raise ValueError("Class weights require the same single target used by the reporter.")
        return aligned_weights(self.values, index)


def aligned_weights(values, index):
    if not isinstance(values, pd.Series) or not values.index.is_unique:
        raise ValueError("Observation weights must be a Series with distinct row identities.")
    if len(values) != len(index) or not values.index.isin(index).all():
        raise ValueError("Observation weights must cover exactly this fit's training rows.")
    result = values.reindex(index).astype(float)
    if not np.isfinite(result).all() or (result < 0).any() or result.sum() <= 0:
        raise ValueError("Observation weights must be finite, nonnegative, with positive total weight.")
    return result.rename("sample_weight")


@dataclass
class ClassWeightPolicy:
    """Compute weights from one fitting population, never from evaluation rows.

    mode: none, balanced, power, custom, or callable. Custom maps ORIGINAL class
    labels to weights (JSON string keys are accepted when unambiguous). A
    calculator(context) returns a Series indexed like context.y. All modes are
    normalized to mean observation weight one. target=None requires one label.
    """
    mode: str = "balanced"
    power: float = 1.0
    class_weights: object = None
    calculator: object = None
    target: str | None = None

    def compute(self, context):
        from sklearn.utils.multiclass import type_of_target
        targets = list(context.y.columns)
        target = self.target
        if target is None:
            if len(targets) != 1:
                raise ValueError("Choose one target for class weighting.")
            target = targets[0]
        if target not in targets:
            raise ValueError(f"Unknown class weighting target {target!r}.")
        y = context.y[target]
        if y.empty or y.isna().any() or type_of_target(y) not in {"binary", "multiclass"}:
            raise ValueError("Class weights require nonmissing categorical or integer class labels.")
        counts = y.value_counts(sort=False)
        counts = counts[counts > 0]
        if self.mode == "none":
            values = pd.Series(1., index=y.index)
        elif self.mode in {"balanced", "power"}:
            power = 1. if self.mode == "balanced" else self.power
            if isinstance(power, bool) or not np.isfinite(power) or power < 0:
                raise ValueError("Balancing power must be finite and nonnegative.")
            # Center logarithms before exponentiation to avoid overflow.
            logs = power * np.log(len(y) / (len(counts) * counts.astype(float)))
            weights = np.exp(logs - logs.max())
            values = y.map(weights).astype(float)
        elif self.mode == "custom":
            if not isinstance(self.class_weights, Mapping):
                raise ValueError("custom mode requires a class_weights mapping.")
            weights = {}
            for label in counts.index:
                if label in self.class_weights:
                    weights[label] = self.class_weights[label]
                elif str(label) in self.class_weights:
                    weights[label] = self.class_weights[str(label)]
                elif isinstance(label, (float, np.floating)) and label.is_integer() and str(int(label)) in self.class_weights:
                    weights[label] = self.class_weights[str(int(label))]
                else:
                    raise ValueError(f"No custom weight supplied for fitting class {label!r}.")
            values = y.map(weights).astype(float)
        elif self.mode == "callable":
            if not callable(self.calculator):
                raise ValueError("callable mode requires calculator(context).")
            values = self.calculator(context)
        else:
            raise ValueError("Weight mode must be none, balanced, power, custom or callable.")
        values = aligned_weights(values, y.index)
        values = values / values.mean()
        grouped = pd.DataFrame({"label": y, "weight": values}).groupby("label", sort=False, observed=True).weight
        table = pd.DataFrame({"class": counts.index, "count": counts.to_numpy(),
                              "frequency": counts.to_numpy() / len(y),
                              "weight": grouped.mean().reindex(counts.index).to_numpy(),
                              "weighted_proportion": grouped.sum().reindex(counts.index).to_numpy() / values.sum()})
        return ObservationWeights(target, values.rename("sample_weight"), table)


def estimator_weight_kwargs(estimator, context, *, method="fit", pipeline=True):
    """Route declared fit weights once, rejecting unsupported/double weighting."""
    if context.sample_weight is None:
        return {}
    from sklearn.base import is_classifier
    native = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    if not is_classifier(native):
        raise TypeError("Class weighting requires a classifier adapter/estimator.")
    params = native.get_params(deep=False)
    if (params.get("class_weight") is not None or params.get("scale_pos_weight", 1) not in (None, 1)
            or params.get("is_unbalance", False)):
        raise ValueError("Disable native class_weight/scale_pos_weight/is_unbalance when using common class weights.")
    if "sample_weight" not in inspect.signature(getattr(native, method)).parameters:
        raise TypeError(f"{type(native).__name__}.{method} does not support sample_weight.")
    values = aligned_weights(context.sample_weight, context.X.index).to_numpy()
    key = f"{estimator.steps[-1][0]}__sample_weight" if pipeline and hasattr(estimator, "steps") else "sample_weight"
    return {key: values}


def report_weights(report, name, fold_id, rows, layout):
    """Resolve a consumed report only when it describes exactly this fit."""
    studies = [s for s in report.studies if s.name == name and s.fold_id == fold_id]
    if not studies:
        studies = [s for s in report.studies if s.name == name and s.fold_id is None]
    if len(studies) != 1 or not isinstance(studies[0].result.weights, ObservationWeights):
        raise ValueError("weights_from needs one ClassWeightReporter result for this fold.")
    study = studies[0]
    if (study.partition != "train" or study.layout != layout or len(study.row_positions) != len(rows)
            or set(study.row_positions) != set(rows)):
        raise ValueError("Consumed weights must use exactly this fold's fitting training rows; use selection_plan.")
    return study.result.weights
