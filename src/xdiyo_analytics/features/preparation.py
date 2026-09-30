"""Small explicit post-assembly operations and frozen identity vocabularies."""

from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import pandas as pd
from .composition import IdentityIndicators


@dataclass
class NumericFeatures:
    """Optional numeric conversion after feature assembly; no fitted statistics."""
    dtype: str = "float32"
    infinities_to_missing: bool = True

    def transform(self, frame):
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("Numeric feature dtype must be float32 or float64.")
        result = frame.astype(self.dtype)
        if self.infinities_to_missing:
            result = result.replace([np.inf, -np.inf], np.nan)
        return result


@dataclass
class IdentityFeatureSpec:
    """Freeze a vocabulary from shared outer-training rows, then add indicators.

    This preparation-level schema matches the notebook's development vocabulary.
    For multiple outer folds, only their common training intersection is used;
    categories seen only in evaluation never create columns. It is intentionally
    a fixed input schema, not a refitted per-fold transformer. Unseen values map
    to zero. Future predictions use the vocabulary retained in model definitions.
    """
    columns: tuple = ("source_league",)
    prefixes: object = None
    alphabetical: bool = True

    def fit(self, metadata):
        values = metadata.sort_values(list(self.columns), kind="stable") if self.alphabetical else metadata
        encoder = IdentityIndicators(self.columns, self.prefixes).fit(values)
        return {"columns": self.columns, "prefixes": dict(self.prefixes or {}),
                "categories": deepcopy(encoder.categories_)}


def common_training_rows(plan):
    if not plan.folds:
        raise ValueError("Feature discovery/identity fitting needs a nonempty split plan.")
    rows = plan.folds[0].train.copy()
    for fold in plan.folds[1:]:
        rows = rows[np.isin(rows, fold.train)]
    if not len(rows):
        raise ValueError("Outer folds have no common training rows for a frozen feature schema. Supply explicit development seasons for preset discovery, or disable preparation-level identity indicators.")
    return rows


def add_identity_features(dataset, state):
    """Apply frozen categories; never inspect evaluation category frequencies."""
    encoder = IdentityIndicators(state["columns"], state["prefixes"])
    encoder.categories_ = state["categories"]
    added = encoder.transform(dataset.metadata)
    overlap = added.columns.intersection(dataset.X.columns)
    if len(overlap):
        # A caller may already have assembled the saved vocabulary. Check values
        # instead of replacing columns silently.
        if not np.array_equal(dataset.X[overlap].to_numpy(), added[overlap].to_numpy(), equal_nan=True):
            raise ValueError("Existing identity features disagree with the saved vocabulary.")
        added = added.drop(columns=overlap)
    dataset.X = pd.concat([dataset.X, added], axis=1)
    dataset.definitions["identity_features"] = deepcopy(state)
    return dataset
