"""Reusable diagnostics for exact-count classifiers."""

from dataclasses import dataclass
from copy import copy
from typing import ClassVar

import numpy as np
import pandas as pd

from .contracts import Artifact, StudyResult
from .post_training import PredictionReporter
from .studies import _columns, _plotting, _style


def _count_values(values, name):
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if (not np.isfinite(numeric).all() or (numeric < 0).any() or
            not np.equal(numeric, np.floor(numeric)).all()):
        raise ValueError(f"{name} must contain finite nonnegative integer counts.")
    return numeric.astype(np.int64)


def _probabilities(frame, target, index):
    if not frame.index.equals(index):
        raise ValueError("Probability rows must have the same index as observed labels.")
    if isinstance(frame.columns, pd.MultiIndex):
        if frame.columns.nlevels != 2 or target not in frame.columns.get_level_values(0):
            raise ValueError("Probability outputs need (target, class) columns.")
        values = frame.xs(target, level=0, axis=1)
    else:
        values = frame
    if not len(values.columns) or not values.columns.is_unique:
        raise ValueError("Probability class columns must be distinct and nonempty.")
    classes = pd.to_numeric(pd.Index(values.columns), errors="coerce").to_numpy(dtype=float)
    if (not np.isfinite(classes).all() or (classes < 0).any() or
            not np.equal(classes, np.floor(classes)).all() or len(np.unique(classes)) != len(classes)):
        raise ValueError("Probability class columns must be distinct nonnegative integer counts.")
    return values.to_numpy(dtype=float), classes.astype(np.int64)


@dataclass(kw_only=True)
class CountClassificationReporter(PredictionReporter):
    """Exact-count classification diagnostics, including unsupported classes.

    ``tolerance`` is an inclusive count-distance band. Probability diagnostics
    use rows whose complete probability vector is finite, nonnegative and sums
    to one. An absent class column means zero support; a present column with a
    missing value makes that row unavailable for probability diagnostics.
    """

    targets: object = None
    output: str = "predict"
    probability_output: str = "predict_proba"
    tolerance: float = 2.0
    epsilon: float = 1e-12
    group_by: object = None
    supported_types: ClassVar[tuple] = ("per_fold", "overall")

    def run(self, context):
        if isinstance(self.tolerance, (bool, np.bool_)) or not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be a finite nonnegative number.")
        if isinstance(self.epsilon, (bool, np.bool_)) or not np.isfinite(self.epsilon) or not 0 < self.epsilon < 1:
            raise ValueError("epsilon must be finite and between zero and one.")
        from sklearn.metrics import f1_score
        result = StudyResult("Count classification diagnostics")
        metric_rows, supports, detail_rows = [], [], []
        # A pooled mean prediction has no single fitting population baseline.
        baseline_by_row = None
        baseline_rows = None
        if getattr(context, "pooling", None) != "mean":
            fold_ids = context.y.index.get_level_values("fold_id") if isinstance(context.y.index, pd.MultiIndex) and "fold_id" in context.y.index.names else pd.Index([getattr(context, "fold_id", None)] * len(context.y))
            baseline_by_row, baseline_rows = {}, {}
            for fold_id in pd.unique(fold_ids):
                fold = context.fold_results.get(fold_id) if hasattr(context, "fold_results") else None
                summary = getattr(fold, "training_summary", {}) if fold is not None else {}
                modes = summary.get("training_target_modes", {})
                rows = summary.get("training_target_mode_rows")
                if not isinstance(modes, dict) or rows is None:
                    continue
                baseline_by_row[fold_id] = modes
                baseline_rows[fold_id] = rows
        for target in _columns(context.y, self.targets, "targets"):
            observed = _count_values(context.y[target], "Observed counts")
            predicted = _count_values(context.predictions[self.output][target], "Predicted counts")
            point_ok = np.isfinite(observed) & np.isfinite(predicted)
            probability = context.predictions.get(self.probability_output)
            if probability is None:
                raise KeyError(f"Missing probability output: {self.probability_output}")
            # Pooled folds can have different fitted class columns. Read each
            # row from its source fold so an absent class is zero support rather
            # than a NaN-created missing probability. Mean pooling intentionally
            # remains unavailable for this distinction.
            p, classes = _probabilities(probability, target, context.y.index)
            fold_ids = context.y.index.get_level_values("fold_id") if isinstance(context.y.index, pd.MultiIndex) and "fold_id" in context.y.index.names else pd.Index([getattr(context, "fold_id", None)] * len(context.y))
            row_positions = context.y.index.get_level_values("row_position") if isinstance(context.y.index, pd.MultiIndex) and "row_position" in context.y.index.names else context.y.index
            row_ps, row_classes = [], []
            for i, (fid, pos) in enumerate(zip(fold_ids, row_positions)):
                source = None if getattr(context, "pooling", None) == "mean" else getattr(context.fold_results.get(fid), "predictions", {}).get(self.probability_output) if hasattr(context, "fold_results") and fid in context.fold_results else None
                if source is None:
                    row_ps.append(p[i]); row_classes.append(classes); continue
                vals, local_classes = _probabilities(source.loc[[pos]], target, source.loc[[pos]].index)
                row_ps.append(vals[0]); row_classes.append(local_classes)
            valid_p = np.array([np.isfinite(v).all() and (v >= 0).all() and np.isclose(v.sum(), 1., atol=1e-6, rtol=0) for v in row_ps])
            supported = np.array([int(value) in set(cls.tolist()) for value, cls in zip(observed, row_classes)])
            actual = np.array([v[list(cls).index(int(value))] if valid and int(value) in set(cls.tolist()) else (0.0 if valid else np.nan) for value, v, cls, valid in zip(observed, row_ps, row_classes, valid_p)])
            weighted = np.array([float(v @ cls) if valid else np.nan for v, cls, valid in zip(row_ps, row_classes, valid_p)])
            classes = np.unique(np.concatenate(row_classes)) if row_classes else classes
            detail = pd.DataFrame({"observed_count": observed, "predicted_count": predicted,
                "absolute_error": np.abs(observed-predicted), "within_tolerance": np.abs(observed-predicted) <= self.tolerance,
                "target_in_fitted_classes": supported, "observed_class_probability": actual,
                "probability_weighted_count": weighted}, index=context.y.index)
            result.tables[f"details::{target}"] = detail.reset_index()
            n = len(observed); mse = float(np.mean((predicted-observed)**2)); supported_rows = valid_p & supported
            metric_rows.append(dict(target=target, n=n, mae=float(np.mean(abs(predicted-observed))), mse=mse,
                rmse=float(np.sqrt(mse)), accuracy=float(np.mean(predicted == observed)),
                macro_f1=float(f1_score(observed, predicted, average="macro", zero_division=0)),
                within_tolerance=float(np.mean(np.abs(predicted-observed) <= self.tolerance)),
                unseen_count_rows=int((~supported).sum()), unseen_count_fraction=float((~supported).mean()),
                probability_rows=int(valid_p.sum()), probability_missing_rows=int((~valid_p).sum()),
                log_loss_clipped_all=float(-np.log(np.maximum(actual[valid_p], self.epsilon)).mean()) if valid_p.any() else np.nan,
                log_loss_supported_only=float(-np.log(np.maximum(actual[supported_rows], self.epsilon)).mean()) if supported_rows.any() else np.nan,
                exact_log_loss_infinite=bool((actual[valid_p] == 0).any()) if valid_p.any() else False,
                probability_weighted_count_mean=float(np.nanmean(weighted)) if valid_p.any() else np.nan,
                log_loss_floor=self.epsilon, tolerance=self.tolerance))
            metric = metric_rows[-1]
            baseline_values = np.array([baseline_by_row.get(fid, {}).get(target, np.nan) for fid in fold_ids], dtype=float) if baseline_by_row is not None else np.full(n, np.nan)
            valid_baseline = np.isfinite(baseline_values)
            metric.update(baseline_majority_count=(float(np.unique(baseline_values[valid_baseline])[0]) if valid_baseline.any() and len(np.unique(baseline_values[valid_baseline])) == 1 else np.nan),
                          baseline_mae=float(np.mean(np.abs(observed[valid_baseline]-baseline_values[valid_baseline]))) if valid_baseline.any() else np.nan,
                          baseline_accuracy=float(np.mean(observed[valid_baseline] == baseline_values[valid_baseline])) if valid_baseline.any() else np.nan,
                          baseline_training_rows=int(sum(baseline_rows.get(fid, 0) for fid in pd.unique(fold_ids) if fid in baseline_rows)) if valid_baseline.any() else np.nan,
                          baseline_status="available" if valid_baseline.all() and valid_baseline.any() else ("partial" if valid_baseline.any() else "unavailable"))
            observed_counts = pd.Series(observed).value_counts().to_dict()
            for value in sorted(set(classes.tolist()) | set(observed.tolist())):
                supports.append(dict(target=target, count_class=int(value), probability_column=any(int(value) in set(cls.tolist()) for cls in row_classes),
                    observed_rows=int(observed_counts.get(value, 0))))
        if self.group_by is not None:
            group_columns = [self.group_by] if isinstance(self.group_by, str) else list(self.group_by)
            missing = set(group_columns) - set(context.metadata.columns)
            if missing:
                raise KeyError(f"Unknown group_by metadata columns: {sorted(missing)}")
            grouped = []
            for identity, positions in context.metadata.groupby(group_columns, sort=True, dropna=False).groups.items():
                sub = self.__class__(type=self.type, partition=self.partition, pooling=self.pooling, targets=self.targets,
                    output=self.output, probability_output=self.probability_output, tolerance=self.tolerance, epsilon=self.epsilon)
                sub_context = copy(context)
                sub_context.y = context.y.loc[positions]
                sub_context.predictions = {k: v.loc[positions] for k, v in context.predictions.items()}
                sub_context.metadata = context.metadata.loc[positions]
                sub_context.fold_id = context.fold_id
                grouped.extend([{**row, **dict(zip(group_columns, identity if isinstance(identity, tuple) else (identity,)))} for row in sub.run(sub_context).tables["metrics"].to_dict("records")])
            result.tables["metrics_by_group"] = pd.DataFrame(grouped)
            result.artifacts.append(Artifact("table", result.tables["metrics_by_group"], "Metrics by group"))
        result.tables["metrics"] = pd.DataFrame(metric_rows)
        result.tables["class_support"] = pd.DataFrame(supports)
        result.artifacts.append(Artifact("table", result.tables["metrics"], "Classification metrics"))
        result.artifacts.append(Artifact("table", result.tables["class_support"], "Class support"))
        result.notes.append("Absent probability class columns represent zero fitted support; present columns with missing values make those rows unavailable for probability diagnostics.")
        return result
