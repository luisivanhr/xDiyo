"""Explicit scopes for reusable post-training studies."""

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..reporting.contracts import AnalysisReport, PostTrainingContext, StudyResult, StudyRun


def _population(training, folds, partition, pooling):
    """Align prediction occurrences, then apply a declared repeated-row policy."""
    truths, metadata, outputs = {}, {}, {}
    for fold in folds:
        positions = fold.score_positions if partition == "score" else fold.test_positions
        truths[fold.fold_id] = fold.y_true.loc[positions].copy()
        metadata[fold.fold_id] = fold.metadata.loc[positions].copy()
        if outputs and set(outputs) != set(fold.predictions):
            raise ValueError("Pooled folds must expose the same named prediction outputs.")
        for name, frame in fold.predictions.items():
            outputs.setdefault(name, {})[fold.fold_id] = frame.loc[positions].copy()
    if not truths:
        raise ValueError("Prediction reporters need at least one selected fitted fold.")
    target_columns = [list(frame.columns) for frame in truths.values()]
    if any(columns != target_columns[0] for columns in target_columns):
        raise ValueError("Pooled folds must have the same target columns/order.")
    y = pd.concat(truths, names=["fold_id", "row_position"])
    meta = pd.concat(metadata, names=["fold_id", "row_position"])
    predictions = {name: pd.concat(frames, names=["fold_id", "row_position"]) for name, frames in outputs.items()}
    repeated = y.index.get_level_values("row_position").duplicated().any()
    if pooling not in {None, "occurrences", "first", "last", "mean"}:
        raise ValueError("pooling must be occurrences, first, last or mean.")
    if repeated and pooling is None:
        raise ValueError("Repeated held-out predictions require an explicit pooling policy.")
    if repeated:
        for frame in (y, meta):
            if frame.groupby(level="row_position", sort=False).nunique(dropna=False).gt(1).any().any():
                raise ValueError("Repeated row positions have conflicting outcomes or metadata.")
    if pooling in {"first", "last", "mean"}:
        keep = ~y.index.get_level_values("row_position").duplicated(keep="last" if pooling == "last" else "first")
        if pooling == "mean":
            pooled = {}
            positions = y.index.get_level_values("row_position")[keep]
            for name, frame in predictions.items():
                if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
                    raise ValueError("Mean pooling requires numeric outputs; use first/last for class labels.")
                groups = frame.groupby(level="row_position", sort=False)
                means = groups.mean().reindex(positions)
                counts = groups.count().reindex(positions)
                means = means.mask(counts.lt(groups.size().reindex(positions), axis=0))
                means.index = pd.MultiIndex.from_arrays([np.full(len(positions), -1), positions],
                                                        names=["fold_id", "row_position"])
                pooled[name] = means
            predictions = pooled
        else:
            predictions = {name: frame.loc[keep].copy() for name, frame in predictions.items()}
        y, meta = y.loc[keep].copy(), meta.loc[keep].copy()
        if pooling == "mean":
            y.index = meta.index = next(iter(predictions.values())).index
    return y, predictions, meta


@dataclass
class PostTrainingAnalysis:
    """Execute zero or more named post-training reporters, returning AnalysisReport.

    Reporter type is per_fold/overall/timeline; prediction partition is test or score.
    score uses Fold.score_positions. Metrics are computed on pooled predictions,
    never implicitly averaged from per-fold metrics. Repeated predictions require
    pooling=occurrences/first/last/mean; no repeats need no policy. first/last use
    selected fold order. Mean combines numeric outputs, not class decisions; use
    predict_proba-only output or first/last when class labels would be ambiguous.
    A missing class/output cell remains missing when mean pooling.

    partition='model' inspects fitted models and their stored training histories,
    with no prediction-row population or pooling. These studies support per_fold
    and overall according to each reporter's contract.

    Experiment reporters declare partition='experiment', type='overall' and read
    the supplied store once; training may be None for that use. No model is fitted,
    no experiment is saved automatically, and exceptions identify their study.
    """
    reporters: dict = field(default_factory=dict)
    title: str = "Post-training analysis"
    fold_ids: object = None

    def run(self, training=None, *, fold_ids=None, experiment=None):
        report = AnalysisReport(title=self.title)
        all_folds = {} if training is None else {fold.fold_id: fold for fold in training.folds}
        fold_ids = self.fold_ids if fold_ids is None else fold_ids
        selected = list(all_folds) if fold_ids is None else list(fold_ids)
        if len(set(selected)) != len(selected) or any(i not in all_folds for i in selected):
            raise ValueError("fold_ids must select distinct fitted fold IDs.")
        for name, reporter in self.reporters.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Reporter names must be nonempty strings.")
            mode, partition = reporter.type, reporter.partition
            if mode not in reporter.supported_types or mode not in {"per_fold", "overall", "timeline"}:
                raise ValueError(f"Unsupported post-training execution type {mode!r}.")
            if partition == "experiment":
                if mode != "overall" or experiment is None:
                    raise ValueError("Experiment studies require overall type and an experiment store.")
                scopes = [(None, [])]
            elif partition in {"test", "score", "model"}:
                if not selected:
                    raise ValueError("Prediction studies require at least one fitted fold.")
                scopes = ([(i, [all_folds[i]]) for i in selected] if mode == "per_fold"
                          else [(None, [all_folds[i] for i in selected])])
            else:
                raise ValueError("Post-training partition must be test, score, model or experiment.")
            for fold_id, folds in scopes:
                pooling = getattr(reporter, "pooling", None)
                if partition in {"experiment", "model"}:
                    index = pd.MultiIndex.from_arrays([[], []], names=["fold_id", "row_position"])
                    y, predictions, meta = pd.DataFrame(index=index), {}, pd.DataFrame(index=index)
                else:
                    y, predictions, meta = _population(training, folds, partition, pooling)
                    if mode == "timeline":
                        times = pd.to_datetime(meta["kickoff_at"], utc=True, errors="raise")
                        if times.isna().any():
                            raise ValueError("Timeline rows require kickoff timestamps.")
                        order = np.argsort(times.to_numpy(), kind="stable")
                        y, meta = y.iloc[order], meta.iloc[order]
                        predictions = {key: frame.iloc[order] for key, frame in predictions.items()}
                context = PostTrainingContext(
                    y, predictions, meta, training.layout if training is not None else "experiment",
                    training.identity_columns if training is not None else (),
                    training.match_columns if training is not None else (), mode, partition, fold_id,
                    pooling or "occurrences", {fold.fold_id: fold.model for fold in folds},
                    deepcopy(training.definitions) if training is not None else {}, experiment,
                    {fold.fold_id: fold for fold in folds})
                scope = y.index.to_frame(index=False)
                positions, count = context.row_positions.copy(), context.n_matches
                try:
                    result = reporter.run(context)
                except Exception as error:
                    error.add_note(f"Post-training reporter {name!r}, type={mode}, partition={partition}, fold={fold_id}")
                    raise
                if not isinstance(result, StudyResult):
                    raise TypeError("Post-training reporters must return StudyResult.")
                report.studies.append(StudyRun(name, mode, partition, fold_id, context.layout,
                                               positions, count, result, scope,
                                               "saved experiment runs" if partition == "experiment" else
                                               "fitted model diagnostics" if partition == "model" else
                                               f"prediction pooling: {pooling or 'occurrences'}"))
        return report
