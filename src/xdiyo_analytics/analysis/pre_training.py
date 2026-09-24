"""Select row scopes and execute explicitly requested, model-free reporters."""

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..datasets import ModelDataset
from ..reporting.contracts import AnalysisContext, AnalysisReport, StudyResult, StudyRun


@dataclass
class PreTrainingAnalysis:
    """Run zero or more named reporters without choosing or fitting a model.

    reporters is an ordered name -> Reporter mapping; {} runs no studies.
    Each reporter must select type and partition explicitly:

    * per_fold: run once for each selected fold and its chosen partition.
    * overall: run once over the chosen population in input order.
    * timeline: run once over that population sorted by UTC kickoff (stable ties).

    partition is train, test, score or all. For per_fold, all is the UNION of that
    fold's train/test rows, excluding unassigned rows such as gaps. For overall
    and timeline, all means the full supplied dataset; other partitions are the
    unique union across selected folds. A union may include nearly every match
    in retrospective/repeated CV: it is not an independent held-out evaluation.
    Without a split plan, only overall/timeline with partition=all is available.
    fold_ids optionally selects zero-based folds; None means every fold.

    Contexts isolate inputs per invocation. Exceptions identify the failed study
    and propagate; no silent omission or substitution of a different analysis.

    A selector is a reporter returning StudyResult.selection. It does not alter
    subsequent inputs automatically. Reporters may specify features_from='name'
    to use an earlier selector's columns for the same fold, or an explicitly
    named overall/consensus selection. Overall selection reuse is not nested CV.
    A per-fold train selection can thereby be applied to that fold's test/score
    rows. source='name' on TopKCorrelationSelector instead reuses an earlier
    correlation calculation, requiring identical partition and row membership.
    Named dependencies must precede consumers in the mapping. No CV aggregation
    of selections or automatic choice of a global selected feature set occurs.
    """

    reporters: dict = field(default_factory=dict)
    title: str = "Pre-training analysis"

    def run(self, dataset, *, split_plan=None, fold_ids=None):
        if not isinstance(dataset, ModelDataset):
            raise TypeError("PreTrainingAnalysis consumes an assembled ModelDataset.")
        if not (dataset.X.index.equals(dataset.metadata.index) and
                dataset.y.index.equals(dataset.metadata.index)):
            raise ValueError("Dataset X, y and metadata must remain aligned.")
        n = len(dataset.metadata)
        if split_plan is not None and split_plan.n_rows != n:
            raise ValueError("Split plan must refer to this dataset's original row order/count.")
        if split_plan is None and fold_ids is not None:
            raise ValueError("fold_ids requires a split plan.")
        selected = [] if split_plan is None else (
            list(range(len(split_plan.folds))) if fold_ids is None else list(fold_ids))
        if len(set(selected)) != len(selected) or any(
                isinstance(i, bool) or not isinstance(i, (int, np.integer)) or
                i < 0 or i >= len(split_plan.folds) for i in selected):
            raise ValueError("fold_ids must contain distinct valid zero-based fold numbers.")
        report = AnalysisReport(title=self.title)
        for name, reporter in self.reporters.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Reporter names must be nonempty strings.")
            mode, partition = reporter.type, reporter.partition
            if mode not in {"per_fold", "overall", "timeline"} or mode not in reporter.supported_types:
                raise ValueError(f"Reporter {name!r} does not support type={mode!r}.")
            if partition not in {"train", "test", "score", "all"}:
                raise ValueError("partition must be train, test, score or all.")
            if (mode == "per_fold" or partition != "all") and not selected:
                raise ValueError(f"Reporter {name!r} requires at least one selected fold.")

            def rows_for(fold):
                rows = (np.union1d(fold.train, fold.test) if partition == "all"
                        else getattr(fold, partition))
                rows = np.asarray(rows)
                if rows.ndim != 1 or rows.dtype.kind not in "iu" or ((rows < 0) | (rows >= n)).any():
                    raise ValueError("Fold row positions are outside the dataset.")
                return np.unique(rows)

            if mode == "per_fold":
                scopes = [(i, rows_for(split_plan.folds[i])) for i in selected]
            else:
                rows = (np.arange(n) if partition == "all" else
                        np.unique(np.concatenate([rows_for(split_plan.folds[i]) for i in selected])))
                if mode == "timeline":
                    times = pd.to_datetime(dataset.metadata.iloc[rows]["kickoff_at"], utc=True)
                    if times.isna().any():
                        raise ValueError("Timeline execution requires nonmissing kickoff timestamps.")
                    rows = rows[np.argsort(times.to_numpy(), kind="stable")]
                scopes = [(None, rows)]
            for fold_id, positions in scopes:
                frames = [frame.iloc[positions].copy(deep=True)
                          for frame in (dataset.X, dataset.y, dataset.metadata)]
                for frame in frames:
                    frame.index = pd.Index(positions, name="row_position")
                feature_source = getattr(reporter, "features_from", None)
                if feature_source is not None:
                    candidates = [run for run in report.studies if run.name == feature_source and run.fold_id == fold_id]
                    if not candidates:
                        candidates = [run for run in report.studies if run.name == feature_source and run.fold_id is None]
                    if len(candidates) != 1 or candidates[0].result.selection is None:
                        raise ValueError(f"features_from={feature_source!r} needs an earlier selector for the same fold/scope.")
                    frames[0] = frames[0].loc[:, list(candidates[0].result.selection.columns)].copy()
                previous = {run.name: run.result for run in report.studies
                            if run.fold_id == fold_id and run.partition == partition and
                            np.array_equal(run.row_positions, positions)}
                fold_rows = {i: rows_for(split_plan.folds[i]) for i in selected}
                fold_results = {}
                for run in report.studies:
                    if (run.fold_id in selected and run.partition == partition and
                            np.array_equal(run.row_positions, fold_rows[run.fold_id])):
                        fold_results.setdefault(run.name, {})[run.fold_id] = run.result
                context = AnalysisContext(
                    *frames, dataset.layout, dataset.match_columns, positions.copy(), mode,
                    partition, fold_id,
                    {} if fold_id is None else deepcopy(split_plan.folds[fold_id].metadata),
                    deepcopy(dataset.definitions), deepcopy(previous), fold_rows, deepcopy(fold_results),
                    fold_metadata_by_id={i: deepcopy(split_plan.folds[i].metadata) for i in selected})
                n_matches = context.n_matches
                try:
                    result = reporter.run(context)
                except Exception as error:
                    error.add_note(f"Reporter {name!r}, type={mode}, partition={partition}, fold={fold_id}")
                    raise
                if not isinstance(result, StudyResult):
                    raise TypeError(f"Reporter {name!r} must return StudyResult.")
                if result.selection is not None:
                    columns = list(result.selection.columns)
                    if len(set(columns)) != len(columns) or set(columns) - set(frames[0].columns):
                        raise ValueError("A selector must return distinct columns from its input features.")
                if feature_source is not None:
                    source_run = candidates[0]
                    source_scope = "overall/consensus" if source_run.fold_id is None else f"fold {source_run.fold_id}"
                    result.notes.append(f"Input features selected by {feature_source!r} ({source_scope}, {source_run.partition}).")
                report.studies.append(StudyRun(name, mode, partition, fold_id, dataset.layout,
                                               positions.copy(), n_matches, result))
        return report
