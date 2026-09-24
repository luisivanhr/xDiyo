"""Combinatorial purged membership and held-out prediction path assembly."""

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from .core import Splitter, _Matches, _duration, _fold, _integer


@dataclass
class CPCV(Splitter):
    """Retrospective combinatorial purged CV, not a past-only deployment replay.

    n_blocks chronological blocks are made from equal-kickoff batches using
    numpy.array_split, so simultaneous fixtures and paired rows stay together.
    All combinations of n_test_blocks are held out. Training intervals overlapping
    ANY test information interval are removed (closed endpoints). Embargo removes
    training kickoffs in (last test end, last test end + embargo] after EACH
    contiguous run of held-out blocks. embargo=None means zero duration.

    folds(..., information_start=None, available_at=None) accepts metadata column
    names or row-aligned datetimes. Defaults are point intervals at kickoff, a
    recorded retrospective assumption. Supply prediction/label-start and actual
    outcome/settlement availability for meaningful interval purging. Within a
    match, intervals are conservatively combined using minimum start/maximum end.

    Stateful features need separate fold-aware reconstruction/exclusion in the
    fitting layer: finite intervals/embargo cannot automatically remove outcomes
    carried indefinitely by ratings. This class never computes features, fits a
    model, settles bets, or calculates DSR, PBO or false-discovery statistics.
    """

    n_blocks: int = 6
    n_test_blocks: int = 2
    embargo: object = None

    @property
    def n_paths(self):
        self._check()
        return comb(self.n_blocks - 1, self.n_test_blocks - 1)

    def _check(self):
        _integer(self.n_blocks, "n_blocks", 2)
        _integer(self.n_test_blocks, "n_test_blocks")
        if self.n_test_blocks >= self.n_blocks:
            raise ValueError("n_test_blocks must be smaller than n_blocks.")

    def folds(self, dataset, *, information_start=None, available_at=None):
        self._check()
        embargo = _duration(self.embargo, "embargo")
        matches = _Matches(dataset)
        kicks = matches.times("kickoff_at", "kickoff_at")
        starts = kicks if information_start is None else matches.times(
            information_start, "information_start", "min")
        ends = kicks if available_at is None else matches.times(
            available_at, "available_at", "max")
        if (starts > ends).any():
            raise ValueError("Information interval starts must not exceed their ends.")
        if (ends < kicks).any():
            raise ValueError("Outcome availability cannot be earlier than kickoff.")
        batches = kicks.unique().sort_values()
        if len(batches) < self.n_blocks:
            raise ValueError("n_blocks exceeds the number of distinct kickoff batches.")
        blocks = [np.flatnonzero(kicks.isin(batch))
                  for batch in np.array_split(batches, self.n_blocks)]
        ids = np.arange(len(kicks))
        folds = []
        for selected in combinations(range(self.n_blocks), self.n_test_blocks):
            test = np.concatenate([blocks[b] for b in selected])
            candidate = np.setdiff1d(ids, test)
            # Merge only overlapping test intervals, not gaps between disjoint events.
            intervals = sorted(zip(starts[test], ends[test]))
            merged = []
            for start, end in intervals:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            purged = np.zeros(len(candidate), dtype=bool)
            for start, end in merged:
                purged |= (starts[candidate] <= end) & (ends[candidate] >= start)
            embargoed = np.zeros(len(candidate), dtype=bool)
            runs = np.split(np.asarray(selected), np.flatnonzero(np.diff(selected) != 1) + 1)
            for run in runs:
                end = ends[np.concatenate([blocks[b] for b in run])].max()
                embargoed |= (kicks[candidate] > end) & (kicks[candidate] <= end + embargo)
            folds.append(_fold(matches, candidate[~(purged | embargoed)], test, {
                "scheme": "cpcv", "retrospective": True,
                "test_blocks": selected,
                "block_rows": {b: matches.rows(blocks[b]) for b in selected},
                "purged_train": matches.rows(candidate[purged]),
                "embargoed_train": matches.rows(candidate[embargoed & ~purged]),
                "information_start": "kickoff_proxy" if information_start is None else "explicit",
                "availability": "kickoff_proxy" if available_at is None else "explicit",
                "embargo": embargo,
            }))
        return folds

    def path_map(self, folds):
        """Assign each block's fold occurrences once across all reconstructed paths.

        Paths reuse observations and models; they are not independent samples.
        Stable lexicographic fold order determines the assignment, as in the
        reference notebook's incidence-matrix procedure.
        """
        occurrences = np.zeros(self.n_blocks, dtype=int)
        records = []
        for fold_id, fold in enumerate(folds):
            for block in fold.metadata["test_blocks"]:
                path_id = int(occurrences[block])
                records.append({"path_id": path_id, "block_id": block, "fold_id": fold_id,
                                "rows": fold.metadata["block_rows"][block].copy()})
                occurrences[block] += 1
        if not (occurrences == self.n_paths).all():
            raise ValueError("Path reconstruction requires the complete CPCV fold collection.")
        return pd.DataFrame(records).sort_values(["path_id", "block_id"]).reset_index(drop=True)


def reconstruct_paths(plan, predictions):
    """Return long-form held-out predictions with path/fold/original-row identities.

    predictions is a sequence (or integer fold_id mapping) of numeric DataFrames.
    Each frame must be indexed by exactly fold.test's ORIGINAL dataset positions,
    in any order, and have identical named prediction columns. Multiple outputs
    (e.g. probabilities) are supported. NaNs remain missing; they never become
    losses or zeros. No training/full-dataset predictions are accepted.

    Output columns are path_id, row_position, fold_id, then the prediction columns.
    Within each path, rows follow kickoff order (stable for ties). The model and
    fitting provenance of the provided values remains the caller's responsibility.
    """
    if plan.paths is None:
        raise ValueError("Path reconstruction needs a CPCV SplitPlan.")
    if len(predictions) != len(plan.folds):
        raise ValueError("Supply predictions for every fold exactly once.")
    frames, columns = [], None
    reserved = {"path_id", "row_position", "fold_id"}
    for fold_id, fold in enumerate(plan.folds):
        frame = predictions[fold_id]
        if not isinstance(frame, pd.DataFrame) or not frame.index.is_unique:
            raise ValueError("Fold predictions must be DataFrames with unique row-position indexes.")
        if (not pd.api.types.is_integer_dtype(frame.index.dtype) or
                set(frame.index) != set(fold.test)):
            raise ValueError("Prediction index must equal the fold's held-out row positions.")
        if not frame.columns.is_unique or not len(frame.columns) or reserved.intersection(frame.columns):
            raise ValueError("Use distinct prediction columns excluding path/row/fold metadata names.")
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
            raise ValueError("Predictions must be numeric (missing values are permitted).")
        if columns is None:
            columns = frame.columns
        elif not columns.equals(frame.columns):
            raise ValueError("All folds must provide identical ordered prediction columns.")
        frames.append(frame)
    pieces = []
    for item in plan.paths.itertuples(index=False):
        part = frames[item.fold_id].loc[item.rows].copy()
        part.insert(0, "fold_id", item.fold_id)
        part.insert(0, "row_position", item.rows)
        part.insert(0, "path_id", item.path_id)
        pieces.append(part.reset_index(drop=True))
    result = pd.concat(pieces, ignore_index=True)
    rank = np.empty(plan.n_rows, dtype=int)
    rank[plan.row_order] = np.arange(plan.n_rows)
    order = np.lexsort((rank[result["row_position"].to_numpy()], result["path_id"].to_numpy()))
    return result.iloc[order].reset_index(drop=True)
