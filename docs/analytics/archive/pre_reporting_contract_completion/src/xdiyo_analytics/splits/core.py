"""Match-aware split records and small, model-independent evaluation plans."""

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
import pandas as pd

from ..datasets import ModelDataset


@dataclass
class Fold:
    """Integer positions into the original dataset, for use with .iloc.

    score is a subset of test; unscored test rows do not become training rows.
    metadata describes the split, including whether it is retrospective.
    """

    train: np.ndarray
    test: np.ndarray
    score: np.ndarray
    metadata: dict = field(default_factory=dict)


@dataclass
class SplitPlan:
    """Prepared folds; no model fitting, selection or refitting is performed.

    Selection and refitting configuration belongs to the future training runner.
    paths maps CPCV (path_id, block_id) to fold_id and row positions. Reporters
    can inspect these folds independently of any model or fitting configuration.
    """

    folds: list[Fold]
    n_rows: int
    row_order: np.ndarray
    paths: object = None

    def split(self, X=None, y=None, groups=None):
        """Yield train/test arrays; pass this iterator as sklearn's cv argument.

        X, if supplied, must retain the dataset's original row order and length.
        The caller owns this alignment; this method never refits or subsets X.
        """
        if X is not None and len(X) != self.n_rows:
            raise ValueError("X must retain the prepared dataset's row count/order.")
        for fold in self.folds:
            yield fold.train.copy(), fold.test.copy()

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.folds)


def create_split_plan(dataset, splitter, **split_options):
    """Prepare model-independent fold membership and optional CPCV path mappings.

    split_options passes row-aligned timing/group inputs to splitter.folds().
    This operation neither computes features nor configures or performs fitting,
    model selection or refitting. Those responsibilities belong to other layers.
    """
    folds = splitter.folds(dataset, **split_options)
    paths = splitter.path_map(folds) if hasattr(splitter, "path_map") else None
    if "kickoff_at" in dataset.metadata:
        times = _times(dataset, "kickoff_at", "kickoff_at")
        order = np.argsort(times.to_numpy(), kind="stable")
    else:
        order = np.arange(len(dataset.metadata))
    return SplitPlan(folds, len(dataset.metadata), order, paths)


class Splitter:
    """Convenience interface; folds() also exposes scoring rows and metadata."""

    def split(self, dataset, **options):
        for fold in self.folds(dataset, **options):
            yield fold.train.copy(), fold.test.copy()

    def get_n_splits(self, dataset, **options):
        return len(self.folds(dataset, **options))


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _duration(value, name):
    result = pd.Timedelta(0) if value is None else pd.Timedelta(value)
    if pd.isna(result) or result < pd.Timedelta(0):
        raise ValueError(f"{name} must be a nonnegative duration.")
    return result


def _times(dataset, value, name):
    """Column name or UTC-convertible row-aligned vector; never infer units."""
    if isinstance(value, str):
        values = dataset.metadata[value]
    else:
        values = value
    if isinstance(values, pd.Series) and not values.index.equals(dataset.metadata.index):
        raise ValueError(f"{name} must have the dataset metadata index/order.")
    if len(values) != len(dataset.metadata):
        raise ValueError(f"{name} must contain one timestamp per dataset row.")
    values = pd.Series(values, index=dataset.metadata.index)
    if pd.api.types.is_numeric_dtype(values.dtype):
        raise ValueError(f"{name} needs datetime values; convert integer epochs explicitly.")
    return pd.to_datetime(values, utc=True, errors="raise")


class _Matches:
    """One exact identity per match and reversible dataset-position membership."""

    def __init__(self, dataset):
        if not isinstance(dataset, ModelDataset):
            raise TypeError("Splitters consume an assembled ModelDataset.")
        meta = dataset.metadata
        if not (len(dataset.X) == len(dataset.y) == len(meta)):
            raise ValueError("X, y and metadata must remain aligned.")
        if not dataset.X.index.equals(meta.index) or not dataset.y.index.equals(meta.index):
            raise ValueError("X, y and metadata must share the same index/order.")
        if not len(meta) or meta[list(dataset.match_columns)].isna().any().any():
            raise ValueError("Splitting needs nonempty data with complete match identities.")
        # Factorize exact tuples: never cast an ID through floating point.
        self.codes, self.keys = pd.factorize(dataset.groups, sort=False)
        positions = [[] for _ in self.keys]
        for row, code in enumerate(self.codes):
            positions[code].append(row)
        self.positions = [np.asarray(rows, dtype=int) for rows in positions]
        self.first = np.array([rows[0] for rows in self.positions], dtype=int)
        self.dataset = dataset
        self.meta = meta.iloc[self.first].reset_index(drop=True)

    def rows(self, matches):
        return np.flatnonzero(np.isin(self.codes, np.asarray(matches, dtype=int)))

    def constant(self, values, name):
        values = np.asarray(values, dtype=object)
        if len(values) != len(self.codes):
            raise ValueError(f"{name} needs one value per dataset row.")
        for rows in self.positions:
            if pd.Series(values[rows]).nunique(dropna=False) != 1:
                raise ValueError(f"{name} must be identical for all rows of a match.")
        return values[self.first]

    def columns(self, names):
        names = (names,) if isinstance(names, str) else tuple(names)
        for name in names:
            values = self.dataset.metadata[name]
            if values.isna().any():
                raise ValueError(f"Grouping column {name!r} contains missing values.")
            self.constant(values, name)
        return names

    def times(self, value, name, reduction=None):
        times = _times(self.dataset, value, name)
        if times.isna().any():
            raise ValueError(f"{name} contains missing timestamps.")
        if reduction is None:
            return pd.DatetimeIndex(self.constant(times, name))
        return pd.DatetimeIndex([getattr(times.iloc[rows], reduction)()
                                 for rows in self.positions])


def _fold(matches, train, test, metadata, score=None):
    train_rows, test_rows = matches.rows(train), matches.rows(test)
    if not len(train_rows) or not len(test_rows):
        raise ValueError("A fold has no training or test matches; adjust spans/purging.")
    return Fold(train_rows, test_rows,
                test_rows.copy() if score is None else matches.rows(score), metadata)


@dataclass
class MatchKFold(Splitter):
    """Non-temporal K-fold over matches, optionally shuffled reproducibly.

    Without shuffle, unique matches follow their first appearance in the input.
    No chronological or out-of-team generalization claim is implied.
    """

    n_splits: int = 5
    shuffle: bool = False
    random_state: object = None

    def folds(self, dataset):
        _integer(self.n_splits, "n_splits", 2)
        matches = _Matches(dataset)
        ids = np.arange(len(matches.keys))
        if len(ids) < self.n_splits:
            raise ValueError("n_splits exceeds the number of matches.")
        if self.shuffle:
            ids = np.random.default_rng(self.random_state).permutation(ids)
        return [_fold(matches, np.setdiff1d(ids, test), test,
                      {"scheme": "match_kfold", "retrospective": True})
                for test in np.array_split(ids, self.n_splits)]


@dataclass
class GroupKFold(Splitter):
    """Keep selected metadata groups disjoint, while always keeping matches whole.

    group_by defaults to competition-season; alternatively pass row-aligned
    groups to folds(). Values must be constant within each match. Groups are
    greedily assigned to balance MATCH counts; shuffle randomizes equal-size
    tie ordering only. This is not sklearn's exact shuffled allocation rule.
    """

    n_splits: int = 5
    group_by: tuple = ("competition_id", "season_id")
    shuffle: bool = False
    random_state: object = None

    def folds(self, dataset, *, groups=None):
        _integer(self.n_splits, "n_splits", 2)
        matches = _Matches(dataset)
        if groups is None:
            names = matches.columns(self.group_by)
            if not names:
                raise ValueError("group_by cannot be empty.")
            keys = pd.MultiIndex.from_frame(matches.meta[list(names)])
        else:
            if isinstance(groups, pd.Series) and not groups.index.equals(dataset.metadata.index):
                raise ValueError("groups must share the dataset index/order.")
            values = pd.Series(list(groups), dtype=object)
            if values.isna().any():
                raise ValueError("groups contains missing values.")
            keys = matches.constant(values, "groups")
        codes, keys = pd.factorize(keys, sort=False)
        if len(keys) < self.n_splits:
            raise ValueError("n_splits exceeds the number of distinct groups.")
        sizes = np.bincount(codes)
        order = np.arange(len(keys))
        if self.shuffle:
            order = np.random.default_rng(self.random_state).permutation(order)
        order = order[np.argsort(-sizes[order], kind="stable")]
        allocation, loads = np.empty(len(keys), dtype=int), np.zeros(self.n_splits, dtype=int)
        for group in order:
            fold = int(np.argmin(loads))
            allocation[group] = fold
            loads[fold] += sizes[group]
        return [_fold(matches, np.flatnonzero(allocation[codes] != i),
                      np.flatnonzero(allocation[codes] == i),
                      {"scheme": "group_kfold", "retrospective": True,
                       "test_groups": tuple(keys[allocation == i])})
                for i in range(self.n_splits)]
