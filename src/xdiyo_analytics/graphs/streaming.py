"""Restartable issuance descriptors; one bounded source-row encoding cache."""
from functools import wraps
import traceback

import numpy as np
import pandas as pd
from .._ownership import isolated_frame


def release_temporary_frames(function):
    """Keep error stacks, but release completed frames' potentially large locals.

    A caught exception (including cancellation) can otherwise retain a snapshot
    through its traceback long after an operation has stopped.
    """
    @wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            # This wrapper is still executing, so clear_frames cannot clear its
            # own arguments (which may themselves include a raw group).
            args = kwargs = None
            raise
    return call


class GroupPlan:
    """Fixed input population for repeatable passes over one immutable store.

    Only fixture metadata, selected fixture inputs and ordered positions persist.
    Copies isolate the plan from subsequent caller edits, as eager tensors did.
    Full timestamps and per-group team populations are retained, not just dates.
    """
    def __init__(self, store, config, context, issue):
        self.store, self.config = store, config
        self.metadata = isolated_frame(context.metadata)
        self.X = isolated_frame(context.X.loc[:, list(config.fixture_features)])
        self.descriptors = tuple((time, positions.copy()) for time, positions
                                 in issue.groupby(issue, sort=True).indices.items())

    def __len__(self):
        return len(self.descriptors)

    @release_temporary_frames
    def raw(self, index, *, columns=None):
        time, positions = self.descriptors[index]
        meta = self.metadata.iloc[positions]
        # Series selection deep-copies pandas attrs. Read through a temporary
        # frame without attrs while returning the original metadata unchanged.
        inputs = pd.DataFrame(meta, copy=False)
        home_ids, away_ids = inputs.home_id.tolist(), inputs.away_id.tolist()
        teams = set(home_ids) | set(away_ids)
        snap = self.store.snapshot(time, self.config, teams, **({'_columns': columns} if columns is not None else {}))
        indices = {t: i for i, t in enumerate(snap.teams)}
        home = np.array([indices[t] for t in home_ids], dtype=np.int64)
        away = np.array([indices[t] for t in away_ids], dtype=np.int64)
        # Receiver positions already identify each edge's team. Reduce only
        # eligible snapshot rows, avoiding a fresh pandas team grouping.
        missing_time = np.iinfo(np.int64).min
        latest = np.full(len(snap.teams), missing_time, dtype=np.int64)
        np.maximum.at(latest, snap.receiver, snap.edges.kickoff_at.array.asi8)

        def rest(t):
            value = latest[indices[t]]
            # Keep scalar Timedelta arithmetic, including submicrosecond
            # truncation, identical to the original pandas groupby path.
            return ((time-pd.Timestamp(value, tz='UTC')).total_seconds()/86400
                    if value != missing_time else np.nan)

        neutral = inputs["neutral_venue"].to_numpy(dtype=float, na_value=np.nan) if "neutral_venue" in inputs else np.full(len(meta), np.nan)
        if np.any(np.isfinite(neutral) & ~np.isin(neutral, [0, 1])):
            raise ValueError("neutral_venue must be 0, 1 or missing.")
        numeric = np.column_stack((neutral, [rest(t) for t in home_ids],
                                   [rest(t) for t in away_ids],
                                   snap.support[home], snap.support[away]))
        if self.config.fixture_features:
            numeric = np.column_stack((numeric, self.X.iloc[positions].to_numpy(dtype=float, na_value=np.nan)))
        return positions, snap, home, away, meta, numeric

    def _summarize(self, index, numeric):
        positions, snap, _, _, _, values = self.raw(index)
        numeric[positions] = values
        return {x for x in snap.league if x is not None}

    @release_temporary_frames
    def fitting_inputs(self):
        numeric = np.empty((len(self.metadata), 5+len(self.config.fixture_features)))
        leagues = set()
        for index in range(len(self)):
            # The helper frame dies before the next snapshot is constructed.
            leagues.update(self._summarize(index, numeric))
        return numeric, leagues


@release_temporary_frames
def prepare_group(raw, preprocessor, device, store=None):
    import torch
    positions, snap, home, away, meta, numeric = raw
    data = preprocessor.native_tensors(snap, device, store) if store is not None else preprocessor.tensors(snap, device)
    return (positions, data,
            torch.as_tensor(home, device=device), torch.as_tensor(away, device=device),
            torch.as_tensor(preprocessor.fixture(meta, numeric), dtype=torch.float32, device=device))


class PreparedGroups:
    """Indexable, repeatable stream. Indexing reconstructs exactly one issuance.

    Python's sequence iterator retains no last yielded group. Native consumers
    additionally process each group inside a helper frame, so the previous
    tensors/autograd graph are gone before requesting the next one.
    """
    def __init__(self, plan, preprocessor, device, input_cache_policy=None):
        self.plan, self.preprocessor, self.device = plan, preprocessor, device
        from ..training.residency import InputResident
        self._resident = InputResident(device, input_cache_policy)
        if self._resident.pool is not None:
            from copy import copy, deepcopy
            self._fixed_plan = copy(plan)
            self._fixed_plan.metadata = isolated_frame(plan.metadata)
            self._fixed_plan.X = isolated_frame(plan.X)
            self._fixed_plan.descriptors = tuple((t, p.copy()) for t, p in plan.descriptors)
            self._fixed_preprocessor = type(preprocessor).from_state(deepcopy(preprocessor.config), deepcopy(preprocessor.state()))

    def lease(self, index):
        def cpu():
            p, prep = self._fixed_plan, self._fixed_preprocessor
            return prepare_group(p.raw(index, columns=prep.snapshot_columns(p.store)), prep, 'cpu', p.store)
        return self._resident.lease(index, cpu, lambda: self[index])

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        return prepare_group(self.plan.raw(index, columns=self.preprocessor.snapshot_columns(self.plan.store)),
                             self.preprocessor, self.device, self.plan.store)
