"""Availability-indexed history and input-only plans for temporal attention.

No learned states persist here. A plan deduplicates (team, query time, layer)
dependencies; the network evaluates them afresh, with gradients, on every pass.
"""
from types import SimpleNamespace
import math

import numpy as np
import pandas as pd

from .config import ENTRY_ORDER, MATCH_KEYS


def time_feature_names(config):
    names = []
    for kind in ('age', 'receiver_gap', 'opponent_gap'):
        scale = config.age_scale_days if kind == 'age' else config.gap_scale_days
        first = ('log1p_days' if config.time_encoding == 'log1p_v1'
                 else f'rational_decay_{scale:g}_days')
        names.append(f'{kind}::{first}')
        names.extend(f'{kind}::half_life_{scale:g}_days' for scale in config.time_scales_days)
        if kind != 'age':
            names.append(kind + '::missing')
    return tuple(names)


def time_encoding_key(config):
    """Immutable prepared-input identity, independent of archive and batch size."""
    return (config.temporal_mode, config.time_encoding, config.age_scale_days,
            config.gap_scale_days, config.time_scales_days)


def rational_decay(days, scale):
    """Fixed-scale tau/(tau + days), without overflow in the sum or ratio."""
    if (isinstance(scale, bool) or not isinstance(scale, (int, float))
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError('Time scale must be finite and positive.')
    days = np.asarray(days, dtype=np.float64)
    if not np.isfinite(days).all() or (days < 0).any():
        raise ValueError('Known durations must be finite and nonnegative.')
    near = days <= scale
    ratio = np.empty_like(days)
    np.divide(days, scale, out=ratio, where=near)
    np.divide(scale, days, out=ratio, where=~near)
    return np.where(near, 1., ratio) / (1. + ratio)


def encode_time_features(age, receiver_gap, opponent_gap, config):
    """Vectorized age + receiver gap + opponent gap; only gaps may be NaN."""
    values = [np.asarray(v, dtype=np.float64) for v in (age, receiver_gap, opponent_gap)]
    if any(v.ndim != 1 or v.shape != values[0].shape for v in values):
        raise ValueError('Temporal durations must be matching one-dimensional arrays.')
    columns = []
    scales = np.asarray(config.time_scales_days)
    for i, duration in enumerate(values):
        missing = np.isnan(duration) if i else np.zeros(duration.shape, dtype=bool)
        known = duration[~missing]
        if not np.isfinite(known).all() or (known < 0).any():
            raise ValueError('Known durations must be finite and nonnegative.')
        days = np.where(missing, 0., duration)
        scale = config.gap_scale_days if i else config.age_scale_days
        first = (np.log1p(days) if config.time_encoding == 'log1p_v1'
                 else rational_decay(days, scale))
        # For tiny valid half-lives an overflowing ratio has the correct limit 0.
        with np.errstate(over='ignore'):
            phi = np.column_stack((first, np.exp2(-days[:, None] / scales)))
        phi[missing] = 0.
        columns.append(phi)
        if i:
            columns.append(missing.astype(float)[:, None])
    return np.concatenate(columns, axis=1).astype(np.float32)


class TemporalHistoryIndex:
    """Immutable per-team release index, including delayed and same-day results."""

    def __init__(self, store, config):
        self.store, self.config = store, config
        self.teams = store._history.team_id.tolist()
        self.opponents = store._history.opponent_id.tolist()
        self.by_team = {}
        self._gaps = {}
        valid = store._finished & (store._available_ns != np.iinfo(np.int64).min)
        # A kickoff at the largest representable timestamp cannot precede any
        # representable cutoff. Exclude it before forming the exclusive bound.
        valid &= store._kickoff_ns < np.iinfo(np.int64).max
        for team, positions in store._history.groupby('team_id', sort=False).indices.items():
            rows = positions[valid[positions]]
            release = np.maximum(store._kickoff_ns[rows] + 1, store._available_ns[rows])
            order = np.argsort(release, kind='stable')
            rows, release = rows[order], release[order]
            latest = np.maximum.accumulate(store._kickoff_ns[rows])
            self.by_team[team] = rows, release, latest

    def cutoff(self, time_ns):
        return self.config.history_cutoff(pd.Timestamp(time_ns, tz='UTC')).value

    def eligible_rows(self, team, cutoff_ns):
        index = self.by_team.get(team)
        if index is None:
            return np.empty(0, dtype=np.int64)
        rows, release, _ = index
        # Restore immutable source order for stable attention reductions.
        return np.sort(rows[:np.searchsorted(release, cutoff_ns, side='right')])

    def gap(self, team, time_ns):
        key = (team, time_ns)
        if key not in self._gaps:
            index = self.by_team.get(team)
            n = np.searchsorted(index[1], self.cutoff(time_ns), side='right') if index is not None else 0
            self._gaps[key] = ((time_ns - int(index[2][n-1])) / 86400e9 if n else np.nan)
        return self._gaps[key]

    def context(self, team, time_ns):
        membership = self.store._membership_by_team.get(team)
        n = np.searchsorted(membership[0], time_ns, side='right') if membership is not None else 0
        return (membership[1][n-1] if n else None,
                membership[2][n-1] if n else 'unknown', n)

    def features(self, rows, times_ns):
        rows = np.asarray(rows, dtype=np.int64)
        kicks = self.store._kickoff_ns[rows]
        age = (np.asarray(times_ns, dtype=np.int64) - kicks) / 86400e9
        receiver_gap = np.array([self.gap(self.teams[r], int(t)) for r, t in zip(rows, kicks)])
        opponent_gap = np.array([self.gap(self.opponents[r], int(t)) for r, t in zip(rows, kicks)])
        return encode_time_features(age, receiver_gap, opponent_gap, self.config)


class TimeIndexedPlan:
    """A finite dependency DAG, batched by depth, with exact root-edge ordering."""

    def __init__(self, index, snapshot, layers):
        self.index = index
        self.issuance_ns = snapshot.issuance.value
        self.queries = [None] * (layers + 1)
        self.batches = [None] * layers
        roots = tuple((team, snapshot.issuance.value) for team in snapshot.teams)
        self.queries[layers] = roots
        raw_queries = {}
        root_count = len(roots)
        for depth in range(layers, 0, -1):
            queries = self.queries[depth]
            lower = dict.fromkeys(queries)
            lower = {key: i for i, key in enumerate(lower)}
            edge_rows, receivers, senders, times, weights, gates = [], [], [], [], [], []
            root_edges = {}
            offset = 0
            for position, (team, time_ns) in enumerate(queries):
                key = (team, time_ns)
                if key not in raw_queries:
                    rows = index.eligible_rows(team, index.cutoff(time_ns))
                    _, _, count = index.context(team, time_ns)
                    age = (time_ns - index.store._kickoff_ns[rows]) / 86400e9
                    crossed = count - index.store._entries_at_event[rows]
                    logw = (-math.log(2) * age / index.config.half_life_days
                            + crossed * math.log(index.config.seasonal_retention))
                    support = np.exp(logw).sum()
                    raw_queries[key] = rows, logw, support / (support + index.config.support_kappa)
                rows, logw, gate = raw_queries[key]
                sender = []
                for row in rows:
                    historical = (index.opponents[row], int(index.store._kickoff_ns[row]))
                    if historical not in lower:
                        lower[historical] = len(lower)
                    sender.append(lower[historical])
                edge_rows.extend(rows)
                receivers.extend([position] * len(rows))
                senders.extend(sender)
                times.extend([time_ns] * len(rows))
                weights.extend(logw)
                gates.append(gate)
                if position < root_count:
                    root_edges.update((int(row), offset + i) for i, row in enumerate(rows))
                offset += len(rows)
            self.queries[depth-1] = tuple(lower)
            self.batches[depth-1] = dict(
                rows=np.asarray(edge_rows, dtype=np.int64),
                times=np.asarray(times, dtype=np.int64),
                receiver=np.asarray(receivers, dtype=np.int64),
                sender=np.asarray(senders, dtype=np.int64),
                self_index=np.asarray([lower[q] for q in queries], dtype=np.int64),
                log_weight=np.asarray(weights, dtype=np.float64),
                gate=np.asarray(gates, dtype=np.float32),
                root_edge_index=np.asarray([root_edges[int(r)] for r in snapshot.edges.index], dtype=np.int64))

    def tensors(self, preprocessor, device):
        import torch

        def tensor(value, dtype=torch.float32):
            return torch.as_tensor(value, device=device, dtype=dtype)

        contexts = [self.index.context(team, time) for team, time in self.queries[0]]
        base = dict(league=tensor([preprocessor.leagues.get(c[0], 0) for c in contexts], torch.long),
                    entry=tensor([ENTRY_ORDER.index(c[1]) for c in contexts], torch.long))
        batches = []
        for batch in self.batches:
            # Reuse only fitted source-row transforms. Timing is query-specific.
            source = SimpleNamespace(store_version=self.index.store.version,
                                     edges=pd.DataFrame(index=batch['rows']))
            x = preprocessor.cached_edges(source, self.index.store)
            x = np.concatenate((x, self.index.features(batch['rows'], batch['times'])), axis=1)
            data = {key: tensor(batch[key], torch.long) for key in
                    ('receiver', 'sender', 'self_index', 'root_edge_index')}
            data.update(x=tensor(x), log_weight=tensor(batch['log_weight'], torch.float64),
                        gate=tensor(batch['gate']))
            batches.append(data)
        return dict(temporal_base=base, temporal_layers=tuple(batches))

    def evidence(self, team, layers):
        if not 0 <= layers < len(self.queries):
            raise ValueError('Evidence depth exceeds the prepared historical query plan.')
        key = (team, self.issuance_ns)
        if key not in self.queries[layers]:
            return ()
        frontier = {self.queries[layers].index(key)}
        rows = set()
        for depth in range(layers, 0, -1):
            batch = self.batches[depth-1]
            mask = np.isin(batch['receiver'], list(frontier))
            rows.update(batch['rows'][mask].tolist())
            frontier = set(batch['sender'][mask].tolist()) | set(batch['self_index'][list(frontier)].tolist())
        frame = self.index.store._history.iloc[sorted(rows)]
        return tuple(sorted(set(frame[list(MATCH_KEYS)].itertuples(index=False, name=None))))
