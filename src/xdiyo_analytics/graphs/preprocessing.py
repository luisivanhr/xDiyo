"""Fit-only numeric transforms and categorical maps; missing values get masks."""
import numpy as np
import pandas as pd
from dataclasses import asdict
import json
from .config import ENTRY_ORDER
from .profiling import phase, timed


class MaskedScale:
    def fit(self, values):
        a = np.asarray(values, dtype=float)
        if np.isinf(a).any():
            raise ValueError("Infinite features are not supported.")
        count = np.isfinite(a).sum(0)
        self.mean = np.nansum(a, axis=0) / np.maximum(count, 1)
        variance = np.nansum((a-self.mean)**2, axis=0) / np.maximum(count, 1)
        self.scale = np.where(variance > 0, np.sqrt(variance), 1.)
        return self

    def transform(self, values):
        a = np.asarray(values, dtype=float)
        if np.isinf(a).any():
            raise ValueError("Infinite features are not supported.")
        missing = np.isnan(a)
        filled = np.where(missing, self.mean, a)
        return np.concatenate(((filled-self.mean)/self.scale, missing.astype(float)), axis=1)

    def state(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state(cls, state):
        obj = cls()
        obj.mean, obj.scale = np.array(state["mean"]), np.array(state["scale"])
        return obj


class GraphPreprocessor:
    _edge_cache_limit = 128 * 1024**2

    def __init__(self, config):
        self.config = config
        goals = ("goals_for", "goals_against") if config.include_goal_features else ()
        self.edge_columns = (*goals, *config.statistics)
        self.fixture_columns = ("neutral_venue", "home_rest_days", "away_rest_days",
                                "home_support", "away_support", *config.fixture_features)
        # One CPU source-row cache shared by every stream using this fitted
        # preprocessor. No graph/device/learned-output cache is retained.
        self._edge_cache = None

    def numeric_edges(self, frame):
        if not set(self.edge_columns) <= set(frame):
            raise ValueError("Selected match-level statistics are absent from history.")
        # Permit only paired native match-stat columns, never rolling histories.
        for col in self.config.statistics:
            if not col.startswith(("team::", "opponent::")):
                raise ValueError("statistics must select build_team_history match-stat columns.")
            role, rest = col.split("::", 1)
            paired = ("opponent" if role == "team" else "team") + "::" + rest
            if paired not in frame:
                raise ValueError("Selected statistics require both paired match perspectives.")
        return frame[list(self.edge_columns)].to_numpy(dtype=float, na_value=np.nan)

    def fit(self, history, snapshots, fixtures, numeric, *, extra_leagues=()):
        self._edge_cache = None
        categories = set(history.competition_id.tolist()) | set(fixtures.competition_id.tolist()) | set(extra_leagues)
        for snap in snapshots:
            categories.update(x for x in snap.league if x is not None)
        self.leagues = {int(key): n+1 for n, key in enumerate(sorted(categories))}
        self.edge_scale = MaskedScale().fit(self.numeric_edges(history))
        self.fixture_scale = MaskedScale().fit(numeric)
        return self

    def one_hot(self, leagues):
        return np.eye(len(self.leagues)+1)[[self.leagues.get(x, 0) for x in leagues]]

    def edges(self, snapshot):
        return self._encode_edges(snapshot.edges)

    def _encode_edges(self, h):
        return np.concatenate((self.edge_scale.transform(self.numeric_edges(h)),
                               h.side.eq("home").to_numpy(dtype=float)[:, None],
                               self.one_hot(h.competition_id)), axis=1)

    @timed('row_encoding')
    def cached_edges(self, snapshot, store):
        """Internal native-snapshot path; arbitrary caller frames use edges().

        Cache only rows actually requested, so invalid future observations do
        not fail an earlier forecast. Arithmetic stays float64 until the same
        float32 cast used by torch. Identity is rechecked after scaler edits,
        store replacement, or schema/configuration changes. Budget overflow
        falls back to the original per-snapshot transform.
        """
        if snapshot.store_version != store.version:
            raise ValueError('Cannot encode a snapshot from a different event store.')
        identity = (id(store), store.version, tuple(self.edge_columns),
                    json.dumps(asdict(self.config), sort_keys=True),
                    json.dumps(self.state(), sort_keys=True))
        cache = self._edge_cache
        if cache is not None and cache[0] != identity:
            self._edge_cache = cache = None
        width = 2*len(self.edge_columns) + len(self.leagues) + 2
        count = len(store._history)
        rows = snapshot.edges.index.to_numpy(dtype=np.int64)
        if count*(width*4+1) > self._edge_cache_limit:
            self._edge_cache = None
            return self._encode_edges(store._history.iloc[rows]).astype(np.float32)
        if not len(rows):
            return self._encode_edges(store._history.iloc[rows]).astype(np.float32)
        if cache is None:
            cache = (identity, np.empty((count, width), dtype=np.float32), np.zeros(count, dtype=bool))
            self._edge_cache = cache
        _, values, ready = cache
        unseen = rows[~ready[rows]]
        # Bound transient float64 encoding as well as the resident float32 cache.
        for start in range(0, len(unseen), 1024):
            part = unseen[start:start+1024]
            values[part] = self._encode_edges(store._history.iloc[part]).astype(np.float32)
            ready[part] = True
        return values[rows]

    def snapshot_columns(self, store):
        """Keep only causal graph metadata and optional deterministic maps."""
        names = {'competition_id', 'season_id', 'event_id', 'team_id', 'opponent_id',
                 'side', 'kickoff_at'}
        if self.config.heatmaps is not None:
            from .spatial import columns
            names.update(columns(self.config.heatmaps))
        # Preserve the source's order and column dtypes. Numeric source columns
        # are encoded from the immutable store, not copied into each snapshot.
        return [c for c in store._history if c in names]

    def fixture(self, fixtures, numeric):
        # Keep caller metadata attrs intact without copying them onto a Series.
        inputs = pd.DataFrame(fixtures, copy=False)
        return np.concatenate((self.fixture_scale.transform(numeric),
                               self.one_hot(inputs.competition_id)), axis=1)

    def tensors(self, snapshot, device):
        import torch
        def tensor(a, dtype=torch.float32):
            with phase('tensor_transfer'):
                return torch.as_tensor(a, dtype=dtype, device=device)
        if hasattr(snapshot, '_encoded_x'):
            encoded = snapshot._encoded_x
        else:
            with phase('row_encoding'):
                encoded = self.edges(snapshot)
        result = dict(x=tensor(encoded),
                    receiver=tensor(snapshot.receiver, torch.long),
                    sender=tensor(snapshot.sender, torch.long),
                    log_weight=tensor(snapshot.log_weight, torch.float64),
                    gate=tensor(snapshot.gate),
                    league=tensor([self.leagues.get(x, 0) for x in snapshot.league], torch.long),
                    entry=tensor([ENTRY_ORDER.index(x) for x in snapshot.entry], torch.long))
        if self.config.heatmaps is not None:
            from .spatial import tensors
            result['maps'], result['map_available'] = tensors(snapshot.edges, self.config.heatmaps, device)
        return result

    def state(self):
        return dict(leagues=list(self.leagues.items()), edge_scale=self.edge_scale.state(),
                    fixture_scale=self.fixture_scale.state())

    @classmethod
    def from_state(cls, config, state):
        obj = cls(config)
        obj.leagues = dict(state["leagues"])
        obj.edge_scale = MaskedScale.from_state(state["edge_scale"])
        obj.fixture_scale = MaskedScale.from_state(state["fixture_scale"])
        return obj
