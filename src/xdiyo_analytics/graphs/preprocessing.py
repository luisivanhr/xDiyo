"""Fit-only numeric transforms and categorical maps; missing values get masks."""
import numpy as np
import pandas as pd
from dataclasses import asdict
import json
from .config import ENTRY_ORDER
from .profiling import phase, timed
from .observed_gap import LEGACY_GAP_ENCODING, BOUNDED_GAP_ENCODING, encode_observed_gap


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
        self._map_cache = None
        self._temporal_index = None

    @property
    def temporal_columns(self):
        if self.config.temporal_mode == 'snapshot':
            return ()
        from .temporal import time_feature_names
        return time_feature_names(self.config)

    @property
    def edge_dim(self):
        return 2*len(self.edge_columns) + len(self.leagues) + 2 + len(self.temporal_columns)

    @property
    def fixture_encoded_columns(self):
        columns = []
        for name in self.fixture_columns:
            if self.config.fixture_gap_encoding == BOUNDED_GAP_ENCODING and name in ('home_rest_days', 'away_rest_days'):
                side = name.split('_', 1)[0]
                columns.extend(f'{side}_{suffix}' for suffix in
                               ('observed_gap_r7', 'no_eligible_history', 'no_eligible_observation_30d'))
            else:
                columns.append(name + '__standardized')
        columns.extend(name + '__missing' for name in self.fixture_columns
                       if self.config.fixture_gap_encoding == LEGACY_GAP_ENCODING or name not in ('home_rest_days', 'away_rest_days'))
        columns.extend(('competition_id=unknown', *(f'competition_id={key}' for key, _ in sorted(self.leagues.items(), key=lambda item: item[1]))))
        return tuple(columns)

    @property
    def fixture_dim(self):
        return len(self.fixture_encoded_columns)

    @property
    def fixture_schema(self):
        return dict(encoding=self.config.fixture_gap_encoding,
                    scale_days=self.config.observed_gap_scale_days,
                    recent_days=self.config.observed_gap_recent_days,
                    encoded_feature_order=list(self.fixture_encoded_columns), encoded_dim=self.fixture_dim)

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
        self._map_cache = None
        categories = set(history.competition_id.tolist()) | set(fixtures.competition_id.tolist()) | set(extra_leagues)
        for snap in snapshots:
            categories.update(x for x in snap.league if x is not None)
        self.leagues = {int(key): n+1 for n, key in enumerate(sorted(categories))}
        self.edge_scale = MaskedScale().fit(self.numeric_edges(history))
        values = np.asarray(numeric, dtype=float).copy()
        if self.config.fixture_gap_encoding == BOUNDED_GAP_ENCODING:
            # These two positions remain in legacy scaler state for compatible
            # column mapping, but the corrected encoding never scales them.
            values[:, 1:3] = np.nan
        self.fixture_scale = MaskedScale().fit(values)
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

    def cached_maps(self, snapshot, store, *, reconstruction_targets=False):
        """Native source-row maps shared across issuances, bounded to 128 MiB.

        These are per-match grids already constructed in the receiving-team
        frame. No point pooling, temporal averaging or learned outputs occur.
        Arbitrary public snapshots retain the validating uncached path.
        Positive-mass KL target checks apply only to auxiliary-loss consumers,
        not ordinary inference. The cache identity separates both modes.
        """
        from .spatial import columns, grid_identity
        from ..neural.inputs import neural_maps
        if snapshot.store_version != store.version:
            raise ValueError('Cannot encode maps from a different event store.')
        cfg = self.config.heatmaps
        reconstruction = cfg.encoder.reconstruction
        validate_kl = reconstruction_targets and reconstruction is not None and reconstruction.loss == 'kl'
        identity = (id(store), store.version, json.dumps(grid_identity(cfg), sort_keys=True), validate_kl)
        cache = self._map_cache
        if cache is not None and cache[0] != identity:
            self._map_cache = cache = None
        n, count = cfg.grid.grid_size, len(store._history)
        rows = snapshot.edges.index.to_numpy(dtype=np.int64)
        def encode(indices):
            maps, available = neural_maps(store._history.iloc[indices][columns(cfg)].to_numpy(
                dtype=float, na_value=np.nan).reshape(len(indices), 2, n, n))
            if validate_kl:
                from ..neural.prepared import validate_kl_targets
                validate_kl_targets(maps, available)
            return maps, available
        if count * (2*n*n*4 + 2*4 + 1) > self._edge_cache_limit or not len(rows):
            self._map_cache = None
            return encode(rows)
        if cache is None:
            cache = (identity, np.empty((count, 2, n, n), dtype=np.float32),
                     np.empty((count, 2), dtype=np.float32), np.zeros(count, dtype=bool))
            self._map_cache = cache
        _, maps, available, ready = cache
        unseen = rows[~ready[rows]]
        for start in range(0, len(unseen), 256):
            part = unseen[start:start+256]
            maps[part], available[part] = encode(part)
            ready[part] = True
        return maps[rows], available[rows]

    def native_tensors(self, snapshot, device, store, *, reconstruction_targets=False):
        snapshot._encoded_x = self.cached_edges(snapshot, store)
        if self.config.temporal_mode != 'snapshot':
            from .temporal import TemporalHistoryIndex, TimeIndexedPlan, time_encoding_key
            index = self._temporal_index
            if index is None or index.store is not store or index.config != self.config:
                self._temporal_index = index = TemporalHistoryIndex(store, self.config)
            snapshot._temporal_x = index.features(snapshot.edges.index, snapshot.issuance.value)
            snapshot._temporal_encoding_key = time_encoding_key(self.config)
        if self.config.heatmaps is not None:
            snapshot._prepared_maps = self.cached_maps(snapshot, store, reconstruction_targets=reconstruction_targets)
        data = self.tensors(snapshot, device)
        if self.config.temporal_mode == 'historical':
            snapshot._temporal_plan = TimeIndexedPlan(index, snapshot, self.config.layers)
            data.update(snapshot._temporal_plan.tensors(self, device))
        return data

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
        values = np.asarray(numeric, dtype=float)
        if values.shape != (len(inputs), len(self.fixture_columns)):
            raise ValueError('Fixture numeric schema has an incompatible shape.')
        if self.config.fixture_gap_encoding == LEGACY_GAP_ENCODING:
            return np.concatenate((self.fixture_scale.transform(values),
                                   self.one_hot(inputs.competition_id)), axis=1)
        ordinary = [i for i in range(len(self.fixture_columns)) if i not in (1, 2)]
        # Transform only unrelated fields. No standardized gap or generic gap
        # missingness coordinate is computed or appended to the model tensor.
        scaler = MaskedScale()
        scaler.mean = self.fixture_scale.mean[ordinary]
        scaler.scale = self.fixture_scale.scale[ordinary]
        scaled = scaler.transform(values[:, ordinary])
        columns = {index: scaled[:, j:j+1] for j, index in enumerate(ordinary)}
        for side, index in (('home', 1), ('away', 2)):
            names = [f'{side}_observed_gap_days', f'{side}_latest_eligible_observed_at',
                     f'{side}_has_eligible_observation']
            if not set(names) <= set(inputs):
                raise ValueError('Corrected fixture gaps require eligible-observation provenance.')
            known = inputs[names[2]].to_numpy()
            timestamps = pd.to_datetime(inputs[names[1]], utc=True)
            if known.dtype.kind != 'b' or not np.array_equal(timestamps.notna().to_numpy(), known):
                raise ValueError('Observed-gap timestamps and eligibility provenance disagree.')
            if not np.array_equal(values[:, index], inputs[names[0]].to_numpy(float), equal_nan=True):
                raise ValueError('Observed-gap duration and provenance disagree.')
            columns[index] = encode_observed_gap(values[:, index], known,
                scale_days=self.config.observed_gap_scale_days, recent_days=self.config.observed_gap_recent_days)
        return np.concatenate((*[columns[i] for i in range(len(self.fixture_columns))],
                               scaled[:, len(ordinary):], self.one_hot(inputs.competition_id)), axis=1)

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
        if self.config.temporal_mode != 'snapshot':
            from .temporal import time_encoding_key
            if (not hasattr(snapshot, '_temporal_x') or
                    getattr(snapshot, '_temporal_encoding_key', None) != time_encoding_key(self.config)):
                raise ValueError('Time features require native_tensors with the full immutable event store.')
            encoded = np.concatenate((encoded, snapshot._temporal_x), axis=1)
        result = dict(x=tensor(encoded),
                    receiver=tensor(snapshot.receiver, torch.long),
                    sender=tensor(snapshot.sender, torch.long),
                    log_weight=tensor(snapshot.log_weight, torch.float64),
                    gate=tensor(snapshot.gate),
                    league=tensor([self.leagues.get(x, 0) for x in snapshot.league], torch.long),
                    entry=tensor([ENTRY_ORDER.index(x) for x in snapshot.entry], torch.long))
        if self.config.heatmaps is not None:
            from .spatial import tensors
            if hasattr(snapshot, '_prepared_maps'):
                result['maps'], result['map_available'] = [torch.as_tensor(v, device=device) for v in snapshot._prepared_maps]
            else:
                result['maps'], result['map_available'] = tensors(snapshot.edges, self.config.heatmaps, device)
        return result

    def state(self):
        state = dict(leagues=list(self.leagues.items()), edge_scale=self.edge_scale.state(),
                     fixture_scale=self.fixture_scale.state())
        if self.config.fixture_gap_encoding == BOUNDED_GAP_ENCODING:
            state['fixture_schema'] = self.fixture_schema
        return state

    @classmethod
    def from_state(cls, config, state):
        obj = cls(config)
        obj.leagues = dict(state["leagues"])
        obj.edge_scale = MaskedScale.from_state(state["edge_scale"])
        obj.fixture_scale = MaskedScale.from_state(state["fixture_scale"])
        if obj.fixture_scale.mean.shape != (len(obj.fixture_columns),) or obj.fixture_scale.scale.shape != (len(obj.fixture_columns),):
            raise ValueError('Saved fixture scaler width disagrees with the input schema.')
        if config.fixture_gap_encoding == BOUNDED_GAP_ENCODING:
            if state.get('fixture_schema') != obj.fixture_schema:
                raise ValueError('Saved corrected fixture schema is missing or inconsistent.')
        elif 'fixture_schema' in state and state['fixture_schema'] != obj.fixture_schema:
            raise ValueError('Saved legacy fixture schema is inconsistent.')
        return obj
