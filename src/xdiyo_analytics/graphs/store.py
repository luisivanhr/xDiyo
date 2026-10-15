"""Immutable copied event inputs, exact identities, and cutoff reconstruction."""
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
import json
import math
import numpy as np
import pandas as pd
from .config import MATCH_KEYS, ENTRY_ORDER
from .profiling import timed
from .._ownership import isolated_frame


def identities(frame, columns):
    return list(frame[list(columns)].itertuples(index=False, name=None))


def _ids(frame, names):
    for name in names:
        if frame[name].isna().any() or any(isinstance(x, bool) or not isinstance(x, Integral)
                                         or x <= 0 for x in frame[name].tolist()):
            raise ValueError(f"{name} needs exact positive integer IDs; float IDs are rejected.")


def _times(values):
    if any(pd.Timestamp(x).tzinfo is None or pd.isna(x) for x in values):
        raise ValueError("Timestamps must be nonmissing and timezone-aware.")
    return pd.to_datetime(values, utc=True).astype('datetime64[ns, UTC]')


@dataclass
class GraphSnapshot:
    issuance: pd.Timestamp
    cutoff: pd.Timestamp
    teams: tuple
    edges: pd.DataFrame
    receiver: np.ndarray
    sender: np.ndarray
    log_weight: np.ndarray
    support: np.ndarray
    gate: np.ndarray
    league: tuple
    entry: tuple
    store_version: str

    def evidence(self, team, layers):
        frontier, event_keys = {team}, set()
        for _ in range(layers):
            selected = self.edges[self.edges.team_id.isin(frontier)]
            event_keys.update(identities(selected, MATCH_KEYS))
            frontier |= set(selected.opponent_id.tolist())
        return tuple(sorted(event_keys))


class TemporalEventStore:
    """Two receiver perspectives per event, plus authoritative season entries.

    history follows build_team_history. team_seasons requires team_id,
    competition_id, season_id, entry_at and movement. entry_at is an effective
    season boundary known at that time, not a result-release timestamp. Include
    intervening seasons even when no matches are available. Only one current
    domestic-league context per team is supported; overlapping memberships must
    be resolved upstream. A full fixture schedule freezes round anchors; an
    explicit round_issue_at preserves the original anchor after rescheduling.
    """
    def __init__(self, history, team_seasons, *, schedule=None, provenance=None):
        h = isolated_frame(history).reset_index(drop=True)
        required = [*MATCH_KEYS, "team_id", "opponent_id", "side", "kickoff_at",
                    "status", "goals_for", "goals_against"]
        if not set(required) <= set(h):
            raise ValueError(f"Missing history fields: {set(required)-set(h)}")
        _ids(h, (*MATCH_KEYS, "team_id", "opponent_id"))
        h["kickoff_at"] = _times(h.kickoff_at)
        if h.duplicated([*MATCH_KEYS, "side"]).any() or not h.side.isin(["home", "away"]).all():
            raise ValueError("Duplicate event perspectives or invalid side.")
        keys = list(MATCH_KEYS)
        home = h[h.side.eq("home")].set_index(keys)
        away = h[h.side.eq("away")].set_index(keys)
        if set(home.index) != set(away.index):
            raise ValueError("Every match requires exactly two paired perspectives.")
        away = away.reindex(home.index)
        pairs = [("team_id", "opponent_id"), ("opponent_id", "team_id"),
                 ("goals_for", "goals_against"), ("goals_against", "goals_for"),
                 ("kickoff_at", "kickoff_at"), ("status", "status")]
        if 'result_available_at' in h:
            pairs.append(('result_available_at', 'result_available_at'))
        for col in h:
            if col.startswith("team::") and col.replace("team::", "opponent::", 1) in h:
                other = col.replace("team::", "opponent::", 1)
                pairs.extend([(col, other), (other, col)])
        for left, right in pairs:
            a, b = home[left], away[right]
            if not (a.eq(b).fillna(False) | (a.isna() & b.isna())).all():
                raise ValueError(f"Inconsistent paired perspectives: {left}/{right}.")
        if h.team_id.eq(h.opponent_id).any():
            raise ValueError("A team cannot play itself.")
        for col in ("goals_for", "goals_against"):
            v = pd.to_numeric(h[col], errors="raise").to_numpy(dtype=float, na_value=np.nan)
            if (np.isfinite(v) & ((v < 0) | (v % 1 != 0))).any() or np.isinf(v).any():
                raise ValueError("Scores must be nonnegative integers or missing.")
        m = isolated_frame(pd.DataFrame(team_seasons))
        required_m = ["team_id", "competition_id", "season_id", "entry_at", "movement"]
        if not set(required_m) <= set(m):
            raise ValueError("team_seasons needs team/competition/season IDs, entry_at and movement.")
        _ids(m, ("team_id", "competition_id", "season_id"))
        m["entry_at"] = _times(m.entry_at)
        if m.duplicated(["team_id", "competition_id", "season_id"]).any() or m.duplicated(["team_id", "entry_at"]).any():
            raise ValueError("Season memberships and team boundaries must be unique.")
        m["movement"] = m.movement.replace({"other_entry": "unknown"})
        if not m.movement.isin(ENTRY_ORDER).all():
            raise ValueError("Unknown season-entry category.")
        self._history = h.sort_values("kickoff_at", kind="stable").reset_index(drop=True)
        from ..data.availability import respect_result_availability
        # Reuse sparse reviewed completion bounds already carried by the
        # library; the ordinary previous-day contract needs no release data.
        self._available = respect_result_availability(self._history, self._history.kickoff_at)
        self._memberships = m[required_m].sort_values("entry_at", kind="stable").reset_index(drop=True)
        if schedule is None:
            schedule = h[h.side.eq("home")].rename(columns={"team_id": "home_id", "opponent_id": "away_id"})
        s = isolated_frame(schedule).reset_index(drop=True)
        _ids(s, (*MATCH_KEYS, "home_id", "away_id"))
        s["kickoff_at"] = _times(s.kickoff_at)
        if s.duplicated(keys).any() or s.home_id.eq(s.away_id).any():
            raise ValueError("Schedule needs unique fixtures with different teams.")
        scheduled = s.set_index(keys)
        for col, expected in [('home_id', home.team_id), ('away_id', home.opponent_id)]:
            actual = scheduled[col].reindex(home.index)
            if not actual.eq(expected).fillna(False).all():
                raise ValueError(f'Schedule disagrees with history: {col}.')
        self._schedule = s
        self.provenance = json.loads(json.dumps(provenance or {}, default=str))
        # Hash only model-relevant source values, without coercing large IDs.
        digest = sha256()
        for f in (self._history, self._memberships, self._schedule):
            digest.update(repr(list(f.columns)).encode())
            # Parquet may change datetime storage units without changing times.
            # Fingerprint the same temporal values independently of those units.
            canonical = f.copy(deep=False)
            for column in f:
                if isinstance(f[column].dtype, pd.DatetimeTZDtype):
                    canonical = canonical.assign(**{column: f[column].dt.tz_convert('UTC').astype('datetime64[ns, UTC]')})
                elif pd.api.types.is_datetime64_dtype(f[column].dtype):
                    canonical = canonical.assign(**{column: f[column].astype('datetime64[ns]')})
            digest.update(pd.util.hash_pandas_object(canonical, index=False).values.tobytes())
        self.version = digest.hexdigest()
        self._round_cache = {}
        # Fixed source arrays, not per-issuance index lists. The explicit unit
        # conversion also handles pandas versions with different time storage.
        self._kickoff_ns = self._history.kickoff_at.array.asi8.copy()
        self._available_ns = self._available.astype('datetime64[ns, UTC]').array.asi8.copy()
        self._finished = self._history.status.eq('finished').to_numpy(dtype=bool, na_value=False)
        # Convert exact IDs to dense source codes once. Arrow-backed integer
        # columns otherwise box every edge ID again at every issuance. Only
        # the codes use int64; original IDs retain their exact Python values.
        source_teams = self._history.team_id.tolist()
        source_opponents = self._history.opponent_id.tolist()
        self._source_nodes = tuple(sorted(set(source_teams) | set(source_opponents)))
        self._source_node_index = {team: i for i, team in enumerate(self._source_nodes)}
        self._receiver_codes = np.fromiter((self._source_node_index[t] for t in source_teams),
                                           dtype=np.int64, count=len(source_teams))
        self._sender_codes = np.fromiter((self._source_node_index[t] for t in source_opponents),
                                         dtype=np.int64, count=len(source_opponents))
        self._membership_by_team = {
            team: (group.entry_at.array.asi8.copy(), tuple(group.competition_id), tuple(group.movement))
            for team, group in self._memberships.groupby('team_id', sort=False)}
        self._entries_at_event = np.zeros(len(self._history), dtype=np.int64)
        for team, positions in self._history.groupby('team_id', sort=False).indices.items():
            membership = self._membership_by_team.get(team)
            if membership is not None:
                self._entries_at_event[positions] = np.searchsorted(
                    membership[0], self._kickoff_ns[positions], side='right')

    def _export_tables(self, directory):
        """Native synchronous traversal; no writable frame alias escapes.

        Arbitrary/custom stores retain the serializer's public-copy path. Do not
        mutate private store fields concurrently with export or preparation.
        """
        from pathlib import Path
        for name, frame in (('history', self._history), ('team_seasons', self._memberships), ('schedule', self._schedule)):
            frame.to_parquet(Path(directory)/(name+'.parquet'), compression='zstd', index=False)

    @property
    def history(self):
        return isolated_frame(self._history)

    @property
    def team_seasons(self):
        return isolated_frame(self._memberships)

    @property
    def schedule(self):
        return isolated_frame(self._schedule)

    def issue_times(self, fixtures, config):
        _ids(fixtures, (*MATCH_KEYS, "home_id", "away_id"))
        kick = _times(fixtures.kickoff_at)
        expected = self._schedule.set_index(list(MATCH_KEYS))
        query_keys = pd.MultiIndex.from_frame(fixtures[list(MATCH_KEYS)])
        known = query_keys.isin(expected.index)
        for column in ('home_id', 'away_id'):
            scheduled = expected[column].reindex(query_keys[known]).tolist()
            if scheduled != fixtures.loc[known, column].tolist():
                raise ValueError(f'Fixture {column} disagrees with the schedule.')
        if config.forecast_policy == "one_day_ahead":
            issue = _times(fixtures.issuance_at) if "issuance_at" in fixtures else kick - pd.Timedelta(hours=24)
        else:
            keys = config.round_keys
            if keys not in self._round_cache:
                s = self._schedule
                if not set(keys) <= set(s) or s[list(keys)].isna().any().any():
                    raise ValueError("By Round requires complete round keys in the authoritative schedule.")
                # Share the library's grouping/anchor implementation.
                from ..features.history import frozen_group_anchors
                times = _times(s.round_issue_at) if "round_issue_at" in s else s.kickoff_at
                anchors = frozen_group_anchors(s, keys, times)
                if "round_issue_at" in s and not anchors.equals(times):
                    raise ValueError("round_issue_at must be identical throughout each round.")
                self._round_cache[keys] = dict(zip(identities(s, MATCH_KEYS), zip(anchors, identities(s, keys))))
            values = []
            for event, round_key in zip(identities(fixtures, MATCH_KEYS), identities(fixtures, keys)):
                if event not in self._round_cache[keys]:
                    raise ValueError("By Round fixture is absent from the authoritative schedule.")
                t, expected = self._round_cache[keys][event]
                if round_key != expected:
                    raise ValueError("Fixture round identity disagrees with the schedule.")
                values.append(t)
            issue = pd.Series(values, index=fixtures.index, dtype="datetime64[ns, UTC]")
            if "issuance_at" in fixtures and not np.array_equal(_times(fixtures.issuance_at), issue):
                raise ValueError("By Round issuance must equal its frozen round anchor.")
        if (issue > kick).any():
            raise ValueError("Issuance must not follow kickoff.")
        return issue

    @timed('snapshot_index')
    def snapshot(self, issuance, config, teams=(), *, _columns=None):
        issuance = pd.Timestamp(issuance).tz_convert("UTC")
        cutoff = config.history_cutoff(issuance)
        rows = self._eligible_rows(cutoff)
        source = self._history if _columns is None else self._history.loc[:, _columns]
        e = source.iloc[rows].copy()
        receiver_codes = self._receiver_codes[rows]
        nodes = tuple(sorted({self._source_nodes[i] for i in np.unique(receiver_codes)} | set(teams)))
        index = {t: n for n, t in enumerate(nodes)}
        local_index = np.full(len(self._source_nodes), -1, dtype=np.int64)
        for team, position in index.items():
            code = self._source_node_index.get(team)
            if code is not None:
                local_index[code] = position
        receiver = local_index[receiver_codes]
        sender = local_index[self._sender_codes[rows]]
        counts = np.zeros(len(nodes), dtype=np.int64)
        leagues, entries = [], []
        for i, team in enumerate(nodes):
            membership = self._membership_by_team.get(team)
            n = np.searchsorted(membership[0], issuance.value, side='right') if membership is not None else 0
            counts[i] = n
            leagues.append(membership[1][n-1] if n else None)
            entries.append(membership[2][n-1] if n else 'unknown')
        age = (issuance - e.kickoff_at).dt.total_seconds().to_numpy() / 86400.
        # Count entries strictly after each event and <= this issuance. Event
        # anchors were indexed once, while current memberships remain specific
        # to the full issuance timestamp (not just its midnight cutoff).
        crossed = (counts[receiver] - self._entries_at_event[rows]).astype(float)
        logw = -math.log(2) * age / config.half_life_days + crossed * math.log(config.seasonal_retention)
        support = np.bincount(receiver, weights=np.exp(logw), minlength=len(nodes))
        return GraphSnapshot(issuance, cutoff, nodes, e, receiver, sender, logw, support,
                             support / (support + config.support_kappa),
                             tuple(leagues), tuple(entries),
                             self.version)

    def eligible(self, cutoff):
        """All completed, previous-day edges, honoring existing reviewed bounds."""
        return self._history.iloc[self._eligible_rows(cutoff)].copy()

    def _eligible_rows(self, cutoff):
        bound = pd.Timestamp(cutoff)
        if bound.tzinfo is None:
            raise ValueError('History cutoff must be timezone-aware.')
        return np.flatnonzero((self._kickoff_ns < bound.value) &
                              (self._available_ns <= bound.value) &
                              (self._available_ns != np.iinfo(np.int64).min) & self._finished)
