"""Immutable copied event inputs, exact identities, and cutoff reconstruction."""
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
import json
import math
import numpy as np
import pandas as pd
from .config import MATCH_KEYS, ENTRY_ORDER


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
    return pd.to_datetime(values, utc=True)


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
        h = history.copy(deep=True).reset_index(drop=True)
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
        m = pd.DataFrame(team_seasons).copy(deep=True)
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
        s = schedule.copy(deep=True).reset_index(drop=True)
        _ids(s, (*MATCH_KEYS, "home_id", "away_id"))
        s["kickoff_at"] = _times(s.kickoff_at)
        if s.duplicated(keys).any() or s.home_id.eq(s.away_id).any():
            raise ValueError("Schedule needs unique fixtures with different teams.")
        self._schedule = s
        self.provenance = json.loads(json.dumps(provenance or {}, default=str))
        # Hash only model-relevant source values, without coercing large IDs.
        digest = sha256()
        for f in (self._history, self._memberships, self._schedule):
            digest.update(repr(list(f.columns)).encode())
            digest.update(pd.util.hash_pandas_object(f, index=False).values.tobytes())
        self.version = digest.hexdigest()
        self._round_cache = {}

    @property
    def history(self):
        return self._history.copy(deep=True)

    @property
    def team_seasons(self):
        return self._memberships.copy(deep=True)

    @property
    def schedule(self):
        return self._schedule.copy(deep=True)

    def issue_times(self, fixtures, config):
        _ids(fixtures, (*MATCH_KEYS, "home_id", "away_id"))
        kick = _times(fixtures.kickoff_at)
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

    def snapshot(self, issuance, config, teams=()):
        issuance = pd.Timestamp(issuance).tz_convert("UTC")
        cutoff = config.history_cutoff(issuance)
        e = self.eligible(cutoff)
        known = self._memberships[self._memberships.entry_at.le(issuance)]
        nodes = tuple(sorted(set(e.team_id.tolist()) | set(teams)))
        index = {t: n for n, t in enumerate(nodes)}
        receiver = np.array([index[t] for t in e.team_id], dtype=np.int64)
        sender = np.array([index[t] for t in e.opponent_id], dtype=np.int64)
        current = {r.team_id: r for r in known.itertuples(index=False)}
        boundaries = {t: rows.entry_at.to_numpy() for t, rows in known.groupby("team_id")}
        age = (issuance - e.kickoff_at).dt.total_seconds().to_numpy() / 86400.
        crossed = np.array([sum(t > when for t in boundaries.get(team, ()))
                            for team, when in zip(e.team_id.tolist(), e.kickoff_at)], dtype=float)
        logw = -math.log(2) * age / config.half_life_days + crossed * math.log(config.seasonal_retention)
        support = np.bincount(receiver, weights=np.exp(logw), minlength=len(nodes))
        return GraphSnapshot(issuance, cutoff, nodes, e, receiver, sender, logw, support,
                             support / (support + config.support_kappa),
                             tuple(current[t].competition_id if t in current else None for t in nodes),
                             tuple(current[t].movement if t in current else "unknown" for t in nodes),
                             self.version)

    def eligible(self, cutoff):
        """All completed, previous-day edges, honoring existing reviewed bounds."""
        h = self._history
        return h[h.kickoff_at.lt(cutoff) & self._available.le(cutoff) & h.status.eq("finished")].copy()
