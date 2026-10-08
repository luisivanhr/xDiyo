"""Fit-local replay plans and numerical evidence, without reporting artifacts.

The plan owns a copy of one validated training population. It is never cached
globally or attached to a mutable DataFrame: each fit (including each fold and
cutoff) builds a new plan. Only parameter coordinates may vary during its life.
The state equations still execute in bayesian_replay's original Python kernel.
"""

from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd

from .bayesian_core import initial_home, initial_team, score_log_probability


class _Schedule:
    def __init__(self, events, entries, frontier=None):
        self.events, self.entries = events, entries
        self.entry_events, self.releases = defaultdict(list), defaultdict(list)
        for entry in entries:
            self.entry_events[entry["entry_at"]].append(entry)
        for time, records in self.entry_events.items():
            self.entry_events[time] = sorted(records, key=lambda r: repr((r["competition_id"], r["team_id"])))
        for event in events:
            self.releases[event["available_at"]].append(event)
        self.times = sorted(set(self.entry_events) | set(self.releases))
        self.frontier = frontier if frontier is not None else max(self.releases, default=None)
        self.steps = {}
        self.batches = {}
        teams, leagues = {}, {}
        for time in self.times:
            real = defaultdict(list)
            for event in self.releases.get(time, ()):
                if event["kickoff_at"] < time:
                    real[event["competition_id"]].append(event)
            before, after = [], []
            for competition, games in real.items():
                initialized = competition in leagues and all(
                    teams.get((competition, team)) == e["season_id"]
                    and leagues[competition] == e["season_id"]
                    for e in games for team in (e["home_id"], e["away_id"]))
                (before if initialized else after).extend(games)
            for entry in self.entry_events.get(time, ()):
                teams[(entry["competition_id"], entry["team_id"])] = entry["season_id"]
                leagues[entry["competition_id"]] = entry["season_id"]
            equal = [e for e in self.releases.get(time, ()) if e["kickoff_at"] == time]
            self.steps[time] = before, after, equal
            for batch in (before, after, equal):
                grouped = defaultdict(list)
                for event in batch:
                    grouped[event["competition_id"]].append(event)
                prepared = []
                for competition, games in sorted(grouped.items(), key=lambda item: repr(item[0])):
                    games.sort(key=lambda e: repr((e["season_id"], e["event_id"])))
                    appearances = Counter(team for e in games for team in (e["home_id"], e["away_id"]))
                    kicks = {team: [e["kickoff_at"] for e in games if team in (e["home_id"], e["away_id"])] for team in appearances}
                    observations = [(e["home_id"], e["away_id"], e["home_goals"], e["away_goals"]) for e in games]
                    prepared.append((competition, games, appearances, kicks, observations))
                self.batches[id(batch)] = prepared


def _ns(value):
    return pd.NaT.value if value is None or pd.isna(value) else value.value


def _tree_set(node, low, high, index, value):
    """Persistent chronological index: a revision copies only O(log N) nodes."""
    if high - low == 1:
        return value
    left, right = node if node is not None else (None, None)
    middle = (low + high) // 2
    if index < middle:
        left = _tree_set(left, low, middle, index, value)
    else:
        right = _tree_set(right, middle, high, index, value)
    return left, right


def _tree_get(node, low, high, index):
    while node is not None and high - low > 1:
        middle = (low + high) // 2
        if index < middle:
            node, high = node[0], middle
        else:
            node, low = node[1], middle
    return node


def _tree_leaves(node, low, high, start=0):
    if node is None or high <= start:
        return
    if high - low == 1:
        yield low, node
        return
    middle = (low + high) // 2
    yield from _tree_leaves(node[0], low, middle, start)
    yield from _tree_leaves(node[1], middle, high, start)


class _TreeField:
    def __init__(self, schedule, field):
        self.schedule, self.field = schedule, field

    def __getitem__(self, time):
        s = self.schedule
        leaf = _tree_get(s.root, 0, s.size, s.positions[time])
        return getattr(leaf, self.field)[time]

    def get(self, time, default=None):
        try:
            return self[time]
        except KeyError:
            return default


class _CanonicalSchedule:
    """One knowledge-time version sharing unchanged chronological batches.

    Late releases can require many numerical refilters. Retaining a complete
    list of every prefix would also cost quadratic *memory*. Persistent roots
    share all untouched batches; only the newly changed kickoff leaves differ.
    """

    events = entries = ()
    releases = {}

    def __init__(self, root, keys, positions, size, batches, frontier):
        self.root, self.keys, self.positions, self.size = root, keys, positions, size
        self.batches, self.frontier = batches, frontier
        self.start = 0
        self.changed = 0
        self.steps = _TreeField(self, "steps")
        self.entry_events = _TreeField(self, "entry_events")

    @property
    def times(self):
        return (self.keys[index] for index, _ in _tree_leaves(self.root, 0, self.size, self.start))


def _versions(events, entries):
    """Prepare availability versions once, with O(N log N) index storage."""
    keys = sorted({*(e["kickoff_at"] for e in events), *(r["entry_at"] for r in entries)})
    positions = {time: i for i, time in enumerate(keys)}
    size = 1 << max(0, (len(keys) - 1).bit_length())
    releases, anchors = defaultdict(list), defaultdict(list)
    for event in events:
        releases[event["available_at"]].append(dict(event, available_at=event["kickoff_at"]))
    for entry in entries:
        anchors[entry["entry_at"]].append(entry)
    known_events, known_entries = defaultdict(list), defaultdict(list)
    root, batches = None, {}
    for time in sorted(set(releases) | set(anchors)):
        changed = set()
        if time in anchors:
            known_entries[time] = anchors[time]
            changed.add(time)
        equal = []
        for event in releases.get(time, ()):
            if event["kickoff_at"] == time:
                equal.append(event)
            else:
                known_events[event["kickoff_at"]].append(event)
                changed.add(event["kickoff_at"])
        for include_equal in (False, True):
            if include_equal:
                if not equal:
                    continue
                known_events[time].extend(equal)
                changed = {time}
            for kickoff in sorted(changed):
                leaf = _Schedule(list(known_events[kickoff]), list(known_entries[kickoff]))
                batches.update(leaf.batches)
                root = _tree_set(root, 0, size, positions[kickoff], leaf)
            schedule = _CanonicalSchedule(root, keys, positions, size, batches, time)
            schedule.changed = min((positions[k] for k in changed), default=len(keys))
            schedule.order = 2 if include_equal else 0
            yield time, schedule


class _States:
    """Compact mathematical state histories; no summaries, frames or journals."""

    def __init__(self, history=True):
        self.keep_history = history
        self.teams, self.leagues = defaultdict(list), defaultdict(list)
        self.usage = set()
        self.kinds = {}

    def team(self, key, time, latest, state, count, kind, season):
        item = (_ns(time), _ns(latest), state, count)
        self.kinds[key] = kind
        if self.keep_history:
            self.teams[key].append(item)
        else:
            self.teams[key] = [item]
        if kind in ("initial", "retained", "bridge", "mirrored"):
            self.usage.add((key[0], season, key[1], kind))

    def home(self, key, time, latest, state):
        item = (_ns(time), _ns(latest), state)
        if self.keep_history:
            self.leagues[key].append(item)
        else:
            self.leagues[key] = [item]

    def publish(self, canonical, time):
        # Equivalent to _refilter_versions' changed-state publication. Call
        # order retains pre-kickoff then post-kickoff versions at equal times.
        for target, source in ((self.teams, canonical.teams), (self.leagues, canonical.leagues)):
            for key, items in source.items():
                item = items[-1]
                if target[key] and target[key][-1][1:] == item[1:]:
                    continue
                target[key].append((_ns(time), *item[1:]))
        self.usage.update(canonical.usage)


def _lookup(items, times, fallback):
    """Batch lookup with the native recorded <= cutoff, latest < cutoff rule."""
    if not items:
        return [fallback] * len(times)
    recorded = np.fromiter((r[0] for r in items), dtype=np.int64)
    latest = np.fromiter((r[1] for r in items), dtype=np.int64)
    positions = np.searchsorted(recorded, times, side="right") - 1
    # Latest contributing kickoffs need not be monotone across season resets.
    while True:
        blocked = (positions >= 0) & (latest[np.maximum(positions, 0)] >= times)
        if not blocked.any():
            break
        positions[blocked] -= 1
    return [items[p][2] if p >= 0 else fallback for p in positions]


class CalibrationReplay:
    """Private immutable-input cache, owned by a single train_bayesian fit.

    History, availability, movement records, anchors, config and training cutoff
    are fixed here. Changing any input requires constructing a new instance.
    No parameter results or mutable fitted states are cached between trials.
    """

    def __init__(self, history, *, config, cutoff, available_at=None,
                 team_seasons=None, season_starts=None):
        from ..features.transitions import TransitionContext
        from .bayesian_replay import _events, _needs_refilter, _entry_key

        self.config, self.cutoff = config, pd.Timestamp(cutoff)
        owned = deepcopy(history)
        if config.score_basis == "regulation" and owned.attrs.get("score_basis") != "regulation":
            raise ValueError("Regulation mode requires caller-verified history.attrs['score_basis']='regulation'.")
        self.events, _, available = _events(owned, available_at)
        if any(e["kickoff_at"] >= self.cutoff or e["available_at"] > self.cutoff for e in self.events):
            raise ValueError("Calibration plan contains observations outside its training cutoff.")
        context = TransitionContext(owned, available_at=available,
                                    team_seasons=deepcopy(team_seasons), season_starts=deepcopy(season_starts))
        entries = [dict(r) for r in context.records.values() if pd.notna(r["entry_at"])]
        entries.sort(key=lambda r: (r["entry_at"], repr(_entry_key(r))))
        self.movements = Counter(r["movement"] for r in context.records.values())
        self.history = owned.iloc[:0].copy()  # score-basis marker only; no per-trial pandas work
        self.refilter = _needs_refilter(self.events, entries, None, config)
        self.schedule = _Schedule(self.events, entries)
        self.versions = list(_versions(self.events, entries)) if self.refilter else []
        self.times = np.array([e["kickoff_at"].value for e in self.events], dtype=np.int64)
        self.queries = defaultdict(list)
        self.league_queries = defaultdict(list)
        for i, event in enumerate(self.events):
            competition = event["competition_id"]
            self.league_queries[competition].append(i)
            for role in ("home", "away"):
                self.queries[(competition, event[f"{role}_id"], role)].append(i)
        self.scores = np.array([(e["home_goals"], e["away_goals"]) for e in self.events])
        self.outcomes = np.where(self.scores[:, 0] > self.scores[:, 1], 0,
                                 np.where(self.scores[:, 0] == self.scores[:, 1], 1, 2))

    def evaluate(self, model, *, objective="outcome_log_loss"):
        from .bayesian_replay import _replay_in_order
        from .bayesian_probability import outcome_probabilities

        if model.config != self.config:
            raise ValueError("Calibration configuration changed; rebuild the replay plan.")
        states = _States()
        if self.refilter:
            for time, schedule, canonical, _ in _suffix_versions(self.history, model, self.versions):
                states.publish(canonical, time)
        else:
            _replay_in_order(self.history, model=model, _plan=self.schedule,
                             _predict=False, _sink=states)
        factors = np.empty((len(self.events), 5))
        for (competition, team, role), indices in self.queries.items():
            values = _lookup(states.teams.get((competition, team)), self.times[indices],
                             initial_team(model.parameters_for(competition)))
            columns = (0, 1) if role == "home" else (2, 3)
            factors[np.ix_(indices, columns)] = [(s.attack.mean, s.defence.mean) for s in values]
        kappas = np.empty(len(self.events))
        for competition, indices in self.league_queries.items():
            parameters = model.parameters_for(competition)
            values = _lookup(states.leagues.get(competition), self.times[indices], initial_home(parameters))
            factors[indices, 4] = [s.mean for s in values]
            kappas[indices] = parameters.kappa if self.config.bivariate else np.nan
        home = factors[:, 0] * factors[:, 3] * factors[:, 4]
        away = factors[:, 2] * factors[:, 1]
        probabilities = outcome_probabilities(home, away, kappas,
            tail_tolerance=self.config.tail_tolerance, max_total=self.config.max_total_goals)
        score_log = np.array([score_log_probability(x, y, h, a, None if np.isnan(k) else k)
                              for (x, y), h, a, k in zip(self.scores, home, away, kappas)])
        log_prob = score_log if objective == "score_log_loss" else np.log(np.maximum(
            probabilities[np.arange(len(self.events)), self.outcomes], np.finfo(float).tiny))
        if not np.isfinite(log_prob).all():
            raise ValueError("Nonfinite prequential likelihood during Bayesian calibration.")
        return CalibrationEvidence(float(-log_prob.mean()), home, away, probabilities,
                                   score_log, states)


class CalibrationEvidence:
    def __init__(self, loss, home, away, probabilities, score_log, states):
        self.loss, self.home, self.away = loss, home, away
        self.probabilities, self.score_log = probabilities, score_log
        self.states = states
        self.entry_counts = Counter(item[-1] for item in states.usage)


def _empty_kernel():
    return dict(states={}, home_states={}, team_season={}, home_season={}, latest={},
                home_latest={}, last_kick={}, counts=Counter(), processed_entries=set(),
                seen_events=set(), closed_sources=set(), usage=set(), iterations=0)


def _copy_kernel(state):
    # Gamma/TeamState values are immutable. Copies isolate each rollback point;
    # no mutable fitted state survives across parameter evaluations.
    return {key: value.copy() if isinstance(value, (dict, set)) else value
            for key, value in state.items()}


def _suffix_versions(history, model, versions, *, stride=128):
    """Resume unchanged prefixes; rewind before the earliest changed leaf.

    Global chronological checkpoints intentionally preserve cross-league bridge
    dependencies, frozen simultaneous entries and shared league clocks. A late
    insertion replays the entire affected suffix, not just its participating teams.
    Only one checkpoint per stride of schedule positions is retained.
    """
    from copy import copy
    from .bayesian_replay import _replay_in_order
    state = _empty_kernel()
    checkpoints = {}
    last = -1
    for time, original in versions:
        schedule = copy(original)  # plan is reusable and immutable across trials
        if schedule.changed <= last:
            for position in list(checkpoints):
                if position >= schedule.changed:
                    del checkpoints[position]
            last = max(checkpoints, default=-1)
            state = _copy_kernel(checkpoints[last]) if last >= 0 else _empty_kernel()
        schedule.start = last + 1
        saved_block = max(checkpoints, default=-1) // stride
        def remember(kickoff, current):
            nonlocal last, saved_block
            last = schedule.positions[kickoff]
            if last // stride > saved_block:
                checkpoints[last] = _copy_kernel(current)
                saved_block = last // stride
        sink = _replay_in_order(history, model=model, _plan=schedule, _predict=False,
                                _sink=_States(history=False), _resume=state, _remember=remember)
        # Canonical usage includes the restored prefix as well as this suffix.
        sink.usage.update(state["usage"])
        yield time, schedule, sink, state


def _kernel_payload(state, boundary):
    """Serialize a current mathematical checkpoint using the native schema."""
    from .bayesian import _identifier
    from .bayesian_replay import _team_payload
    def date(value):
        return None if value is None or pd.isna(value) else value.isoformat()
    return dict(processed_through=date(boundary),
        teams=[dict(competition_id=_identifier(key[0]), team_id=_identifier(key[1]),
                    season_id=_identifier(state["team_season"][key]), state=_team_payload(value),
                    latest_kickoff_at=date(state["latest"].get(key)),
                    last_kickoff_at=date(state["last_kick"].get(key)), games_seen=state["counts"][key])
               for key, value in sorted(state["states"].items(), key=lambda x: repr(x[0]))],
        leagues=[dict(competition_id=_identifier(key), season_id=_identifier(state["home_season"][key]),
                      state=[value.shape, value.rate], latest_kickoff_at=date(state["home_latest"].get(key)))
                 for key, value in sorted(state["home_states"].items(), key=lambda x: repr(x[0]))],
        **{target: [list(map(_identifier, item)) for item in sorted(state[source], key=repr)]
           for target, source in (("entries", "processed_entries"), ("event_ids", "seen_events"),
                                  ("closed_sources", "closed_sources"))})
