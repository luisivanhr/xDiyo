"""Availability-aware GAP replay and native, vintage-safe numeric rating runs.

Wheatcroft (2021), pp.82-85. Reconstructing history under a new parameter
vintage is an explicit caller choice, not a claim about an unspecified detail
of the paper. Prediction scheduling is supplied separately through cutoffs.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from numbers import Number

from .gap_core import ALGORITHM, FIELDS, GAPParameters, GAPState, WheatcroftGAP, donor_mean
from .gap_data import GAPSeason, GAPStatistic, gap_observations, gap_prediction_times
from .snapshots import RatingRun


def _time(value):
    import pandas as pd
    if isinstance(value, Number):
        raise ValueError('Use explicit datetimes, not numeric timestamps.')
    result = pd.to_datetime(value, utc=True)
    if result is None or pd.isna(result):
        raise ValueError('A finite datetime is required.')
    return result


@dataclass(frozen=True)
class GAPVintage:
    """One pooled triple per statistic for seasons entering in [start, end).

    fit_cutoff is the latest information used for parameter estimation. A
    synthetic example must say so in provenance; there are no default values.
    A season keeps that vintage even for fixtures postponed beyond end.
    Adjacent vintages can be supplied to build_gap_ratings without fitting.
    """
    parameters: tuple
    fit_cutoff: str
    valid_from: str
    valid_until: str
    provenance: str

    def __post_init__(self):
        from .gap_core import STATISTICS
        values = tuple((name, p) for name, p in self.parameters)
        names = [name for name, p in values]
        if not names or len(set(names)) != len(names) or set(names) - set(STATISTICS):
            raise ValueError('Vintage requires distinct paper statistics.')
        if not all(isinstance(p, GAPParameters) for name, p in values):
            raise TypeError('Supply GAPParameters for each statistic.')
        object.__setattr__(self, 'parameters', values)
        fit, start, end = map(_time, (self.fit_cutoff, self.valid_from, self.valid_until))
        if fit > start or start >= end:
            raise ValueError('Require fit_cutoff <= valid_from < valid_until.')
        for name, value in zip(('fit_cutoff', 'valid_from', 'valid_until'), (fit, start, end)):
            object.__setattr__(self, name, value.isoformat())
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError('Parameter provenance is required, including synthetic choices.')

    def to_dict(self):
        return {**asdict(self), 'parameters': [[s, asdict(p)] for s, p in self.parameters]}

    @classmethod
    def from_dict(cls, data):
        return cls(**{**data, 'parameters': tuple((s, GAPParameters(**p)) for s, p in data['parameters'])})


class _Replay:
    """Reference replay by kickoff within the currently available information set.

    A delayed release re-filters that season's known games in kickoff order.
    Transition donor states are frozen at entry, so later prior-season releases
    cannot retrospectively rewrite a new season's initial state. Deliberately
    simple prefix replay: correctness reference, not a real-data rebuild job.
    """
    def __init__(self, observations, seasons, parameters):
        import pandas as pd
        self.observations = observations
        self.parameters = dict(parameters)
        self.algorithm = WheatcroftGAP()
        self.plans = {s.key: s for s in seasons if s.statistic in self.parameters}
        if len(self.plans) != len([s for s in seasons if s.statistic in self.parameters]):
            raise ValueError('Duplicate GAP season/league/statistic plans.')
        # A missing predecessor must never quietly reset a later season.
        coverage = {}
        for plan in self.plans.values():
            coverage.setdefault((plan.competition_id, plan.statistic), []).append(plan)
        for plans in coverage.values():
            previous = None
            for plan in sorted(plans, key=lambda p: _time(p.entry_at)):
                expected = None if previous is None else previous.season_id
                if plan.previous_season_id != expected:
                    raise ValueError('GAP requires the immediately preceding season; only first-stat coverage initializes at zero.')
                if previous is not None and _time(previous.entry_at) == _time(plan.entry_at):
                    raise ValueError('League/statistic seasons must have distinct entry cutoffs.')
                previous = plan
        self.cache, self.initials = {}, {}
        self.events = {}
        for key, plan in self.plans.items():
            self.events[key] = sorted(
                observations.loc[observations.competition_id.eq(key[0]) & observations.season_id.eq(key[1])].to_dict('records'),
                key=lambda r: (r['kickoff_at'], repr(r['event_id'])),
            )
            members = dict(plan.members)
            for r in self.events[key]:
                if plan.closed_at is not None and r['kickoff_at'] >= _time(plan.closed_at):
                    raise ValueError('Season closure contradicts a fixture at or after its closed_at boundary.')
                if r['kickoff_at'] < _time(plan.entry_at):
                    raise ValueError('Season entry cannot follow a fixture kickoff.')
                if r['home_id'] not in members or r['away_id'] not in members:
                    raise ValueError('GAP initialization unavailable: fixture team missing from known membership.')
            if plan.previous_season_id is not None:
                prior = self.plans.get((key[0], plan.previous_season_id, key[2]))
                if prior is None or _time(prior.entry_at) >= _time(plan.entry_at):
                    raise ValueError('GAP requires an earlier explicit previous-season plan.')
                old = dict(prior.members)
                for team, movement in plan.members:
                    if movement == 'retained' and team not in old:
                        raise ValueError('Retained team is absent from previous membership.')
                    if movement != 'retained' and team in old:
                        raise ValueError('Incoming team was already in this league; movement evidence contradicts membership.')
                if any(m == 'promoted' for _, m in plan.members) and not plan.relegated_out:
                    raise ValueError('GAP initialization unavailable: empty donor group for promoted entrants.')
                if any(m == 'relegated' for _, m in plan.members) and not plan.promoted_out:
                    raise ValueError('GAP initialization unavailable: empty donor group for relegated entrants.')
                if (set(plan.promoted_out) | set(plan.relegated_out)) - set(old):
                    raise ValueError('Donor IDs must belong to this league in the previous season.')
        for r in observations.to_dict('records'):
            for stat in self.parameters:
                if (r['competition_id'], r['season_id'], stat) not in self.plans:
                    # Absent statistic coverage can precede its explicitly declared first season.
                    if pd.notna(r[f'{stat}_home']) or pd.notna(r[f'{stat}_away']):
                        raise ValueError(f'Missing GAP season metadata for observed {stat}.')

    def initial(self, key):
        import pandas as pd
        if key in self.initials:
            return self.initials[key]
        plan = self.plans[key]
        latest = _time(plan.entry_at) - pd.Timedelta(1, 'ns')
        if plan.previous_season_id is None:
            states = {team: GAPState() for team, movement in plan.members}
        else:
            old, _, _, prior_latest = self.state_at((key[0], plan.previous_season_id, key[2]), _time(plan.entry_at))
            states = {}
            for team, movement in plan.members:
                donors = (team,) if movement == 'retained' else (
                    plan.relegated_out if movement == 'promoted' else plan.promoted_out)
                states[team] = donor_mean(old[t] for t in donors)
            latest = max(latest, prior_latest)
        self.initials[key] = states, latest
        return states, latest

    def state_at(self, key, cutoff):
        import pandas as pd
        cutoff = _time(cutoff)
        cached = key, cutoff
        if cached in self.cache:
            return self.cache[cached]
        plan = self.plans[key]
        if cutoff < _time(plan.entry_at):
            raise ValueError('Membership/initialization not available at the requested cutoff.')
        base, latest = self.initial(key)
        states = dict(base)
        games, updates = dict.fromkeys(states, 0), dict.fromkeys(states, 0)
        seen_kickoffs = set()
        stat = key[2]
        for r in self.events[key]:
            if not r['finished'] or pd.isna(r['available_at']) or r['available_at'] > cutoff or r['kickoff_at'] >= cutoff:
                continue
            h, a = r['home_id'], r['away_id']
            for team in (h, a):
                tie = team, r['kickoff_at']
                if tie in seen_kickoffs:
                    raise ValueError('A team has two matches at the same kickoff; chronological GAP order is undefined.')
                seen_kickoffs.add(tie)
                games[team] += 1
            latest = max(latest, r['kickoff_at'])
            yh, ya = r[f'{stat}_home'], r[f'{stat}_away']
            if pd.isna(yh) or pd.isna(ya):
                continue
            states[h], states[a] = self.algorithm.update(states[h], states[a], float(yh), float(ya), self.parameters[stat])
            updates[h] += 1
            updates[a] += 1
        result = states, games, updates, latest
        self.cache[cached] = result
        return result


class GAPRatingRun:
    """Compose RatingRun's numeric lookup with native vintage and membership guards."""
    def __init__(self, snapshots, vintages, statistics, seasons, metadata):
        self.snapshots = snapshots
        self.vintages = tuple(vintages)
        self.statistics = tuple(statistics)
        self.seasons = tuple(seasons)
        self.metadata = metadata
        self.streams = tuple(s.name for s in self.statistics)
        self.scope = ('competition_id', 'season_id', 'gap_vintage')
        self.initial_state = dict.fromkeys(FIELDS, 0.0)
        self._numeric = RatingRun(snapshots, {**self.initial_state, 'initialized': 0.0, 'games_seen': 0.0, 'stat_updates': 0.0},
                                  self.scope, self.streams, metadata)
        _validate_vintages(self.vintages, self.streams)
        if (metadata.get('algorithm') != ALGORITHM or metadata.get('state_policy') != 'replay'
                or metadata.get('schema') != 2 or metadata.get('vintage_selection') != 'season_entry'):
            raise ValueError('Unsupported GAP artifact algorithm/state policy.')

    def _query(self, history, cutoffs):
        import pandas as pd
        times = gap_prediction_times(history, cutoffs)
        query = history.copy()
        entries = _season_entries(self.seasons)
        indexes = []
        for row, time in zip(history.itertuples(), times):
            entry = entries.get((row.competition_id, row.season_id))
            if pd.isna(time) or entry is None:
                indexes.append(-1)
                continue
            candidates = [i for i, v in enumerate(self.vintages) if _time(v.valid_from) <= entry < _time(v.valid_until)]
            if len(candidates) != 1:
                raise ValueError('No parameter vintage for this target season; later-season fits cannot serve earlier rows.')
            i = candidates[0]
            if time < _time(self.vintages[i].fit_cutoff) or time < _time(self.vintages[i].valid_from):
                raise ValueError('GAP parameter vintage was not available at the prediction cutoff.')
            indexes.append(i)
        query['gap_vintage'] = indexes
        return query, times

    def features(self, history, *, cutoffs=None, side='both', fields=None):
        fields = FIELDS if fields is None else tuple(fields)
        if not fields or len(set(fields)) != len(fields) or set(fields) - set(FIELDS):
            raise ValueError('Select distinct GAP HA/HD/AA/AD state fields; diagnostics are separate.')
        query, times = self._query(history, cutoffs)
        values = self._numeric.features(query, cutoffs=times, side=side, fields=(*fields, 'initialized'))
        for stat in self.streams:
            for role in ('team', 'opponent'):
                column = f'{stat}::{role}::initialized'
                if column in values:
                    missing = values.pop(column).ne(1)
                    values.loc[missing, [f'{stat}::{role}::{f}' for f in fields]] = float('nan')
        return values

    def fixture_features(self, history, *, cutoffs=None, statistics=None, fields=('difference',)):
        import pandas as pd
        statistics = self.streams if statistics is None else tuple(statistics)
        fields = tuple(fields)
        if not statistics or len(set(statistics)) != len(statistics) or set(statistics) - set(self.streams):
            raise ValueError('Select distinct available GAP statistic streams.')
        if not fields or len(set(fields)) != len(fields) or set(fields) - {'home', 'away', 'difference'}:
            raise ValueError('GAP fixture fields are home, away or difference.')
        if not history.side.isin(['home', 'away']).all():
            raise ValueError('Fixture perspective must be home or away.')
        states = self.features(history, cutoffs=cutoffs)
        home_row = history.side.eq('home').to_numpy()
        output = {}
        import numpy as np
        for stat in statistics:
            def state(role, field):
                return states[f'{stat}::{role}::{field}'].to_numpy()
            home = np.where(home_row, state('team', 'HA')/2 + state('opponent', 'AD')/2,
                            state('opponent', 'HA')/2 + state('team', 'AD')/2)
            away = np.where(home_row, state('opponent', 'AA')/2 + state('team', 'HD')/2,
                            state('team', 'AA')/2 + state('opponent', 'HD')/2)
            values = {'home': home, 'away': away, 'difference': home-away}
            for field in fields:
                output[f'gap_{stat}_{"diff" if field == "difference" else field}'] = values[field]
        return pd.DataFrame(output, index=history.index)

    def diagnostics(self, history, *, cutoffs=None):
        """Never predictor columns. Burn-in chooses the literal p.85 pre-count > 6."""
        import numpy as np
        import pandas as pd
        query, times = self._query(history, cutoffs)
        values = self._numeric.features(query, cutoffs=times, fields=('initialized', 'games_seen', 'stat_updates'))
        home = history.side.eq('home').to_numpy()
        output = {}
        for stat in self.streams:
            for field in ('games_seen', 'stat_updates', 'initialized'):
                team, opponent = (values[f'{stat}::{role}::{field}'].to_numpy() for role in ('team', 'opponent'))
                output[f'{stat}::home_{field}'] = np.where(home, team, opponent)
                output[f'{stat}::away_{field}'] = np.where(home, opponent, team)
            ready = (output[f'{stat}::home_initialized'] == 1) & (output[f'{stat}::away_initialized'] == 1)
            output[f'{stat}::status'] = np.where(ready, 'available', 'initialization_unavailable')
            output[f'{stat}::paper_burn_in_eligible'] = ready & (output[f'{stat}::home_games_seen'] > 6)
        return pd.DataFrame(output, index=history.index)

    def save(self, directory):
        """Native checksum artifact; deliberately no generic ratings.json file."""
        import hashlib
        from uuid import uuid4
        from ..experiments.store import _publish
        target = Path(directory)
        if target.exists():
            raise FileExistsError(f'GAP artifact already exists: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = target.parent / f'.pending-gap-{uuid4()}'
        stage.mkdir()
        try:
            self.snapshots.to_parquet(stage / 'snapshots.parquet', index=False)
            state = dict(schema=2, vintages=[v.to_dict() for v in self.vintages],
                         statistics=[asdict(s) for s in self.statistics], seasons=[asdict(s) for s in self.seasons], metadata=self.metadata)
            def scalar(value):
                if hasattr(value, 'item'):
                    return value.item()
                raise TypeError(f'Unsupported JSON value {type(value).__name__}')
            (stage / 'state.json').write_text(json.dumps(state, indent=2, allow_nan=False, default=scalar), encoding='utf-8')
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.iterdir()}
            (stage / 'gap.json').write_text(json.dumps({'schema': 2, 'files': hashes}, indent=2), encoding='utf-8')
            _publish(stage, target)
        except Exception:
            import shutil
            if stage.exists() and not stage.is_symlink() and stage.resolve().parent == target.parent.resolve():
                shutil.rmtree(stage)
            raise
        return target

    @classmethod
    def load(cls, directory):
        import hashlib
        import pandas as pd
        root = Path(directory).resolve()
        manifest = json.loads((root / 'gap.json').read_text(encoding='utf-8'))
        if manifest.get('schema') != 2 or set(manifest.get('files', {})) != {'snapshots.parquet', 'state.json'}:
            raise ValueError('Invalid or legacy GAP manifest; rebuild schema-1 artifacts with the audited schema-2 producer.')
        for name, digest in manifest['files'].items():
            path = root / name
            if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'GAP artifact file missing or changed: {name}')
        state = json.loads((root / 'state.json').read_text(encoding='utf-8'))
        if state.get('schema') != 2:
            raise ValueError('Unsupported GAP artifact schema.')
        return cls(pd.read_parquet(root / 'snapshots.parquet'), tuple(GAPVintage.from_dict(v) for v in state['vintages']),
                   tuple(GAPStatistic(**s) for s in state['statistics']), tuple(GAPSeason(**s) for s in state['seasons']), state['metadata'])


def _validate_vintages(vintages, streams):
    if not vintages:
        raise ValueError('Supply explicit parameter vintages; GAP has no production defaults.')
    previous = None
    for vintage in vintages:
        if set(dict(vintage.parameters)) != set(streams):
            raise ValueError('Every vintage must supply exactly the selected statistic families.')
        if previous is not None and _time(vintage.valid_from) < _time(previous.valid_until):
            raise ValueError('Parameter vintages must be chronological and nonoverlapping.')
        previous = vintage


def _season_entries(seasons):
    entries = {}
    for plan in seasons:
        key, entry = (plan.competition_id, plan.season_id), _time(plan.entry_at)
        if key in entries and entries[key] != entry:
            raise ValueError('Statistic plans must agree on the league-season entry cutoff.')
        entries[key] = entry
    return entries


def build_gap_ratings(history, *, statistics, seasons, vintages, data_cutoff,
                      state_policy, available_at=None):
    """Produce opt-in numeric snapshots. Never tune parameters or load datasets.

    state_policy='replay' must be explicitly chosen: reconstruct all prior
    history under each new vintage. Continuing the old accumulated state is a
    separate unresolved replication choice and is intentionally not guessed.
    Provide production pre-round cutoffs to feature evaluation, independently
    of update chronology. Repeat-team batches get frozen predictions, followed
    by ordinary sequential paired updates; no residual pooling.
    """
    import pandas as pd
    if state_policy != 'replay':
        raise ValueError('Explicitly choose state_policy="replay"; state continuation is not specified by this implementation.')
    statistics, seasons, vintages = tuple(statistics), tuple(seasons), tuple(vintages)
    _validate_vintages(vintages, tuple(s.name for s in statistics))
    data_cutoff = _time(data_cutoff)
    # Future membership/predecessor claims are not prerequisites for past states.
    seasons = tuple(s for s in seasons if s.statistic in {stat.name for stat in statistics}
                    and _time(s.entry_at) <= data_cutoff)
    _season_entries(seasons)
    observations = gap_observations(history, statistics, available_at=available_at)
    # No values released after the source cutoff enter any state or artifact.
    observed = observations.loc[observations.available_at.le(data_cutoff) & observations.finished].copy()
    snapshots = []
    for index, vintage in enumerate(vintages):
        start, end = _time(vintage.valid_from), _time(vintage.valid_until)
        if start > data_cutoff:
            raise ValueError('Vintage starts after the declared data cutoff.')
        replay = _Replay(observed, seasons, vintage.parameters)
        target_plans = {key: plan for key, plan in replay.plans.items() if start <= _time(plan.entry_at) < end}
        if not target_plans:
            raise ValueError('Parameter vintage has no target season; do not introduce mid-season vintages.')
        # An observation released at kickoff is usable only strictly afterward.
        releases = [max(r.available_at, r.kickoff_at + pd.Timedelta(1, 'ns')) for r in observed.itertuples()]
        times = sorted({start, *(_time(s.entry_at) for s in seasons), *releases})
        times = [t for t in times if start <= t <= data_cutoff]
        for time in times:
            for key, plan in target_plans.items():
                if _time(plan.entry_at) > time:
                    continue
                states, counts, updates, latest = replay.state_at(key, time)
                for team, state in states.items():
                    snapshots.append(dict(competition_id=key[0], season_id=key[1], stream=key[2], gap_vintage=index,
                                          team_id=team, recorded_at=time, latest_kickoff_at=latest,
                                          games_seen=counts[team], stat_updates=updates[team], initialized=1., **asdict(state)))
    columns = ['competition_id', 'season_id', 'stream', 'gap_vintage', 'team_id', 'recorded_at', 'latest_kickoff_at',
               'games_seen', 'stat_updates', 'initialized', *FIELDS]
    table = pd.DataFrame(snapshots, columns=columns)
    import hashlib
    source_digest = hashlib.sha256(observed.sort_values(['kickoff_at', 'competition_id', 'season_id', 'event_id']).to_json(date_format='iso', date_unit='ns').encode()).hexdigest()
    metadata = dict(observations_sha256=source_digest, algorithm=ALGORITHM, schema=2, vintage_selection='season_entry', state_policy=state_policy, data_cutoff=data_cutoff.isoformat(),
                    initialization='zero_first_stat_season_then_retention_or_opposite_outgoing_cohort_mean',
                    missing_observations='skip_incomplete_pair_per_statistic', chronology='kickoff_replay_of_available_prefix',
                    transition_boundary='freeze_prior_season_at_entry', burn_in='home_pre_match_games_seen > 6',
                    feature_schema=[f'gap_{s.name}_diff' for s in statistics],
                    availability='explicit_with_reviewed_result_bounds',
                    paper='Wheatcroft 2021, doi:10.3233/JSA-200462, pp.82-85, equations 6-9')
    return GAPRatingRun(table, vintages, statistics, seasons, metadata)
