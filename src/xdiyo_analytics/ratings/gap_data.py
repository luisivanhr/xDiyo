"""Explicit, count-preserving GAP adapter. No provider definitions are guessed."""
from dataclasses import dataclass
from .gap_core import STATISTICS


@dataclass(frozen=True)
class GAPStatistic:
    """Prepared-history columns in focal-team/opponent coordinates.

    Supply provider/period/field definitions, including whether goals exclude
    extra time and penalties. Derivation is deliberately a separate operation.
    """
    name: str
    for_column: str
    against_column: str
    definition: str

    def __post_init__(self):
        if self.name not in STATISTICS:
            raise ValueError(f'Unsupported paper statistic: {self.name}')
        if not all(isinstance(x, str) and x.strip() for x in (self.for_column, self.against_column, self.definition)):
            raise ValueError('Supply distinct columns and an explicit statistic definition.')
        if self.for_column == self.against_column:
            raise ValueError('Team and opponent columns must differ.')


    @classmethod
    def from_stat(cls, history, name, source, *, definition):
        """Resolve an exact existing Stat identity, preserving provider metadata."""
        from ..features.expressions import Stat
        if not isinstance(source, Stat) or source.period is None:
            raise ValueError('GAP needs one explicitly selected statistic period.')
        metadata = history.attrs.get('stat_columns', {})
        own = [column for column, info in metadata.items()
               if info.get('role') == 'team' and info.get('period') == source.period
               and info.get('group_name') == source.group and info.get('key') == source.key
               and info.get('field') == source.field]
        if len(own) != 1:
            raise ValueError('GAP needs one exact team statistic column.')
        opponent = [column for column, info in metadata.items() if info == {**metadata[own[0]], 'role': 'opponent'}]
        if len(opponent) != 1:
            raise ValueError('GAP needs one exact mirrored opponent statistic column.')
        return cls(name, own[0], opponent[0], definition)


def derive_shots_off_target(total, on_target, *, total_is_on_plus_off, definition_evidence):
    """Only subtract after caller verifies the provider's additive definition.

    Totals including a third blocked-shot category are NOT suitable. Evidence
    should identify the provider document/version and consistent period/units.
    Missing values stay missing; negatives and nonfinite observations fail.
    """
    import numpy as np
    import pandas as pd
    if total_is_on_plus_off is not True or not isinstance(definition_evidence, str) or not definition_evidence.strip():
        raise ValueError('Verified provider definition evidence is required before deriving shots off target.')
    if isinstance(total, pd.Series) and isinstance(on_target, pd.Series) and not total.index.equals(on_target.index):
        raise ValueError('Shot columns must align.')
    a, b = np.asarray(total, dtype=float), np.asarray(on_target, dtype=float)
    if a.shape != b.shape:
        raise ValueError('Shot columns must have identical shape.')
    for values in (a, b):
        if np.isinf(values).any() or (values < 0).any():
            raise ValueError('Invalid shot counts.')
    result = a - b
    if (result < 0).any():
        raise ValueError('Total shots minus on-target is negative; check provider definitions.')
    if isinstance(total, pd.Series):
        return pd.Series(result, index=total.index, name='shots_off_target')
    return result


def gap_observations(history, statistics, *, available_at=None):
    """Validate two perspective rows and produce one paired observation per game.

    Explicit availability is required (or result_available_at); no kickoff proxy.
    Complete-pair omission is per statistic, while games_seen counts completed
    games even when that statistic is missing. Zero is an observation.
    """
    import numpy as np
    import pandas as pd
    from ..features.history import aligned_times, availability_times
    statistics = tuple(statistics)
    names = [s.name for s in statistics]
    if not names or len(set(names)) != len(names):
        raise ValueError('Choose distinct GAP statistic families.')
    keys = ['competition_id', 'season_id', 'event_id']
    needed = [*keys, 'team_id', 'opponent_id', 'side', 'status', 'kickoff_at']
    if history[needed[:-1]].isna().any().any():
        raise ValueError('GAP match and team identifiers must be nonmissing.')
    if not history.side.isin(['home', 'away']).all() or history.duplicated(keys + ['side']).any():
        raise ValueError('GAP requires distinct home and away perspective rows.')
    if not history.groupby(keys).size().eq(2).all():
        raise ValueError('GAP requires both perspective rows.')
    if available_at is None:
        if 'result_available_at' not in history:
            raise ValueError('GAP requires explicit result availability; no kickoff proxy is assumed.')
        available_at = 'result_available_at'
    kick = aligned_times(history, None, default='kickoff_at')
    release = availability_times(history, available_at)
    if kick.isna().any() or (release < kick).any():
        raise ValueError('Kickoff must be known and availability cannot precede kickoff.')
    frame = history.copy()
    frame['_kick'], frame['_release'] = kick.to_numpy(), release.to_numpy()
    for stat in statistics:
        for column in (stat.for_column, stat.against_column):
            if frame[column].map(lambda x: isinstance(x, (bool, np.bool_))).any():
                raise ValueError(f'Boolean values are not GAP counts: {column}.')
            values = pd.to_numeric(frame[column], errors='raise').to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(values).any() or (values < 0).any():
                raise ValueError(f'Invalid negative/nonfinite observations in {column}.')
            frame[column] = values
    records = []
    def same(a, b):
        return (pd.isna(a) and pd.isna(b)) or a == b
    for identity, rows in frame.groupby(keys, sort=False):
        h, a = rows.loc[rows.side.eq('home')].iloc[0], rows.loc[rows.side.eq('away')].iloc[0]
        if h.team_id == h.opponent_id or h.team_id != a.opponent_id or h.opponent_id != a.team_id:
            raise ValueError('Inconsistent reciprocal team identities.')
        if any(not same(h[c], a[c]) for c in ('_kick', '_release', 'status')):
            raise ValueError('Both perspectives must agree on kickoff, availability and status.')
        record = dict(zip(keys, identity))
        record.update(home_id=h.team_id, away_id=a.team_id, kickoff_at=h['_kick'],
                      available_at=h['_release'], finished=h.status == 'finished')
        for stat in statistics:
            yh, ya = h[stat.for_column], h[stat.against_column]
            if not same(yh, a[stat.against_column]) or not same(ya, a[stat.for_column]):
                raise ValueError(f'Inconsistent mirrored {stat.name} counts.')
            record[f'{stat.name}_home'], record[f'{stat.name}_away'] = yh, ya
        records.append(record)
    return pd.DataFrame(records, columns=[*keys, 'home_id', 'away_id', 'kickoff_at', 'available_at', 'finished',
                                        *(f'{s}_{side}' for s in names for side in ('home', 'away'))])


@dataclass(frozen=True)
class GAPSeason:
    """Known-at-cutoff membership for ONE league, season and statistic.

    members are (team_id, movement) pairs. First-stat seasons use 'initial';
    subsequent ones use proven retained/promoted/relegated movements. Donor IDs
    are promoted/relegated OUT OF this league's previous season. Evidence must
    identify reliable league/team mapping and membership provenance.
    """
    competition_id: object
    season_id: object
    statistic: str
    entry_at: str
    known_at: str
    members: tuple
    evidence: str
    previous_season_id: object = None
    promoted_out: tuple = ()
    relegated_out: tuple = ()

    def __post_init__(self):
        import pandas as pd
        if self.statistic not in STATISTICS or not self.evidence.strip():
            raise ValueError('Season requires a paper statistic and membership evidence.')
        for value in (self.competition_id, self.season_id):
            if pd.isna(value):
                raise ValueError('Reliable league and season identities are required.')
        entry, known = pd.to_datetime(self.entry_at, utc=True), pd.to_datetime(self.known_at, utc=True)
        if pd.isna(entry) or pd.isna(known) or known > entry:
            raise ValueError('Membership must be known at the season entry cutoff.')
        object.__setattr__(self, 'entry_at', entry.isoformat())
        object.__setattr__(self, 'known_at', known.isoformat())
        object.__setattr__(self, 'members', tuple(tuple(m) for m in self.members))
        for name in ('promoted_out', 'relegated_out'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        ids = [t for t, movement in self.members]
        if not ids or len(set(ids)) != len(ids) or any(pd.isna(t) for t in ids):
            raise ValueError('Season membership requires distinct nonmissing team IDs.')
        allowed = {'initial'} if self.previous_season_id is None else {'retained', 'promoted', 'relegated'}
        if any(m not in allowed for t, m in self.members):
            raise ValueError('GAP initialization unavailable: missing or incompatible movement evidence.')
        if self.previous_season_id == self.season_id:
            raise ValueError('A season cannot precede itself.')
        if set(self.promoted_out) & set(self.relegated_out):
            raise ValueError('Donor groups cannot overlap.')
        for donors in (self.promoted_out, self.relegated_out):
            if len(set(donors)) != len(donors) or any(pd.isna(t) for t in donors):
                raise ValueError('Donor IDs must be distinct and nonmissing.')
        if (set(self.promoted_out) | set(self.relegated_out)) & set(ids):
            raise ValueError('Departing donor teams cannot also remain in this league.')

    @property
    def key(self):
        return self.competition_id, self.season_id, self.statistic


def gap_season_from_context(context, *, competition_id, season_id, statistic,
                            known_at, evidence, first_stat_season=False,
                            promoted_out=(), relegated_out=()):
    """Use validated TransitionContext/build_team_seasons evidence, not Glicko math.

    Caller supplies dated, verified outgoing cohorts because the context's
    current-league rows alone do not establish complete outgoing membership.
    Unknown entrants are rejected by GAPSeason. first_stat_season is an explicit
    coverage assertion, never inferred from a team's first dataset appearance.
    """
    key = competition_id, season_id
    previous = context.predecessors[key]
    if not first_stat_season and previous is None:
        raise ValueError('GAP requires an adjacent previous season or explicit first-stat coverage.')
    records = [r for (c, s, t), r in context.records.items() if (c, s) == key]
    return GAPSeason(competition_id, season_id, statistic, context.anchors[key], known_at,
                     tuple((r['team_id'], 'initial' if first_stat_season else r['movement']) for r in records),
                     evidence, None if first_stat_season else previous[1], tuple(promoted_out), tuple(relegated_out))
