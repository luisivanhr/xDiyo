"""Schedule-derived season clock, independent of results and row filtering."""
from dataclasses import dataclass
from numbers import Integral

from .expressions import Expr


@dataclass(frozen=True)
class SeasonCountdown(Expr):
    """Scheduled season round, adjusting outliers to the nearest round cluster.

    Each round uses its largest scheduled cluster spanning at most 96 hours.
    Fixtures inside their own cluster retain their assigned ordinal; outliers
    use the closest cluster interval, ties going to the later round, with their
    original ordinal as a lower bound. Playoffs are excluded. Scheduled context
    is used before kickoff; results and played status are irrelevant.
    """
    total_rounds: int | None = None
    format: str = 'auto'
    team_count: int | None = None

    def __post_init__(self):
        if self.format not in ('auto', 'double', 'single', 'scotland', 'japan_transition'):
            raise ValueError('SeasonCountdown format must be auto, double, single, scotland or japan_transition.')
        for name in ('total_rounds', 'team_count'):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 2):
                raise ValueError(f'SeasonCountdown {name} must be an integer of at least 2.')


_DOUBLE = frozenset(('Bundesliga', 'Bundesliga_2', 'Championship', 'Eredivisie',
    'J1', 'J2', 'La_Liga', 'La_Liga_2', 'Ligue_1', 'Ligue_2', 'Premier_League',
    'Pro_League', 'Serie_A', 'Serie_B'))
_KEYS = ['competition_id', 'season_id', 'event_id']


def _schedule(frame):
    import numpy as np
    import pandas as pd
    missing = set(_KEYS + ['round']) - set(frame)
    if missing:
        raise ValueError(f'SeasonCountdown needs schedule identity columns: {sorted(missing)}')
    names = _KEYS + ['round'] + [c for c in ('kickoff_at', 'kickoff_utc', 'tournament_id',
             'source_league', 'source_season', 'team_id', 'home_id', 'away_id', 'side', 'opponent_id', 'season_phase') if c in frame]
    data = frame[names].reset_index(drop=True).copy()
    data.attrs = {}
    if {'side', 'team_id', 'opponent_id'} <= set(data) and 'home_id' not in data:
        data['home_id'] = data.team_id.where(data.side.eq('home'), data.opponent_id)
        data['away_id'] = data.opponent_id.where(data.side.eq('home'), data.team_id)
    if data[_KEYS].isna().any().any():
        raise ValueError('SeasonCountdown schedule identities must not be missing.')
    time = 'kickoff_at' if 'kickoff_at' in data else 'kickoff_utc'
    if time not in data:
        raise ValueError('SeasonCountdown needs kickoff_at or kickoff_utc.')
    data['kickoff_at'] = (pd.to_datetime(data[time], unit='s', utc=True, errors='coerce')
                          if time == 'kickoff_utc' else pd.to_datetime(data[time], utc=True, errors='coerce', format='mixed'))
    data['round'] = pd.to_numeric(data['round'], errors='coerce').astype(float)
    invalid = data['round'].notna() & (~np.isfinite(data['round']) | (data['round'] < 1) | (data['round'] % 1 != 0))
    if invalid.any():
        raise ValueError('SeasonCountdown numeric rounds must be finite positive integers.')
    fields = ['round', 'kickoff_at'] + [c for c in ('source_league', 'source_season', 'tournament_id', 'season_phase', 'home_id', 'away_id') if c in data]
    if (data.groupby(_KEYS, dropna=False)[fields].nunique(dropna=False) > 1).any().any():
        raise ValueError('SeasonCountdown found inconsistent fixture schedule copies.')
    return data


def _round_clusters(calendar):
    """Largest <=96-hour window of unique fixtures; earliest start wins ties.

    Return sorted ordinals and nanosecond interval endpoints. Only the native
    schedule is used, including upcoming fixtures. Build once per calendar.
    """
    import numpy as np
    usable = calendar.loc[calendar._eligible & calendar._ordinal.notna()
                          & calendar.kickoff_at.notna()].drop_duplicates(_KEYS)
    clusters = []
    span = 96 * 60 * 60 * 1_000_000_000
    for ordinal, group in usable.groupby('_ordinal', sort=True):
        times = np.sort(group.kickoff_at.astype('datetime64[ns, UTC]').astype('int64').to_numpy())
        left = best_left = best_right = 0
        for right in range(len(times)):
            while int(times[right]) - int(times[left]) > span:
                left += 1
            # Strict improvement preserves the earliest cluster on a size tie.
            if right - left > best_right - best_left:
                best_left, best_right = left, right
        clusters.append((ordinal, times[best_left], times[best_right]))
    return (np.array([c[0] for c in clusters], dtype=float),
            np.array([c[1] for c in clusters], dtype='int64'),
            np.array([c[2] for c in clusters], dtype='int64'))


def _nearest_rounds(ordinals, instants, clusters):
    """Vectorized interval distances, bounded across queries rather than per row.

    Compare unsigned nanosecond distances to avoid signed overflow even across
    the complete pandas timestamp range. Later ordinal wins an exact tie.
    """
    import numpy as np
    rounds, starts, ends = clusters
    result = ordinals.copy()
    if not len(rounds):
        return result
    # Flip the sign bit: signed timestamps become ordered unsigned integers.
    def ordered(values):
        return values.view('uint64') ^ np.uint64(1 << 63)
    starts_u, ends_u = ordered(starts), ordered(ends)
    for first in range(0, len(instants), 4096):
        last = first + 4096
        time = ordered(instants[first:last])[:, None]
        distance = np.maximum(starts_u, time) - np.minimum(ends_u, time)
        # Distance is zero for times inside the interval (not unsigned wrap).
        distance[(time >= starts_u) & (time <= ends_u)] = 0
        assigned = ordinals[first:last]
        own = (assigned[:, None] == rounds) & (time >= starts_u) & (time <= ends_u)
        nearest = rounds[len(rounds) - 1 - distance[:, ::-1].argmin(axis=1)]
        result[first:last] = np.where(own.any(axis=1), assigned, np.maximum(assigned, nearest))
    return result


def season_countdown(metadata, expression, *, schedule_history=None, team_seasons=None, cutoffs=None):
    """Resolve scheduled rounds before kickoff, without results or activation.

    schedule_history is the schedule available to the prediction (not a snapshot
    archive). Historical exports may contain revised dates; this function cannot
    reconstruct what dates were known at an earlier issuance. Cutoffs retain the
    native alignment/validation contract but do not gate scheduled context.
    """
    import numpy as np
    import pandas as pd
    from .history import aligned_times
    query = _schedule(metadata)
    cutoff_frame = metadata.assign(kickoff_at=query.kickoff_at.to_numpy())
    times = aligned_times(cutoff_frame, cutoffs, default='kickoff_at').reset_index(drop=True)
    if (times.notna() & query.kickoff_at.notna() & times.gt(query.kickoff_at)).any():
        raise ValueError('SeasonCountdown cutoffs must be no later than target kickoff.')
    source = _schedule(metadata if schedule_history is None else schedule_history)
    output = np.full(len(query), np.nan)
    for key, rows in query.groupby(_KEYS[:2], sort=False):
        population = source.loc[source.competition_id.eq(key[0]) & source.season_id.eq(key[1])]
        games = population.drop_duplicates(_KEYS).copy()
        leagues = games.get('source_league', pd.Series(dtype=str)).dropna().unique()
        seasons = games.get('source_season', pd.Series(dtype=str)).dropna().unique()
        if len(leagues) > 1 or len(seasons) > 1:
            raise ValueError('SeasonCountdown has conflicting league-season metadata.')
        league = str(leagues[0]) if len(leagues) else ''
        season = str(seasons[0]) if len(seasons) else ''
        mode = expression.format
        if mode == 'auto':
            mode = ('double' if league in _DOUBLE else 'single' if league.startswith('Liga_MX_')
                    else 'scotland' if league == 'Premiership' else 'japan_transition' if league == 'J1_Transition' else None)
            if mode is None and expression.total_rounds is None:
                raise ValueError(f'SeasonCountdown has no verified format for {league or key}; supply format or total_rounds.')
        # Stage identifiers are provider tournament IDs, not unique competition IDs.
        eligible = pd.Series(True, index=games.index)
        regular = {'Pro_League': 38, 'Liga_MX_Apertura': 28, 'Liga_MX_Clausura': 16753,
            'Bundesliga': 42, 'Bundesliga_2': 41, 'Championship': 2, 'Eredivisie': 39,
            'J1': 82, 'J2': 3034, 'La_Liga': 36, 'La_Liga_2': 37, 'Ligue_1': 4,
            'Ligue_2': 19, 'Premier_League': 1, 'Serie_A': 33, 'Serie_B': 34}
        if league in regular and ('tournament_id' in games or league in ('Pro_League', 'Liga_MX_Apertura', 'Liga_MX_Clausura')):
            if 'tournament_id' not in games:
                raise ValueError(f'SeasonCountdown needs tournament_id to exclude {league} playoffs.')
            regular_id = 28 if league == 'Liga_MX_Clausura' and season == '10_11' else regular[league]
            eligible &= games.tournament_id.eq(regular_id).fillna(False)
        if mode == 'scotland':
            if 'tournament_id' not in games:
                raise ValueError('Scotland countdown needs tournament_id to identify regular and split stages.')
            eligible &= games.tournament_id.isin([54, 92066, 92067])
        if mode == 'japan_transition':
            if 'tournament_id' not in games:
                raise ValueError('Japan transition countdown needs regional tournament_id.')
            eligible &= games.tournament_id.isin([171852, 171853])
        n = expression.team_count
        fixed = 12 if mode == 'scotland' else 20 if mode == 'japan_transition' else None
        if fixed is not None:
            if n is not None and n != fixed:
                raise ValueError(f'SeasonCountdown {mode} requires {fixed} season teams.')
            n = fixed
        if n is None and expression.total_rounds is None:
            if team_seasons is None:
                raise ValueError('SeasonCountdown needs full team_seasons or team_count to infer total_rounds.')
            roster = team_seasons.loc[team_seasons.competition_id.eq(key[0]) & team_seasons.season_id.eq(key[1])]
            observed = set()
            for column in ('team_id', 'home_id', 'away_id'):
                if column in population:
                    observed.update(population[column].dropna().unique())
            if not observed.issubset(set(roster.team_id.dropna())):
                raise ValueError(f'SeasonCountdown team membership omits scheduled teams for {key}.')
            n = roster.team_id.dropna().nunique()
            if n < 2:
                raise ValueError(f'SeasonCountdown has no complete team membership for {key}.')
        total = expression.total_rounds
        if total is None:
            total = 38 if mode == 'scotland' else 18 if mode == 'japan_transition' else (n - 1 if n % 2 == 0 else n) * (2 if mode == 'double' else 1)
        if total < 2:
            raise ValueError('SeasonCountdown requires at least two planned rounds.')
        ordinal = games['round'].copy()
        if league == 'J1' and season == '16_16':
            phases = _japan_phases(games.loc[eligible])
            ordinal.loc[eligible] += (phases - 1) * 17
        if (eligible & ordinal.gt(total)).any():
            raise ValueError(f'SeasonCountdown round exceeds total_rounds={total} for {key}.')
        games['_ordinal'] = ordinal
        games['_eligible'] = eligible
        fields = ['round', 'kickoff_at'] + [c for c in ('source_league', 'source_season', 'tournament_id', 'season_phase', 'home_id', 'away_id') if c in query and c in games]
        joined = rows.merge(games[_KEYS + fields + ['_ordinal','_eligible']], on=_KEYS,
            how='left', suffixes=('', '_source'), validate='many_to_one', indicator=True, sort=False)
        if joined['_merge'].ne('both').any():
            raise ValueError('SeasonCountdown query fixture is absent from schedule_history.')
        for field in fields:
            a, b = joined[field], joined[field + '_source']
            if (~(a.eq(b) | (a.isna() & b.isna()))).any():
                raise ValueError(f'SeasonCountdown query {field} conflicts with schedule_history.')
        effective = joined._ordinal.to_numpy(dtype=float, na_value=np.nan, copy=True)
        query_times = times.iloc[rows.index].reset_index(drop=True)
        # Scotland's split remains continuous; Japanese transition groups have
        # independent calendars even when their scheduled intervals overlap.
        calendars = games.groupby('tournament_id', sort=False, dropna=False) if mode == 'japan_transition' else [(None, games)]
        for calendar_id, calendar in calendars:
            selected = joined.tournament_id.eq(calendar_id).fillna(False) if mode == 'japan_transition' else pd.Series(True, index=joined.index)
            selected &= joined.kickoff_at.notna() & joined._ordinal.notna() & joined._eligible.fillna(False)
            positions = np.flatnonzero(selected)
            if not len(positions):
                continue
            clusters = _round_clusters(calendar)
            instants = joined.kickoff_at.iloc[positions].astype('datetime64[ns, UTC]').astype('int64').to_numpy()
            effective[positions] = _nearest_rounds(effective[positions], instants, clusters)
        effective[~(joined._eligible.fillna(False) & query_times.notna() & joined.kickoff_at.notna()).to_numpy()] = np.nan
        output[rows.index] = (total - effective) / (total - 1)
    return pd.Series(output, index=metadata.index, dtype=float)


def _japan_phases(games):
    """Immutable original phase; never derive it from a rescheduled kickoff."""
    from importlib.resources import files
    import pandas as pd
    keys = [*_KEYS, 'round', 'home_id', 'away_id']
    known = pd.Series(float('nan'), index=games.index)
    if set(keys) <= set(games):
        with files('xdiyo_analytics.features').joinpath('data/j1_2016_phases.csv').open('rb') as stream:
            reference = pd.read_csv(stream)
        matched = games[keys].merge(reference, on=keys, how='left', validate='one_to_one', sort=False)
        known = pd.Series(matched.season_phase.to_numpy(), index=games.index)
    if 'season_phase' in games:
        explicit = pd.to_numeric(games.season_phase, errors='coerce')
        if (games.season_phase.notna() & ~explicit.isin([1,2])).any():
            raise ValueError('Japan 2016 season_phase must be 1 or 2.')
        if (explicit.notna() & known.notna() & explicit.ne(known)).any():
            raise ValueError('Japan 2016 phase conflicts with verified fixture identity.')
        known = explicit.fillna(known)
    if known.isna().any():
        raise ValueError('Japan 2016 requires original season_phase (1 or 2) or a verified packaged fixture identity; kickoff cannot identify phase.')
    return known
