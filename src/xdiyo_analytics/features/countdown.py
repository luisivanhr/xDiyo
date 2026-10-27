"""Schedule-derived season clock, independent of results and row filtering."""
from dataclasses import dataclass
from numbers import Integral

from .expressions import Expr


@dataclass(frozen=True)
class SeasonCountdown(Expr):
    """Remaining season rounds in [0, 1], using the round at actual kickoff.

    Auto recognizes published league formats. Supply complete team_seasons and
    schedule_history to evaluate a filtered population. Explicit team_count or
    total_rounds handles custom/partial sources. Playoffs have missing values;
    Scotland's five split rounds are part of its 38-round league season.
    Round anchors are the first kickoff in each round's largest 96-hour cluster
    (earliest cluster wins ties). as_of requires complete timestamped snapshots;
    retrospective explicitly accepts final revised schedules, which are not
    evidence of what was published at an earlier prediction cutoff.
    """
    total_rounds: int | None = None
    format: str = 'auto'
    team_count: int | None = None
    schedule_mode: str = 'as_of'

    def __post_init__(self):
        if self.format not in ('auto', 'double', 'single', 'scotland', 'japan_transition'):
            raise ValueError('SeasonCountdown format must be auto, double, single, scotland or japan_transition.')
        if self.schedule_mode not in ('as_of', 'retrospective'):
            raise ValueError('SeasonCountdown schedule_mode must be as_of or retrospective.')
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


def _anchor(times):
    """Linear sliding cluster on sorted timestamps; isolated catch-ups lose."""
    import pandas as pd
    if times.notna().sum() == 0:
        return pd.NaT
    width = pd.Timedelta(days=4).value
    # pandas may use microseconds; normalize explicitly to nanoseconds.
    values = times.dropna().sort_values().astype('datetime64[ns, UTC]').astype('int64').to_numpy()
    end, best, start = 0, 0, 0
    for i in range(len(values)):
        while end < len(values) and int(values[end]) - int(values[i]) <= width:
            end += 1
        if end - i > best:
            best, start = end - i, i
    return pd.Timestamp(values[start], tz='UTC')


def _retrospective_countdown(metadata, expression, *, schedule_history=None, team_seasons=None):
    """Return one value per query row; use full schedule solely for the clock."""
    import numpy as np
    import pandas as pd
    query = _schedule(metadata)
    source = _schedule(metadata if schedule_history is None else schedule_history)
    records = []
    for key, population in source.groupby(_KEYS[:2], sort=False, dropna=False):
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
        if n is None and expression.total_rounds is None and mode not in ('scotland', 'japan_transition'):
            if team_seasons is None:
                raise ValueError('SeasonCountdown team-count inference needs full team_seasons; supply team_count for a custom schedule.')
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
        games['_total'] = total
        games['_effective'] = ordinal.where(eligible & games.kickoff_at.notna())
        calendars = games.groupby('tournament_id', sort=False, dropna=False) if mode == 'japan_transition' else [(None, games)]
        for _, calendar in calendars:
            valid = calendar[calendar._eligible & calendar._ordinal.notna() & calendar.kickoff_at.notna()]
            anchors = [(r, _anchor(g.kickoff_at)) for r, g in valid.groupby('_ordinal', sort=True)]
            for r, anchor in anchors:
                selected = valid.index[valid.kickoff_at.ge(anchor)]
                old = games.loc[selected, '_effective'].fillna(0)
                games.loc[selected, '_effective'] = np.maximum(old, r)
            idx = valid.index
            games.loc[idx, '_effective'] = np.maximum(games.loc[idx, '_effective'], games.loc[idx, '_ordinal'])
        games['_value'] = (total - games['_effective']) / (total - 1)
        records.append(games)
    lookup = pd.concat(records, ignore_index=True) if records else source.assign(_value=np.nan, _effective=np.nan, _total=np.nan)
    # Reject conflicting query metadata instead of silently using another schedule.
    comparison = ['round', 'kickoff_at'] + [c for c in ('source_league', 'source_season', 'tournament_id', 'season_phase', 'home_id', 'away_id') if c in query and c in lookup]
    joined = query.merge(lookup[_KEYS + comparison + ['_value', '_effective', '_total']], on=_KEYS,
                         how='left', suffixes=('', '_source'), validate='many_to_one', indicator=True, sort=False)
    if joined['_merge'].ne('both').any():
        raise ValueError('SeasonCountdown query fixture is absent from schedule_history.')
    for field in comparison:
        a, b = joined[field], joined[field + '_source']
        if (~(a.eq(b) | (a.isna() & b.isna()))).any():
            raise ValueError(f'SeasonCountdown query {field} conflicts with schedule_history.')
    result = pd.Series(joined['_value'].to_numpy(), index=metadata.index, dtype=float)
    return result


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


def season_countdown(metadata, expression, *, schedule_history=None, team_seasons=None, cutoffs=None):
    """Use an issuance-known complete snapshot, or explicit retrospective input.

    as_of requires schedule_history.schedule_snapshot_at on every row. Each
    timestamp identifies a complete competition/season snapshot, including its
    planned future fixtures. Select the latest snapshot at or before each query
    cutoff; later revisions never enter those anchors. Callers must supply
    truthful publication times and complete snapshots, not per-row capture times.
    """
    import numpy as np
    import pandas as pd
    if expression.schedule_mode == 'retrospective':
        return _retrospective_countdown(metadata, expression, schedule_history=schedule_history, team_seasons=team_seasons)
    if schedule_history is None or 'schedule_snapshot_at' not in schedule_history:
        raise ValueError('SeasonCountdown as_of needs complete schedule_history snapshots with schedule_snapshot_at; choose schedule_mode="retrospective" explicitly to use final revised schedules.')
    query = _schedule(metadata)
    from .history import aligned_times
    cutoff_frame = metadata.assign(kickoff_at=query.kickoff_at.to_numpy())
    times = aligned_times(cutoff_frame, cutoffs, default='kickoff_at').reset_index(drop=True)
    if times.isna().any() or times.gt(query.kickoff_at).any():
        raise ValueError('SeasonCountdown requires known cutoffs no later than target kickoff.')
    source = schedule_history.copy(deep=False)
    source = source.assign(schedule_snapshot_at=pd.to_datetime(source.schedule_snapshot_at, utc=True, errors='raise', format='mixed'))
    if source.schedule_snapshot_at.isna().any():
        raise ValueError('SeasonCountdown schedule_snapshot_at must be known for every snapshot row.')
    output = np.full(len(query), np.nan)
    for key, rows in query.groupby(_KEYS[:2], sort=False):
        population = source.loc[source.competition_id.eq(key[0]) & source.season_id.eq(key[1])]
        # Match precision explicitly: pandas 3 may parse snapshots in microseconds
        # while an issuance cutoff carries nanoseconds. Never round a cutoff.
        snapshots = population.schedule_snapshot_at.drop_duplicates().sort_values().astype('datetime64[ns, UTC]')
        if snapshots.empty:
            raise ValueError(f'No schedule snapshot for {key}.')
        chosen = snapshots.searchsorted(times.iloc[rows.index], side='right') - 1
        if (chosen < 0).any():
            raise ValueError(f'No schedule snapshot known at the requested cutoff for {key}.')
        for position in np.unique(chosen):
            selected = rows.index[chosen == position]
            snapshot = population.loc[population.schedule_snapshot_at.eq(snapshots.iloc[position])]
            values = _retrospective_countdown(metadata.iloc[selected], expression, schedule_history=snapshot, team_seasons=team_seasons)
            output[selected] = values.to_numpy()
    return pd.Series(output, index=metadata.index, dtype=float)
