"""Ordered, configurable football feature banks used by notebooks and builders.

Availability is an explicit input: callers discover identities only in their
chosen development population. Building the preset does not inspect outcomes.
"""
from dataclasses import dataclass, replace
from .expressions import Stat, ForAgainst, Lag, RollingMean, RollingStd, RollingSkewness, RollingZScore, EMA, H2H, NormalizedStanding
from .league import League, LeaveOneOut
from .ratings import MatchResultGlicko, StatGlicko
from .warmup import WarmStart
from .composition import Column, Sum, Difference
from .contextual import CalendarFeature

STAT_CHOICES = (
    ('Match overview', 'cornerKicks'), ('Match overview', 'ballPossession'),
    ('Match overview', 'totalShotsOnGoal'), ('Match overview', 'expectedGoals'),
    ('Match overview', 'bigChanceCreated'), ('Match overview', 'goalkeeperSaves'),
    ('Match overview', 'totalTackle'), ('Shots', 'shotsOnGoal'),
    ('Shots', 'shotsOffGoal'), ('Shots', 'blockedScoringAttempt'),
    ('Shots', 'totalShotsInsideBox'), ('Shots', 'totalShotsOutsideBox'),
    ('Attack', 'touchesInOppBox'), ('Attack', 'bigChanceMissed'),
    ('Attack', 'offsides'), ('Passes', 'accurateCross'),
    ('Passes', 'finalThirdEntries'), ('Passes', 'finalThirdPhaseStatistic'),
    ('Defending', 'ballRecovery'), ('Defending', 'interceptionWon'),
    ('Defending', 'totalClearance'), ('Defending', 'errorsLeadToShot'),
)
HALF_KEYS = ('cornerKicks', 'ballPossession', 'totalShotsOnGoal',
             'shotsOnGoal', 'shotsOffGoal', 'touchesInOppBox')
@dataclass(frozen=True)
class FeatureBankPreset:
    """Broad football bank with stable names, formulas and insertion order.

    build returns historical expressions only. The caller adds RestDays before
    match assembly when include_rest is enabled, then postassembly_definitions
    and calendar_definitions. No fitted identity vocabulary is inferred here.
    Warm policies add columns while preserving every unwarmed baseline.
    """
    stats: tuple = STAT_CHOICES
    half_keys: tuple = HALF_KEYS
    windows: tuple = (3, 5, 10, 20)
    lags: tuple = (1, 2, 3)
    spans: tuple = (5, 10)
    periods: tuple = ('ALL', '1ST', '2ND')
    std_windows: tuple = (5, 10)
    z_windows: tuple = (5,)
    half_lags: tuple = (1,)
    half_windows: tuple = (5, 10)
    league_windows: tuple = (3, 5)
    include_ratings: bool = True
    include_loo: bool = True
    loo_windows: tuple = (3, 5)
    loo_reducers: tuple = ('mean',)
    include_h2h: bool = True
    h2h_windows: tuple = (3, 5)
    h2h_reducers: tuple = ('mean',)
    warm_policy: object = None
    warm_stat_keys: tuple | None = None
    rating_warm_policy: object = None
    include_rest: bool = True
    include_calendar: bool = True
    include_combinations: bool = True
    skew_windows: tuple = ()

    def build(self, stat_identities):
        """Return ordered expressions from (period, group, key) identities."""
        windows = self.windows
        lags = self.lags
        spans = self.spans
        periods = self.periods
        include_ratings = self.include_ratings
        include_loo = self.include_loo
        loo_windows = self.loo_windows
        loo_reducers = self.loo_reducers
        include_h2h = self.include_h2h
        h2h_windows = self.h2h_windows
        h2h_reducers = self.h2h_reducers
        warm_policy = self.warm_policy
        warm_stat_keys = self.warm_stat_keys
        rating_warm_policy = self.rating_warm_policy
        available = set(map(tuple, stat_identities))
        definitions = {}
        for group, key in self.stats:
            for period in periods:
                if (period, group, key) not in available or (period != 'ALL' and key not in self.half_keys):
                    continue
                for side in ('for', 'against'):
                    prefix = f'{period}_{group}_{key}_{side}'
                    source = ForAgainst(Stat(period, group, key), side=side)
                    for lag in lags if period == 'ALL' else self.half_lags:
                        definitions[f'{prefix}_lag{lag}'] = Lag(source, periods=lag)
                    for window in windows if period == 'ALL' else self.half_windows:
                        definitions[f'{prefix}_mean{window}'] = RollingMean(source, window=window)
                    if period == 'ALL':
                        for window in self.std_windows:
                            definitions[f'{prefix}_std{window}'] = RollingStd(source, window=window, ddof=1)
                        for window in self.z_windows:
                            definitions[f'{prefix}_z{window}'] = RollingZScore(source, window=window, ddof=1)
                        for window in self.skew_windows:
                            definitions[f'{prefix}_skew{window}'] = RollingSkewness(source, window=window)
                        for span in spans:
                            definitions[f'{prefix}_ema{span}'] = EMA(source, span=span)
        corners = Stat('ALL', 'Match overview', 'cornerKicks')
        if include_h2h:
            for side in ('for', 'against'):
                source = ForAgainst(corners, side=side)
                for window in h2h_windows:
                    for reducer in h2h_reducers:
                        definitions[f'h2h_corners_{side}_{reducer}{window}'] = _reducer(
                            reducer, H2H(source), window)
        population = League(corners, unit='team', schedule='completed_rounds', window_unit='rounds')
        for window in self.league_windows:
            definitions[f'league_corners_mean{window}'] = RollingMean(population, window=window)
            definitions[f'league_corners_std{window}'] = RollingStd(population, window=window, ddof=1)
        if include_loo:
            for window in loo_windows:
                for reducer in loo_reducers:
                    definitions[f'loo_corners_{reducer}{window}'] = _reducer(
                        reducer, LeaveOneOut(population), window,
                        reference=Lag(ForAgainst(corners, side='for')))
        definitions['standing'] = NormalizedStanding(side='for', missing_value=0.)
        if include_ratings:
            definitions['result_glicko'] = MatchResultGlicko(side='for', fields=('rating', 'rd'))
            definitions['corners_glicko'] = StatGlicko(corners, side='for', fields=('rating', 'rd'))
        # Additional columns retain all unwarmed definitions for direct comparison.
        if warm_policy is not None:
            for name, operator in list(definitions.items()):
                if not isinstance(operator, (RollingMean, RollingStd, RollingSkewness, RollingZScore)):
                    continue
                source = operator.source
                h2h = isinstance(source, H2H)
                if h2h:
                    source = source.source
                raw = source
                while isinstance(raw, (ForAgainst, League, LeaveOneOut)):
                    raw = raw.source
                if warm_stat_keys is not None and raw.key not in warm_stat_keys:
                    continue
                wrapped = WarmStart(replace(operator, source=source), warm_policy)
                definitions[f'warm::{name}'] = H2H(wrapped) if h2h else wrapped
        if include_ratings and rating_warm_policy is not None:
            for name in ('result_glicko', 'corners_glicko'):
                definitions[f'warm::{name}'] = WarmStart(definitions[name], rating_warm_policy)
        return definitions

    def postassembly_definitions(self, history_columns, assembled_columns, *, feature_scopes=None):
        """Historical team-side sums, differences, and mean3-minus-mean20 trends.

        feature_scopes maps evaluated history column names to 'team' or 'fixture',
        as emitted in evaluate_features(...).attrs['feature_scopes']. Fixture
        outputs never participate, regardless of their aliases. Missing entries
        retain the legacy team-level behavior.
        """
        if not self.include_combinations:
            return {}
        scopes = feature_scopes if feature_scopes is not None else {}
        if any(scope not in ('team', 'fixture') for scope in scopes.values()):
            raise ValueError("Feature scopes must be 'team' or 'fixture'.")
        available = set(assembled_columns)
        combinations = {}
        for name in history_columns:
            if scopes.get(name, 'team') == 'fixture':
                continue
            if any(token in name for token in ('_mean5', '_mean10', 'standing', 'glicko', 'rest_days')):
                home_name, away_name = f'home::{name}', f'away::{name}'
                if home_name not in available or away_name not in available:
                    raise ValueError('Preset combinations require match layout with home/away feature columns.')
                home, away = Column(home_name), Column(away_name)
                combinations[f'sum::{name}'] = Sum(home, away)
                combinations[f'difference::{name}'] = Difference(home, away)
        for side in ('home', 'away'):
            for group, key in self.stats:
                for role in ('for', 'against'):
                    history_stem = f'ALL_{group}_{key}_{role}_mean'
                    if any(scopes.get(history_stem + window, 'team') == 'fixture' for window in ('3', '20')):
                        continue
                    stem = f'{side}::{history_stem}'
                    if stem + '3' in available and stem + '20' in available:
                        combinations[f'trend::{stem}3_vs_20'] = Difference(Column(stem + '3'), Column(stem + '20'))
        return combinations

    def calendar_definitions(self):
        """Known fixture calendar columns, intended once after match assembly."""
        return ({f'calendar::{kind}': CalendarFeature(kind)
                 for kind in ('month_sin', 'month_cos', 'weekday', 'round')}
                if self.include_calendar else {})

    def inventory(self, columns):
        """Summarize actual assembled column names by family, operator and scope."""
        from .inventory import summarize_feature_columns
        return summarize_feature_columns(columns)


def _reducer(name, source, window, reference=None):
    if name == 'mean':
        return RollingMean(source, window=window)
    if name == 'std':
        return RollingStd(source, window=window, ddof=1)
    if name == 'skew':
        return RollingSkewness(source, window=window)
    if name == 'z':
        return RollingZScore(source, window=window, ddof=1, reference=reference)
    raise ValueError('Population reducers must be mean, std, skew or z.')

