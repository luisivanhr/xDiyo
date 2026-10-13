"""Opt-in temporal memory and supervised future-statistic configuration."""
from dataclasses import dataclass, asdict
import math
import pandas as pd


@dataclass(frozen=True)
class BoundaryMemory:
    """Fixed global chronological segments; None in TemporalGraphConfig disables."""
    policy: str = 'season'
    days: int = 90
    anchor: str = '1970-01-01T00:00:00Z'
    retention: float = 1.

    def __post_init__(self):
        if self.policy not in ('season', 'fixed_days'):
            raise ValueError('Boundary policy must be season or fixed_days.')
        if type(self.days) is not int or self.days < 1:
            raise ValueError('Segment days must be a positive integer.')
        t = pd.Timestamp(self.anchor)
        if pd.isna(t) or t.tzinfo is None:
            raise ValueError('Segment anchor must be timezone-aware.')
        if not math.isfinite(self.retention) or not 0 <= self.retention <= 1:
            raise ValueError('Carry retention must lie in [0, 1].')


# Exact native statistics, never fuzzy aliases or period substitution.
STATISTICS = {
    'goals': None,
    'cornerKicks': 'Match overview',
    'totalShotsOnGoal': 'Match overview',
    'shotsOnGoal': 'Shots',
    'ballPossession': 'Match overview',
    'accuratePasses': 'Passes',
}


@dataclass(frozen=True)
class StatisticTarget:
    statistic: str = 'cornerKicks'
    side: str = 'home'
    period: str = 'ALL'
    family: str = 'point'
    weight: float = 1.
    quantiles: tuple[float, ...] = (.25, .5, .75)

    def __post_init__(self):
        try:
            levels = tuple(float(q) for q in self.quantiles)
        except (TypeError, ValueError) as e:
            raise ValueError('Quantiles must be a nonempty list of finite unique levels in (0,1).') from e
        if not levels or any(not math.isfinite(q) or not 0 < q < 1 for q in levels) or len(set(levels)) != len(levels):
            raise ValueError('Quantiles must be nonempty, finite, unique and strictly between zero and one.')
        object.__setattr__(self, 'quantiles', tuple(sorted(levels)))
        if self.statistic not in STATISTICS or self.side not in ('home', 'away') or self.period not in ('ALL', '1ST', '2ND'):
            raise ValueError('Unsupported statistic, side or period for statistic target.')
        if self.statistic == 'goals' and self.period != 'ALL':
            raise ValueError('Canonical goals targets currently support ALL only; no half substitution.')
        if self.family not in ('point', 'poisson', 'negative_binomial', 'inflated_beta', 'quantile'):
            raise ValueError('Unsupported target family.')
        if self.proportion and self.family not in ('point', 'inflated_beta', 'quantile'):
            raise ValueError('Possession requires point, inflated_beta or quantile.')
        if not self.proportion and self.family == 'inflated_beta':
            raise ValueError('inflated_beta requires a proportion target.')
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError('Target weights must be positive and finite.')

    @property
    def proportion(self):
        return self.statistic == 'ballPossession'

    @property
    def column(self):
        return 'goals_for' if self.statistic == 'goals' else f'team::{self.period}::{STATISTICS[self.statistic]}::{self.statistic}::value'

    @property
    def name(self):
        return f'{self.side}::{self.period}::{self.statistic}'


@dataclass(frozen=True)
class StatisticsForecast:
    targets: tuple[StatisticTarget, ...] = (StatisticTarget(),)
    route: str = 'auxiliary'
    weight: float = 1.
    point_loss: str = 'huber'

    def __post_init__(self):
        object.__setattr__(self, 'targets', tuple(self.targets))
        if not self.targets or any(not isinstance(t, StatisticTarget) for t in self.targets):
            raise ValueError('targets must be a nonempty ordered list of StatisticTarget.')
        if len({t.name for t in self.targets}) != len(self.targets):
            raise ValueError('Statistic targets must be distinct.')
        if self.route not in ('auxiliary', 'combined', 'bottleneck'):
            raise ValueError('Statistic route must be auxiliary, combined or bottleneck.')
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError('Statistics supervision weight must be positive and finite.')
        if self.point_loss not in ('huber', 'mse'):
            raise ValueError('Point loss must be huber or mse (in training-standardized transformed units).')

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value['targets'] = tuple(StatisticTarget(**t) for t in value['targets'])
        return cls(**value)


def target_schema(target):
    """Stable identity: unused quantile defaults do not alter existing models."""
    value = asdict(target)
    if target.family != 'quantile':
        del value['quantiles']
    else:
        value['quantiles'] = list(target.quantiles)
    return value


def quantile_column(target, level):
    return target.name + '::quantile::' + repr(float(level))
