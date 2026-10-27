"""Opt-in temporal memory and supervised future-statistic configuration."""
from dataclasses import dataclass, asdict
import math
import pandas as pd


@dataclass(frozen=True)
class GoalAuxiliary:
    """Parallel raw-count supervision; None in TemporalGraphConfig disables it."""
    beta: float = .1
    architecture: str = 'linear'
    rate_floor: float = 1e-6
    detach_encoder: bool = False
    likelihood: str = 'poisson'
    competitions: tuple[int, ...] = ()
    initial_alpha: float = .1
    dispersion_learning_rate: float = .05

    def __post_init__(self):
        if self.architecture not in ('linear', 'attack_defence'):
            raise ValueError('Goal auxiliary architecture must be linear or attack_defence.')
        if self.likelihood not in ('poisson', 'independent_nb', 'shared_gamma'):
            raise ValueError('Unknown goal likelihood.')
        if self.architecture == 'linear' and self.likelihood != 'poisson':
            raise ValueError('The legacy linear goal head uses the Poisson objective.')
        competitions = tuple(self.competitions)
        if any(type(c) is not int for c in competitions) or len(set(competitions)) != len(competitions):
            raise ValueError('Goal competitions must be unique integer identities.')
        if self.architecture == 'attack_defence' and not competitions:
            raise ValueError('Structured goal heads require supported target competitions.')
        object.__setattr__(self, 'competitions', competitions)
        for name in ('initial_alpha', 'dispersion_learning_rate'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive.')
        for name, positive in (('beta', False), ('rate_floor', True)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 or (positive and value == 0)):
                raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}.')
            object.__setattr__(self, name, float(value))
        if not 1.1754943508222875e-38 <= self.rate_floor <= 3.4028234663852886e38:
            raise ValueError('rate_floor must be a positive normal float32 value.')
        if type(self.detach_encoder) is not bool:
            raise ValueError('detach_encoder must be a boolean.')


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
class NumericStatisticTarget:
    """Exact own-team numeric observation, supervised as ordered quantiles.

    Values and predictions use declared native units. Bounds are required only
    for bounded_continuous. Transformation and scaling fit no held-out data.
    """
    semantic_name: str
    source_column: str
    units: str
    support: str
    side: str = 'home'
    period: str = 'ALL'
    component: str = 'value'
    transform: str = 'identity'
    scaling: str = 'standard'
    lower: float | None = None
    upper: float | None = None
    weight: float = 1.
    quantiles: tuple[float, ...] = (.25, .5, .75)
    family: str = 'quantile'

    def __post_init__(self):
        for key in ('semantic_name', 'source_column', 'units', 'component'):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f'{key} must be an explicit nonempty string.')
        if self.source_column != 'goals_for' and not self.source_column.startswith('team::'):
            raise ValueError('source_column must identify an own-team channel, never an opponent copy.')
        if self.side not in ('home', 'away') or self.period not in ('ALL', '1ST', '2ND'):
            raise ValueError('Invalid numeric target side or period.')
        if self.source_column == 'goals_for' and self.period != 'ALL':
            raise ValueError('Canonical goals_for requires period ALL; no half substitution.')
        parts = self.source_column.split('::')
        if len(parts) > 2 and parts[1] in ('ALL','1ST','2ND'):
            if parts[1] != self.period or parts[-1] != self.component:
                raise ValueError('Exact source period/component differs from declared target.')
        if self.support not in ('count', 'nonnegative_continuous', 'signed_continuous', 'bounded_continuous'):
            raise ValueError('Invalid target support.')
        if self.transform not in ('identity', 'log1p', 'signed_log1p', 'affine'):
            raise ValueError('Invalid target transform.')
        if self.scaling not in ('standard', 'robust', 'none'):
            raise ValueError('Invalid fitting-only scaling.')
        if self.support == 'bounded_continuous':
            if self.lower is None or self.upper is None or not all(math.isfinite(x) for x in (self.lower,self.upper)) or not self.lower < self.upper:
                raise ValueError('Bounded targets require finite increasing bounds.')
            if self.transform != 'affine':
                raise ValueError('Bounded targets require the explicit affine transform.')
        elif self.lower is not None or self.upper is not None or self.transform == 'affine':
            raise ValueError('Bounds/affine transform require bounded_continuous support.')
        if self.support == 'signed_continuous' and self.transform == 'log1p':
            raise ValueError('Signed targets cannot use plain log1p.')
        if self.family != 'quantile':
            raise ValueError('Numeric targets currently require family=quantile.')
        # Reuse the legacy quantile/weight validation without changing its schema.
        checked = StatisticTarget(weight=self.weight, family='quantile', quantiles=self.quantiles)
        object.__setattr__(self, 'quantiles', checked.quantiles)

    @property
    def column(self):
        return self.source_column

    @property
    def name(self):
        return f'{self.side}::{self.period}::{self.semantic_name}'

    @property
    def proportion(self):
        return False


@dataclass(frozen=True)
class StatisticsForecast:
    targets: tuple[StatisticTarget | NumericStatisticTarget, ...] = (StatisticTarget(),)
    route: str = 'auxiliary'
    weight: float = 1.
    point_loss: str = 'huber'

    def __post_init__(self):
        object.__setattr__(self, 'targets', tuple(self.targets))
        if not self.targets or any(not isinstance(t, (StatisticTarget, NumericStatisticTarget)) for t in self.targets):
            raise ValueError('targets must be a nonempty ordered list of StatisticTarget.')
        if len({t.name for t in self.targets}) != len(self.targets):
            raise ValueError('Statistic targets must be distinct.')
        identities = [(t.column, t.side) for t in self.targets]
        if len(set(identities)) != len(identities):
            raise ValueError('Duplicate source/side component targets (aliases are not independent targets).')
        if self.route not in ('auxiliary', 'combined', 'bottleneck'):
            raise ValueError('Statistic route must be auxiliary, combined or bottleneck.')
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError('Statistics supervision weight must be positive and finite.')
        if self.point_loss not in ('huber', 'mse'):
            raise ValueError('Point loss must be huber or mse (in training-standardized transformed units).')

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value['targets'] = tuple((NumericStatisticTarget if 'source_column' in t else StatisticTarget)(**t) for t in value['targets'])
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
