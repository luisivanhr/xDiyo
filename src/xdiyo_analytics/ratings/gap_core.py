"""Wheatcroft (2021), pp.82-83, equations (6)-(9). No fitted defaults."""
from dataclasses import dataclass
import math
from numbers import Real

ALGORITHM = 'wheatcroft_2021_gap_v1'
STATISTICS = ('goals', 'shots_on_target', 'shots_off_target', 'corners')
FIELDS = ('HA', 'HD', 'AA', 'AD')


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative.')
    return float(value)


@dataclass(frozen=True)
class GAPParameters:
    learning_rate: float
    phi1: float
    phi2: float

    def __post_init__(self):
        if nonnegative(self.learning_rate, 'lambda') == 0:
            raise ValueError('lambda must be positive.')
        for name in ('phi1', 'phi2'):
            if not 0 < nonnegative(getattr(self, name), name) < 1:
                raise ValueError(f'{name} must be strictly between zero and one.')


@dataclass(frozen=True)
class GAPState:
    HA: float = 0.0
    HD: float = 0.0
    AA: float = 0.0
    AD: float = 0.0

    def __post_init__(self):
        for name in FIELDS:
            nonnegative(getattr(self, name), name)


@dataclass(frozen=True)
class GAPPrediction:
    home: float
    away: float

    @property
    def difference(self):
        return self.home - self.away


@dataclass(frozen=True)
class WheatcroftGAP:
    """Small strategy boundary: pure initial state, predict and paired update.

    Cross-venue weights follow equations (6)-(7), not the less precise prose.
    Future algorithms can implement these methods without changing adapters.
    """
    algorithm: str = ALGORITHM

    def __post_init__(self):
        if self.algorithm != ALGORITHM:
            raise ValueError('Only wheatcroft_2021_gap_v1 is implemented.')

    def initial_state(self):
        return GAPState()

    def predict(self, home: GAPState, away: GAPState):
        # Halve first to avoid overflow in the intermediate sum.
        return GAPPrediction(home.HA / 2 + away.AD / 2, away.AA / 2 + home.HD / 2)

    def update(self, home: GAPState, away: GAPState, y_home, y_away, parameters: GAPParameters):
        forecast = self.predict(home, away)
        eh = nonnegative(y_home, 'home count') - forecast.home
        ea = nonnegative(y_away, 'away count') - forecast.away
        rate, p, q = parameters.learning_rate, parameters.phi1, parameters.phi2
        def clipped(value):
            if not math.isfinite(value):
                raise ValueError('GAP update overflowed; use finite, numerically stable parameters.')
            return max(value, 0.0)
        return (
            GAPState(clipped(home.HA + rate*p*eh), clipped(home.HD + rate*p*ea),
                     clipped(home.AA + rate*(1-p)*eh), clipped(home.AD + rate*(1-p)*ea)),
            GAPState(clipped(away.HA + rate*(1-q)*ea), clipped(away.HD + rate*(1-q)*eh),
                     clipped(away.AA + rate*q*ea), clipped(away.AD + rate*q*eh)),
        )


def donor_mean(states):
    """Componentwise donor average (p.85); no empty-cohort fallback."""
    states = tuple(states)
    if not states:
        raise ValueError('GAP initialization unavailable: empty donor group.')
    return GAPState(**{field: math.fsum(getattr(s, field) / len(states) for s in states) for field in FIELDS})
