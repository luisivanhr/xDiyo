"""Gamma variational filtering for the Ridall et al. goal model.

All Gamma distributions use *shape/rate*, not shape/scale.  Defence is a
vulnerability: a larger value increases the opponent's expected goals.  This
module has no chronology, data-frame, serialization or orchestration policy;
the rating adapter supplies priors and a simultaneous set of eligible matches.
SciPy is imported only by summaries and score prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from numbers import Integral
from typing import Hashable, Iterable, Mapping


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        raise ValueError(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} number")
    return result


def _discount(value: float, name: str) -> float:
    result = _positive(value, name)
    if result > 1:
        raise ValueError(f"{name} must lie in (0, 1]")
    return result


@dataclass(frozen=True)
class GammaState:
    """A Gamma distribution in shape/rate parameterization."""

    shape: float
    rate: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _positive(self.shape, "shape"))
        object.__setattr__(self, "rate", _positive(self.rate, "rate"))
        if not math.isfinite(self.mean) or self.mean <= 0:
            raise ValueError("Gamma mean must be representable as a finite positive number")

    @property
    def mean(self) -> float:
        return self.shape / self.rate

    @property
    def variance(self) -> float:
        return self.mean / self.rate

    @property
    def var(self) -> float:
        return self.variance

    @property
    def sd(self) -> float:
        return math.sqrt(self.shape) / self.rate

    def discount(self, factor: float) -> GammaState:
        """Preserve the mean, increasing the variance by ``1 / factor``."""
        factor = _discount(factor, "discount factor")
        return GammaState(self.shape * factor, self.rate * factor)

    def scale(self, multiplier: float) -> GammaState:
        """Distribution of ``multiplier * Z`` for this Gamma variable Z."""
        multiplier = _positive(multiplier, "scale multiplier")
        return GammaState(self.shape, self.rate / multiplier)

    def summary(self) -> dict[str, float]:
        return {"shape": self.shape, "rate": self.rate, "mean": self.mean,
                "variance": self.variance, "sd": self.sd}


@dataclass(frozen=True)
class TeamState:
    attack: GammaState
    defence: GammaState

    def __post_init__(self) -> None:
        if not isinstance(self.attack, GammaState) or not isinstance(self.defence, GammaState):
            raise TypeError("attack and defence must be GammaState instances")

    def discount(self, factor: float) -> TeamState:
        return TeamState(self.attack.discount(factor), self.defence.discount(factor))


@dataclass(frozen=True)
class BayesianParameters:
    """Fixed filter parameters; default values are starting values, not a fit."""

    team_discount: float = .987
    season_discount: float = .737
    home_discount: float = .999
    home_season_discount: float = .911
    kappa: float = 6.323
    attack_mean: float = 1.0
    defence_mean: float = 1.0
    home_mean: float = 1.0
    attack_shape: float = 10.0
    defence_shape: float = 10.0
    home_shape: float = 10.0
    entry_attack_shift: float = .2
    entry_defence_shift: float = .13
    entry_attack_shape: float = 20.0
    entry_defence_shape: float = 30.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name.endswith("discount"):
                value = _discount(value, field.name)
            else:
                value = _positive(value, field.name, allow_zero=field.name.endswith("shift"))
            object.__setattr__(self, field.name, value)


def initial_team(parameters: BayesianParameters, movement: str = "unknown") -> TeamState:
    """Destination-league prior with mirrored log shifts for moving teams."""
    if movement not in {"unknown", "retained", "promoted", "relegated"}:
        raise ValueError(f"Unsupported movement {movement!r}")
    attack_mean, defence_mean = parameters.attack_mean, parameters.defence_mean
    attack_shape, defence_shape = parameters.attack_shape, parameters.defence_shape
    if movement in {"promoted", "relegated"}:
        direction = -1 if movement == "promoted" else 1
        try:
            attack_mean *= math.exp(direction * parameters.entry_attack_shift)
            defence_mean *= math.exp(-direction * parameters.entry_defence_shift)
        except OverflowError as exc:
            raise ValueError("Entry shifts produce an unrepresentable Gamma mean") from exc
        attack_shape, defence_shape = parameters.entry_attack_shape, parameters.entry_defence_shape
    attack_mean = _positive(attack_mean, "entry attack mean")
    defence_mean = _positive(defence_mean, "entry defence mean")
    return TeamState(GammaState(attack_shape, attack_shape / attack_mean),
                     GammaState(defence_shape, defence_shape / defence_mean))


def initial_home(parameters: BayesianParameters) -> GammaState:
    return GammaState(parameters.home_shape, parameters.home_shape / parameters.home_mean)


def _count(value: float, name: str) -> int:
    number = _positive(value, name, allow_zero=True)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer goal count")
    return int(number)


def _key(team_id: Hashable) -> tuple[str, str]:
    # IDs are normally integers. This also supports mixed integral/string keys
    # without letting caller insertion order determine floating point sums.
    return type(team_id).__qualname__, repr(team_id)


def update_matches(
    states: Mapping[Hashable, TeamState],
    home: GammaState,
    matches: Iterable[tuple[Hashable, Hashable, int, int]],
    parameters: BayesianParameters,
    *,
    bivariate: bool = True,
    method: str = "vb",
    tolerance: float = 1e-9,
    max_iterations: int = 200,
) -> tuple[dict[Hashable, TeamState], GammaState, dict[str, object]]:
    """Update one joint batch with a shared home-advantage Gamma factor.

    ``vb`` performs coordinate ascent in epsilon, attack, defence, home blocks.
    Every iteration conditions on the SAME input prior; it never counts a
    result again. ``one_step`` implements the epsilon-first, frozen-prior-mean
    approximation in equation (2.7). With ``bivariate=False``, epsilon is one.

    Time/season discounts must be applied by the caller before this function.
    Nonconvergence is returned in diagnostics rather than silently accepted.
    Inputs are never mutated, including unobserved teams.
    """
    if method not in {"vb", "one_step"}:
        raise ValueError("method must be 'vb' or 'one_step'")
    if not isinstance(bivariate, bool):
        raise ValueError("bivariate must be a bool")
    tolerance = _positive(tolerance, "tolerance")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not isinstance(home, GammaState):
        raise TypeError("home must be a GammaState")
    if any(not isinstance(state, TeamState) for state in states.values()):
        raise TypeError("states values must be TeamState instances")

    batch = []
    for home_id, away_id, home_goals, away_goals in matches:
        if home_id == away_id:
            raise ValueError("A team cannot play itself")
        if home_id not in states or away_id not in states:
            raise ValueError("Every match team must have an input state")
        batch.append((home_id, away_id, _count(home_goals, "home_goals"),
                      _count(away_goals, "away_goals")))
    batch.sort(key=lambda row: (_key(row[0]), _key(row[1]), row[2], row[3]))
    diagnostics: dict[str, object] = {
        "converged": True, "iterations": 0, "max_relative_change": 0.0,
        "method": method, "bivariate": bivariate, "match_count": len(batch),
    }
    result = dict(states)
    if not batch:
        return result, home, diagnostics

    active = sorted({team for match in batch for team in match[:2]}, key=_key)
    scored: dict[Hashable, list[int]] = {team: [] for team in active}
    conceded: dict[Hashable, list[int]] = {team: [] for team in active}
    for i, j, x, y in batch:
        scored[i].append(x)
        scored[j].append(y)
        conceded[i].append(y)
        conceded[j].append(x)
    attack_shape = {team: states[team].attack.shape + sum(scored[team]) for team in active}
    defence_shape = {team: states[team].defence.shape + sum(conceded[team]) for team in active}
    home_shape = home.shape + sum(x for _, _, x, _ in batch)
    attack = {team: states[team].attack for team in active}
    defence = {team: states[team].defence for team in active}
    home_factor = home
    epsilon = [1.0] * len(batch)

    def epsilon_means() -> list[float]:
        if not bivariate:
            return [1.0] * len(batch)
        return [(parameters.kappa + x + y) / (parameters.kappa
                + attack[i].mean * defence[j].mean * home_factor.mean
                + attack[j].mean * defence[i].mean) for i, j, x, y in batch]

    def attack_block() -> dict[Hashable, GammaState]:
        increments: dict[Hashable, list[float]] = {team: [] for team in active}
        for (i, j, _, _), effect in zip(batch, epsilon):
            increments[i].append(defence[j].mean * home_factor.mean * effect)
            increments[j].append(defence[i].mean * effect)
        return {team: GammaState(attack_shape[team], states[team].attack.rate
                                 + math.fsum(increments[team])) for team in active}

    def defence_block() -> dict[Hashable, GammaState]:
        increments: dict[Hashable, list[float]] = {team: [] for team in active}
        for (i, j, _, _), effect in zip(batch, epsilon):
            increments[i].append(attack[j].mean * effect)
            increments[j].append(attack[i].mean * home_factor.mean * effect)
        return {team: GammaState(defence_shape[team], states[team].defence.rate
                                 + math.fsum(increments[team])) for team in active}

    def home_block() -> GammaState:
        rate = home.rate + math.fsum(attack[i].mean * defence[j].mean * effect
                                    for (i, j, _, _), effect in zip(batch, epsilon))
        return GammaState(home_shape, rate)

    if method == "one_step":
        epsilon = epsilon_means()
        # All three blocks read the frozen prior factor means. Assignment only
        # happens after evaluating every block (unlike coordinate ascent).
        attack, defence, home_factor = attack_block(), defence_block(), home_block()
        diagnostics["iterations"] = 1
    else:
        diagnostics["converged"] = False
        for iteration in range(1, max_iterations + 1):
            previous = ([attack[team].mean for team in active]
                        + [defence[team].mean for team in active]
                        + [home_factor.mean] + epsilon)
            epsilon = epsilon_means()
            attack = attack_block()
            defence = defence_block()
            home_factor = home_block()
            current = ([attack[team].mean for team in active]
                       + [defence[team].mean for team in active]
                       + [home_factor.mean] + epsilon)
            change = max(abs(new - old) / max(abs(new), abs(old), 1e-300)
                         for new, old in zip(current, previous))
            diagnostics.update(iterations=iteration, max_relative_change=change)
            if change <= tolerance:
                diagnostics["converged"] = True
                break
    result.update({team: TeamState(attack[team], defence[team]) for team in active})
    diagnostics["random_effect_means"] = epsilon
    return result, home_factor, diagnostics


def team_summary(state: TeamState) -> dict[str, float]:
    """Numeric outputs; log features use expected logs, not logs of means."""
    from scipy.special import digamma

    log_attack = float(digamma(state.attack.shape)) - math.log(state.attack.rate)
    log_defence_strength = math.log(state.defence.rate) - float(digamma(state.defence.shape))
    return {
        "attack_mean": state.attack.mean,
        "defence_vulnerability_mean": state.defence.mean,
        "attack_sd": state.attack.sd,
        "defence_vulnerability_sd": state.defence.sd,
        "log_attack": log_attack,
        "log_defence_strength": log_defence_strength,
        "strength_index": log_attack + log_defence_strength,
        "attack_shape": state.attack.shape, "attack_rate": state.attack.rate,
        "defence_shape": state.defence.shape, "defence_rate": state.defence.rate,
    }


def predict_score_summary(
    lambda_home: float,
    lambda_away: float,
    kappa: float | None = None,
    *,
    tail_tolerance: float = 1e-10,
    max_total: int = 10000,
) -> dict[str, float]:
    """Conditional score probabilities with an explicit controlled tail.

    The total is negative binomial (Poisson when ``kappa=None``); conditional
    on the total, home goals are binomial. Summation of total-goal strata is
    exact apart from the reported tail. Outcome probabilities are NOT
    renormalized. Both-score and over-2.5 probabilities have closed forms.
    Team-state posterior uncertainty is not integrated by this function.
    """
    lambda_home = _positive(lambda_home, "lambda_home", allow_zero=True)
    lambda_away = _positive(lambda_away, "lambda_away", allow_zero=True)
    if kappa is not None:
        kappa = _positive(kappa, "kappa")
    tail_tolerance = _positive(tail_tolerance, "tail_tolerance")
    if tail_tolerance >= 1:
        raise ValueError("tail_tolerance must lie in (0, 1)")
    if isinstance(max_total, bool) or not isinstance(max_total, Integral) or max_total < 0:
        raise ValueError("max_total must be a nonnegative integer")
    total_mean = lambda_home + lambda_away
    if not math.isfinite(total_mean):
        raise ValueError("Total expected goals must be finite")
    if total_mean == 0:
        return {"expected_home_goals": 0.0, "expected_away_goals": 0.0,
                "p_home_win": 0.0, "p_draw": 1.0, "p_away_win": 0.0,
                "p_both_score": 0.0, "p_over_2_5": 0.0, "omitted_mass": 0.0}

    import numpy as np
    from scipy.stats import binom, nbinom, poisson

    if kappa is None:
        total_distribution = poisson(total_mean)
        zero_home, zero_away, zero_both = (math.exp(-lambda_home),
                                          math.exp(-lambda_away), math.exp(-total_mean))
    else:
        total_distribution = nbinom(kappa, kappa / (kappa + total_mean))
        zero_home = math.exp(-kappa * math.log1p(lambda_home / kappa))
        zero_away = math.exp(-kappa * math.log1p(lambda_away / kappa))
        zero_both = math.exp(-kappa * math.log1p(total_mean / kappa))

    # A bounded integer search using sf avoids cancellation in 1-cdf and
    # quantile failures for very small tail tolerances.
    upper = min(max_total, max(1, int(math.ceil(total_mean))))
    while float(total_distribution.sf(upper)) > tail_tolerance and upper < max_total:
        upper = min(max_total, max(upper + 1, upper * 2))
    omitted = float(total_distribution.sf(upper))
    if not math.isfinite(omitted) or omitted > tail_tolerance:
        raise ValueError("max_total cannot achieve the requested score probability tail_tolerance")
    low, high = 0, upper
    while low < high:
        middle = (low + high) // 2
        if float(total_distribution.sf(middle)) <= tail_tolerance:
            high = middle
        else:
            low = middle + 1
    upper = low
    omitted = float(total_distribution.sf(upper))
    totals = np.arange(upper + 1)
    mass = total_distribution.pmf(totals)
    fraction = lambda_home / total_mean
    draw = np.where(totals % 2 == 0, binom.pmf(totals // 2, totals, fraction), 0.0)
    home_win = binom.sf(totals // 2, totals, fraction)
    away_win = binom.cdf((totals - 1) // 2, totals, fraction)
    return {
        "expected_home_goals": lambda_home, "expected_away_goals": lambda_away,
        "p_home_win": float(np.dot(mass, home_win)),
        "p_draw": float(np.dot(mass, draw)),
        "p_away_win": float(np.dot(mass, away_win)),
        "p_both_score": min(1.0, max(0.0, 1.0 - zero_home - zero_away + zero_both)),
        "p_over_2_5": float(total_distribution.sf(2)), "omitted_mass": omitted,
    }
