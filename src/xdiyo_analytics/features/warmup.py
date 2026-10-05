"""Explicit warm-start policies and consistent mean/variance calculations."""

from dataclasses import dataclass, replace
from numbers import Integral
import math

from .expressions import Expr, Stat, ForAgainst, RollingMean, RollingStd, RollingSkewness, RollingZScore
from .league import League, LeaveOneOut, league_values


def _count(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


@dataclass(frozen=True)
class Hard:
    """Use the seeded EMA until this many league rounds have completed."""
    rounds: int = 1

    def __post_init__(self):
        _count(self.rounds, "rounds")

    def weight(self, rounds, observations):
        return float(rounds < self.rounds)


@dataclass(frozen=True)
class LinearFade:
    """Fade the EMA weight from one at start to zero at start+rounds."""
    start: int = 1
    rounds: int = 2

    def __post_init__(self):
        _count(self.start, "start")
        _count(self.rounds, "rounds", 1)

    def weight(self, rounds, observations):
        return min(1., max(0., 1. - (rounds - self.start) / self.rounds))


@dataclass(frozen=True)
class ObservationCount:
    """Weight the EMA by strength/(strength+valid new observations)."""
    strength: float = 5.

    def __post_init__(self):
        if not math.isfinite(self.strength) or self.strength <= 0:
            raise ValueError("strength must be finite and positive.")

    def weight(self, rounds, observations):
        return self.strength / (self.strength + observations)


@dataclass(frozen=True)
class SeededEMA:
    """Prior-seeded moments, with an explicit handoff to ordinary rolling.

    Omitted mode preserves legacy behavior below. uniform uses the final own
    boundary rolling window; w_league_prior replaces mover seeds with a cohort.
    New modes interpolate variance estimates during fade. See explicit_warmup.md.

    In legacy mode, alpha controls each observation. league_weight blends a mover's
    previous-team mean toward the destination league mean. Variance always uses
    matching previous-season league observations. No rating rank cohorts apply.
    When no mean prior exists, use the ordinary rolling result. A missing spread
    remains missing during warm-up. Weighted variances are population moments;
    after the handoff the original operator's ddof applies unchanged.
    """
    alpha: float = .5
    handoff: object = Hard()
    league_weight: float = .5
    round_keys: tuple = ("competition_id", "season_id", "round")
    mode: str = "legacy"
    bottom: int = 3
    top: int = 3
    variance_prior: str = "within_team"
    variance_estimator: str = "population"
    prior_strength: float | None = None
    variance_fade: str = "estimate_interpolation"

    def __post_init__(self):
        if not math.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError("alpha must lie in (0, 1].")
        if not math.isfinite(self.league_weight) or not 0 <= self.league_weight <= 1:
            raise ValueError("league_weight must lie in [0, 1].")
        if not isinstance(self.handoff, (Hard, LinearFade, ObservationCount)):
            raise TypeError("Choose Hard, LinearFade or ObservationCount for handoff.")
        if self.mode not in ('legacy', 'uniform', 'w_league_prior'):
            raise ValueError('mode must be legacy, uniform or w_league_prior.')
        for name in ('bottom', 'top'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or (value != -1 and value < 1):
                raise ValueError(f'{name} must be a positive integer or -1 (all eligible donors).')
        if self.variance_prior != 'within_team' or self.variance_fade != 'estimate_interpolation':
            raise ValueError('New modes use within_team variance and estimate_interpolation fade.')
        if self.variance_estimator not in ('population', 'weighted_sample'):
            raise ValueError('variance_estimator must be population or weighted_sample.')
        if self.prior_strength is not None and (isinstance(self.prior_strength, bool)
                or not math.isfinite(self.prior_strength) or self.prior_strength <= 1):
            raise ValueError('prior_strength must be finite and greater than 1, or None.')


@dataclass(frozen=True)
class WarmStart(Expr):
    """Opt in for this expression only; policy=None is an explicit no-op."""
    source: Expr
    policy: object = None


def blend_moments(mean_a, variance_a, mean_b, variance_b, weight):
    """Mixture of population moments, including the between-means term."""
    if not 0 <= weight <= 1:
        raise ValueError("Moment weight must lie in [0, 1].")
    if weight == 1:
        return mean_a, variance_a
    if weight == 0:
        return mean_b, variance_b
    mean = weight * mean_a + (1 - weight) * mean_b
    variance = (weight * variance_a + (1 - weight) * variance_b
                + weight * (1 - weight) * (mean_a - mean_b) ** 2)
    return mean, variance


def seeded_moments(mean, variance, observations, alpha):
    """Update both moments with the same weights, skipping missing values."""
    for value in observations:
        if not math.isfinite(value):
            continue
        difference = value - mean
        mean += alpha * difference
        # A missing prior spread stays unknown, including at alpha=1 only
        # until a complete replacement observation supplies its own moment.
        variance = 0. if alpha == 1 else (1 - alpha) * (variance + alpha * difference ** 2)
    return mean, variance


def evaluate_warm_start(node, context, evaluate, candidates, *, h2h=False, group_by=()):
    """Evaluate seeded moments without mutating observed history or expressions."""
    import numpy as np
    import pandas as pd
    operator, policy = node.source, node.policy
    from .moments import sample_moments, mix_moments, skewness
    is_skew = isinstance(operator, RollingSkewness)
    if policy.mode != 'legacy':
        from .seeded import evaluate_seeded
        return evaluate_seeded(node, context, evaluate, candidates, h2h=h2h, group_by=group_by)
    if not isinstance(operator, (RollingMean, RollingStd, RollingSkewness, RollingZScore)):
        raise TypeError("SeededEMA wraps RollingMean, RollingStd, RollingSkewness or RollingZScore.")
    ordinary, scope = evaluate(operator, h2h)
    if isinstance(policy.handoff, Hard) and policy.handoff.rounds == 0:
        return ordinary, scope
    source = operator.source
    selection = source if isinstance(source, LeaveOneOut) else None
    league = selection.source if selection is not None else source if isinstance(source, League) else None
    raw = league.source if league is not None else source
    if not isinstance(raw, (Stat, ForAgainst)):
        raise TypeError("SeededEMA needs a Stat/ForAgainst observation source or its League population.")
    frame, _ = evaluate(raw, h2h)
    if league is not None:
        frame = league_values(context.history, raw, frame, league.unit)
    values = frame.to_numpy(dtype=float, na_value=np.nan)
    output = ordinary.to_numpy(dtype=float, na_value=np.nan, copy=True)
    p, history = context.population, context.history
    if league is not None:
        exclude = selection.exclude if selection is not None else None
        windows = p.rows(league, operator.window, exclude)
        all_rows = p.rows(replace(league, window_unit="matches"), 2 ** 31 - 1, exclude)
    else:
        all_rows = candidates(scope, operator.venue)
        windows = [rows[-operator.window:] for rows in all_rows]
    reference = None
    if isinstance(operator, RollingZScore) and operator.reference is not None:
        reference, _ = evaluate(operator.reference, h2h)
        reference = reference.to_numpy(dtype=float, na_value=np.nan)
    prior_cache = {}
    for row in range(len(history)):
        if pd.isna(p.cutoff.iloc[row]) or pd.isna(p.kickoff.iloc[row]):
            continue
        record = context.record(row)
        season, boundary = context.seasons[row], record["entry_at"]
        time = p.cutoff.iloc[row]
        round_keys = league.round_keys if league is not None else policy.round_keys
        if league is not None and league.schedule == "completed_rounds":
            info, row_round = p.round_info(round_keys)
            time = info[row_round[row]][1]
        rounds = 0 if isinstance(policy.handoff, ObservationCount) else context.completed_rounds(row, round_keys, time=time)
        # Hard handoff avoids even looking up a prior once warm-up has ended.
        if isinstance(policy.handoff, (Hard, LinearFade)) and policy.handoff.weight(rounds, 0) == 0:
            continue
        extras = tuple(x for x in group_by if x not in {"team_id", "competition_id", "season_id"})
        if operator.venue == 'same' and 'side' not in extras:
            extras += ('side',)
        if h2h and "opponent_id" not in extras:
            extras += ("opponent_id",)
        extra_values = tuple(history[x].iloc[row] for x in extras) if league is None else ()
        signature = season, record["team_id"], extra_values
        if signature not in prior_cache:
            prior_league = context.predecessors[season]
            league_rows = context.prior_rows(prior_league, boundary)
            if league is not None:
                if league.unit == "match":
                    league_rows = league_rows[history.iloc[league_rows]["side"].eq("home").to_numpy(dtype=bool)]
                if selection is not None:
                    keep = p.teams[league_rows] != record["team_id"]
                    if selection.exclude == "fixtures":
                        keep &= p.opponents[league_rows] != record["team_id"]
                    league_rows = league_rows[keep]
                team_rows = league_rows
            else:
                old = (record["previous_competition_id"], record["previous_season_id"])
                team_rows = context.prior_rows(old, boundary)
                team_rows = team_rows[p.teams[team_rows] == record["team_id"]]
                for name, value in zip(extras, extra_values):
                    # Object comparison preserves unsigned IDs beyond int64;
                    # Arrow's scalar inference can otherwise overflow here.
                    team_rows = team_rows[history[name].iloc[team_rows].astype(object).eq(value).fillna(False).to_numpy(dtype=bool)]
            priors = []
            for col in range(values.shape[1]):
                team_sample, league_sample = values[team_rows, col], values[league_rows, col]
                team_sample = team_sample[np.isfinite(team_sample)]
                league_sample = league_sample[np.isfinite(league_sample)]
                team_mean = team_sample.mean() if len(team_sample) else np.nan
                league_mean = league_sample.mean() if len(league_sample) else np.nan
                mean = team_mean if np.isfinite(team_mean) else league_mean
                if league is None and record["movement"] in {"promoted", "relegated"} and np.isfinite(league_mean):
                    mean = policy.league_weight * league_mean + (1 - policy.league_weight) * mean
                variance = league_sample.var(ddof=0) if len(league_sample) else np.nan
                # Legacy priors use the destination spread about the chosen
                # team/mover mean. Translate the league distribution, retaining
                # its central third moment just as we retain its variance.
                third = sample_moments(league_sample)[2] if is_skew else np.nan
                if is_skew and len(league_sample) < max(3, operator.min_periods):
                    variance, third = np.nan, np.nan
                priors.append((mean, variance, third))
            prior_cache[signature] = priors
        positions = all_rows[row]
        new = np.array([i for i in positions if context.seasons[i] == season and p.kickoff.iloc[i] >= boundary], dtype=int)
        for col, (mean, variance, third) in enumerate(prior_cache[signature]):
            if not np.isfinite(mean):
                continue
            sample = values[new, col]
            n = int(np.isfinite(sample).sum())
            weight = policy.handoff.weight(rounds, n)
            if weight == 0:
                continue
            if is_skew:
                moments = (mean, variance, third)
                for value in sample[np.isfinite(sample)]:
                    moments = mix_moments(moments, (float(value), 0., 0.), 1-policy.alpha)
                rolling = values[windows[row], col]
                rolling = rolling[np.isfinite(rolling)]
                if weight < 1 and len(rolling) >= max(3, operator.min_periods):
                    moments = mix_moments(moments, sample_moments(rolling), weight)
                output[row, col] = skewness(moments)
                continue
            mean, variance = seeded_moments(mean, variance, sample, policy.alpha)
            rolling = values[windows[row], col]
            rolling = rolling[np.isfinite(rolling)]
            # Partial rolling windows are allowed. An absent rolling component
            # leaves only the available prior; it is never replaced by zero.
            if weight < 1 and len(rolling) >= operator.min_periods:
                mean, variance = blend_moments(mean, variance, rolling.mean(), rolling.var(ddof=0), weight)
            if isinstance(operator, RollingMean):
                output[row, col] = mean
            elif isinstance(operator, RollingStd):
                output[row, col] = np.sqrt(max(0., variance)) if np.isfinite(variance) else np.nan
            else:
                value = (reference[row, col] if reference is not None else
                         values[positions[-1], col] if len(positions) else np.nan)
                output[row, col] = ((value - mean) / np.sqrt(variance)
                                    if np.isfinite(variance) and variance > 0 and np.isfinite(value) else np.nan)
    return pd.DataFrame(output, index=ordinary.index, columns=ordinary.columns), scope
