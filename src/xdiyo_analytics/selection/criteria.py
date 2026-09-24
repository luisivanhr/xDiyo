"""Selection rules over comparable numerical evidence, without model fitting."""

from dataclasses import dataclass, field
from numbers import Real

import numpy as np
import pandas as pd

from ..experiments import rank_runs


@dataclass(frozen=True)
class FitStatistics:
    """Adapter-supplied fitted log likelihood, complexity and sample convention.

    log_likelihood must be the unpenalized fitted data log likelihood, including
    constants needed for comparisons. n_parameters includes the adapter's stated
    nuisance/intercept/effective-DoF convention. likelihood_id identifies response
    units, likelihood normalization and parameter-count convention. Comparable
    candidates must use the same convention; it is not a model-family identifier.
    n_observations is the likelihood's sampling-unit count, not inferred from X.
    The adapter owns applicability to its fit; no likelihood or DoF is guessed
    from MSE, a regularization penalty, or the count of nonzero coefficients.
    """
    log_likelihood: float
    n_parameters: float
    n_observations: int
    likelihood_id: str


def information_criteria(statistics):
    """Return AIC=-2*logL+2*k and BIC=-2*logL+k*log(n).

    These are per-fit criteria. Mean-fold comparison is separately labeled by
    ModelSelection and is not presented as a single full-population AIC/BIC.
    """
    if not isinstance(statistics, FitStatistics):
        raise TypeError("fit_statistics must supply FitStatistics.")
    values = (statistics.log_likelihood, statistics.n_parameters, statistics.n_observations)
    if any(isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) for value in values):
        raise ValueError("Fit statistics must be finite numeric values.")
    if statistics.n_parameters < 0 or statistics.n_observations <= 0 or int(statistics.n_observations) != statistics.n_observations:
        raise ValueError("Fit statistics need nonnegative complexity and a positive integer sample count.")
    if not isinstance(statistics.likelihood_id, str) or not statistics.likelihood_id:
        raise ValueError("Provide the likelihood/parameter-count comparison convention.")
    return {"aic": -2*statistics.log_likelihood+2*statistics.n_parameters,
            "bic": -2*statistics.log_likelihood+statistics.n_parameters*np.log(statistics.n_observations)}


@dataclass
class SelectionDecision:
    """Chosen trial ID plus reusable comparison table; no refitting side effects."""
    winner_id: str
    table: pd.DataFrame


def _comparable(table):
    eligible = table.loc[table.status.eq("ranked")]
    if eligible.empty:
        raise ValueError("No candidate has complete, finite evidence for this decision rule.")
    if eligible.comparison_group.nunique() != 1:
        raise ValueError("Candidates have incompatible evaluation samples or metric definitions; select a comparable population.")
    return eligible


def _finish(table, winner, order):
    table = table.copy()
    table["selected"] = table.run_id.eq(winner)
    table["candidate_order"] = table.run_id.map(order)
    return SelectionDecision(winner, table)


@dataclass
class MetricSelection:
    """Choose the best raw scalar metric; exact ties follow candidate order.

    selector disambiguates target/study/output if necessary; defaults select one
    overall metric. direction=None uses the metric registry's declared direction.
    """
    metric: str
    direction: str | None = None
    selector: dict = field(default_factory=dict)

    def decide(self, trials):
        records = [trial.record for trial in trials]
        table = rank_runs(records, {self.metric: 1}, selectors={self.metric: self.selector},
                          directions={} if self.direction is None else {self.metric: self.direction})
        eligible = _comparable(table)
        order = {trial.trial_id: i for i, trial in enumerate(trials)}
        # One-metric percentile ranking preserves the raw order and exact ties.
        best = eligible.loc[eligible.score.eq(eligible.score.max())]
        winner = min(best.run_id, key=order.get)
        return _finish(table, winner, order)


@dataclass
class WeightedSelection:
    """Use the same weighted metric utility calculation as the run leaderboard.

    Percentile scaling is relative to this candidate set; fixed scaling uses
    declared reference bounds. Weights are normalized, minimizing metrics are
    reversed, and missing required metrics never receive renormalized weights.
    """
    weights: dict
    selectors: dict = field(default_factory=dict)
    scaling: str = "percentile"
    reference_scales: dict = field(default_factory=dict)
    directions: dict = field(default_factory=dict)

    def decide(self, trials):
        table = rank_runs([trial.record for trial in trials], self.weights, selectors=self.selectors,
                          scaling=self.scaling, reference_scales=self.reference_scales, directions=self.directions)
        eligible = _comparable(table)
        order = {trial.trial_id: i for i, trial in enumerate(trials)}
        best = eligible.loc[eligible.score.eq(eligible.score.max())]
        return _finish(table, min(best.run_id, key=order.get), order)


@dataclass
class ParsimonySelection(MetricSelection):
    """Among candidates within absolute tolerance of the best metric, pick simpler.

    complexity names a candidate-supplied measure; smaller is simpler. Measures
    are arithmetic means across inner fitted folds. Missing complexity excludes
    that candidate from the parsimony choice (still visible in the comparison).
    Remaining complexity ties prefer the better metric, then candidate order.
    This is a declared practical tolerance, not a one-standard-error rule or test.
    """
    complexity: str = "n_parameters"
    tolerance: float = 0.0

    def decide(self, trials):
        if isinstance(self.tolerance, bool) or not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("Parsimony tolerance must be finite and nonnegative.")
        decision = super().decide(trials)
        table = decision.table.copy()
        complexity = {trial.trial_id: trial.complexity.get(self.complexity, np.nan) for trial in trials}
        table["complexity"] = table.run_id.map(complexity)
        best = table.loc[table.run_id.eq(decision.winner_id), f"raw::{self.metric}"].iloc[0]
        table["within_tolerance"] = table.status.eq("ranked") & (table[f"raw::{self.metric}"]-best).abs().le(self.tolerance)
        eligible = table.loc[table.within_tolerance & np.isfinite(table.complexity)].sort_values(
            ["complexity", "score", "candidate_order"], ascending=[True, False, True], kind="stable")
        if eligible.empty:
            raise ValueError("No near-best candidate supplies the requested complexity measure.")
        winner = eligible.iloc[0].run_id
        table["selected"] = table.run_id.eq(winner)
        return SelectionDecision(winner, table)
