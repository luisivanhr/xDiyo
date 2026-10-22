"""Explicit, bounded Nelder-Mead statistic-MAE fitting (Wheatcroft pp.82-83).

No caller invokes this module during feature evaluation. No production starting
point, tolerance or sample-mask policy is invented here.
"""
from dataclasses import dataclass
import math
from .gap_core import GAPParameters
from .gap import _Replay, _time
from .gap_data import GAPSeason, gap_observations, gap_prediction_times


def _samples(history, statistic, cutoffs, objective_mask, available_at):
    import numpy as np
    import pandas as pd
    observations = gap_observations(history, (statistic,), available_at=available_at)
    times = gap_prediction_times(history, cutoffs)
    if times.isna().any() or (times > pd.to_datetime(history.kickoff_at, utc=True)).any():
        raise ValueError('Objective cutoffs must be known and no later than fixture kickoff.')
    keys = ['competition_id', 'season_id', 'event_id']
    frame = history[keys].copy()
    frame['cutoff'] = times.to_numpy()
    if objective_mask is None:
        raise ValueError('An explicit objective sample mask is required; burn-in fitting policy is unresolved.')
    if isinstance(objective_mask, pd.Series) and not objective_mask.index.equals(history.index):
        raise ValueError('Objective mask must align with history.')
    mask = np.asarray(objective_mask)
    if mask.shape != (len(history),) or mask.dtype.kind != 'b':
        raise ValueError('Objective mask must be a nonmissing boolean per perspective row.')
    frame['mask'] = mask
    if frame.groupby(keys)[['cutoff', 'mask']].nunique(dropna=False).gt(1).any().any():
        raise ValueError('Both fixture perspectives must have the same objective cutoff and mask.')
    observations = observations.merge(frame.drop_duplicates(keys), on=keys, validate='one_to_one', sort=False)
    return observations


def _loss(observations, statistic, seasons, parameters):
    import pandas as pd
    replay = _Replay(observations, seasons, ((statistic.name, parameters),))
    errors = []
    for r in observations.to_dict('records'):
        yh, ya = r[f'{statistic.name}_home'], r[f'{statistic.name}_away']
        if not r['mask'] or not r['finished'] or pd.isna(r['available_at']) or pd.isna(yh) or pd.isna(ya):
            continue
        key = r['competition_id'], r['season_id'], statistic.name
        states, _, _, _ = replay.state_at(key, r['cutoff'])
        prediction = replay.algorithm.predict(states[r['home_id']], states[r['away_id']])
        errors.append(abs(yh-prediction.home) + abs(ya-prediction.away))
    if not errors:
        raise ValueError('No eligible paired observations in GAP objective.')
    return math.fsum(errors) / len(errors), len(errors)


def gap_mae(history, *, statistic, seasons, parameters, cutoffs, objective_mask, available_at=None):
    """Equation (9): paired absolute-error sum divided by N matches, never 2N.

    All available complete pairs update states, independently of the loss mask.
    For pre-round experiments pass the real frozen cutoffs explicitly.
    """
    observations = _samples(history, statistic, cutoffs, objective_mask, available_at)
    return _loss(observations, statistic, tuple(seasons), parameters)[0]


@dataclass(frozen=True)
class GAPFitResult:
    statistic: str
    parameters: GAPParameters
    fit_cutoff: str
    loss: float
    success: bool
    message: str
    iterations: int
    evaluations: int
    provenance: dict


def tune_gap(history, *, statistic, seasons, fit_cutoff, target_start, initial,
             cutoffs, objective_mask, maxiter, maxfev, xatol, fatol, available_at=None, initial_simplex=None,
             target_seasons=None):
    """Fit one statistic pooled over the supplied preceding league-seasons.

    Training plans require explicit closed_at evidence; target_seasons names
    the disjoint upcoming league-seasons and target_start must match their
    earliest entry. No held-out observations may enter even masked out.

    A transformed unconstrained simplex uses log(lambda), logit(phi1/phi2).
    Unrepresentable transformed candidates get infinite loss, not clipping.
    This numerical technique/settings are implementation choices, not published
    constants. Every setting is explicit and recorded; no automatic real fit.
    """
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit
    if not isinstance(initial, GAPParameters):
        raise TypeError('Supply explicit optimizer starting GAPParameters.')
    for value in (maxiter, maxfev):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Positive integer optimizer limits are required.')
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in (xatol, fatol)):
        raise ValueError('Positive finite optimizer tolerances are required.')
    fit, start = _time(fit_cutoff), _time(target_start)
    if fit > start:
        raise ValueError('Parameter fit cutoff must not follow target season start.')
    seasons = tuple(s for s in seasons if s.statistic == statistic.name)
    targets = tuple(target_seasons or ())
    if not targets or any(not isinstance(s, GAPSeason) or s.statistic != statistic.name for s in targets):
        raise ValueError('Declare the held-out target seasons for the fitted statistic; a date alone is insufficient.')
    if len({s.key for s in targets}) != len(targets) or len({s.competition_id for s in targets}) != len(targets):
        raise ValueError('Target seasons must identify one upcoming season per league.')
    if {s.key for s in seasons} & {s.key for s in targets}:
        raise ValueError('Held-out target seasons must not appear in parameter training.')
    if start != min(_time(s.entry_at) for s in targets):
        raise ValueError('target_start must equal the earliest declared target season entry.')
    if not seasons or any(s.closed_at is None or _time(s.closed_at) > fit for s in seasons):
        raise ValueError('Every training season must have explicit closed_at evidence no later than the fit cutoff.')
    if any(_time(s.entry_at) >= start for s in seasons):
        raise ValueError('Only preceding seasons belong in parameter training.')
    observations = _samples(history, statistic, cutoffs, objective_mask, available_at)
    if (observations.kickoff_at >= fit).any() or (observations.available_at > fit).any():
        raise ValueError('Training history includes observations after the parameter fit cutoff.')
    if any((r.competition_id, r.season_id, statistic.name) not in {s.key for s in seasons}
           for r in observations.itertuples()):
        raise ValueError('Every training fixture must belong to a declared closed training season.')
    # Verify exact target identities/predecessors without assimilating target outcomes.
    _Replay(observations, (*seasons, *targets), ((statistic.name, initial),))
    # Validate the full objective contract before the optimizer catches numerical overflow.
    initial_loss, count = _loss(observations, statistic, seasons, initial)
    def decode(x):
        return GAPParameters(float(np.exp(x[0])), float(expit(x[1])), float(expit(x[2])))
    def objective(x):
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            try:
                parameters = decode(x)
                return _loss(observations, statistic, seasons, parameters)[0]
            except (OverflowError, FloatingPointError):
                return float('inf')
            except ValueError as error:
                # Only invalid transformed numerical candidates/overflow are penalized.
                if any(s in str(error) for s in ('lambda', 'phi1', 'phi2', 'overflowed')):
                    return float('inf')
                raise
    x0 = np.array([math.log(initial.learning_rate), math.log(initial.phi1/(1-initial.phi1)),
                   math.log(initial.phi2/(1-initial.phi2))])
    if initial_simplex is None:
        # Explicit classic simplex construction, in transformed coordinates.
        simplex = np.tile(x0, (4, 1))
        for i in range(3):
            simplex[i+1, i] = 1.05*x0[i] if x0[i] != 0 else .00025
    else:
        simplex = np.asarray(initial_simplex, dtype=float)
    if simplex.shape != (4, 3) or not np.isfinite(simplex).all() or np.linalg.matrix_rank(simplex[1:]-simplex[0]) < 3:
        raise ValueError('Initial simplex must be a finite nondegenerate 4x3 array in transformed coordinates.')
    options = dict(maxiter=maxiter, maxfev=maxfev, xatol=xatol, fatol=fatol,
                   adaptive=False, initial_simplex=simplex.tolist())
    result = minimize(objective, x0, method='Nelder-Mead', options=options)
    import hashlib
    from dataclasses import asdict
    import scipy
    provenance = dict(scipy_version=scipy.__version__, transformed_start=x0.tolist(), method='Nelder-Mead', domain='log_lambda_logit_phi; invalid floats have infinite loss',
                      initial=asdict(initial), options=options, initial_loss=initial_loss, eligible_matches=count,
                      sample_mask=observations['mask'].tolist(),
                      training_identities=observations[['competition_id', 'season_id', 'event_id']].to_dict('records'),
                      cutoff_policy='explicit_per_fixture', cutoffs=[t.isoformat() for t in observations.cutoff],
                      observations_sha256=hashlib.sha256(observations.to_json(date_format='iso', date_unit='ns').encode()).hexdigest(),
                      target_start=start.isoformat(), target_seasons=[asdict(s) for s in targets], state_policy='replay', statistic_definition=asdict(statistic),
                      season_plans=[asdict(s) for s in seasons], objective='sum_paired_absolute_errors / N')
    return GAPFitResult(statistic.name, decode(result.x), fit.isoformat(), float(result.fun), bool(result.success),
                        str(result.message), int(result.nit), int(result.nfev), provenance)
