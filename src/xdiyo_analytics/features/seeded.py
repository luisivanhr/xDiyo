"""Explicit boundary-window warm starts; legacy warm-up lives in warmup.py.

Population moments (ddof=0) or conventional weighted-sample correction (ddof=1).
Fade interpolates variance estimates, not distributions, so no mixture or
overlap-weight correction is applied to the two branches.
"""
from dataclasses import asdict, replace
import math
import numpy as np
import pandas as pd

from .expressions import RollingMean, RollingStd, RollingZScore, H2H
from .league import League, LeaveOneOut
from .warmup import Hard, LinearFade, ObservationCount
from .transitions import _season_start_year


def moment_update(mean, central, q, value, alpha):
    if alpha == 1:
        if not math.isfinite(value):
            raise ValueError('Nonfinite warm-start observation.')
        return float(value), 0., 1.
    delta = value - mean
    mean = mean + alpha * delta
    central = (1-alpha) * (central + alpha * delta * delta)
    q = (1-alpha)**2 * q + alpha**2
    if not math.isfinite(mean) or (not math.isnan(central) and (not math.isfinite(central) or central < 0)):
        raise ValueError('Nonfinite or negative warm-start moment state.')
    return mean, central, q


def evaluate_seeded(node, context, evaluate, candidates, *, h2h, group_by):
    op, policy = node.source, node.policy
    if isinstance(op, H2H):
        return evaluate_seeded(replace(node, source=op.source), context, evaluate,
                               candidates, h2h=True, group_by=group_by)
    if not isinstance(op, (RollingMean, RollingStd, RollingZScore)):
        raise TypeError('Explicit SeededEMA modes wrap RollingMean, RollingStd or RollingZScore; '
                        'Lag retains exact-lag meaning and direct EMA has a separate span/state contract.')
    if isinstance(op.source, (League, LeaveOneOut)):
        raise ValueError('Explicit team seed modes cannot map a team predecessor/cohort onto a pooled '
                         'League/LOO population. Use ordinary League/LOO or legacy warm-up.')
    if not {'team_id', 'competition_id'}.issubset(group_by):
        raise ValueError('Explicit team seeds require team_id and competition_id grouping.')
    dispersion = isinstance(op, (RollingStd, RollingZScore))
    if dispersion:
        expected = 0 if policy.variance_estimator == 'population' else 1
        if op.ddof != expected:
            raise ValueError(f'{policy.variance_estimator} requires ddof={expected}; new modes support ddof 0 or 1.')
        if policy.mode == 'w_league_prior' and expected == 1 and policy.prior_strength is None:
            raise ValueError('weighted_sample cohort dispersion requires explicit prior_strength > 1.')
    ordinary, scope = evaluate(op, h2h)
    source, _ = evaluate(op.source, h2h)
    values = source.to_numpy(dtype=float, na_value=np.nan)
    out = ordinary.copy(deep=True)
    output = out.to_numpy(dtype=float, na_value=np.nan, copy=True)
    p, history = context.population, context.history
    # Two different events for one team at the same kickoff are ambiguous.
    # Reject only on this new path; do not change legacy historical tie handling.
    if history.loc[p.kickoff.notna().to_numpy()].duplicated(['competition_id', 'team_id', 'kickoff_at']).any():
        raise ValueError('Explicit warm-start modes require unambiguous team/kickoff observations.')
    def ordered(rows):
        return np.array(sorted(rows, key=lambda i: (p.kickoff.iloc[i], str(history.event_id.iloc[i]))), dtype=int)
    all_rows = [ordered(rows) for rows in candidates(scope, op.venue)]
    extras = tuple(k for k in group_by if k not in ('team_id', 'competition_id', 'season_id'))
    if scope and 'opponent_id' not in extras:
        extras += ('opponent_id',)
    if op.venue == 'same' and 'side' not in extras:
        extras += ('side',)
    reference = None
    if isinstance(op, RollingZScore) and op.reference is not None:
        reference = evaluate(op.reference, h2h)[0].to_numpy(dtype=float, na_value=np.nan)
    audits, seeds = [], {}
    years = {key: _season_start_year(history.iloc[rows]) for key, rows in context.groups.items()}
    origin_seasons = {}

    def boundary_window(team, previous, boundary, row):
        if previous is None or any(pd.isna(v) for v in previous) or previous not in context.groups:
            return np.array([], dtype=int)
        # Require membership in the identified predecessor before extending its
        # competition-scoped window into still older seasons.
        if team not in set(p.teams[context.groups[previous]]):
            return np.array([], dtype=int)
        rows = np.flatnonzero(p.valid & (p.teams == team) & (p.competitions == previous[0])
                              & (p.kickoff < boundary).to_numpy() & (p.available <= boundary).to_numpy())
        if previous not in origin_seasons:
            prior_year = years[previous]
            origin_seasons[previous] = {key for key, year in years.items()
                if key == previous or (key[0] == previous[0] and prior_year is not None
                                       and year is not None and year < prior_year)}
        rows = rows[np.array([context.seasons[i] in origin_seasons[previous] for i in rows], dtype=bool)]
        if 'season_id' in group_by:
            rows = rows[history.season_id.iloc[rows].astype(object).eq(previous[1]).to_numpy()]
        for name in extras:
            rows = rows[history[name].iloc[rows].astype(object).eq(history[name].iloc[row]).fillna(False).to_numpy(dtype=bool)]
        return ordered(rows)[-op.window:]

    def state(rows, col):
        sample = values[rows, col]
        sample = sample[np.isfinite(sample)]
        count = len(sample)
        if count < op.min_periods or (dispersion and count <= op.ddof):
            return None
        mean = float(sample.mean())
        central = float(np.mean((sample-mean)**2))
        if not np.isfinite(mean) or not np.isfinite(central):
            raise ValueError('Nonfinite boundary warm-start moments.')
        return mean, central, 1./count, count

    def variance(s):
        mean, central, q, count = s
        return central if not dispersion or policy.variance_estimator == 'population' else (
            central / (1-q) if q < 1 else np.nan)

    def evidence(rows):
        return dict(events=[int(history.event_id.iloc[i]) for i in rows],
                    latest_kickoff=None if not len(rows) else p.kickoff.iloc[rows].max().isoformat(),
                    latest_available=None if not len(rows) else p.available.iloc[rows].max().isoformat())

    for row in range(len(history)):
        record = context.record(row)
        season, boundary = context.seasons[row], record['entry_at']
        if pd.isna(boundary) or pd.isna(p.cutoff.iloc[row]) or pd.isna(p.kickoff.iloc[row]):
            continue
        rounds = 0 if isinstance(policy.handoff, ObservationCount) else context.completed_rounds(row, policy.round_keys)
        if isinstance(policy.handoff, (Hard, LinearFade)) and policy.handoff.weight(rounds, 0) == 0:
            continue
        signature = season, record['team_id'], tuple(history[k].iloc[row] for k in extras)
        if signature not in seeds:
            previous = (record['previous_competition_id'], record['previous_season_id'])
            own_rows = boundary_window(record['team_id'], previous, boundary, row)
            donor_rows = []
            destination = context.predecessors[season]
            moved = record['movement'] in ('promoted', 'relegated')
            requested = policy.bottom if record['movement'] == 'promoted' else policy.top
            if policy.mode == 'w_league_prior' and moved and destination is not None:
                old_rows = context.prior_rows(destination, boundary)
                for team in sorted(set(p.teams[old_rows]), key=lambda t: str(t)):
                    if team == record['team_id']:
                        continue
                    entrant = context.movement_evidence.get((*season, team), {})
                    if entrant.get('movement') in ('promoted', 'relegated', 'other_entry'):
                        continue
                    ranking_rows = ordered([i for i in old_rows if p.teams[i] == team])
                    position = None
                    if 'team_position' in history:
                        for i in ranking_rows[::-1]:
                            value = history.team_position.iloc[i]
                            if pd.notna(value) and np.isfinite(value) and value >= 1:
                                position = float(value)
                                break
                    if requested != -1 and position is None:
                        continue
                    donor_rows.append((team, position, boundary_window(team, destination, boundary, row), ranking_rows))
            column_seeds = []
            for col, column in enumerate(source.columns):
                own = state(own_rows, col)
                seed, fallback = own, 'own_boundary' if own is not None else 'ordinary_no_own_seed'
                donors = [(t, rank, rows, ranking, state(rows, col)) for t, rank, rows, ranking in donor_rows]
                donors = [d for d in donors if d[-1] is not None]
                available = len(donors)
                if requested != -1:
                    # Identity breaks ties in either ranking direction.
                    donors.sort(key=lambda d: ((-d[1] if record['movement'] == 'promoted' else d[1]), str(d[0])))
                    donors = donors[:requested]
                if donors:
                    mean = float(np.mean([d[-1][0] for d in donors]))
                    v = float(np.mean([variance(d[-1]) for d in donors]))
                    q = 1./policy.prior_strength if dispersion and policy.variance_estimator == 'weighted_sample' else 0.
                    seed = (mean, v*(1-q), q, None)
                    fallback = 'destination_cohort'
                elif policy.mode == 'w_league_prior' and moved:
                    fallback = 'own_no_usable_cohort' if own is not None else 'ordinary_no_seed_or_cohort'
                column_seeds.append(seed)
                audits.append(dict(team_id=int(record['team_id']), competition_id=int(season[0]), season_id=int(season[1]),
                    movement=record['movement'], evidence=str(record.get('evidence', '')), source=str(record.get('source', '')),
                    boundary=boundary.isoformat(), feature=repr(op), column=str(column), mode=policy.mode,
                    parameters=asdict(policy), scope={k: str(history[k].iloc[row]) for k in extras},
                    predecessor=[None if pd.isna(v) else int(v) for v in previous],
                    seed_mean=None if seed is None else seed[0], seed_variance=None if seed is None else variance(seed),
                    seed_q=None if seed is None else seed[2], own_valid_count=None if own is None else own[3],
                    fallback=fallback, requested=requested if policy.mode == 'w_league_prior' and moved else None,
                    available_donors=available, own_window=evidence(own_rows),
                    donors=[dict(team_id=int(t), position=rank, mean=s[0], variance=variance(s), valid_count=s[3],
                                 window=evidence(rows), ranking=evidence(ranking)) for t,rank,rows,ranking,s in donors]))
            seeds[signature] = column_seeds
        positions = all_rows[row]
        new = [i for i in positions if context.seasons[i] == season and p.kickoff.iloc[i] >= boundary]
        for col, seed in enumerate(seeds[signature]):
            if seed is None:
                continue  # Frozen fallback stays on the ordinary path all season.
            mean, central, q, _ = seed
            count = 0
            for i in new:
                value = values[i, col]
                if np.isfinite(value):
                    mean, central, q = moment_update(mean, central, q, value, policy.alpha)
                    count += 1
            weight = policy.handoff.weight(rounds, count)
            if weight == 0:
                continue
            v = variance((mean, central, q, count))
            rolling = state(positions[-op.window:], col)
            if weight < 1 and rolling is not None:
                mean = weight*mean + (1-weight)*rolling[0]
                v = weight*v + (1-weight)*variance(rolling)
            if isinstance(op, RollingMean):
                output[row, col] = mean
            elif isinstance(op, RollingStd):
                output[row, col] = np.sqrt(v) if np.isfinite(v) and v >= 0 else np.nan
            else:
                x = reference[row, col] if reference is not None else values[positions[-1], col] if len(positions) else np.nan
                output[row, col] = (x-mean)/np.sqrt(v) if np.isfinite(x) and np.isfinite(v) and v > 0 else np.nan
    result = pd.DataFrame(output, index=ordinary.index, columns=ordinary.columns)
    result.attrs = dict(ordinary.attrs)
    result.attrs['warm_start_audit'] = audits
    import hashlib
    result.attrs['warm_start_source_hash'] = hashlib.sha256(
        pd.util.hash_pandas_object(source, index=True).to_numpy().tobytes()).hexdigest()
    return result, scope
