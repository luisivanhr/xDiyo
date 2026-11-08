"""Causal observed-member form ranks, explicitly distinct from official tables.

Only previously played regular-stage fixtures establish the ranking population.
No final roster, final schedule, later result or inferred stage enters the query.
Without an as-of schedule publication, countdown is deliberately unavailable.
"""
from collections import defaultdict
import numpy as np
import pandas as pd
from .phase_context import ACTUAL_DATE_POLICY, PhaseRankLookup, KEYS, VALUES, WINDOWS
from .store import _ids, _times
from ..ratings.standings import competition_ranks
from ..data.availability import respect_result_availability


def build_actual_date_context(fixtures, *, timezone, provenance):
    required = [*KEYS, 'home_id', 'away_id', 'kickoff_at', 'goals_for', 'goals_against', 'phase_key']
    source = fixtures.copy(deep=True).reset_index(drop=True)
    if not set(required) <= set(source):
        raise ValueError('Missing actual-date context inputs.')
    _ids(source, (*KEYS, 'home_id', 'away_id'))
    source['kickoff_at'] = _times(source.kickoff_at)
    if source.duplicated(list(KEYS)).any() or source.home_id.eq(source.away_id).any():
        raise ValueError('Context fixtures must be unique and correctly oriented.')
    scores = source[['goals_for','goals_against']].to_numpy(float)
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores % 1).any():
        raise ValueError('Context requires finite nonnegative integer scores.')
    source['_release'] = respect_result_availability(source, source.kickoff_at)
    cuts = source.kickoff_at.dt.tz_convert(timezone).dt.normalize().dt.tz_convert('UTC')
    result = source[[*KEYS, 'home_id', 'away_id', 'kickoff_at']].copy()
    result['cutoff_at'] = cuts
    for name in VALUES:
        result[name] = np.nan
    # Stage membership is a retained annotation; unresolved or reset stages are
    # not silently promoted into a league table. Ranks are descriptive form,
    # not official positions, and carry/halving rules are never invented.
    regular = source.loc[source.phase_key.eq('regular_season').fillna(False)]
    for _, group in regular.groupby(['competition_id', 'season_id'], sort=True):
        past = defaultdict(list)
        released = group.loc[group._release.notna()].sort_values(['_release','kickoff_at','event_id'])
        events = list(released.itertuples())
        cursor = 0
        for cutoff, queries in group.groupby(cuts.loc[group.index], sort=True):
            while cursor < len(events) and source.loc[events[cursor].Index, '_release'] < cutoff:
                event = events[cursor]
                cursor += 1
                if event.kickoff_at >= cutoff:
                    continue
                for team, gf, ga in ((event.home_id,event.goals_for,event.goals_against),
                                     (event.away_id,event.goals_against,event.goals_for)):
                    contribution = np.array([1, gf>ga, gf==ga, gf<ga,
                        3*(gf>ga)+(gf==ga), gf, ga, gf-ga], dtype=np.int64)
                    past[team].append((event.kickoff_at,event.event_id,contribution))
                    past[team].sort(key=lambda x:(x[0],x[1]))
            teams = sorted(past)
            if len(teams) < 2:
                continue
            for window in WINDOWS:
                totals = [sum((item[2] for item in past[t][-window:]), np.zeros(8,dtype=np.int64)) for t in teams]
                ranks = dict(zip(teams, ((len(teams)-r)/(len(teams)-1) for r in competition_ranks(totals))))
                for side in ('home','away'):
                    result.loc[queries.index,f'{side}_rank_{window}'] = queries[f'{side}_id'].map(ranks)
    return PhaseRankLookup(result, provenance=dict(provenance,
        timezone=timezone, rank_population='previously observed regular-stage participants only',
        rank_semantics='3/1/0 last-w own matches; points/gd/gf/ga competition ranks; (N-r)/(N-1)',
        countdown='missing: no historical schedule publication log',
        result_availability='actual kickoff proxy, respecting later reviewed bounds; strictly before cutoff',
        unsupported_stages='missing; no guessed carry/reset rules'), policy=ACTUAL_DATE_POLICY)
