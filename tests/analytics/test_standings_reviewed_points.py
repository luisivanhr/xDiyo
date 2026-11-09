"""Administrative points are reviewed input, separate from goals and targets."""
import json

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.data import SeasonData
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import DerivedStandings, evaluate_features
from xdiyo_analytics.ratings import build_standings, StandingsRun
from xdiyo_analytics.ratings.standings import REVIEWED_POLICY, REVIEWED_DAILY_POLICY, TOTALS
from xdiyo_analytics.graphs import build_phase_context
from xdiyo_analytics.graphs.phase_context import PhaseRankLookup, PhaseRankFeatures, REVIEWED_DAILY_POLICY as PHASE_POLICY
from test_derived_standings import population


def reviewed_population():
    h, m = population(8, seasons=(1, 2))
    event = h.event_id.iloc[0]
    h['is_awarded'] = False
    mask = h.event_id.eq(event)
    h.loc[mask, 'is_awarded'] = True
    # The vendor's awarded 0-3 is not the official standings-goal total.
    for side, scores, points in [('home', (0, 3), (0, 3)), ('away', (3, 0), (3, 0))]:
        rows = mask & h.side.eq(side)
        h.loc[rows, ['goals_for', 'goals_against']] = scores
        h.loc[rows, ['standings_goals_for', 'standings_goals_against']] = (0, 0)
        h.loc[rows, ['standings_points_for', 'standings_points_against']] = points
    h.loc[mask, 'result_review_json'] = json.dumps({'reviewed': True, 'source': 'verified fixture ruling', 'decision': 'penalty loss, actual goals retained'})
    h['standings_result_available_at'] = pd.Series(pd.NaT, index=h.index, dtype='datetime64[ns, UTC]')
    h.loc[mask, 'standings_result_available_at'] = h.loc[mask, 'release']
    h['source_league'] = 'Premier_League'
    h['source_season'] = h.season_id.map({1: '20_21', 2: '21_22'})
    return h, m, event


@pytest.mark.parametrize('timing', ['available_at_cutoff', 'kickoff_day'])
@pytest.mark.parametrize('venue', ['all', 'home', 'away'])
def test_reviewed_replay_matches_independent_totals_and_ranks(tmp_path, timing, venue):
    h, m, event = reviewed_population()
    before = h.copy(deep=True)
    run = build_standings(h, team_seasons=m, normalized_ranks=[1, 10], timing=timing,
                          available_at='release' if timing != 'kickoff_day' else None, venues=[venue])
    assert run.metadata['policy'] == (REVIEWED_DAILY_POLICY if timing == 'kickoff_day' else REVIEWED_POLICY)
    assert run.metadata['reviewed_points'][0]['event'] == event
    for (comp, season, at, window), table in run.snapshots.groupby(['competition_id', 'season_id', 'recorded_at', 'window']):
        source = h.loc[h.competition_id.eq(comp) & h.season_id.eq(season) & h.kickoff_at.lt(at)]
        if timing != 'kickoff_day':
            source = source.loc[source.release.le(at)]
        if venue != 'all':
            source = source.loc[source.side.eq(venue)]
        expected = {}
        for team in m.loc[m.competition_id.eq(comp) & m.season_id.eq(season), 'team_id']:
            past = source.loc[source.team_id.eq(team)].sort_values(['kickoff_at', 'event_id'])
            if window:
                past = past.tail(window)
            wins = draws = losses = gf = ga = 0
            for match in past.itertuples():
                if match.event_id == event:
                    wins += match.side == 'away'
                    losses += match.side == 'home'
                    # Both actual standings goals are zero.
                else:
                    wins += match.goals_for > match.goals_against
                    draws += match.goals_for == match.goals_against
                    losses += match.goals_for < match.goals_against
                    gf += match.goals_for
                    ga += match.goals_against
            expected[team] = [len(past), wins, draws, losses, 3*wins+draws, gf, ga, gf-ga]
        keys = {t: (v[4], v[7], v[5], -v[6]) for t, v in expected.items()}
        for row in table.itertuples():
            assert [getattr(row, c) for c in TOTALS] == expected[row.team_id]
            assert row.rank == 1+sum(k > keys[row.team_id] for k in keys.values())
    run.save(tmp_path/'saved')
    loaded = StandingsRun.load(tmp_path/'saved')
    pd.testing.assert_frame_equal(run.snapshots, loaded.snapshots)
    expr = DerivedStandings([1, 10], side='both', venue=venue)
    pd.testing.assert_frame_equal(run.features(h, expr), loaded.features(h, expr))
    pd.testing.assert_frame_equal(h, before)


def test_daily_reviewed_points_same_day_isolation_and_prior_day_inclusion():
    h, m, event = reviewed_population()
    day = h.loc[h.event_id.eq(event), 'kickoff_at'].iloc[0].normalize()
    query = h.loc[h.kickoff_at.dt.normalize().eq(day)]
    def run(source, q=query):
        return build_standings(source, team_seasons=m, queries=q, normalized_ranks=[1, 10], timing='kickoff_day')
    baseline = run(h)
    changed = h.copy()
    mask = changed.event_id.eq(event)
    changed.loc[mask, ['standings_points_for', 'standings_points_against']] = changed.loc[mask, ['standings_points_against', 'standings_points_for']].to_numpy()
    pd.testing.assert_frame_equal(baseline.snapshots, run(changed).snapshots)
    assert all(event not in p['contributing_events'] for p in baseline.publications)
    # A result from the previous calendar date is available even with a later ruling date.
    h.loc[h.event_id.eq(event), 'standings_result_available_at'] = pd.Timestamp('2099-01-01', tz='UTC')
    later = h.loc[h.kickoff_at.dt.normalize().gt(day)].iloc[:2]
    retrospective = run(h, later)
    assert any(event in p['contributing_events'] for p in retrospective.publications)
    with pytest.raises(ValueError, match='known ruling/result release'):
        build_standings(h.drop(columns='standings_result_available_at'), team_seasons=m)


@pytest.mark.parametrize('failure', ['partial_points', 'missing_goals', 'missing_review', 'unreviewed', 'missing_source', 'bad_pair', 'negative', 'paired_conflict'])
def test_rejects_unreviewed_incomplete_or_unsupported_points(failure):
    h, m, event = reviewed_population()
    mask = h.event_id.eq(event)
    if failure == 'partial_points':
        h.loc[h.index[mask][0], 'standings_points_for'] = np.nan
    elif failure == 'missing_goals':
        h.loc[mask, ['standings_goals_for', 'standings_goals_against']] = np.nan
        h.loc[mask, 'standings_result_available_at'] = pd.NaT
    elif failure == 'missing_review':
        h.loc[mask, 'result_review_json'] = np.nan
    elif failure == 'unreviewed':
        h.loc[mask, 'result_review_json'] = '{"reviewed":false,"source":"ref"}'
    elif failure == 'missing_source':
        h.loc[mask, 'result_review_json'] = '{"reviewed":true}'
    elif failure == 'bad_pair':
        h.loc[mask, ['standings_points_for', 'standings_points_against']] = 3
    elif failure == 'negative':
        h.loc[mask & h.side.eq('home'), 'standings_points_for'] = -1
        h.loc[mask & h.side.eq('away'), 'standings_points_against'] = -1
    else:
        h.loc[mask & h.side.eq('home'), 'standings_points_for'] = 1
    with pytest.raises(ValueError, match='points|nonnegative'):
        build_standings(h, team_seasons=m, timing='kickoff_day')


def test_public_history_and_feature_paths_preserve_separate_points_and_goals():
    matches = pd.DataFrame(dict(event_id=[1], home_id=[11], away_id=[12], kickoff_utc=[1600000000],
        status=['finished'], home_score_current=[0], away_score_current=[3],
        home_standings_goals=[0], away_standings_goals=[0],
        home_standings_points=[0], away_standings_points=[3],
        result_review_json=['{"reviewed":true,"source":"verified ruling"}']))
    h = build_team_history(SeasonData({'matches': matches}, {}))
    home = h.loc[h.side.eq('home')].iloc[0]
    assert (home.goals_for, home.goals_against, home.result) == (0, 3, 'L')
    assert (home.standings_goals_for, home.standings_goals_against) == (0, 0)
    assert (home.standings_points_for, home.standings_points_against) == (0, 3)
    # The general feature evaluator also uses the reviewed timestamp policy.
    source, members, _ = reviewed_population()
    evaluated = evaluate_features(source, {'r': DerivedStandings([1, 10])},
        standings_history=source, team_seasons=members, standings_available_at='release')
    assert evaluated.attrs['derived_standings']['r'][0]['source']['policy'] == REVIEWED_POLICY


def test_phase_lookup_version_and_both_rank_consumers(tmp_path):
    h, m, _ = reviewed_population()
    h = h.loc[h.season_id.eq(1)].reset_index(drop=True)
    m = m.loc[m.season_id.eq(1)].reset_index(drop=True)
    fixtures = h.loc[h.side.eq('home')].rename(columns={'team_id':'home_id','opponent_id':'away_id'}).reset_index(drop=True)
    cuts = fixtures.kickoff_at.dt.normalize()
    lookup = build_phase_context(h, m, fixtures, cutoffs=cuts, provenance={'review':'test'})
    assert lookup.policy == PHASE_POLICY
    lookup.save(tmp_path)
    loaded = PhaseRankLookup.load(tmp_path)
    ranks = build_standings(h, team_seasons=m, normalized_ranks=[1,10], timing='kickoff_day').features(h, DerivedStandings([1,10], side='both'))
    np.testing.assert_array_equal(loaded.edges(h)[:,:2], ranks[['team::form_rank_1','opponent::form_rank_1']])
    fixtures['_phase_cutoff'] = cuts
    expected = ranks.loc[h.side.eq('home'), ['team::form_rank_10','opponent::form_rank_10']]
    np.testing.assert_array_equal(loaded.fixture(fixtures, PhaseRankFeatures('D10', loaded.source_id, 10, PHASE_POLICY))[:,:2], expected)


def test_known_official_review_is_narrow_idempotent_and_preserves_labels():
    from xdiyo_analytics.ratings import apply_known_standings_reviews
    h = pd.DataFrame([dict(competition_id=182, season_id=77357, event_id=14064792,
        team_id=t, opponent_id=o, side=s, round=16, kickoff_at=pd.Timestamp('2025-12-05T19:00Z'),
        status='finished', is_awarded=True, goals_for=gf, goals_against=ga)
        for s,t,o,gf,ga in [('home',1655,52874,0,3),('away',52874,1655,3,0)]])
    original = h.copy(deep=True)
    corrected = apply_known_standings_reviews(h)
    pd.testing.assert_frame_equal(apply_known_standings_reviews(corrected), corrected)
    pd.testing.assert_frame_equal(h, original)
    pd.testing.assert_frame_equal(corrected[h.columns], h)
    assert corrected.standings_goals_for.eq(0).all()
    assert corrected.standings_points_for.tolist() == [0,3]
    assert 'standings_result_available_at' not in corrected
    other = h.copy(); other['event_id'] += 1
    pd.testing.assert_frame_equal(apply_known_standings_reviews(other), other)
    bad = corrected.copy(); bad['standings_goals_for'] = 3
    with pytest.raises(ValueError, match='Conflicting approved'):
        apply_known_standings_reviews(bad)
    bad = h.copy(); bad['round'] = 15
    with pytest.raises(ValueError, match='original round'):
        apply_known_standings_reviews(bad)
