"""Daily ranks: independent arithmetic oracle and both graph consumption paths."""
import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.features import DerivedStandings
from xdiyo_analytics.ratings import build_standings, StandingsRun
from xdiyo_analytics.ratings.standings import DAILY_POLICY, TOTALS
from xdiyo_analytics.graphs import build_phase_context
from xdiyo_analytics.graphs.phase_context import PhaseRankLookup, PhaseRankFeatures, DAILY_POLICY as PHASE_POLICY
from test_derived_standings import population, oracle


def daily_population():
    h, m = population(8, seasons=(1, 2))
    # A round-one match is played on a later round's actual day, without a flag.
    postponed = h.loc[h.season_id.eq(1) & h['round'].eq(1), 'event_id'].iloc[0]
    day = h.loc[h.season_id.eq(1) & h['round'].eq(4), 'kickoff_at'].min().normalize()
    h.loc[h.event_id.eq(postponed), 'kickoff_at'] = day + pd.Timedelta(hours=9)
    # Previous-day late results are included, even within 24 hours of forecast.
    h.loc[h.season_id.eq(1) & h['round'].eq(3), 'kickoff_at'] = day - pd.Timedelta(minutes=1)
    h['release'] = h.kickoff_at
    other = h.copy(); other['competition_id'] = 99
    other_m = m.copy(); other_m['competition_id'] = 99
    return pd.concat([h, other], ignore_index=True), pd.concat([m, other_m], ignore_index=True), day, postponed


def test_daily_tables_match_oracle_for_all_teams_windows_and_scopes():
    h, m, day, postponed = daily_population()
    before = h.copy(deep=True)
    run = build_standings(h, team_seasons=m, normalized_ranks=[1, 10], timing='kickoff_day')
    assert run.metadata['policy'] == DAILY_POLICY
    assert run.snapshots.recorded_at.eq(run.snapshots.recorded_at.dt.normalize()).all()
    for row in run.snapshots.itertuples(index=False):
        query = h.loc[h.competition_id.eq(row.competition_id) & h.season_id.eq(row.season_id)
                      & h.kickoff_at.dt.normalize().eq(row.recorded_at)].iloc[0]
        totals, ranks, _ = oracle(h, m, query, row.recorded_at, row.window)
        assert row.rank == ranks[row.team_id]
        assert [getattr(row, c) for c in TOTALS] == totals[row.team_id]
    pubs = [p for p in run.publications if p['scope'] == [17, 1]]
    at = next(p for p in pubs if pd.Timestamp(p['cutoff']) == day)
    assert postponed not in at['contributing_events']
    assert set(h.loc[h.competition_id.eq(17) & h.season_id.eq(1) & h['round'].eq(3), 'event_id']) <= set(at['contributing_events'])
    later = next(p for p in pubs if pd.Timestamp(p['cutoff']) > day)
    assert postponed in later['contributing_events']
    assert run.metadata['table_computations'] == h.groupby(['competition_id', 'season_id']).kickoff_at.apply(lambda s: s.dt.normalize().nunique()).sum()
    pd.testing.assert_frame_equal(h, before)


def test_daily_same_day_future_isolation_prior_day_and_release_independence():
    h, m, day, _ = daily_population()
    query = h.loc[h.kickoff_at.dt.normalize().eq(day)]
    def values(source):
        run = build_standings(source, team_seasons=m, normalized_ranks=[1, 10], queries=query,
                              cutoffs=query.kickoff_at, timing='kickoff_day')
        return run.features(query, DerivedStandings([1, 10], side='both'), cutoffs=query.kickoff_at), run
    expected, run = values(h)
    changed = h.copy()
    changed.loc[changed.kickoff_at.ge(day), ['goals_for', 'goals_against']] += 30
    changed['result_available_at'] = pd.Timestamp('2099-01-01', tz='UTC')
    actual, _ = values(changed)
    np.testing.assert_array_equal(expected, actual)
    # Even missing future results cannot revise an earlier forecast.
    changed.loc[changed.kickoff_at.ge(day), ['goals_for', 'goals_against']] = np.nan
    np.testing.assert_array_equal(expected, values(changed)[0])
    # Same day forecasts share publication IDs, including the postponed fixture.
    audit = expected.attrs['standings_audit']['rows']
    assert len({r['publication_id'] for row in audit for r in row}) == 2  # two competitions
    # A preceding season or another league never contributes to this table.
    assert all(p['eligible_result_count'] == 0 for p in run.publications if p['scope'][1] == 2)
    with pytest.raises(ValueError, match='target kickoff calendar date'):
        build_standings(h, team_seasons=m, queries=query, cutoffs=day-pd.Timedelta(days=1), timing='kickoff_day')
    with pytest.raises(ValueError, match='do not supply available_at'):
        build_standings(h, team_seasons=m, timing='kickoff_day', available_at='release')


def test_daily_persistence_and_historical_edge_decoder_windows(tmp_path):
    h, m, _, _ = daily_population()
    # Use a single complete supported scope to exercise the native countdown too.
    h = h.loc[h.competition_id.eq(17) & h.season_id.eq(1)].reset_index(drop=True)
    m = m.loc[m.competition_id.eq(17) & m.season_id.eq(1)].reset_index(drop=True)
    h['source_league'] = 'Premier_League'
    h['source_season'] = '20_21'
    fixtures = h.loc[h.side.eq('home')].rename(columns={'team_id':'home_id', 'opponent_id':'away_id'}).reset_index(drop=True)
    cuts = fixtures.kickoff_at.dt.normalize()
    lookup = build_phase_context(h, m, fixtures, cutoffs=cuts, provenance={'test':'daily'})
    assert lookup.policy == PHASE_POLICY
    lookup.save(tmp_path/'phase')
    loaded = PhaseRankLookup.load(tmp_path/'phase')
    assert loaded.source_id == lookup.source_id
    run = build_standings(h, team_seasons=m, normalized_ranks=[1, 10], timing='kickoff_day')
    run.save(tmp_path/'standings')
    restored = StandingsRun.load(tmp_path/'standings')
    pd.testing.assert_frame_equal(run.features(h, DerivedStandings([1, 10])), restored.features(h, DerivedStandings([1, 10])))
    ranks = run.features(h, DerivedStandings([1, 10], side='both'))
    np.testing.assert_array_equal(loaded.edges(h)[:, :2], ranks[['team::form_rank_1', 'opponent::form_rank_1']])
    query = fixtures.copy(); query['_phase_cutoff'] = cuts
    spec = PhaseRankFeatures('D10', loaded.source_id, 10, PHASE_POLICY)
    expected = ranks.loc[h.side.eq('home'), ['team::form_rank_10', 'opponent::form_rank_10']]
    np.testing.assert_array_equal(loaded.fixture(query, spec)[:, :2], expected)
    assert loaded.edges(h)[:, 2:].sum() == 0
    assert loaded.fixture(query, spec)[:, 2:].sum() == 0
    assert all(run._fixtures[(17, 1, r.event_id)]['round'] == r.round for r in h.itertuples())

