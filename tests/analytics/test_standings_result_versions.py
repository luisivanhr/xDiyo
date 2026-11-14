"""Independent as-of arithmetic, revision chronology and legacy regressions."""
import json
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.features import DerivedStandings
from xdiyo_analytics.ratings import build_standings, StandingsRun
from xdiyo_analytics.ratings.standings import TOTALS, VERSIONED_POLICY, VERSIONED_DAILY_POLICY
from xdiyo_analytics.ratings.standings_versions import FIELDS
from test_derived_standings import population


def source():
    history, members = population(4)
    event = history.event_id.iloc[0]
    mask = history.event_id.eq(event)
    history['is_awarded'] = False
    history.loc[mask, 'is_awarded'] = True
    history.loc[mask, ['goals_for', 'goals_against']] = 1
    home = history.loc[mask & history.side.eq('home')].iloc[0]
    release = home.kickoff_at + pd.Timedelta(days=7)
    common = dict(competition_id=home.competition_id, season_id=home.season_id, event_id=event,
                  home_id=home.team_id, away_id=home.opponent_id, kickoff_at=home.kickoff_at,
                  eligible=True, source='verified sporting record and later administrative decision')
    journal = pd.DataFrame([
        dict(common, revision=0, available_at=home.release, goals_home=1, goals_away=1, points_home=1, points_away=1),
        dict(common, revision=1, available_at=release, goals_home=0, goals_away=2, points_home=0, points_away=3)])
    return history, members, journal, home, release


def oracle(history, journal, team, cutoff, window, venue, timing):
    past = []
    for row in history.loc[history.team_id.eq(team)].sort_values(['kickoff_at', 'event_id']).itertuples():
        if row.kickoff_at >= cutoff or (venue != 'all' and row.side != venue):
            continue
        versions = journal.loc[journal.event_id.eq(row.event_id) & journal.available_at.le(cutoff)].sort_values('revision')
        if row.event_id in set(journal.event_id):
            if versions.empty or not versions.iloc[-1].eligible:
                continue
            v = versions.iloc[-1]
            gf, ga, points = ((v.goals_home, v.goals_away, v.points_home) if row.side == 'home'
                              else (v.goals_away, v.goals_home, v.points_away))
        else:
            if timing != 'kickoff_day' and row.release > cutoff:
                continue
            gf, ga = row.goals_for, row.goals_against
            points = 3 * int(gf > ga) + int(gf == ga)
        past.append((1, int(points == 3), int(points == 1), int(points == 0), points, gf, ga, gf - ga))
    selected = past[-window:] if window else past
    return [sum(r[i] for r in selected) for i in range(8)]


@pytest.mark.parametrize('timing', ['kickoff_day', 'available_at_cutoff'])
@pytest.mark.parametrize('venue', ['all', 'home', 'away'])
def test_versioned_replay_matches_independent_every_snapshot(timing, venue, tmp_path):
    h, members, journal, home, release = source()
    before, journal_before = h.copy(deep=True), journal.copy(deep=True)
    run = build_standings(h, team_seasons=members, result_versions=journal, normalized_ranks=[1, 2, 20],
                          venues=[venue], timing=timing, available_at='release' if timing != 'kickoff_day' else None)
    assert run.metadata['policy'] == (VERSIONED_DAILY_POLICY if timing == 'kickoff_day' else VERSIONED_POLICY)
    for (cutoff, window), table in run.snapshots.groupby(['recorded_at', 'window']):
        totals = {t: oracle(h, journal, t, cutoff, window, venue, timing) for t in members.team_id}
        keys = {t: (v[4], v[7], v[5], -v[6]) for t, v in totals.items()}
        for row in table.itertuples():
            assert [getattr(row, c) for c in TOTALS] == totals[row.team_id]
            assert row.rank == 1 + sum(k > keys[row.team_id] for k in keys.values())
    for publication in run.publications:
        assert len(publication['contributing_events']) == len(set(publication['contributing_events'])) == publication['eligible_result_count']
    run.save(tmp_path/'result')
    restored = StandingsRun.load(tmp_path/'result')
    pd.testing.assert_frame_equal(run.snapshots, restored.snapshots)
    pd.testing.assert_frame_equal(run.features(h, DerivedStandings([1, 2], side='both', venue=venue)),
                                  restored.features(h, DerivedStandings([1, 2], side='both', venue=venue)))
    pd.testing.assert_frame_equal(h, before)
    pd.testing.assert_frame_equal(journal, journal_before)


def test_boundary_and_replacement_does_not_make_old_event_recent():
    h, members, journal, home, release = source()
    query = h.loc[h.kickoff_at.eq(h.kickoff_at.max())]
    def at(t):
        return build_standings(h, team_seasons=members, result_versions=journal, queries=query,
                              cutoffs=t, available_at='release', normalized_ranks=[1, 2, 20])
    before, after = at(release-pd.Timedelta(1, 'ns')), at(release)
    # Two newer sporting matches already occupy the first two form slots.
    for w in (1, 2):
        pd.testing.assert_frame_equal(before.snapshots.loc[before.snapshots.window.eq(w)].reset_index(drop=True),
                                      after.snapshots.loc[after.snapshots.window.eq(w)].assign(recorded_at=before.snapshots.recorded_at.iloc[0]).reset_index(drop=True))
    cumulative_before = before.snapshots.loc[before.snapshots.window.eq(0)].set_index('team_id')
    cumulative_after = after.snapshots.loc[after.snapshots.window.eq(0)].set_index('team_id')
    assert cumulative_after.loc[home.team_id, 'points'] == cumulative_before.loc[home.team_id, 'points'] - 1
    assert cumulative_after.loc[home.opponent_id, 'points'] == cumulative_before.loc[home.opponent_id, 'points'] + 2
    assert cumulative_after.played.equals(cumulative_before.played)


def test_future_revision_poison_does_not_rewrite_earlier_snapshot():
    h, members, journal, home, release = source()
    query = h.loc[h.kickoff_at.eq(h.kickoff_at.max())]
    early = release-pd.Timedelta(1, 'ns')
    def run(v):
        return build_standings(h, team_seasons=members, result_versions=v, queries=query,
                              cutoffs=early, available_at='release', normalized_ranks=[1, 20])
    changed = journal.copy();changed.loc[changed.revision.eq(1), ['goals_home', 'goals_away']] = [999, 0]
    changed.loc[changed.revision.eq(1), ['points_home', 'points_away']] = [3, 0]
    baseline, poisoned = run(journal), run(changed)
    pd.testing.assert_frame_equal(baseline.snapshots, poisoned.snapshots)
    assert baseline.metadata['source_sha256'] != poisoned.metadata['source_sha256']


def test_initial_abandoned_result_and_reviewed_points_are_distinct():
    h, members, journal, home, release = source()
    journal.loc[journal.revision.eq(0), 'eligible'] = False
    journal.loc[:, ['goals_home', 'goals_away']] = 0
    query = h.loc[h.kickoff_at.eq(h.kickoff_at.max())]
    def at(t):return build_standings(h, team_seasons=members, result_versions=journal, queries=query, cutoffs=t, available_at='release')
    before, after = at(release-pd.Timedelta(1, 'ns')), at(release)
    assert home.event_id not in before.publications[0]['contributing_events']
    assert after.publications[0]['contributing_events'].count(home.event_id) == 1
    for t, points in [(home.team_id, 0), (home.opponent_id, 3)]:
        a = after.snapshots.loc[after.snapshots.window.eq(0)&after.snapshots.team_id.eq(t)].iloc[0]
        b = before.snapshots.loc[before.snapshots.window.eq(0)&before.snapshots.team_id.eq(t)].iloc[0]
        assert a.points-b.points == points and a.gf == b.gf and a.ga == b.ga


@pytest.mark.parametrize('fault', ['naive_time', 'missing_time', 'missing_source', 'wrong_home', 'wrong_kickoff',
 'duplicate_revision', 'revision_gap', 'same_time', 'invalid_eligible', 'invalid_points', 'fractional_score', 'unknown_event'])
def test_rejects_unknown_or_ambiguous_journal(fault):
    h, members, journal, home, release = source()
    j = journal.copy()
    if fault == 'naive_time':j['available_at'] = j.available_at.dt.tz_localize(None)
    elif fault == 'missing_time':j.loc[1, 'available_at'] = pd.NaT
    elif fault == 'missing_source':j.loc[1, 'source'] = ''
    elif fault == 'wrong_home':j.loc[1, 'home_id'] = 999
    elif fault == 'wrong_kickoff':j.loc[1, 'kickoff_at'] += pd.Timedelta(1, 's')
    elif fault == 'duplicate_revision':j.loc[1, 'revision'] = 0
    elif fault == 'revision_gap':j.loc[1, 'revision'] = 3
    elif fault == 'same_time':j.loc[1, 'available_at'] = j.available_at.iloc[0]
    elif fault == 'invalid_eligible':j['eligible'] = ['yes', 'no']
    elif fault == 'invalid_points':j.loc[1, ['points_home', 'points_away']] = [3, 3]
    elif fault == 'fractional_score':j['goals_home'] = [1., .5]
    else:j.loc[1, 'event_id'] = 999
    with pytest.raises(ValueError):build_standings(h, team_seasons=members, result_versions=j, timing='kickoff_day')


def test_unknown_admin_is_not_silently_retrospective():
    h, members, journal, home, release = source()
    other = h.loc[h.event_id.ne(home.event_id), 'event_id'].iloc[0]
    h.loc[h.event_id.eq(other), 'is_awarded'] = True
    with pytest.raises(ValueError, match='unknown provenance'):
        build_standings(h, team_seasons=members, result_versions=journal, timing='kickoff_day')


def test_empty_version_mode_and_no_journal_legacy_contract():
    h, members = population(4)
    journal = pd.DataFrame(columns=FIELDS)
    ordinary = build_standings(h, team_seasons=members, timing='kickoff_day', normalized_ranks=[1, 20])
    versioned = build_standings(h, team_seasons=members, timing='kickoff_day', result_versions=journal, normalized_ranks=[1, 20])
    pd.testing.assert_frame_equal(ordinary.snapshots, versioned.snapshots)
    # Explicit legacy daily policy intentionally ignores the later release.
    h, members, journal, home, release = source()
    for side, values in [('home', (0, 2)), ('away', (2, 0))]:
        take = h.event_id.eq(home.event_id)&h.side.eq(side)
        h.loc[take, ['standings_goals_for', 'standings_goals_against']] = values
    h['standings_result_available_at'] = pd.Series(pd.NaT, index=h.index, dtype='datetime64[ns, UTC]')
    h.loc[h.event_id.eq(home.event_id), 'standings_result_available_at'] = pd.Timestamp('2099-01-01', tz='UTC')
    legacy = build_standings(h, team_seasons=members, timing='kickoff_day')
    assert legacy.metadata['policy'] != VERSIONED_DAILY_POLICY
    assert any(home.event_id in p['contributing_events'] for p in legacy.publications)


def test_phase_context_consumers_keep_original_cutoff_rows(tmp_path):
    from xdiyo_analytics.graphs import build_phase_context, PhaseRankLookup
    from xdiyo_analytics.graphs.phase_context import PhaseRankFeatures, VERSIONED_DAILY_POLICY as PHASE_POLICY
    h, members, journal, home, release = source()
    h['source_league'] = 'Premier_League'
    h['source_season'] = '20_21'
    fixtures = h.loc[h.side.eq('home')].rename(columns={'team_id': 'home_id', 'opponent_id': 'away_id'}).reset_index(drop=True)
    cuts = fixtures.kickoff_at.dt.normalize()
    lookup = build_phase_context(h, members, fixtures, cutoffs=cuts, provenance={'review': 'version regression'}, result_versions=journal)
    assert lookup.policy == PHASE_POLICY
    lookup.save(tmp_path/'phase');loaded=PhaseRankLookup.load(tmp_path/'phase')
    np.testing.assert_array_equal(loaded.edges(h), lookup.edges(h))
    fixtures['_phase_cutoff'] = cuts
    specification = PhaseRankFeatures('D1', lookup.source_id, 1, PHASE_POLICY)
    np.testing.assert_array_equal(loaded.fixture(fixtures, specification), lookup.fixture(fixtures, specification))
    assert lookup.state()['provenance']['result_versions'][1]['available_at'] == release.isoformat()


@pytest.mark.parametrize('timing', ['available_at_cutoff', 'kickoff_day'])
@pytest.mark.parametrize('fault', ['timing', 'ordinary_timing', 'journal'])
def test_saved_versioned_contract_rejects_inconsistent_metadata(tmp_path, timing, fault):
    h, members, journal, home, release = source()
    run = build_standings(h, team_seasons=members, result_versions=journal, timing=timing,
                          available_at='release' if timing != 'kickoff_day' else None)
    run.save(tmp_path/'saved')
    path = tmp_path/'saved/standings.json';body=json.loads(path.read_text())
    if fault == 'timing':body['metadata']['timing'] = 'kickoff_day' if timing != 'kickoff_day' else 'available_at_cutoff'
    elif fault == 'ordinary_timing':body['metadata']['ordinary_result_timing'] = 'unrecognized'
    else:body['metadata']['result_versions'][1]['available_at'] = '2010-01-01T00:00:00+00:00'
    path.write_text(json.dumps(body),encoding='utf8')
    with pytest.raises(ValueError, match='schema|fingerprint'):
        StandingsRun.load(tmp_path/'saved')


@pytest.mark.parametrize('fault', ['remove_initial', 'empty', 'missing_source', 'wrong_team', 'same_time', 'changed_score'])
def test_saved_journal_semantics_and_source_identity_cannot_be_bypassed(tmp_path, fault):
    h, members, journal, home, release = source()
    run = build_standings(h, team_seasons=members, result_versions=journal, timing='kickoff_day')
    run.save(tmp_path/'saved')
    path = tmp_path/'saved/standings.json';body=json.loads(path.read_text());rows=body['metadata']['result_versions']
    if fault == 'remove_initial':rows.pop(0)
    elif fault == 'empty':rows.clear()
    elif fault == 'missing_source':rows[1].pop('source')
    elif fault == 'wrong_team':rows[1]['home_id'] = 999
    elif fault == 'same_time':rows[1]['available_at'] = rows[0]['available_at']
    else:rows[1]['goals_home'] = 100
    # Rehashing just the journal must not bypass schema/fixture/source checks.
    body['metadata']['result_versions_sha256'] = sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(body),encoding='utf8')
    with pytest.raises(ValueError):StandingsRun.load(tmp_path/'saved')
