"""Local civil-day cutoffs and prepared population identity."""
from copy import deepcopy

import pandas as pd
import pytest

from xdiyo_analytics.data.league_day import day_boundary
from xdiyo_analytics.features import DerivedStandings
from xdiyo_analytics.ratings import StandingsRun, build_standings
from xdiyo_analytics.ui.recipe import prepare_recipe, run_recipe, node
from test_ui_recipe_cutoffs import daily_recipe
from test_native_query_preflight import publication, recipe as native_recipe


def test_calendar_completion_dst_and_multiple_competitions():
    frame = pd.DataFrame(dict(competition_id=[17, 11621, 17],
                              completion_date=[None, None, '2025-04-03']))
    times = pd.Series(pd.to_datetime(['2025-03-30T23:30Z'] * 3))
    result = day_boundary(frame, times, 'native')
    assert result.tolist() == list(pd.to_datetime(['2025-03-30T23:00Z', '2025-03-30T06:00Z',
                                                   '2025-03-30T23:00Z']))
    assert day_boundary(frame, times, 'native', after_match=True).iloc[-1] == pd.Timestamp('2025-04-03T23:00Z')
    with pytest.raises(ValueError, match='Missing league timezone'):
        day_boundary(frame, times, {'17': 'Europe/London'})


def test_prepared_recipe_inputs_differ_but_estimator_and_reports_can_change(daily_recipe, monkeypatch):
    recipe, _ = daily_recipe
    prepared = prepare_recipe(recipe)
    changed = deepcopy(recipe)
    changed['features']['corners_mean']['params']['window'] = 1
    with pytest.raises(ValueError, match='features'):
        run_recipe(changed, prepared=prepared)
    changed = deepcopy(recipe)
    changed['cutoff_hours'] = 2
    with pytest.raises(ValueError, match='cutoff_hours'):
        run_recipe(changed, prepared=prepared)
    recipe['model']['params']['alpha'] = 0.5
    recipe['post_reporters'] = {}
    monkeypatch.setattr('xdiyo_analytics.experiments.football.FootballExperiment.run',
                        lambda self, actual, **options: actual)
    assert run_recipe(recipe, prepared=prepared) is prepared


def test_daily_tabular_recipe_and_saved_standings(export,tmp_path):
    publication(export)
    r = native_recipe(export)
    r['league_timezones'] = {'17': 'America/Mexico_City'}
    r['features'] = {'rank': node('features.DerivedStandings')}
    r['split'] = node('splits.TemporalSplit', train_size=2, test_size=1, unit='kickoffs')
    prepared = prepare_recipe(r)
    h = prepared.outputs['history']
    for fold in prepared.split_plan.folds:
        meta = prepared.dataset.metadata
        cut = meta.kickoff_at.iloc[fold.test].min().tz_convert('America/Mexico_City').normalize()
        assert fold.metadata['fit_at'] == cut
        assert (meta.kickoff_at.iloc[fold.train] < cut).all()
    from xdiyo_analytics.data import load_season
    members = load_season(export.root, 'Premier_League_24_25', tables=['team_seasons'])['team_seasons']
    run = build_standings(h,team_seasons=members,timing='kickoff_day',league_timezones=r['league_timezones'])
    earlier = h.kickoff_at - pd.Timedelta(days=3)
    frozen = build_standings(h,team_seasons=members,timing='kickoff_day',cutoffs=earlier,league_timezones=r['league_timezones'])
    assert set(frozen.snapshots.recorded_at) == set(day_boundary(h, earlier, r['league_timezones']))
    run.save(tmp_path/'standings')
    saved = StandingsRun.load(tmp_path/'standings')
    assert saved.metadata['calendar_policy'] == 'league_day_v1'
    pd.testing.assert_frame_equal(run.features(h,DerivedStandings()),saved.features(h,DerivedStandings()))


def test_only_prior_local_completion_days_enter_table():
    from test_derived_standings import population
    h, roster = population(4)
    ids = h.event_id.drop_duplicates().tolist()
    times = ['2025-01-02T02:00Z', '2025-01-01T17:00Z', '2025-01-02T17:00Z',
             '2025-01-02T23:30Z', '2025-01-04T12:00Z']
    for event, instant in zip(ids, times):
        h.loc[h.event_id.eq(event), 'kickoff_at'] = pd.Timestamp(instant)
    h.loc[~h.event_id.isin(ids[:5]), 'kickoff_at'] += pd.DateOffset(years=10)
    h['completion_date'] = None
    h.loc[h.event_id.eq(ids[1]), 'completion_date'] = '2025-01-03'
    qs = h[h.event_id.isin(ids[2:5])]
    result = build_standings(h, team_seasons=roster, queries=qs,
                             timing='kickoff_day', league_timezones={17:'America/Mexico_City'})
    observed = {pd.Timestamp(row['cutoff']):set(row['contributing_events']) for row in result.publications}
    assert observed == {pd.Timestamp('2025-01-02T06:00Z'):{ids[0]},
                        pd.Timestamp('2025-01-04T06:00Z'):set(ids[:4])}
