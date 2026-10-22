"""Chronology, data quality, transitions and vintage tests on tiny fixtures."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.ratings import GAPParameters, GAPRatingRun, GAPVintage, GAPSeason, RatingRun
from xdiyo_analytics.ratings.gap_data import gap_observations
from gap_fixtures import fixture, statistic, season_plan, vintage, build


def test_pre_round_freezing_repeated_team_and_future_invariance():
    h = pd.DataFrame(fixture(1,'2020-01-02') + fixture(2,'2020-01-03',yh=10,ya=1) + fixture(3,'2020-01-05'))
    cutoffs = pd.Series(pd.Timestamp('2020-01-02',tz='UTC'), index=h.index)
    cutoffs.iloc[4:] = pd.Timestamp('2020-01-05',tz='UTC')
    run = build(h)
    x = run.fixture_features(h,cutoffs=cutoffs,fields=('home','away','difference'))
    assert (x.iloc[:4] == 0).all().all()
    # First update from zero: home HA=.6, HD=.3; away AA=.24, AD=.48.
    # Second update predictions .54/.27, residuals 9.46/.73:
    # home HA=2.019, HD=.4095, away AA=.3276, AD=1.6152.
    assert x.iloc[4].to_dict() == pytest.approx(dict(gap_goals_home=1.8171, gap_goals_away=.36855, gap_goals_diff=1.44855))
    changed = h.copy()
    changed.loc[h.event_id.eq(3), ['goals_for','goals_against']] = 100.
    y = build(changed).fixture_features(changed, cutoffs=cutoffs,fields=('home','away','difference'))
    pd.testing.assert_frame_equal(x,y)
    # Alter a same-round result: other same-round predictions remain frozen.
    changed.loc[h.event_id.eq(1), ['goals_for','goals_against']] = 50.
    y = build(changed).fixture_features(changed, cutoffs=cutoffs)
    assert y.iloc[:4].eq(0).all().all()


def test_delayed_release_replays_kickoff_order_and_preserves_earlier_features():
    h = pd.DataFrame(fixture(1,'2020-01-02',release='2020-01-06',yh=4,ya=2) +
                     fixture(2,'2020-01-03',yh=10,ya=1) + fixture(3,'2020-01-05') + fixture(4,'2020-01-07'))
    # Query before and exactly at the late release of match 1; match 3 is unseen.
    h.loc[h.event_id.eq(3),'status'] = 'scheduled'
    run = build(h)
    query = h.loc[h.event_id.eq(4)].copy()
    early = run.fixture_features(query,cutoffs=pd.Timestamp('2020-01-05',tz='UTC'),fields=('home','away'))
    assert early.iloc[0].tolist() == pytest.approx([1.35,.135])
    at_release = run.fixture_features(query,cutoffs=pd.Timestamp('2020-01-06',tz='UTC'),fields=('home','away'))
    assert at_release.iloc[0].tolist() == pytest.approx([1.8171,.36855])
    changed = h.copy()
    changed.loc[h.event_id.eq(1),['goals_for','goals_against']] = 99.
    pd.testing.assert_frame_equal(early, build(changed).fixture_features(query,cutoffs=pd.Timestamp('2020-01-05',tz='UTC'),fields=('home','away')))


def test_postponed_round_is_governed_by_real_cutoff_and_availability():
    h = pd.DataFrame(fixture(1,'2020-01-02') + fixture(2,'2020-01-20'))
    h['round'] = 1
    run = build(h)
    # Nominal same round can have a new actual production cutoff for the postponement.
    assert run.fixture_features(h, cutoffs='kickoff_at').iloc[2,0] == pytest.approx(.27)
    frozen = run.fixture_features(h,cutoffs=pd.Timestamp('2020-01-01',tz='UTC'))
    assert frozen.eq(0).all().all()


def test_equal_kickoff_release_is_not_visible_until_strictly_after():
    h = pd.DataFrame(fixture(1,'2020-01-02',release='2020-01-02') + fixture(2,'2020-01-03'))
    run = build(h)
    q = h.iloc[2:]
    assert run.fixture_features(q,cutoffs=pd.Timestamp('2020-01-02',tz='UTC')).eq(0).all().all()
    assert run.fixture_features(q,cutoffs=pd.Timestamp('2020-01-02',tz='UTC')+pd.Timedelta(1,'ns')).iloc[0,0] == pytest.approx(.27)


def test_stat_and_league_independence_missing_and_zero():
    h = pd.DataFrame(fixture(1,'2020-01-02',yh=np.nan,ya=2,corners=(6,3)) + fixture(2,'2020-01-03',yh=0,ya=0) +
                     fixture(3,'2020-01-04') + fixture(4,'2020-01-02',league=20,yh=100,ya=0))
    plans = [season_plan(league=l,stat=s) for l in (10,20) for s in ('goals','corners')]
    run = build(h,stats=('goals','corners'),plans=plans)
    result = run.fixture_features(h,fields=('home','away'), cutoffs='kickoff_at')
    assert result.loc[h.event_id.eq(3),'gap_goals_home'].eq(0).all()
    assert result.loc[h.event_id.eq(2),'gap_corners_home'].iloc[0] == pytest.approx(.81)
    diag = run.diagnostics(h, cutoffs='kickoff_at')
    at = h.event_id.eq(3)
    assert diag.loc[at,'goals::home_games_seen'].eq(2).all()
    assert diag.loc[at,'goals::home_stat_updates'].eq(1).all()  # zero updated; missing skipped
    assert diag.loc[at,'corners::home_stat_updates'].eq(2).all()
    changed = h.copy()
    changed.loc[h.competition_id.eq(20),['goals_for','goals_against']] = 10000.
    pd.testing.assert_frame_equal(result.loc[h.competition_id.eq(10)],
        build(changed,stats=('goals','corners'),plans=plans).fixture_features(changed,fields=('home','away'), cutoffs='kickoff_at').loc[h.competition_id.eq(10)])


def test_burn_in_literal_p85_boundary_and_updates_not_suppressed():
    h = pd.DataFrame(sum((fixture(i,f'2020-01-{i+1:02d}') for i in range(1,10)),[]))
    run = build(h)
    d = run.diagnostics(h, cutoffs='kickoff_at').iloc[::2]
    assert d['goals::home_games_seen'].tolist() == list(range(9))
    assert d['goals::paper_burn_in_eligible'].tolist() == [False]*7+[True]*2
    assert d['goals::home_stat_updates'].tolist() == list(range(9))
    assert run.fixture_features(h, cutoffs='kickoff_at').notna().all().all()


def test_season_carryover_and_both_opposite_donor_rules():
    h = pd.DataFrame(fixture(1,'2020-01-02',h=1,a=2) + fixture(2,'2020-01-03',h=3,a=4,yh=8,ya=6) +
                     fixture(3,'2021-01-02',h=1,a=5,season=2021) + fixture(4,'2021-01-03',h=4,a=6,season=2021))
    plans = [season_plan(members=tuple((i,'initial') for i in (1,2,3,4))),
             season_plan(season=2021,entry='2021-01-01',previous=2020,
                         members=((1,'retained'),(4,'retained'),(5,'promoted'),(6,'relegated')),
                         relegated_out=(2,),promoted_out=(3,))]
    run = build(h,plans=plans)
    q = h.loc[h.season_id.eq(2021)]
    states = run.features(q,cutoffs=pd.Timestamp('2021-01-01',tz='UTC'),side='both')
    # Retained 1 and promoted 5 (donor 2) reproduce previous home/away pair.
    assert states.iloc[0].tolist() == pytest.approx([.6,.3,.2,.1, .16,.32,.24,.48])
    # Relegated 6 inherits former home team 3, all four components.
    assert states.loc[q.event_id.eq(4), 'goals::opponent::HA'].iloc[0] == pytest.approx(1.2)
    assert run.diagnostics(q,cutoffs=pd.Timestamp('2021-01-01',tz='UTC'))['goals::home_games_seen'].eq(0).all()
    # Empty donors are explicit failures, not a fallback to the league mean.
    with pytest.raises(ValueError,match='empty donor'):
        build(h,plans=[plans[0],replace(plans[1],relegated_out=())])
    with pytest.raises(ValueError,match='movement'):
        replace(plans[1],members=((1,'retained'),(5,'unknown')))
    with pytest.raises(ValueError,match='known'):
        replace(plans[1],known_at='2021-02-01')


def test_donor_state_frozen_before_late_previous_season_release():
    h = pd.DataFrame(fixture(1,'2020-12-30',release='2021-01-05') + fixture(2,'2021-01-02',season=2021))
    h.loc[h.event_id.eq(2),'status'] = 'scheduled'
    plans = [season_plan(),season_plan(season=2021,entry='2021-01-01',previous=2020,members=((1,'retained'),(2,'retained')))]
    run = build(h,plans=plans)
    q = pd.DataFrame(fixture(3,'2021-01-10',season=2021))
    assert run.fixture_features(q, cutoffs='kickoff_at').eq(0).all().all()


def test_parameter_vintages_route_chronologically_and_survive_reload(tmp_path):
    h = pd.DataFrame(fixture(1,'2020-01-02') + fixture(2,'2021-01-02',season=2021))
    plans = [season_plan(),season_plan(season=2021,entry='2021-01-01',previous=2020,members=((1,'retained'),(2,'retained')))]
    vintages = [vintage(end='2021-01-01'),vintage(start='2021-01-01',fit='2021-01-01',parameters=GAPParameters(.4,.75,.6))]
    run = build(h,plans=plans,vintages=vintages)
    x = run.fixture_features(h,fields=('home','away','difference'), cutoffs='kickoff_at')
    assert x.iloc[2,0] == pytest.approx(1.08)  # explicit new-vintage replay, not continuation
    run.save(tmp_path/'run')
    loaded = GAPRatingRun.load(tmp_path/'run')
    pd.testing.assert_frame_equal(loaded.snapshots,run.snapshots)
    pd.testing.assert_frame_equal(loaded.fixture_features(h,fields=('home','away','difference'), cutoffs='kickoff_at'),x)
    assert loaded.streams == run.streams and loaded.metadata == run.metadata
    with pytest.raises(FileNotFoundError):
        RatingRun.load(tmp_path/'run')
    with pytest.raises(FileExistsError):
        run.save(tmp_path/'run')
    with pytest.raises(ValueError,match='vintage'):
        loaded.fixture_features(h,cutoffs=pd.Timestamp('2019-12-30',tz='UTC'))
    late_only = build(h,plans=plans,vintages=vintages[1:])
    with pytest.raises(ValueError,match='vintage'):
        late_only.fixture_features(h, cutoffs='kickoff_at')
    late_only.save(tmp_path/'late')
    with pytest.raises(ValueError,match='vintage'):
        GAPRatingRun.load(tmp_path/'late').features(h, cutoffs='kickoff_at')
    (tmp_path/'run'/'state.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        GAPRatingRun.load(tmp_path/'run')


def test_vintage_validation_and_unknown_membership():
    with pytest.raises(ValueError,match='fit_cutoff'):
        vintage(start='2020-01-01',fit='2020-02-01')
    h = pd.DataFrame(fixture(1,'2020-01-02'))
    with pytest.raises(ValueError,match='nonoverlapping'):
        build(h,vintages=[vintage(),vintage()])
    with pytest.raises(ValueError,match='missing from known'):
        build(h,plans=[season_plan(members=((1,'initial'),(3,'initial')))])
    q = pd.DataFrame(fixture(2,'2020-01-03',h=9,a=2))
    run = build(h)
    assert run.fixture_features(q, cutoffs='kickoff_at').isna().all().all()
    assert run.diagnostics(q, cutoffs='kickoff_at')['goals::status'].eq('initialization_unavailable').all()


@pytest.mark.parametrize('kind',['mirror','missing_side','duplicate','negative','infinite','availability','team','status','no_availability'])
def test_adapter_validation(kind):
    h = pd.DataFrame(fixture(1,'2020-01-02'))
    if kind=='mirror': h.loc[1,'goals_for']=99
    elif kind=='missing_side': h=h.iloc[:1]
    elif kind=='duplicate': h=pd.concat([h,h])
    elif kind=='negative': h.loc[0,'goals_for']=-1
    elif kind=='infinite': h.loc[0,'goals_for']=np.inf
    elif kind=='availability': h.loc[0,'result_available_at']=pd.Timestamp('2020-01-01',tz='UTC')
    elif kind=='team': h.loc[0,'team_id']=2
    elif kind=='status': h.loc[0,'status']='scheduled'
    elif kind=='no_availability': h=h.drop(columns='result_available_at')
    with pytest.raises(ValueError):
        gap_observations(h,(statistic(),))


def test_explicit_availability_cannot_override_reviewed_completion_bound():
    h = pd.DataFrame(fixture(1,'2020-01-02',release='2020-01-05')+fixture(2,'2020-01-04'))
    run = build(h,available_at=h.kickoff_at)
    assert run.fixture_features(h, cutoffs='kickoff_at').eq(0).all().all()


def test_data_cutoff_excludes_later_results_and_no_parameter_fitting(monkeypatch):
    import xdiyo_analytics.ratings.gap_training as training
    monkeypatch.setattr(training,'tune_gap',lambda *a,**k: pytest.fail('Unexpected fitting'))
    h = pd.DataFrame(fixture(1,'2020-01-02')+fixture(2,'2020-01-04'))
    run = build(h,cutoff='2020-01-01')
    assert run.fixture_features(h, cutoffs='kickoff_at').eq(0).all().all()
    assert run.snapshots.recorded_at.max() <= pd.Timestamp('2020-01-01',tz='UTC')


def test_input_permutation_large_ids_and_duplicate_index():
    big = 2**53+1
    h = pd.DataFrame(fixture(big,'2020-01-02',h=big+2,a=big+4)+fixture(big+1,'2020-01-04',h=big+2,a=big+4))
    h.index = [0,0,1,1]
    plans = [season_plan(members=((big+2,'initial'),(big+4,'initial')))]
    run = build(h,plans=plans)
    x = run.fixture_features(h, cutoffs='kickoff_at')
    shuffled = h.iloc[[3,1,2,0]]
    y = build(shuffled,plans=plans).fixture_features(shuffled, cutoffs='kickoff_at')
    assert y.iloc[:,0].tolist() == [x.iloc[i,0] for i in [3,1,2,0]]
    assert set(run.snapshots.team_id) == {big+2,big+4}


def test_missing_release_does_not_become_kickoff_and_unknown_cutoff_is_missing():
    h = pd.DataFrame(fixture(1,'2020-01-02')+fixture(2,'2020-01-04'))
    h.loc[h.event_id.eq(1),'result_available_at'] = pd.NaT
    run = build(h)
    assert run.fixture_features(h, cutoffs='kickoff_at').eq(0).all().all()
    times = h.kickoff_at.copy()
    times.iloc[:2] = pd.NaT
    assert run.fixture_features(h,cutoffs=times).iloc[0].isna().all()
    with pytest.raises(ValueError,match='follow target'):
        run.fixture_features(h,cutoffs=h.kickoff_at+pd.Timedelta(days=1))


def test_first_coverage_season_is_separate_for_each_statistic():
    h = pd.DataFrame(fixture(1,'2020-01-02',corners=(np.nan,np.nan)) + fixture(2,'2021-01-02',season=2021))
    plans = [season_plan(),season_plan(season=2021,entry='2021-01-01',previous=2020,members=((1,'retained'),(2,'retained'))),
             season_plan(season=2021,entry='2021-01-01',stat='corners')]
    run = build(h,stats=('goals','corners'),plans=plans)
    result = run.fixture_features(h,fields=('home',), cutoffs='kickoff_at')
    assert result.iloc[:2].gap_corners_home.isna().all()
    assert result.iloc[2].gap_corners_home == 0
    assert result.iloc[2].gap_goals_home == pytest.approx(.54)


def test_multi_donor_componentwise_mean():
    h = pd.DataFrame(fixture(1,'2020-01-02',h=1,a=2)+fixture(2,'2020-01-03',h=3,a=4,yh=8,ya=6)+
                     fixture(3,'2021-01-02',h=5,a=6,season=2021))
    plans = [season_plan(members=tuple((i,'initial') for i in (1,2,3,4))),
             season_plan(season=2021,entry='2021-01-01',previous=2020,members=((5,'promoted'),(6,'relegated')),
                         relegated_out=(1,3),promoted_out=(2,4))]
    run = build(h,plans=plans)
    state = run.features(h.iloc[4:],cutoffs=pd.Timestamp('2021-01-01',tz='UTC'),side='for').iloc[0]
    assert state.tolist() == pytest.approx([.9,.6,.3,.2])


def test_subsequent_season_cannot_silently_reset_or_switch_parameters_midseason():
    h = pd.DataFrame(fixture(1,'2020-01-02')+fixture(2,'2021-01-02',season=2021))
    with pytest.raises(ValueError,match='only first-stat'):
        build(h,plans=[season_plan(),season_plan(season=2021,entry='2021-01-01')])
    one_season = pd.DataFrame(fixture(1,'2020-01-02')+fixture(2,'2020-08-01'))
    with pytest.raises(ValueError,match='target season'):
        build(one_season,vintages=[vintage(end='2020-07-01'),vintage(start='2020-07-01',fit='2020-07-01')])


def test_later_vintage_parameters_do_not_change_earlier_training_features():
    h = pd.DataFrame(fixture(1,'2020-01-02')+fixture(2,'2020-02-01')+fixture(3,'2021-01-02',season=2021))
    plans=[season_plan(),season_plan(season=2021,entry='2021-01-01',previous=2020,members=((1,'retained'),(2,'retained')))]
    old=vintage(end='2021-01-01')
    new=vintage(start='2021-01-01',fit='2021-01-01')
    before=build(h,plans=plans,vintages=[old,new]).fixture_features(h, cutoffs='kickoff_at')
    after=build(h,plans=plans,vintages=[old,replace(new,parameters=(('goals',GAPParameters(.9,.1,.95)),))]).fixture_features(h, cutoffs='kickoff_at')
    pd.testing.assert_frame_equal(before.iloc[:4],after.iloc[:4])
    assert before.iloc[4,0] != after.iloc[4,0]


def test_boolean_counts_and_missing_status_are_invalid():
    h = pd.DataFrame(fixture(1,'2020-01-02'))
    h['goals_for'] = h.goals_for.astype(object)
    h.loc[0,'goals_for'] = True
    with pytest.raises(ValueError,match='Boolean'):
        build(h)
    h = pd.DataFrame(fixture(1,'2020-01-02'))
    h.loc[0,'status'] = None
    with pytest.raises(ValueError):
        build(h)
