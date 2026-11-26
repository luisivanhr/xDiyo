import pandas as pd
import pytest

from xdiyo_analytics.data.scoreline_history import ARCHIVE_EVENT, prepare
from xdiyo_analytics.features.transitions import TransitionContext
from xdiyo_analytics.ratings.bayesian_replay import _events


def _fixture(key, league, year, day, home, away, *, native=False, replay=True, competition=None):
    dt = pd.Timestamp(day, tz='UTC')
    return dict(fixture_key=f'fixture:{key}', league=league, season=f'{year}/{year+1}', phase=None,
                competition_id=competition if competition is not None else {'first':1,'second':2}[league],
                season_id=year if native else None, event_id=key if native else None,
                source='SofaScore' if native else 'archive', source_fixture_id=str(key),
                kickoff_utc=dt+pd.Timedelta(hours=18) if native else pd.NaT,
                source_reported_date=dt.date(), replay_available_at=dt+pd.Timedelta(days=1),
                replay_eligible=replay, home_model_team_id=home, away_model_team_id=away,
                home_team=f'club {home}', away_team=f'club {away}', home_goals=2, away_goals=1,
                date_precision='second' if native else 'date_only', score_basis='as_reported',
                replay_basis='source_reported_prior_day', team_identity='reviewed')


def _sample():
    return pd.DataFrame([
        _fixture(10,'second',2017,'2017-08-01',11,12),
        _fixture(11,'first',2019,'2019-08-01',12,13),
        _fixture(12,'second',2019,'2019-08-01',14,15),
        _fixture(13,'second',2020,'2020-08-01',11,12,native=True),
        _fixture(14,'second',2020,'2020-08-02',14,16,native=True),
        _fixture(15,'first',2020,'2020-08-02',13,19,native=True),
        _fixture(16,'second',2020,'2020-08-03',2000000001,16,native=True,replay=False),
    ])


LEAGUES = {'first': ('x', 1), 'second': ('x', 2)}


def test_prepared_history_contract_and_no_same_day():
    scores = _sample()
    history, moves, deferred = prepare(scores, {(1,2020,19):'other_entry'}, LEAGUES)
    events, _, effective = _events(history, None)  # schema validation only, no rating updates
    assert effective.eq(history.replay_available_at).all()
    assert len(events) == 6
    assert list(deferred.fixture_key) == ['fixture:16']
    archive = history[history.source.eq('archive')]
    assert archive.kickoff_utc.isna().all()
    assert archive.event_id.ge(ARCHIVE_EVENT).all()
    assert archive.kickoff_at.eq(archive.prediction_at).all()
    assert (history.replay_available_at >= history.prediction_at + pd.Timedelta(days=1)).all()
    now = moves[moves.season_id.eq(2020)].set_index('team_id')
    assert now.loc[11,'movement'] == 'promoted'
    assert pd.isna(now.loc[11,'previous_season_id'])
    assert now.loc[12,'movement'] == 'relegated'
    assert now.loc[12,'previous_season_id'] == history.loc[history.fixture_key.eq('fixture:11'),'season_id'].iloc[0]
    assert now.loc[14,'movement'] == 'retained'
    assert now.loc[16,'movement'] == 'unknown'
    assert now.loc[19,'movement'] == 'other_entry'
    ctx = TransitionContext(history, cutoffs='prediction_at', available_at='replay_available_at',
                            team_seasons=moves)
    assert ctx.records[(2,2020,11)]['movement'] == 'promoted'
    assert 'evidence' not in moves
    shuffled = scores.sample(frac=1, random_state=5)
    other, _, _ = prepare(shuffled, {(1,2020,19):'other_entry'}, LEAGUES)
    assert history.equals(other)


def test_no_accidental_admission_or_self_opponent():
    scores = _sample()
    scores.loc[0,'away_model_team_id'] = 11
    with pytest.raises(ValueError, match='self-opposed'):
        prepare(scores, {}, LEAGUES)
    scores.loc[0,'away_model_team_id'] = 12
    scores.loc[0,'replay_available_at'] -= pd.Timedelta(days=1)
    with pytest.raises(ValueError, match='prior-day'):
        prepare(scores, {}, LEAGUES)


def test_unsplit_archive_does_not_pick_arbitrary_provider_phase():
    scores = _sample()
    scores.loc[1,'competition_id'] = None
    extra = _fixture(17,'first',2020,'2020-12-01',18,19,native=True,competition=88)
    scores.loc[len(scores)] = extra
    history, _, _ = prepare(scores, {}, LEAGUES)
    assert int(history[history.fixture_key.eq('fixture:11')].competition_id.iloc[0]) >= 1 << 60
    assert set(history[history.source.eq('SofaScore')].competition_id) == {1,2,88}


def test_no_predecessor_pointer_for_club_held_out_of_prepared_history():
    scores = _sample()
    scores.loc[1, 'replay_eligible'] = False
    scores.loc[len(scores)] = _fixture(17, 'first', 2019, '2019-09-01', 80, 81)
    history, moves, _ = prepare(scores, {}, LEAGUES)
    club = moves[moves.season_id.eq(2020) & moves.team_id.eq(12)].iloc[0]
    assert club.movement == 'relegated'
    assert pd.isna(club.previous_season_id) and pd.isna(club.previous_competition_id)
    TransitionContext(history, cutoffs='prediction_at', team_seasons=moves)
