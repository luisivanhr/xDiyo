from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from test_temporal_graph import history, sample, context
from xdiyo_analytics.graphs import TemporalEventStore, TemporalGraphConfig
from xdiyo_analytics.graphs.actual_date_context import build_actual_date_context
from xdiyo_analytics.graphs.research_profiles import GATActualDateBNV1, GATResearchBNV1, profile_key
from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
from xdiyo_analytics.graphs.phase_context import PhaseRankLookup


def cfg():
    return GATActualDateBNV1()


@pytest.mark.parametrize('kick,expected', [
    ('2025-04-27T12:00Z','2025-04-26T22:00Z'),
    ('2025-02-27T19:45Z','2025-02-26T23:00Z'),
    ('2024-03-31T15:00Z','2024-03-30T23:00Z'),
    ('2024-10-27T15:00Z','2024-10-26T22:00Z')])
def test_playing_day_dst_and_postponed_round(kick, expected):
    s = sample(); f = s.schedule
    f['round'] = pd.array([pd.NA]*len(f),dtype='Int64')
    f.loc[0,'kickoff_at'] = pd.Timestamp(kick)
    f['round_issue_at'] = pd.Timestamp('2019-01-01',tz='UTC')
    s = TemporalEventStore(s.history,s.team_seasons,schedule=f)
    got = s.issue_times(f.iloc[[0]],cfg()).iloc[0]
    assert got == pd.Timestamp(expected)
    assert cfg().labels_known_at(f.kickoff_at).iloc[0] == pd.Timestamp(kick)
    assert f['round'].isna().all()
    wrong = f.iloc[[0]].assign(issuance_at=pd.Timestamp('2019-01-01',tz='UTC'))
    with pytest.raises(ValueError,match='local midnight'): s.issue_times(wrong,cfg())
    wrong = f.iloc[[0]].copy(); wrong.kickoff_at += pd.Timedelta(days=1)
    with pytest.raises(ValueError,match='disagrees'): s.issue_times(wrong,cfg())


def test_old_modes_and_profile_unchanged():
    s=sample(); f=s.schedule
    legacy=GATResearchBNV1()
    assert legacy.forecast_policy=='by_round' and len(legacy.statistics)==4
    assert legacy.labels_known_at(f.kickoff_at).iloc[0] > f.kickoff_at.iloc[0]
    assert len(cfg().statistics)==6 and profile_key(cfg())=='graphs.GATActualDateBNV1'
    pd.testing.assert_series_equal(s.issue_times(f,TemporalGraphConfig('Europe/Berlin',forecast_policy='one_day_ahead')),
                                  f.kickoff_at-pd.Timedelta(hours=24))
    from xdiyo_analytics.graphs import RetainedGATExecution
    with pytest.raises(ValueError,match='exact supported'):
        RetainedGATExecution().validate_config(cfg())
    with pytest.raises(ValueError,match='optional history'):
        replace(cfg(),temporal_mode='historical')


def test_strict_result_and_statistic_boundaries():
    s=sample(); h=s.history
    boundary=pd.Timestamp('2020-01-10T23:00Z')
    h['result_available_at']=h.kickoff_at
    h.loc[h.event_id.eq(1),'result_available_at']=boundary
    store=TemporalEventStore(h,s.team_seasons)
    assert 1 not in set(store.snapshot(boundary,cfg()).edges.event_id)
    assert 1 in set(store.snapshot(boundary,TemporalGraphConfig('Europe/Berlin')).edges.event_id)
    col=cfg().statistics[0]; pair=col.replace('team::','opponent::')
    h[col]=h[pair]=3.; h['available_at::'+col]=boundary
    spec=replace(cfg(),statistics=(col,))
    pre=GraphPreprocessor(spec)
    assert np.isnan(pre._available_numeric_edges(h,boundary)[:,2]).all()
    assert np.isfinite(pre._available_numeric_edges(h,boundary+pd.Timedelta(1,'ns'))[:,2]).all()


def table():
    s=sample(); f=s.schedule
    f['phase_key']='regular_season'
    return f


def test_context_future_mutation_invariance_and_saved_round_separation(tmp_path):
    f=table(); a=build_actual_date_context(f,timezone='Europe/Berlin',provenance={})
    changed=f.copy(); changed.loc[changed.event_id.ge(5),'goals_for']=99
    changed.loc[changed.event_id.ge(5),'home_id']=999
    changed['round']=999
    b=build_actual_date_context(changed,timezone='Europe/Berlin',provenance={})
    pd.testing.assert_frame_equal(a.frame.iloc[:4],b.frame.iloc[:4])
    assert a.frame.countdown.isna().all()
    assert a.frame.iloc[0].filter(like='rank').isna().all()
    assert a.frame.iloc[1].filter(like='rank').notna().all()
    a.save(tmp_path); assert PhaseRankLookup.load(tmp_path).source_id==a.source_id


def test_late_result_new_teams_and_unknown_stages_do_not_leak():
    f=table(); f['result_available_at']=f.kickoff_at
    f.loc[0,'result_available_at']=f.loc[2,'kickoff_at']
    f.loc[1,'home_id']=999
    f.loc[2,'phase_key']='knockout'
    a=build_actual_date_context(f,timezone='Europe/Berlin',provenance={}).frame
    assert a.loc[1,'home_rank_1'] != a.loc[1,'home_rank_1']
    assert a.iloc[2].filter(like='rank').isna().all()


def test_training_selection_boundary_strictness():
    from xdiyo_analytics.graphs.adapter import _Backend
    backend=_Backend(cfg(),sample(),17,'cpu')
    backend.trained_through=pd.Timestamp('2022-06-30T22:00Z')
    with pytest.raises(ValueError,match='boundary'):
        backend._boundary(pd.Series([backend.trained_through]))
    backend._boundary(pd.Series([backend.trained_through+pd.Timedelta(days=1)]))


def test_postponed_graph_uses_intervening_history_and_excludes_same_day():
    s=sample(); f=s.schedule
    f['round']=1
    f['round_issue_at']=f.kickoff_at.min()
    store=TemporalEventStore(s.history,s.team_seasons,schedule=f)
    target=f.iloc[[-1]]
    now=store.issue_times(target,cfg()).iloc[0]
    old=store.issue_times(target,GATResearchBNV1()).iloc[0] if 'round_stage' in f else store.issue_times(target,TemporalGraphConfig('Europe/Berlin')).iloc[0]
    assert store.snapshot(old,cfg()).edges.empty
    current=store.snapshot(now,cfg())
    assert set(current.edges.event_id)==set(range(1,9))
    assert (current.edges.kickoff_at<now).all()
