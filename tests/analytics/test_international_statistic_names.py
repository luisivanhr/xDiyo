"""Native naming/orientation without changing values, masks or identities."""
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.data import SeasonData, homogenize_international_statistics, international_statistic_mapping
from xdiyo_analytics.data.statistics_backfill import STATISTICS, PERIODS
from xdiyo_analytics.histories import build_team_history

# Independent expectation, including the misleading native total-shots key.
EXPECTED = {
    'corners': ('Match overview', 'cornerKicks'),
    'yellow_cards': ('Match overview', 'yellowCards'),
    'red_cards': ('Match overview', 'redCards'),
    'xg': ('Match overview', 'expectedGoals'),
    'ball_possession': ('Match overview', 'ballPossession'),
    'total_shots': ('Match overview', 'totalShotsOnGoal'),
    'shots_on_target': ('Shots', 'shotsOnGoal'),
    'shots_off_target': ('Shots', 'shotsOffGoal'),
    'fouls': ('Match overview', 'fouls'),
    'goalkeeper_saves': ('Match overview', 'goalkeeperSaves'),
}


@pytest.mark.parametrize('field', EXPECTED)
@pytest.mark.parametrize('period,native', [('ft', 'ALL'), ('1h', '1ST'), ('2h', '2ND')])
@pytest.mark.parametrize('perspective', ['home', 'away'])
def test_each_field_period_side_and_lossless_roundtrip(field, period, native, perspective):
    group, key = EXPECTED[field]
    frame = pd.DataFrame({f'home_{field}_{period}': pd.Series([0, pd.NA, 63.25], dtype='Float64'),
                          f'away_{field}_{period}': pd.Series([pd.NA, 7, 0], dtype='Int64'),
                          'match_id': [2**53+1, 2**53+2, 2**53+3],
                          'home_win_opening_odds': [1.1, np.nan, 2.3]})
    frame.attrs = {'existing': {'do_not_change': [1]}}
    before = deepcopy(frame)
    out = homogenize_international_statistics(frame, perspective=perspective)
    for side in ('home', 'away'):
        role = 'team' if side == perspective else 'opponent'
        pd.testing.assert_series_equal(out[f'{role}::{native}::{group}::{key}::value'],
                                       frame[f'{side}_{field}_{period}'], check_names=False)
    pd.testing.assert_series_equal(out.home_win_opening_odds, frame.home_win_opening_odds)
    lineage = json.loads(json.dumps(out.attrs['international_statistic_names']))
    again = homogenize_international_statistics(out,perspective=perspective)
    pd.testing.assert_frame_equal(again,out)
    assert again.attrs == out.attrs
    with pytest.raises(ValueError,match='restore source names'):
        homogenize_international_statistics(out,perspective='away' if perspective=='home' else 'home')
    restored = out.rename(columns={v:k for k,v in lineage['mapping'].items()})
    pd.testing.assert_frame_equal(restored, frame)
    pd.testing.assert_frame_equal(frame, before)
    out.attrs['existing']['do_not_change'].append(2)
    assert frame.attrs == before.attrs


def fixture_data():
    f = pd.DataFrame(dict(event_id=[2**53+1, 2**53+2, 2**53+3], home_id=[10,20,10], away_id=[20,10,20],
        kickoff_utc=[1577836800,1578096000,1578355200], competition_id=17, season_id=1,
        round=[1,2,3], status='finished', home_score_current=[1,0,2], away_score_current=[0,0,3]))
    stats = []
    for period, native in PERIODS.items():
        for i,(field,(group,key)) in enumerate(EXPECTED.items()):
            for side in ('home','away'):
                values = [0., np.nan, i+0.25] if side == 'home' else [i+1., 0., np.nan]
                f[f'{side}_{field}_{period}'] = values
                for row,value in zip(f.itertuples(),values):
                    stats.append(dict(event_id=row.event_id, team_id=getattr(row,side+'_id'), side=side,
                                      period=native,group_name=group,key=key,value=value))
    return f,pd.DataFrame(stats)


def test_native_history_and_gat_preparation_parity_no_training():
    from xdiyo_analytics.graphs import TemporalGraphConfig, TemporalGraphModel
    from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
    f,stats = fixture_data()
    original = f.copy(deep=True)
    wide = build_team_history(SeasonData({'matches': f}, {}))
    native = build_team_history(SeasonData({'matches':f.drop(columns=list(international_statistic_mapping('home'))),
                                          'statistics':stats}, {}))
    assert wide.attrs['stat_columns'] == native.attrs['stat_columns']
    for c in native:
        # The legacy long-table merge materializes side as object on pandas 2.
        if c == 'side':
            pd.testing.assert_series_equal(wide[c].astype('string'),native[c].astype('string'))
        else:
            pd.testing.assert_series_equal(wide[c],native[c])
    with pytest.raises(ValueError,match='xG is excluded'):
        TemporalGraphConfig('UTC', statistics=tuple(native.attrs['stat_columns']))
    cols = tuple(sorted(c for c in native.attrs['stat_columns'] if 'expectedGoals' not in c))
    config = TemporalGraphConfig('UTC', statistics=cols)
    processor = GraphPreprocessor(config)
    np.testing.assert_array_equal(processor.numeric_edges(wide), processor.numeric_edges(native))
    membership = pd.DataFrame([dict(team_id=t,competition_id=17,season_id=1,
        entry_at=pd.Timestamp('2019-07-01',tz='UTC'),movement='retained') for t in (10,20)])
    model = TemporalGraphModel(config, membership_policy='explicit')
    store = model.event_store(wide,membership)
    baseline = model.event_store(native,membership)
    np.testing.assert_array_equal(processor.numeric_edges(store.history),processor.numeric_edges(baseline.history))
    pd.testing.assert_frame_equal(f,original)
    shuffled = build_team_history(SeasonData({'matches':f.sample(frac=1,random_state=3)},{}))
    pd.testing.assert_frame_equal(wide,shuffled)


def test_native_features_for_against_and_future_mutation():
    from xdiyo_analytics.features import RollingMean, Stat, ForAgainst, evaluate_features
    f,stats = fixture_data()
    wide = build_team_history(SeasonData({'matches':f},{}))
    native = build_team_history(SeasonData({'matches':f.drop(columns=list(international_statistic_mapping('home'))),
                                          'statistics':stats},{}))
    features = {side:RollingMean(ForAgainst(Stat('ALL','Match overview','totalShotsOnGoal'),side=side),window=2,min_periods=1)
                for side in ('for','against')}
    expected = evaluate_features(native,features)
    pd.testing.assert_frame_equal(evaluate_features(wide,features),expected)
    changed = f.copy()
    changed.loc[2,'home_total_shots_ft'] = 999
    changed = build_team_history(SeasonData({'matches':changed},{}))
    pd.testing.assert_frame_equal(evaluate_features(changed,features),expected)


def test_unknown_fields_odds_and_ambiguous_names_stay_unchanged():
    f = pd.DataFrame({'home_shots_ft':[4], 'home_attacks_ft':[3], 'away_win_closing_odds':[2.2],
                      'home_goals_ft':[1], 'arbitrary':[None]})
    out = homogenize_international_statistics(f,perspective='home')
    pd.testing.assert_frame_equal(f,out)
    assert out.attrs['international_statistic_names']['unresolved'] == ['home_shots_ft','home_attacks_ft']
    assert out.attrs['international_statistic_names']['mapping'] == {}


def test_collisions_rejected_even_when_values_agree():
    target = international_statistic_mapping('home')['home_corners_ft']
    with pytest.raises(ValueError,match='collision'):
        homogenize_international_statistics(pd.DataFrame({'home_corners_ft':[2],target:[2]}),perspective='home')
    f,stats = fixture_data()
    with pytest.raises(ValueError,match='collision'):
        build_team_history(SeasonData({'matches':f,'statistics':stats},{}))
    with pytest.raises(ValueError,match='Duplicate'):
        homogenize_international_statistics(pd.DataFrame([[1,2]],columns=['a','a']),perspective='home')
    with pytest.raises(ValueError,match='perspective'):
        international_statistic_mapping('unknown')


def test_single_source_side_preserves_counterpart_missingness():
    f,_ = fixture_data()
    f = f.drop(columns=[c for c in international_statistic_mapping('home') if c != 'home_corners_ft'])
    h = build_team_history(SeasonData({'matches':f},{}))
    team = 'team::ALL::Match overview::cornerKicks::value'
    opponent = team.replace('team::','opponent::')
    assert h.loc[h.side.eq('away'),team].isna().all()
    assert h.loc[h.side.eq('home'),opponent].isna().all()
    assert h.loc[h.side.eq('home'),team].iloc[0] == 0
