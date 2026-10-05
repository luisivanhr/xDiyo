"""Independent weighted-observation oracles for historical third moments."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from scipy.stats import skew

from xdiyo_analytics.features import (RollingSkewness, WarmStart, SeededEMA,
    Hard, ObservationCount, LinearFade, League, LeaveOneOut, ForAgainst, H2H,
    FeatureBankPreset, evaluate_features)
from xdiyo_analytics.features.moments import mix_moments, sample_moments, skewness
from transition_samples import A, B, C, D, STAT, OWN, START, history_from_games, league_games, movement
from test_ui_workflow import ui_recipe


def weighted(values, weights):
    x, w = np.asarray(values, float), np.asarray(weights, float)
    w = w/w.sum()
    mean = np.dot(w, x)
    v = np.dot(w, (x-mean)**2)
    return np.dot(w, (x-mean)**3)/v**1.5 if v > 0 else np.nan


def toy():
    return history_from_games(
        [dict(day=i, season=1, values=(v, 2*v+1)) for i,v in enumerate([1,2,8])]
        + [dict(day=10+2*i, season=2, round=i+1, values=(v, 2*v+1))
           for i,v in enumerate([4,14,3,1000])])


def test_mixture_third_moment_and_translation():
    a, b = [1,2,11], [-5,3,4,8]
    for weight in [0, .17, .5, .9, 1]:
        moments = mix_moments(sample_moments(a), sample_moments(b), weight)
        assert skewness(moments) == pytest.approx(weighted(a+b, [weight/3]*3+[(1-weight)/4]*4))
    # Direct raw-cube subtraction would lose this small spread at a large offset.
    assert skewness(sample_moments(np.array(a)+1e10)) == pytest.approx(skew(a), abs=2e-6)


def test_team_history_is_past_only_and_window_counts_missing_matches():
    h=toy(); op=RollingSkewness(STAT, window=4)
    r=evaluate_features(h, {'s':op}).s
    assert pd.isna(r.iloc[4])
    assert r.iloc[6] == pytest.approx(skew([1,2,8], bias=True))
    assert r.iloc[10] == pytest.approx(skew([2,8,4,14], bias=True))
    h.loc[6, OWN]=np.nan
    r=evaluate_features(h, {'s':RollingSkewness(STAT, window=3)}).s
    assert pd.isna(r.iloc[10])  # Do not refill with an older observation.
    h.loc[12:, OWN]=1e8
    assert pd.isna(evaluate_features(h, {'s':RollingSkewness(STAT,3)}).s.iloc[10])


@pytest.mark.parametrize('values', [[1,1,1], [np.nan,1,2], [np.inf,1,2], [1,2]])
def test_undefined_windows(values):
    h=history_from_games([dict(values=(v,v)) for v in values+[99]])
    assert pd.isna(evaluate_features(h, {'s':RollingSkewness(STAT,5,min_periods=1)}).s.iloc[-2])


def test_late_release_simultaneous_cutoffs_and_shuffle():
    h=toy(); h.loc[4:5,'available_at']=START+pd.Timedelta(days=11)
    r=evaluate_features(h, {'s':RollingSkewness(STAT,5)}, available_at='available_at')
    assert pd.isna(r.s.iloc[6])
    assert r.s.iloc[8] == pytest.approx(skew([1,2,8,4]))
    cut=h.kickoff_at.copy(); cut.iloc[8:10]=START+pd.Timedelta(days=10)
    frozen=evaluate_features(h, {'s':RollingSkewness(STAT,5)}, available_at='available_at',cutoffs=cut)
    assert pd.isna(frozen.s.iloc[8])
    h.loc[6:7,'kickoff_at']=h.kickoff_at.iloc[4]
    same=evaluate_features(h, {'s':RollingSkewness(STAT,5)})
    assert pd.isna(same.s.iloc[6])
    shuffled=h.sample(frac=1,random_state=9)
    pd.testing.assert_frame_equal(evaluate_features(shuffled,{'s':RollingSkewness(STAT,5)}).sort_index(),same.sort_index())


@pytest.mark.parametrize('exclude,expected', [
    (None,[2,8,4,6,10,20,30,40]),
    ('team_contributions',[8,4,6,20,30,40]),
    ('fixtures',[4,6,30,40]),
])
def test_league_loo_removes_all_moments(exclude, expected):
    h=history_from_games(league_games())
    pop=League(STAT)
    source=LeaveOneOut(pop,exclude) if exclude else pop
    actual=evaluate_features(h,{'s':RollingSkewness(source,2)}).s.iloc[8]
    assert actual == pytest.approx(skew(expected, bias=True))


def test_loo_match_totals_and_against_h2h():
    h=history_from_games(league_games())
    pop=LeaveOneOut(League(STAT,unit='match'),exclude='fixtures')
    assert pd.isna(evaluate_features(h,{'s':RollingSkewness(pop,2)}).s.iloc[8])
    h=toy()
    actual=evaluate_features(h,{'s':RollingSkewness(H2H(ForAgainst(STAT,'against')),3)}).s.iloc[6]
    assert actual == pytest.approx(skew([3,5,17]))


@pytest.mark.parametrize('handoff', [Hard(10),ObservationCount(2),LinearFade(start=0,rounds=4)])
def test_uniform_seed_update_and_fade_are_moment_mixtures(handoff):
    h=toy(); p=SeededEMA(mode='uniform',alpha=.25,handoff=handoff)
    expr=WarmStart(RollingSkewness(STAT,3),p)
    actual=evaluate_features(h, {'s':expr})
    assert actual.s.iloc[6] == pytest.approx(skew([1,2,8]))
    # At the third fixture two new observations have been incorporated.
    vals=[1,2,8,4,14]
    weights=[.75**2/3]*3+[.25*.75,.25]
    w=handoff.weight(2,2)
    assert actual.s.iloc[10] == pytest.approx(weighted(vals+[8,4,14],
                                      [v*w for v in weights]+[(1-w)/3]*3))
    audit=[a for a in actual.attrs['warm_start_audit']['s'] if a['team_id']==A and a['season_id']==2][0]
    assert audit['seed_third_central_moment'] == pytest.approx(np.mean((np.array([1,2,8])-11/3)**3))


def test_zero_handoff_missing_and_alpha_one():
    h=toy(); op=RollingSkewness(STAT,3)
    base=evaluate_features(h, {'s':op})
    for mode in ['legacy','uniform','w_league_prior']:
        pd.testing.assert_frame_equal(evaluate_features(h,{'s':WarmStart(op,SeededEMA(mode=mode,handoff=Hard(0)))}),base)
    h.loc[6,OWN]=np.nan
    p=SeededEMA(mode='uniform',handoff=Hard(10))
    r=evaluate_features(h,{'s':WarmStart(op,p)})
    assert r.s.iloc[8] == pytest.approx(r.s.iloc[6])
    r=evaluate_features(toy(),{'s':WarmStart(op,replace(p,alpha=1))})
    assert pd.isna(r.s.iloc[8])


def test_uniform_seed_honors_boundary_releases_and_sample_policy_is_irrelevant():
    h=toy(); p=SeededEMA(mode='uniform',handoff=Hard(10))
    op=RollingSkewness(STAT,3)
    # A late predecessor is unavailable at the frozen entry boundary. With
    # only two seed observations, fallback remains ordinary all season.
    h.loc[4:5,'available_at']=START+pd.Timedelta(days=11)
    ordinary=evaluate_features(h, {'s':op}, available_at='available_at')
    actual=evaluate_features(h, {'s':WarmStart(op,p)}, available_at='available_at')
    pd.testing.assert_frame_equal(actual, ordinary)
    population=evaluate_features(toy(),{'s':WarmStart(op,p)})
    corrected=evaluate_features(toy(),{'s':WarmStart(op,replace(p,variance_estimator='weighted_sample'))})
    pd.testing.assert_frame_equal(corrected,population)


@pytest.mark.parametrize('kwargs', [{'window':0},{'window':True},{'min_periods':0},{'min_periods':True},{'window':3,'min_periods':4}])
def test_invalid_team_windows(kwargs):
    with pytest.raises(ValueError):
        evaluate_features(toy(),{'s':RollingSkewness(STAT,**kwargs)})


@pytest.mark.parametrize('exclude', ['team_contributions','fixtures'])
def test_legacy_loo_seed_and_new_observations(exclude):
    games=[dict(day=i,season=1,home=C,away=D,values=pair) for i,pair in enumerate([(1,3),(2,8),(4,15)])]
    games += [dict(day=4,season=1,home=A,away=B,values=(90,7)),
              dict(day=10,season=2,round=1,home=A,away=B,values=(100,11)),
              dict(day=12,season=2,round=2,home=A,away=C,values=(999,999))]
    h=history_from_games(games)
    source=LeaveOneOut(League(STAT,schedule='kickoff',window_unit='matches'),exclude)
    expr=WarmStart(RollingSkewness(source,5),SeededEMA(alpha=.25,handoff=Hard(10)))
    result=evaluate_features(h,{'s':expr}).s
    prior=[1,3,2,8,4,15]+([7] if exclude=='team_contributions' else [])
    assert result.iloc[8] == pytest.approx(skew(prior))
    expected=weighted(prior+[11], [.75/len(prior)]*len(prior)+[.25]) if exclude=='team_contributions' else skew(prior)
    assert result.iloc[10] == pytest.approx(expected)


@pytest.mark.parametrize('direction', ['promoted','relegated'])
def test_mover_cohort_recentered_shapes(direction):
    games=[dict(day=i,competition=20,season=11,label='23_24',home=A,away=C,values=(v,0))
           for i,v in enumerate([40,41,60])]
    games += [dict(day=i,competition=10,season=1,home=B,away=D,values=pair)
              for i,pair in enumerate([(1,20),(2,23),(9,24)])]
    games += [dict(day=10,competition=10,season=2,round=1,home=A,away=B,values=(6,7)),
              dict(day=12,competition=10,season=2,round=2,home=A,away=B,values=(100,100))]
    h=history_from_games(games); records=movement(); records[0]['movement']=direction
    p=SeededEMA(mode='w_league_prior',bottom=-1,top=-1,alpha=.5,handoff=Hard(10))
    r=evaluate_features(h,{'s':WarmStart(RollingSkewness(STAT,3),p)},team_seasons=records)
    a,b=np.array([1,2,9]),np.array([20,23,24]); mean=(a.mean()+b.mean())/2
    prior=np.r_[a-a.mean()+mean,b-b.mean()+mean]
    assert r.s.iloc[12] == pytest.approx(skew(prior))
    assert r.s.iloc[14] == pytest.approx(weighted([*prior,6],[1/12]*6+[.5]))


def test_ui_recipe_export_and_preset(ui_recipe):
    from xdiyo_analytics.ui.catalog import default_catalog
    from xdiyo_analytics.ui.recipe import prepare_recipe, export_python
    from xdiyo_analytics.ui.inventory import inventory
    cat=default_catalog(); expr=WarmStart(RollingSkewness(STAT,5),SeededEMA(mode='uniform'))
    assert cat.build(cat.encode(expr)) == expr
    ui_recipe['features']={'skew':cat.encode(expr),'loo_skew':cat.encode(RollingSkewness(LeaveOneOut(League(STAT)),2))}
    prepared=prepare_recipe(ui_recipe)
    assert {'home::skew','away::skew','home::loo_skew','away::loo_skew'} <= set(prepared.dataset.X)
    code=export_python(ui_recipe)
    scope={}
    exec(compile(code.split('result = run_recipe',1)[0],'<recipe>','exec'),scope)
    pd.testing.assert_frame_equal(scope['prepared'].dataset.X,prepared.dataset.X)
    schema=inventory()['components']
    assert 'features.RollingSkewness' in schema
    fields={f['name']:f for f in schema['features.WarmStart']['fields']}
    assert 'features.RollingSkewness' in fields['source']['components']
    preset=FeatureBankPreset(skew_windows=(5,),loo_reducers=('skew',),h2h_reducers=('skew',),h2h_windows=(3,))
    definitions=preset.build([('ALL','Match overview','cornerKicks')])
    assert any(isinstance(v,RollingSkewness) for v in definitions.values())
    inv=preset.inventory(['home::ALL_Match overview_cornerKicks_for_skew5','home::loo_corners_skew3'])
    assert inv['unclassified_columns']==0
