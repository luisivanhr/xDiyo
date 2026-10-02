"""Native scores, synthetic temporal boundaries and preparation only; no fitting."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.data import SeasonData
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import (MatchScore, Lag, RollingMean, RollingStd,
    RollingZScore, EMA, H2H, WarmStart, ForAgainst, Difference, Sum, Ratio, Constant,
    evaluate_features)
from xdiyo_analytics.labels import Outcome, create_labels
from xdiyo_analytics.datasets import assemble_dataset
from xdiyo_analytics.ui.recipe import (catalog_for_ui, default_recipe, node,
    prepare_recipe, export_python, export_notebook)

A,B=2**63+51,2**63+53


def score_data(n=6):
    dates=pd.date_range('2023-12-28 12:00',periods=n,tz='UTC')
    home=[A if i%2==0 else B for i in range(n)]
    matches=pd.DataFrame(dict(event_id=pd.array([2**63+101+i for i in range(n)],dtype='uint64'),
        home_id=pd.array(home,dtype='uint64'),
        away_id=pd.array([B if x==A else A for x in home],dtype='uint64'),
        kickoff_utc=[x.timestamp() for x in dates],competition_id=17,tournament_id=50,
        season_id=[1 if i<3 else 2 for i in range(n)],round=np.arange(n)+1,status='finished',
        home_score_current=[2*i+1 if x==A else 2*i+2 for i,x in enumerate(home)],
        away_score_current=[2*i+2 if x==A else 2*i+1 for i,x in enumerate(home)]))
    return SeasonData({'matches':matches},{})


def definitions():
    return {f'goals_{side}_r20':RollingMean(MatchScore(score_field='current',side=side),
                                         window=20,min_periods=1,venue='all')
            for side in ('for','against')}


def test_perspectives_rolling_means_metadata_and_no_mutation():
    data=score_data()
    original=data.matches.copy(deep=True)
    history=build_team_history(data)
    before=deepcopy(history)
    result=evaluate_features(history,definitions())
    for team,gf,ga in [(A,[np.nan,1,2,3,4,5],[np.nan,2,3,4,5,6]),
                       (B,[np.nan,2,3,4,5,6],[np.nan,1,2,3,4,5])]:
        rows=history.team_id.eq(team)
        np.testing.assert_allclose(result.loc[rows,'goals_for_r20'],gf,equal_nan=True)
        np.testing.assert_allclose(result.loc[rows,'goals_against_r20'],ga,equal_nan=True)
    both=evaluate_features(history,{'both':Lag(MatchScore(side='both')),
                                  'conceded':Lag(ForAgainst(MatchScore(),side='against'))})
    assert both.columns.tolist()==['both::goals_for','both::goals_against','conceded']
    pd.testing.assert_series_equal(both['both::goals_against'],both.conceded,check_names=False)
    assert history.attrs['stat_columns']=={}
    assert history.attrs['score_columns']['goals_for']['score_field']=='current'
    assert "score_field='current'" in result.attrs['features']['goals_for_r20']
    pd.testing.assert_frame_equal(history,before)
    pd.testing.assert_frame_equal(data.matches,original)


@pytest.mark.parametrize('basis',['display','period1','period2','normaltime','penalties',None])
def test_unsupported_basis_never_falls_back(basis):
    with pytest.raises(ValueError,match='score_field'):
        MatchScore(score_field=basis)


def test_current_ignores_disagreeing_halves_extra_time_and_penalties():
    data=score_data()
    baseline=build_team_history(data)
    for side in ('home','away'):
        for field in ('display','period1','period2','extra1','extra2','overtime','penalties'):
            data.matches[f'{side}_score_{field}']=999 if side=='home' else np.nan
    changed=build_team_history(data)
    pd.testing.assert_frame_equal(changed,baseline)
    pd.testing.assert_frame_equal(evaluate_features(changed,definitions()),evaluate_features(baseline,definitions()))
    # A 3-4 native current score remains 3-4 even with incompatible halves.
    data.matches.loc[0,['home_score_current','away_score_current']]=[3,4]
    h=build_team_history(data)
    assert h.loc[(h.event_id==data.matches.event_id.iloc[0]) & h.team_id.eq(A),'goals_for'].item()==3
    data.matches.drop(columns=['home_score_current','away_score_current'],inplace=True)
    absent=build_team_history(data)
    assert evaluate_features(absent,definitions()).isna().all().all()


@pytest.mark.parametrize('missing',[np.inf,-np.inf,np.nan,None])
def test_missing_nonfinite_window_and_min_periods(missing):
    h=build_team_history(score_data(22))
    h['goals_for']=h['round']-1.
    h.loc[h['round']==1,'goals_for']=1000.
    h.loc[h['round']==21,'goals_for']=missing
    result=evaluate_features(h,{'mean':RollingMean(MatchScore(),20),
        'strict':RollingMean(MatchScore(),20,min_periods=20),'lag':Lag(MatchScore())})
    last=h['round'].eq(22)
    assert result.loc[last,'mean'].eq(10.).all()  # 1..19; missing 20 consumes a slot
    assert result.loc[last,'strict'].isna().all()
    assert result.loc[last,'lag'].isna().all()
    assert result.loc[h['round']==1].isna().all().all()


def test_frozen_round_boundaries_availability_postponements_and_perturbations():
    data=score_data(10)
    m=data.matches
    dates=pd.to_datetime(['2024-01-01 12:00','2024-01-02 12:00','2024-01-03 12:00',
        '2024-01-04 12:00','2024-01-05 12:00','2024-01-10 11:00',
        '2024-01-10 12:00','2024-01-20 12:00','2024-01-21 12:00','2024-01-10 12:00'],utc=True)
    m['kickoff_utc']=[x.timestamp() for x in dates]
    m.loc[1,'competition_id']=99
    m.loc[4,'status']='inprogress'
    m.loc[[6,7,9],'round']=30
    h=build_team_history(data)
    h['cutoff']=h.groupby(['competition_id','season_id','tournament_id','round']).kickoff_at.transform('min')-pd.Timedelta(hours=1)
    h['available']=h.kickoff_at+pd.Timedelta(hours=3)
    boundary=pd.Timestamp('2024-01-10 11:00',tz='UTC')
    h.loc[h.event_id==m.event_id.iloc[2],'available']=boundary
    h.loc[h.event_id==m.event_id.iloc[3],'available']=boundary+pd.Timedelta(nanoseconds=1)
    # Per-row explicit cutoff series also survive shuffled histories.
    out=evaluate_features(h,definitions(),cutoffs='cutoff',available_at='available')
    targets=h['round'].eq(30)&h.team_id.eq(A)
    assert out.loc[targets,'goals_for_r20'].eq(3.).all()  # scores 1 and 5
    assert out.loc[targets,'goals_against_r20'].eq(4.).all()  # scores 2 and 6
    mutated=h.copy()
    mutated.loc[h.kickoff_at>=boundary,['goals_for','goals_against']]=9999
    later=evaluate_features(mutated,definitions(),cutoffs='cutoff',available_at='available')
    pd.testing.assert_frame_equal(out.loc[targets],later.loc[targets])
    # An earlier duplicate of the target identity cannot contribute to itself.
    duplicate=h.loc[targets].iloc[[0]].copy()
    duplicate['kickoff_at']=pd.Timestamp('2024-01-06',tz='UTC')
    duplicate['available']=duplicate.kickoff_at
    duplicate['cutoff']=duplicate.kickoff_at
    duplicate['goals_for']=9999
    appended=pd.concat([h,duplicate],ignore_index=True)
    check=evaluate_features(appended,definitions(),cutoffs='cutoff',available_at='available')
    i=np.flatnonzero(targets)[0]
    assert check.goals_for_r20.iloc[i]==3.


@pytest.mark.parametrize('raw',[MatchScore(),Sum(MatchScore(),Constant(1)),
    H2H(MatchScore()),WarmStart(MatchScore()),ForAgainst(MatchScore()),
    Difference(Lag(MatchScore()),MatchScore()),Ratio(MatchScore(),Constant(2))])
def test_raw_scores_fail_closed_in_wrappers_and_arithmetic(raw):
    h=build_team_history(score_data())
    with pytest.raises(ValueError,match='Observed'):
        evaluate_features(h,{'unsafe':raw})
    with pytest.raises(ValueError,match='reference'):
        evaluate_features(h,{'unsafe':RollingZScore(MatchScore(),reference=raw)})


def test_valid_operators_and_keyed_assembly_survive_shuffling():
    h=build_team_history(score_data())
    features=definitions()
    catalog=catalog_for_ui()
    for expression in [Lag(MatchScore()),RollingStd(MatchScore()),EMA(MatchScore()),
        H2H(RollingMean(MatchScore())),WarmStart(RollingMean(MatchScore())),
        RollingZScore(MatchScore(),reference=Lag(MatchScore())),
        Difference(RollingMean(MatchScore()),RollingMean(MatchScore(side='against')))]:
        restored=catalog.build(json.loads(json.dumps(catalog.encode(expression))))
        assert expression==restored
        evaluate_features(h,{'safe':restored})
    labels=create_labels(h,{'outcome':Outcome(perspective='home')})['outcome']
    expected=assemble_dataset(evaluate_features(h,features,keyed=True),labels,layout='match')
    shuffled=h.sample(frac=1,random_state=4)
    actual=assemble_dataset(evaluate_features(shuffled,features,keyed=True).sample(frac=1,random_state=7),labels,layout='match')
    pd.testing.assert_frame_equal(actual.X,expected.X)
    pd.testing.assert_frame_equal(actual.metadata,expected.metadata)
    assert set(actual.X)=={f'{side}::goals_{kind}_r20' for side in ('home','away') for kind in ('for','against')}
    assert set(actual.metadata.event_id)==set(score_data().matches.event_id)


def score_recipe():
    recipe=default_recipe('data/xDiyo_data')
    recipe.update(name='Native current goals',features=catalog_for_ui().encode(definitions()),
                  labels={'outcome':node('labels.Outcome',perspective='home')},target='outcome',
                  post_reporters={},split=node('splits.TemporalSplit',train_size=3,test_size=1,unit='kickoffs'))
    recipe['data']['tables']=['matches']
    return recipe


def test_recipe_exports_execute_only_preparation_with_native_scores(monkeypatch):
    data=score_data()
    def load(**options):
        assert options['tables']==['matches']
        return data
    monkeypatch.setattr('xdiyo_analytics.data.load_seasons',load)
    def forbidden(*args,**kwargs):
        raise AssertionError('Training or auxiliary score input requested')
    monkeypatch.setattr('xdiyo_analytics.ui.recipe.run_recipe',forbidden)
    monkeypatch.setattr('xdiyo_analytics.ui.recipe._context_file',forbidden)
    recipe=score_recipe()
    expected=prepare_recipe(recipe)
    for source in [export_python(recipe).split('result = run_recipe')[0],
                   ''.join(export_notebook(recipe)['cells'][1]['source'])]:
        namespace={}
        exec(compile(source,'<preparation-only>','exec'),namespace)
        pd.testing.assert_frame_equal(namespace['prepared'].dataset.X,expected.dataset.X)
        assert len(namespace['prepared'].dataset.X.columns)==4
    assert recipe==json.loads(json.dumps(recipe))


def test_score_ui_schema():
    schema={s['id']:s for s in catalog_for_ui().schema()}
    fields={f['name']:f for f in schema['features.MatchScore']['fields']}
    assert fields['score_field']['choices']==['current']
    assert fields['side']['choices']==['for','against','both']
    assert fields['score_field']['primary'] and fields['side']['primary']
    with pytest.raises(ValueError,match='side'):
        MatchScore(side='home')
