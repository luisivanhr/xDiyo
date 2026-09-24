"""Independent observed/predicted comparisons and immutable reporting populations."""
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.reporting import MatchResultReporter, TeamCatalog
from xdiyo_analytics.labels import Outcome, Above, BetOption
from match_results_samples import fixture_context, result_from_context, HOME, AWAY

def run(ctx,**kwargs):return MatchResultReporter(type='overall',partition='test',**kwargs).run(ctx)

@pytest.mark.parametrize('layout',['match','team_match'])
def test_numeric_boundary_signed_error_exact_ids_and_input_copies(layout):
    ctx=fixture_context(layout);before=deepcopy(ctx)
    report=run(ctx,tolerance=.25);table=report.tables['matches::count']
    expected=np.resize([0.,.25,-.5,.1],len(table))
    np.testing.assert_allclose(table.error,expected)
    assert table.status.tolist()==['within tolerance' if abs(v)<=.25 else 'outside tolerance' for v in expected]
    assert table.home_id.tolist()==[HOME]*len(table) and table.away_id.tolist()==[AWAY]*len(table)
    assert table.event_id.tolist()==ctx.metadata.event_id.tolist()
    assert table.fixture.nunique()==4 and table.fold_id.tolist()==[4]*len(table)
    pd.testing.assert_frame_equal(ctx.y,before.y);pd.testing.assert_frame_equal(ctx.metadata,before.metadata)
    pd.testing.assert_frame_equal(ctx.predictions['predict'],before.predictions['predict'])
    assert report.artifacts[0].options['numeric'] is True


@pytest.mark.parametrize('definition',[None,Outcome(),Above(Outcome(),0),BetOption(Outcome(),selection='win')])
def test_auto_comparison_recognizes_built_in_label_definitions(definition):
    ctx=fixture_context();ctx.y['count']=[0,1,0,1];ctx.predictions['predict']['count']=[0,0,1,1]
    if definition is not None:ctx.definitions={'label':definition}
    report=run(ctx)
    assert report.artifacts[0].options['numeric'] is (definition is None)
    assert report.tables['matches::count'].status.tolist()==(
        ['within tolerance','outside tolerance','outside tolerance','within tolerance'] if definition is None
        else ['correct','incorrect','incorrect','correct'])


def test_custom_numeric_classes_and_multitarget_panels_are_independent():
    ctx=fixture_context();ctx.y['class']=[1,0,1,0];ctx.predictions['predict']['class']=[1,1,0,0]
    result=run(ctx,comparison='categorical')
    assert list(result.tables)==['matches::count','matches::class'] and len(result.artifacts)==2
    assert result.tables['matches::class'].status.tolist()==['correct','incorrect','incorrect','correct']
    assert result.tables['matches::count'].status.tolist()==['correct','incorrect','incorrect','incorrect']
    selected=run(ctx,target=['class','count']);assert list(selected.tables)==['matches::class','matches::count']


def test_string_classes_use_categorical_comparison():
    ctx=fixture_context();ctx.y['count']=['win','draw','loss',None]
    ctx.predictions['predict']['count']=['win','loss','loss','win']
    assert run(ctx).tables['matches::count'].status.tolist()==['correct','incorrect','correct','unavailable']


def test_missing_nonfinite_and_push_void_never_report_success():
    ctx=fixture_context(n=8);ctx.y['count']=[1,np.nan,np.inf,1,0,1,1,pd.NA]
    ctx.predictions['predict']['count']=[np.inf,1,2,np.nan,0,1,1,1]
    ctx.metadata['settlement::count']=[None,None,None,None,'push','void','missing',None]
    table=run(ctx,tolerance=100).tables['matches::count']
    assert table.status.tolist()==['unavailable']*4+['push','void','missing','unavailable']
    assert table.error.isna().all()


@pytest.mark.parametrize('field',['league','season','team','round'])
@pytest.mark.parametrize('value',[None,0,'unknown'])
def test_initial_filters_never_drop_stored_rows(field,value):
    ctx=fixture_context();result=run(ctx,**{field:value})
    assert len(result.tables['matches::count'])==4
    assert result.artifacts[0].options['initial'][field] == (None if value is None else str(value))


def test_source_columns_and_explicit_column_overrides():
    ctx=fixture_context();ctx.metadata['source_league']='Example';ctx.metadata['source_season']='24_25'
    table=run(ctx).tables['matches::count']
    assert table.league.eq('Example').all() and table.season.eq('24_25').all()
    table=run(ctx,league_column='competition_id',season_column='season_id').tables['matches::count']
    assert table.league.eq(17).all() and table.season.eq(2025).all()
    with pytest.raises(KeyError):run(ctx,league_column='absent')


def test_metadata_and_catalog_name_fallbacks_and_one_lookup_per_team(monkeypatch):
    ctx=fixture_context();catalog=TeamCatalog({HOME:'Catalog home'})
    calls=[];original=catalog.display
    def counted(*args,**kwargs):calls.append(args[0]);return original(*args,**kwargs)
    monkeypatch.setattr(catalog,'display',counted)
    result=run(ctx,catalog=catalog)
    assert calls==[HOME,AWAY]
    assert result.tables['matches::count'].home.eq('Catalog home').all()
    assert result.tables['matches::count'].away.eq('Away club').all()
    ctx.metadata=ctx.metadata.drop(columns=['away_name'])
    result=run(ctx,catalog=catalog)
    assert result.tables['matches::count'].away.eq(f'Team {AWAY}').all()
    assert any('No display name' in note for note in result.notes)


@pytest.mark.parametrize('name_fields',[{}, {'name':None}, {'name':''}])
def test_catalog_record_without_name_uses_available_metadata(name_fields):
    ctx=fixture_context();catalog=TeamCatalog({HOME:{'badge_path':'missing.svg',**name_fields}})
    table=run(ctx,catalog=catalog).tables['matches::count']
    assert table.home.eq('Home club').all()


@pytest.mark.parametrize('problem',['duplicate','opponent','league','season','round','stage','bad_side'])
def test_duplicate_or_conflicting_paired_fixture_rejected(problem):
    ctx=fixture_context('team_match')
    if problem=='duplicate':ctx.metadata.loc[(4,1),'side']='home';ctx.metadata.loc[(4,1),'team_id']=HOME;ctx.metadata.loc[(4,1),'opponent_id']=AWAY
    elif problem=='bad_side':ctx.metadata.loc[(4,1),'side']='unknown'
    else:
        column={'opponent':'opponent_id','league':'competition_id','season':'season_id'}.get(problem,problem)
        # Competition and season are also match keys: an extra context field must
        # differ while the same declared event identity remains fixed.
        if problem in ('league','season'):
            ctx.match_columns=('event_id',)
        ctx.metadata.loc[(4,1),column]='Changed' if problem=='stage' else 999
    with pytest.raises(ValueError):run(ctx)


def test_partial_paired_rows_remain_available_for_neutral_missing_side_display():
    ctx=fixture_context('team_match');keep=[0,2,3]
    ctx.y=ctx.y.iloc[keep];ctx.metadata=ctx.metadata.iloc[keep];ctx.predictions={'predict':ctx.predictions['predict'].iloc[keep]}
    table=run(ctx).tables['matches::count']
    assert len(table)==3 and table.fixture.nunique()==2 and table.side.tolist()==['home','home','away']


@pytest.mark.parametrize('pooling,rows,folds', [('occurrences',8,{4,9}),('first',4,{4}),('last',4,{9}),('mean',4,{-1})])
def test_repeated_folds_need_explicit_pooling_and_keep_occurrence_identity(pooling,rows,folds):
    training=result_from_context(fixture_context(),repeat=True)
    with pytest.raises(ValueError,match='Repeated'):PostTrainingAnalysis({'matches':MatchResultReporter(type='overall')}).run(training)
    report=PostTrainingAnalysis({'matches':MatchResultReporter(type='overall',pooling=pooling)}).run(training)
    table=report.studies[0].result.tables['matches::count']
    assert len(table)==rows and set(table.fold_id)==folds and table.fixture.nunique()==rows


def test_per_fold_selection_and_empty_score_scope():
    training=result_from_context(fixture_context(),repeat=True)
    analysis=PostTrainingAnalysis({'matches':MatchResultReporter(type='per_fold')})
    assert [s.fold_id for s in analysis.run(training,fold_ids=[9,4]).studies]==[9,4]
    for fold in training.folds:fold.score_positions=np.array([],dtype=int)
    report=analysis.run(training)
    assert all(s.result.tables['matches::count'].empty for s in report.studies)
    assert 'match-results' in report.to_html()


@pytest.mark.parametrize('kwargs',[dict(comparison='guess'),dict(tolerance=-1),dict(tolerance=True),dict(tolerance=np.inf),
    dict(page_size=0),dict(page_size=True),dict(page_size=1.5),dict(decision='vote'),dict(threshold=.5),
    dict(threshold=2,positive_class=1),dict(decision='argmax',threshold=.5,positive_class=1),dict(decision='argmax')])
def test_invalid_options_rejected(kwargs):
    with pytest.raises(ValueError):run(fixture_context(),**kwargs)
