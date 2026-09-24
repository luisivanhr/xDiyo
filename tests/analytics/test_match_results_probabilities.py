"""Explicit probability decisions and missing-class rows, without inference."""
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.reporting import MatchResultReporter
from match_results_samples import fixture_context, result_from_context

def probabilities(values,classes=(1,0),observed=None):
    ctx=fixture_context(n=len(values))
    ctx.y['count']=([1]*len(values) if observed is None else observed)
    ctx.predictions['predict_proba']=pd.DataFrame(values,index=ctx.y.index,
        columns=pd.MultiIndex.from_product([['count'],classes],names=['target','class']))
    return ctx

def run(ctx,**kwargs):return MatchResultReporter(type='overall',partition='test',output='predict_proba',**kwargs).run(ctx).tables['matches::count']


def test_probabilities_are_displayed_without_invented_decisions():
    ctx=probabilities([[.7,.3],[.5,.5],[np.nan,.3],[.8,.8],[1.1,-.1],[np.inf,0]],observed=[1,0,1,1,1,1])
    before=deepcopy(ctx);table=run(ctx)
    assert table.prediction.isna().all() and table.error.isna().all()
    assert table.status.tolist()==['no decision','no decision']+['unavailable']*4
    assert json.loads(table.probabilities.iloc[0])=={'1':.7,'0':.3}
    assert table.probabilities.iloc[2:].isna().all()
    pd.testing.assert_frame_equal(ctx.predictions['predict_proba'],before.predictions['predict_proba'])


def test_argmax_first_stored_class_tie_and_threshold_inclusive_boundary():
    ctx=probabilities([[.5,.5],[.6,.4],[.599,.401],[.1,.9]],observed=[1,1,0,0])
    assert run(ctx,decision='argmax').prediction.tolist()==[1,1,1,0]
    threshold=run(ctx,threshold=.6,positive_class=1)
    assert threshold.prediction.tolist()==[0,1,0,0]
    assert threshold.status.tolist()==['incorrect','correct','correct','correct']


def test_string_class_identity_and_missing_observation():
    ctx=probabilities([[.1,.9],[.5,.5]],classes=('win','loss'),observed=['loss',None])
    table=run(ctx,decision='argmax')
    assert table.prediction.tolist()==['loss','win'] and table.status.tolist()==['correct','unavailable']


@pytest.mark.parametrize('all_invalid',[False,True,'empty'])
@pytest.mark.parametrize('problem',['three_classes','unknown_positive'])
def test_binary_threshold_contract_does_not_depend_on_row_availability(all_invalid,problem):
    classes=(0,1,2) if problem=='three_classes' else (0,1)
    values=[] if all_invalid=='empty' else [[np.nan]*len(classes)] if all_invalid else [[1/len(classes)]*len(classes)]
    ctx=probabilities(values,classes=classes)
    with pytest.raises(ValueError,match='two classes'):
        run(ctx,threshold=.5,positive_class=1 if problem=='three_classes' else 9)


def test_point_predictions_remain_independent_of_companion_probability_availability():
    ctx=probabilities([[.8,.2],[np.nan,.2]])
    ctx.predictions['predict']['count']=[1,1]
    reporter=MatchResultReporter(type='overall',partition='test',comparison='categorical')
    table=reporter.run(ctx).tables['matches::count']
    assert table.status.tolist()==['correct','correct']
    assert json.loads(table.probabilities.iloc[0])=={'1':.8,'0':.2} and pd.isna(table.probabilities.iloc[1])
    table=MatchResultReporter(type='overall',partition='test',probability_output=None).run(ctx).tables['matches::count']
    assert table.probabilities.isna().all()


def test_missing_pooled_class_columns_are_unavailable_not_renormalized():
    ctx=probabilities([[.7,.3],[.3,.7]],classes=(0,1),observed=[0,1])
    result=result_from_context(ctx,repeat=True)
    result.folds[1].predictions['predict_proba'].columns=pd.MultiIndex.from_product([['count'],[1,2]],names=['target','class'])
    report=PostTrainingAnalysis({'matches':MatchResultReporter(type='overall',output='predict_proba',
        decision='argmax',pooling='occurrences')}).run(result)
    table=report.studies[0].result.tables['matches::count']
    assert table.status.tolist()==['unavailable']*4 and table.probabilities.isna().all()


@pytest.mark.parametrize('problem',['duplicate_classes','one_class','bad_index','flat_columns'])
def test_invalid_probability_frames_rejected(problem):
    ctx=probabilities([[.8,.2],[.1,.9]])
    p=ctx.predictions['predict_proba']
    if problem=='duplicate_classes':p.columns=pd.MultiIndex.from_product([['count'],[1,1]])
    elif problem=='one_class':ctx.predictions['predict_proba']=p.iloc[:,:1]
    elif problem=='bad_index':p.index=p.index[::-1]
    else:p.columns=['count','extra']
    with pytest.raises((ValueError,KeyError)):
        run(ctx,decision='argmax')
