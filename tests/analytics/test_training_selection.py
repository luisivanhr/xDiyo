from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from training_samples import MeanAdapter,sample,plan
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.reporting import AnalysisReport,FeatureSelection,StudyResult,StudyRun,TopKCorrelationSelector
from xdiyo_analytics.training import TrainingRunner


def selection(data,folds,*,fold_id=None,columns=('signal',),rows=None,layout=None):
    rows=folds.folds[0].train.copy() if rows is None else np.array(rows)
    return StudyRun('choice','overall' if fold_id is None else 'per_fold','train',fold_id,
                    data.layout if layout is None else layout,rows,len(rows),
                    StudyResult('Columns',selection=FeatureSelection(columns,pd.DataFrame({'feature':columns,'score':np.arange(len(columns))}))))


@pytest.mark.parametrize('layout',['match','team_match'])
def test_consume_actual_training_selector_without_recomputation(layout,monkeypatch):
    data=sample(layout=layout); folds=plan(data)
    report=PreTrainingAnalysis({'choice':TopKCorrelationSelector(type='per_fold',partition='train',k=1)}).run(data,split_plan=folds)
    before=deepcopy(report)
    def forbidden(*args,**kwargs):raise AssertionError('Selector recalculated during training')
    monkeypatch.setattr(TopKCorrelationSelector,'select',forbidden)
    result=TrainingRunner(MeanAdapter).run(data,folds,analysis_report=report,features_from='choice')
    for fitted,study in zip(result.folds,report.studies):
        assert fitted.feature_columns==study.result.selection.columns==('signal',)
        assert fitted.selection is not study
        assert fitted.selection.fold_id==fitted.fold_id
        assert list(fitted.model.fit_context.X)==list(fitted.feature_columns)
        assert set(study.row_positions)==set(fitted.train_positions)
    result.folds[0].selection.result.selection.ranking.iloc[0,0]='changed'
    result.folds[0].selection.row_positions[0]=999
    for actual,original in zip(report.studies,before.studies):
        pd.testing.assert_frame_equal(actual.result.selection.ranking,original.result.selection.ranking)
        np.testing.assert_array_equal(actual.row_positions,original.row_positions)


def test_matching_fold_takes_precedence_and_overall_falls_back():
    data=sample(); folds=plan(data)
    report=AnalysisReport([selection(data,folds),selection(data,folds,fold_id=0,columns=('curve','signal'))])
    result=TrainingRunner(MeanAdapter).run(data,folds,analysis_report=report,features_from='choice')
    assert result.folds[0].feature_columns==('curve','signal')
    assert result.folds[1].feature_columns==('signal',)
    assert result.folds[1].selection.fold_id is None


@pytest.mark.parametrize('problem',['outside_train','test_rows','wrong_layout','missing_name','no_selection','ambiguous','empty_columns','unknown_column'])
def test_unsafe_or_invalid_consumed_selection_rejected(problem):
    data=sample(); folds=plan(data); study=selection(data,folds,fold_id=0)
    report=AnalysisReport([study]); name='choice'
    if problem=='outside_train':study.row_positions=np.array([0,5])
    elif problem=='test_rows':study.row_positions=folds.folds[0].test.copy()
    elif problem=='wrong_layout':study.layout='team_match'
    elif problem=='missing_name':name='absent'
    elif problem=='no_selection':study.result.selection=None
    elif problem=='ambiguous':report.studies.append(deepcopy(study))
    elif problem=='empty_columns':study.result.selection.columns=()
    else:study.result.selection.columns=('absent',)
    with pytest.raises((ValueError,KeyError)):
        TrainingRunner(MeanAdapter).run(data,folds,fold_ids=[0],analysis_report=report,features_from=name)


def test_actual_pooled_consensus_cannot_leak_into_earlier_fold():
    data=sample(); folds=plan(data)
    report=PreTrainingAnalysis({'choice':TopKCorrelationSelector(type='overall',partition='train',across_folds=True,k=1)}).run(data,split_plan=folds)
    with pytest.raises(ValueError,match='only this fold'):
        TrainingRunner(MeanAdapter).run(data,folds,fold_ids=[0],analysis_report=report,features_from='choice')
    result=TrainingRunner(MeanAdapter).run(data,folds,fold_ids=[1],analysis_report=report,features_from='choice')
    assert len(result.folds)==1


@pytest.mark.parametrize('mode',['missing_report','unused_report','columns_and_selection'])
def test_selection_consumption_requires_explicit_unambiguous_options(mode):
    data=sample(); folds=plan(data)
    report=AnalysisReport([selection(data,folds)])
    kwargs={'features_from':'choice','analysis_report':report}
    features=None
    if mode=='missing_report':kwargs.pop('analysis_report')
    elif mode=='unused_report':kwargs.pop('features_from')
    else:features=['signal']
    with pytest.raises(ValueError):TrainingRunner(MeanAdapter,feature_columns=features).run(data,folds,**kwargs)
