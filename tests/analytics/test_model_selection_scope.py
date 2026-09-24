from copy import deepcopy
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from model_selection_samples import sample,plan,development,outer,candidate,FixedAdapter
from split_samples import rows_for,unchanged
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.reporting import FeatureSelector,FeatureSelection,StudyResult
from xdiyo_analytics.selection import Candidate,ModelSelection
from xdiyo_analytics.splits import Fold,SplitPlan

@pytest.mark.parametrize('layout',['match','team_match'])
@pytest.mark.parametrize('shuffle',[False,True])
def test_noncontiguous_original_positions_exact_ids_and_pooled_score_arithmetic(layout,shuffle):
    data=sample(layout=layout,shuffle=shuffle);before=deepcopy(data);splits=plan(data);dev=development(data)
    result=ModelSelection([candidate('near'),candidate('shifted',5.,2.)],metrics='mse').run(data,splits,development_positions=dev)
    assert result.winner.candidate.name=='near' and result.development_positions.tolist()==dev.tolist()
    errors=[]
    for stored,expected in zip(result.winner.training.folds,splits.folds):
        for field,values in [('train_positions',expected.train),('test_positions',expected.test),('score_positions',expected.score)]:
            np.testing.assert_array_equal(getattr(stored,field),values)
        assert stored.predictions['predict'].index.tolist()==expected.test.tolist()
        assert stored.y_true.index.tolist()==expected.test.tolist()
        assert stored.metadata.event_id.tolist()==data.metadata.iloc[expected.test].event_id.tolist()
        assert all(int(v)>2**63 for v in stored.metadata.event_id)
        assert set(stored.model.context.metadata.case)<=set(range(8))
        assert set(stored.model.prediction_context.metadata.case)<=set([6,7,10,11])
        y=data.y.target.iloc[expected.score].to_numpy();x=data.X.signal.iloc[expected.score].to_numpy()
        errors.extend((1+2*x-y)**2)
    raw=result.comparison.set_index('name').loc['near','raw::mse']
    assert raw==pytest.approx(np.mean(errors))
    fold_means=[np.mean((1+2*data.X.signal.iloc[f.score]-data.y.target.iloc[f.score])**2) for f in splits.folds]
    assert not np.isclose(raw,np.mean(fold_means))
    view=result.to_report().studies[0]
    assert view.row_positions.tolist()==dev.tolist() and view.n_matches==12
    unchanged(data,before)

def test_outer_labels_and_features_cannot_change_inner_winner_or_metrics():
    data=sample();changed=deepcopy(data);external=rows_for(data,range(12,16))
    changed.y.iloc[external]=1e9;changed.X.iloc[external]=-1e9
    search=ModelSelection([candidate('near'),candidate('shifted',5.,2.)],metrics='mse')
    first=search.run(data,plan(data),development_positions=development(data))
    second=search.run(changed,plan(changed),development_positions=development(changed))
    assert first.winner.candidate.name==second.winner.candidate.name
    for a,b in zip(first.trials,second.trials):assert a.record['metrics']==b.record['metrics']

@pytest.mark.parametrize('form',['array','dict','selector'])
@pytest.mark.parametrize('layout',['match','team_match'])
def test_monitoring_validation_mapping_and_selector_fitting_scope(form,layout):
    data=sample(layout=layout,shuffle=True);dev=development(data);splits=plan(data);events=[];observed=[]
    class Selector(FeatureSelector):
        def select(self,context):
            observed.append((context.fold_id,context.metadata.case.tolist(),context.row_positions.tolist()))
            return StudyResult('picked',selection=FeatureSelection(('signal',),pd.DataFrame({'feature':['signal']})))
    class Validation:
        def select(self,scoped,train_positions):
            events.append((len(scoped.X),set(scoped.metadata.case),scoped.X.index.tolist()))
            return np.flatnonzero(scoped.metadata.case.eq(4))
    valid=rows_for(data,[4])
    validation=valid if form=='array' else {0:valid,1:valid} if form=='dict' else Validation()
    item=candidate('candidate',validation=validation,pre_analysis=PreTrainingAnalysis({'pick':Selector(type='per_fold',partition='train')}),features_from='pick')
    result=ModelSelection([item],metrics='mse').run(data,splits,development_positions=dev)
    for fold,original in zip(result.winner.training.folds,splits.folds):
        expected_valid=dev[data.metadata.iloc[dev].case.eq(4).to_numpy()] if form=='selector' else valid
        np.testing.assert_array_equal(fold.validation_positions,expected_valid)
        np.testing.assert_array_equal(fold.fit_positions,original.train[~np.isin(original.train,valid)])
        assert 4 not in fold.model.context.metadata.case.tolist()
        assert set(fold.model.context.validation.metadata.case)=={4}
        assert set(fold.selection.row_positions)==set(fold.fit_positions)
        if fold.selection.scope is not None:
            assert set(fold.selection.scope.row_position)==set(fold.fit_positions)
        assert fold.feature_columns==('signal',)
    assert all(4 not in cases for _,cases,_ in observed)
    if form=='selector':assert events and all(n==len(dev) and cases==set(range(12)) and idx==list(range(len(dev))) for n,cases,idx in events)

@pytest.mark.parametrize('problem',['outside_train','outside_test','outside_score','wrong_n','empty_dev','duplicate_dev','partial_match'])
def test_development_scope_errors_fail_before_candidate_factory(problem):
    data=sample(layout='team_match');splits=plan(data);dev=development(data);calls=[]
    if problem.startswith('outside_'):
        field=problem.removeprefix('outside_');setattr(splits.folds[0],field,rows_for(data,[13]))
    elif problem=='wrong_n':splits.n_rows+=1
    elif problem=='empty_dev':dev=np.array([],dtype=int)
    elif problem=='duplicate_dev':dev=np.r_[dev,dev[0]]
    else:dev=dev[:-1]
    def make():calls.append(1);return FixedAdapter()
    with pytest.raises(ValueError):ModelSelection([Candidate('x',make)],metrics='mse',on_error='record').run(data,splits,development_positions=dev)
    assert calls==[]

@pytest.mark.parametrize('bad_part',['all','test'])
def test_candidate_exploration_scope_configuration_propagates_in_record_mode(bad_part):
    data=sample()
    class Selector(FeatureSelector):
        def select(self,context):return StudyResult('picked',selection=FeatureSelection(('signal',),pd.DataFrame()))
    bad=candidate('bad preparation',pre_analysis=PreTrainingAnalysis({'pick':Selector(type='overall',partition=bad_part)}),features_from='pick')
    with pytest.raises(ValueError,match="partition='train'"):
        ModelSelection([candidate('good'),bad],metrics='mse',on_error='record').run(data,plan(data),development_positions=development(data))

def test_evaluate_requires_untouched_test_and_original_row_count():
    data=sample();result=ModelSelection([candidate('near')],metrics='mse').run(data,plan(data),development_positions=development(data))
    for invalid in [Fold(rows_for(data,range(5)),rows_for(data,[9]),rows_for(data,[9])),
                    Fold(rows_for(data,[13]),rows_for(data,[14]),rows_for(data,[14]))]:
        with pytest.raises(ValueError,match='untouched|within development'):result.evaluate(data,invalid)
    smaller=replace(data,X=data.X.iloc[:-1],y=data.y.iloc[:-1],metadata=data.metadata.iloc[:-1])
    with pytest.raises(ValueError,match='original dataset'):result.evaluate(smaller,outer(data))
    evaluated=result.evaluate(data,outer(data))
    assert evaluated.folds[0].test_positions.tolist()==outer(data).test.tolist()
    assert not any(evaluated.folds[0].model is inner.model for inner in result.winner.training.folds)
