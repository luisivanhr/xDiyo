from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from training_samples import MeanAdapter, sample, plan
from split_samples import unchanged, rows_for
from xdiyo_analytics.splits import Fold, SplitPlan, CPCV, create_split_plan, reconstruct_paths
from xdiyo_analytics.training import TrainingRunner, fit_predict, EstimatorAdapter


@pytest.mark.parametrize('layout',['match','team_match'])
@pytest.mark.parametrize('shuffle',[False,True])
@pytest.mark.parametrize('targets',[None,['second','first']])
def test_original_positions_shapes_groups_and_all_test_rows(layout,shuffle,targets):
    data=sample(layout=layout,shuffle=shuffle)
    for frame in (data.X,data.y,data.metadata):frame.index=pd.Index(['row-'+str(i) for i in range(len(frame))])
    folds=plan(data)
    results=TrainingRunner(MeanAdapter,feature_columns=['curve','signal'],target_columns=targets).run(data,folds,fold_ids=[1,0])
    assert [fold.fold_id for fold in results.folds]==[1,0]
    expected_targets=['first','second'] if targets is None else targets
    assert results.layout==layout and results.match_columns==data.match_columns
    for result in results.folds:
        original=folds.folds[result.fold_id]
        assert result.model.calls==['fit','predict']
        training,prediction=result.model.fit_context,result.model.predict_context
        assert training.X.index.tolist()==original.train.tolist()
        assert prediction.X.index.tolist()==original.test.tolist()
        assert training.X.index.name==prediction.X.index.name=='row_position'
        assert list(training.X)==list(prediction.X)==['curve','signal']
        assert 'settlement::first' in prediction.metadata and 'settlement::first' not in prediction.X
        assert not hasattr(prediction,'y')
        assert list(training.y)==expected_targets
        np.testing.assert_array_equal(result.train_positions,original.train)
        np.testing.assert_array_equal(result.test_positions,original.test)
        np.testing.assert_array_equal(result.score_positions,original.score)
        expected=np.tile(data.y.iloc[original.train][expected_targets].mean().to_numpy(),(len(original.test),1))
        np.testing.assert_allclose(result.predictions['predict'],expected)
        expected_y=data.y.iloc[original.test][expected_targets].copy()
        expected_y.index=pd.Index(original.test,name='row_position')
        pd.testing.assert_frame_equal(result.y_true,expected_y)
        expected_ids=data.metadata.event_id.iloc[original.test].tolist()
        assert result.metadata.event_id.tolist()==expected_ids
        assert all(int(item)>2**63 for item in expected_ids)
        assert prediction.groups.tolist()==[tuple(data.metadata.iloc[row][list(data.match_columns)].tolist()) for row in original.test]
        assert prediction.groups.index.equals(prediction.X.index)
    assert results.folds[0].model is not results.folds[1].model
    scored=results.predictions_by_fold(scored_only=True)
    assert scored[1].empty and scored[0].index.tolist()==folds.folds[0].score.tolist()
    combined=results.prediction_frame()
    assert combined.index.names==['fold_id','row_position']
    assert len(combined)==sum(len(fold.test) for fold in folds.folds)
    assert combined.index.get_level_values('fold_id').unique().tolist()==[1,0]


def test_context_result_plan_and_dataset_copies_are_isolated():
    data=sample(); folds=plan(data); before=deepcopy(data); fold_before=deepcopy(folds)
    results=TrainingRunner(lambda:MeanAdapter(mutate=True)).run(data,folds)
    unchanged(data,before)
    for result,original in zip(results.folds,fold_before.folds):
        assert result.model.predict_context.fold_metadata==original.metadata
        assert result.model.predict_context.definitions==data.definitions
        assert result.fold_metadata==original.metadata
        assert result.metadata['case'].min()>=0
        stored=result.predictions['predict'].copy()
        result.model.returned['predict'].iloc[:,:]=-1
        pd.testing.assert_frame_equal(result.predictions['predict'],stored)
        output=results.predictions_by_fold()[result.fold_id]
        output.iloc[:,:]=-5
        pd.testing.assert_frame_equal(result.predictions['predict'],stored)
    results.folds[0].train_positions[0]=99
    results.folds[0].fold_metadata['nested'].append('result mutation')
    results.definitions['features']['signal']['note'].append('result mutation')
    for actual,original in zip(folds.folds,fold_before.folds):
        np.testing.assert_array_equal(actual.train,original.train)
        assert actual.metadata==original.metadata
    unchanged(data,before)


def test_missing_unselected_train_target_allowed_missing_test_target_retained():
    data=sample(); folds=plan(data)
    data.y.loc[folds.folds[0].train,'second']=np.nan
    data.y.loc[folds.folds[0].test,'first']=np.nan
    result=fit_predict(data,folds.folds[0],MeanAdapter,target_columns='first',feature_columns='signal',fold_id=12)
    assert result.fold_id==12 and result.y_true.isna().all().all()
    assert len(result.predictions['predict'])==len(folds.folds[0].test)
    assert result.feature_columns==('signal',) and result.target_columns==('first',)


def test_selected_missing_training_target_fails_before_factory():
    data=sample(); folds=plan(data); data.y.iloc[folds.folds[0].train[0],0]=np.nan
    calls=[]
    with pytest.raises(ValueError,match='Training targets contain missing'):
        fit_predict(data,folds.folds[0],lambda:calls.append('called'))
    assert calls==[]


@pytest.mark.parametrize('mutation,match',[
    (lambda d,f:setattr(f,'train',[]),'train'),
    (lambda d,f:setattr(f,'test',[]),'test'),
    (lambda d,f:setattr(f,'train',[0.0,1.0]),'integer'),
    (lambda d,f:setattr(f,'train',[True,False]),'integer'),
    (lambda d,f:setattr(f,'train',[[0,1]]),'integer'),
    (lambda d,f:setattr(f,'train',[-1,0]),'outside'),
    (lambda d,f:setattr(f,'train',[0,99]),'outside'),
    (lambda d,f:setattr(f,'test',[6,6]),'repeated'),
    (lambda d,f:setattr(f,'test',[0,6]),'disjoint'),
    (lambda d,f:setattr(f,'score',[2]),'subset'),
    (lambda d,f:setattr(f,'score',[7,7]),'repeated'),
    (lambda d,f:setattr(d.y,'index',d.y.index[::-1]),'shared index'),
    (lambda d,f:setattr(d.X,'columns',['same']*3),'distinct column'),
])
def test_reject_bad_scopes_before_fit(mutation,match):
    data=sample(); fold=plan(data).folds[0]; mutation(data,fold)
    with pytest.raises(ValueError,match=match):fit_predict(data,fold,MeanAdapter)


@pytest.mark.parametrize('field',['train','test','score'])
def test_every_selected_scope_retains_whole_matches(field):
    data=sample(layout='team_match'); fold=plan(data).folds[0]
    setattr(fold,field,getattr(fold,field)[:-1])
    with pytest.raises(ValueError,match='all rows of each selected match'):fit_predict(data,fold,MeanAdapter)


@pytest.mark.parametrize('ids',[[0,0],[-1],[2],[True],[0.0]])
def test_invalid_fold_ids(ids):
    data=sample()
    with pytest.raises(ValueError,match='fold_ids'):TrainingRunner(MeanAdapter).run(data,plan(data),fold_ids=ids)


def test_empty_requested_folds_and_empty_plan_do_not_construct_models():
    data=sample(); calls=[]
    runner=TrainingRunner(lambda:calls.append('called'))
    for folds,ids in [(plan(data),[]),(SplitPlan([],len(data.X),np.arange(len(data.X))),None)]:
        result=runner.run(data,folds,fold_ids=ids)
        assert result.folds==[] and result.predictions_by_fold()=={}
        assert result.prediction_frame().index.names==['fold_id','row_position']
        assert result.prediction_frame().empty
    assert calls==[]


@pytest.mark.parametrize('setting',['feature_columns','target_columns'])
@pytest.mark.parametrize('columns',[[],['signal','signal'],['absent']])
def test_column_validation(setting,columns):
    data=sample()
    if setting=='target_columns':columns=['first' if item=='signal' else item for item in columns]
    with pytest.raises((ValueError,KeyError)):TrainingRunner(MeanAdapter,**{setting:columns}).run(data,plan(data))


@pytest.mark.parametrize('phase',['fit','predict'])
def test_model_failure_preserves_exception_and_adds_original_fold(phase):
    failure=RuntimeError('adapter failure')
    class Broken(MeanAdapter):
        def fit(self,context):
            if phase=='fit':raise failure
            super().fit(context)
        def predict(self,context):raise failure
    data=sample()
    with pytest.raises(RuntimeError) as caught:TrainingRunner(Broken).run(data,plan(data),fold_ids=[1])
    assert caught.value is failure
    assert any('fold 1' in note for note in caught.value.__notes__)


def test_adapter_and_wrapped_estimator_reuse_rejected():
    data=sample(); shared=MeanAdapter()
    with pytest.raises(ValueError,match='reused an adapter'):TrainingRunner(lambda:shared).run(data,plan(data))
    class Estimator:
        def fit(self,X,y):self.targets=2; return self
        def predict(self,X):return np.zeros((len(X),self.targets))
    estimator=Estimator()
    with pytest.raises(ValueError,match='reused an estimator'):
        TrainingRunner(lambda:EstimatorAdapter(estimator)).run(data,plan(data))


@pytest.mark.parametrize('output',[None,{},[],{'':pd.DataFrame()}, {'predict':np.zeros((3,2))}])
def test_adapter_output_mapping_contract(output):
    class Invalid(MeanAdapter):
        def predict(self,context):return output
    data=sample()
    with pytest.raises(TypeError):TrainingRunner(Invalid).run(data,plan(data),fold_ids=[0])


@pytest.mark.parametrize('kind',['reversed','extra_row','duplicate_column','zero_columns'])
def test_adapter_output_must_keep_exact_order_and_named_columns(kind):
    class Invalid(MeanAdapter):
        def predict(self,context):
            frame=super().predict(context)['predict']
            if kind=='reversed':frame=frame.iloc[::-1]
            elif kind=='extra_row':frame=pd.concat([frame,frame.iloc[:1]])
            elif kind=='duplicate_column':frame.columns=['same','same']
            else:frame=frame.iloc[:,:0]
            return {'predict':frame}
    data=sample()
    with pytest.raises(ValueError,match='exact test row'):TrainingRunner(Invalid).run(data,plan(data),fold_ids=[0])


def test_generic_non_target_output_columns_and_no_automatic_feature_imputation():
    class Custom(MeanAdapter):
        def predict(self,context):
            assert context.X.isna().any().any()
            return {'embedding':pd.DataFrame({'component':context.X['signal'],'tag':'retained'},index=context.X.index)}
    data=sample(); folds=plan(data); data.X.loc[folds.folds[0].test[0],'signal']=np.nan
    result=TrainingRunner(Custom).run(data,folds,fold_ids=[0])
    assert result.prediction_frame('embedding')['component'].isna().sum()==1
    assert (result.prediction_frame('embedding')['tag']=='retained').all()
    with pytest.raises(KeyError):result.predictions_by_fold('predict')


def test_runner_type_plan_length_and_factory_contract():
    data=sample(); folds=plan(data)
    with pytest.raises(TypeError):TrainingRunner(MeanAdapter).run(object(),folds)
    with pytest.raises(TypeError):fit_predict(data,object(),MeanAdapter)
    folds.n_rows+=1
    with pytest.raises(ValueError,match='row count'):TrainingRunner(MeanAdapter).run(data,folds)
    with pytest.raises(TypeError,match='ModelAdapter'):fit_predict(data,plan(data).folds[0],lambda:object())


def test_cpcv_reconstruction_retains_fold_occurrences():
    data=sample()
    split=create_split_plan(data,CPCV(n_blocks=4,n_test_blocks=2))
    for fold in split.folds:fold.score=fold.test[:1].copy()
    result=TrainingRunner(MeanAdapter,target_columns='first').run(data,split)
    reconstructed=reconstruct_paths(split,result.predictions_by_fold())
    assert result.prediction_frame().index.is_unique
    assert len(result.prediction_frame())>len(data.X)
    assert len(reconstructed)==3*len(data.X)
    for item in reconstructed.itertuples(index=False):
        assignment=split.paths[(split.paths.path_id==item.path_id)&split.paths.rows.map(lambda positions:item.row_position in positions)]
        assert len(assignment)==1 and int(assignment.iloc[0].fold_id)==item.fold_id
        expected=data.y.iloc[split.folds[item.fold_id].train]['first'].mean()
        assert item.first==expected
    with pytest.raises(ValueError,match='held-out row positions'):
        reconstruct_paths(split,result.predictions_by_fold(scored_only=True))
