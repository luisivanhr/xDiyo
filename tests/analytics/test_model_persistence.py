from copy import deepcopy
from dataclasses import replace
import hashlib,json
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from model_selection_samples import sample,plan
from split_samples import rows_for
from model_persistence_samples import NativeLinear,NativeSerializer
from xdiyo_analytics.training import EstimatorAdapter,TrainingRunner,refit_model,save_model,load_model,JoblibSerializer
from xdiyo_analytics.experiments.recovery import pack,unpack

@pytest.mark.parametrize('layout',['match','team_match'])
@pytest.mark.parametrize('classifier',[False,True])
def test_whole_fitted_pipeline_roundtrip_predicts_without_refit_on_null_labels(tmp_path,monkeypatch,layout,classifier):
    dataset=sample(layout=layout,shuffle=True);dataset.X.loc[0,'wave']=np.nan
    if classifier:dataset.y['target']=(dataset.metadata.case%2).map({0:'away',1:'home'})
    def factory():
        model=LogisticRegression(C=.4,random_state=31) if classifier else Ridge(alpha=.4)
        return EstimatorAdapter(make_pipeline(SimpleImputer(),StandardScaler(),model),('predict','predict_proba') if classifier else ('predict',))
    fitted=refit_model(dataset,factory,train_positions=rows_for(dataset,range(12)),feature_columns=['wave','signal'])
    future=deepcopy(dataset)
    future.y=pd.DataFrame({name:pd.Series(pd.NA,index=dataset.y.index,dtype='string' if classifier else 'Float64') for name in dataset.y})
    rows=rows_for(dataset,range(12,16))[::-1];before=deepcopy(future)
    expected=fitted.predict(future,positions=rows)
    path=fitted.save(tmp_path/'fitted-model');manifest=json.loads((path/'model.json').read_text())
    assert manifest['format_id']==JoblibSerializer.format_id and 'metadata.json' in manifest['files']
    def forbidden(*args,**kwargs):raise AssertionError('restoring/predicting must not fit')
    monkeypatch.setattr(Ridge,'fit',forbidden);monkeypatch.setattr(LogisticRegression,'fit',forbidden)
    monkeypatch.setattr(StandardScaler,'fit',forbidden);monkeypatch.setattr(SimpleImputer,'fit',forbidden)
    restored=load_model(path);actual=restored.predict(future,positions=rows)
    assert restored.feature_columns==('wave','signal') and restored.target_columns==('target',)
    assert restored.layout==layout and restored.match_columns==fitted.match_columns
    for method in expected:pd.testing.assert_frame_equal(actual[method],expected[method])
    if classifier:assert actual['predict_proba'].columns.tolist()==[('target','away'),('target','home')]
    np.testing.assert_array_equal(restored.model.estimator.named_steps['standardscaler'].mean_,fitted.model.estimator.named_steps['standardscaler'].mean_)
    for name in ['X','y','metadata']:pd.testing.assert_frame_equal(getattr(future,name),getattr(before,name))

def test_training_result_requires_explicit_fold_choice_and_retains_its_contract(tmp_path):
    dataset=sample();training=TrainingRunner(lambda:EstimatorAdapter(Ridge(alpha=.2))).run(dataset,plan(dataset))
    with pytest.raises(ValueError,match='fold_id'):save_model(training,tmp_path/'ambiguous')
    with pytest.raises(KeyError,match='No fitted fold'):save_model(training,tmp_path/'absent',fold_id=99)
    path=save_model(training,tmp_path/'chosen',fold_id=1);restored=load_model(path);fold=training.folds[1]
    np.testing.assert_array_equal(restored.train_positions,fold.train_positions)
    pd.testing.assert_frame_equal(restored.predict(dataset,positions=fold.test_positions)['predict'],fold.predictions['predict'])
    single=replace(training,folds=[training.folds[0]])
    assert save_model(single,tmp_path/'single').exists()

def test_explicit_native_serializer_restores_prediction_state_and_requires_matching_format(tmp_path,monkeypatch):
    dataset=sample();fitted=refit_model(dataset,NativeLinear,train_positions=rows_for(dataset,range(12)))
    with pytest.raises(TypeError,match='native serializer'):save_model(fitted,tmp_path/'no-native')
    native=NativeSerializer();path=save_model(fitted,tmp_path/'native',serializer=native)
    with pytest.raises(ValueError,match='native serializer'):load_model(path)
    with pytest.raises(ValueError,match='format'):load_model(path,serializer=JoblibSerializer())
    expected=fitted.predict(dataset)
    monkeypatch.setattr(NativeLinear,'fit',lambda *args:pytest.fail('unexpected native fit'))
    restored=load_model(path,serializer=native)
    pd.testing.assert_frame_equal(restored.predict(dataset)['predict'],expected['predict'])
    assert not list(path.rglob('*.joblib'))

@pytest.mark.parametrize('failure',['writer_raises','writer_empty','bad_format'])
def test_failed_serialization_never_publishes_destination(tmp_path,failure):
    dataset=sample();fitted=refit_model(dataset,NativeLinear,train_positions=np.arange(10))
    class Serializer:
        format_id='' if failure=='bad_format' else 'test.failure.v1'
        def save(self,adapter,path):
            if failure=='writer_raises':
                (path/'partial.txt').write_text('partial');raise OSError('injected native write failure')
    destination=tmp_path/'model'
    with pytest.raises((TypeError,ValueError,OSError)):save_model(fitted,destination,serializer=Serializer())
    assert not destination.exists()

@pytest.mark.parametrize('problem',['changed_bytes','missing_file','path_escape','wrong_schema','missing_manifest','bad_adapter'])
def test_load_rejects_invalid_artifacts_before_returning_a_model(tmp_path,problem):
    dataset=sample();fitted=refit_model(dataset,NativeLinear,train_positions=np.arange(10));serializer=NativeSerializer()
    path=save_model(fitted,tmp_path/'model',serializer=serializer);manifest=json.loads((path/'model.json').read_text())
    if problem=='changed_bytes':(path/'model/state.npz').write_bytes(b'corrupted')
    elif problem=='missing_file':(path/'model/state.npz').unlink()
    elif problem=='path_escape':manifest['files']['../outside']='unused';(path/'model.json').write_text(json.dumps(manifest))
    elif problem=='wrong_schema':manifest['schema']=99;(path/'model.json').write_text(json.dumps(manifest))
    elif problem=='missing_manifest':path=tmp_path/'absent'
    else:
        class Bad(NativeSerializer):
            def load(self,directory):return object()
        serializer=Bad()
    with pytest.raises((ValueError,FileNotFoundError,TypeError)):load_model(path,serializer=serializer)

def test_saved_destination_is_never_overwritten_and_metadata_only_results_cannot_be_saved(tmp_path):
    dataset=sample();fitted=refit_model(dataset,lambda:EstimatorAdapter(Ridge()),train_positions=np.arange(10))
    path=save_model(fitted,tmp_path/'model');before={p.relative_to(path):p.read_bytes() for p in path.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):save_model(fitted,path)
    assert before=={p.relative_to(path):p.read_bytes() for p in path.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='no fitted model'):save_model(unpack(pack(fitted)),tmp_path/'metadata-only')
    with pytest.raises(ValueError,match='fold_id'):save_model(fitted,tmp_path/'wrong-fold',fold_id=0)

def test_checkpoint_proxy_saves_underlying_fitted_prediction_pipeline(tmp_path):
    from xdiyo_analytics.experiments import FootballExperiment,PreparedExperiment
    from xdiyo_analytics.selection import Candidate
    from xdiyo_analytics.training import CheckpointPolicy
    from xdiyo_analytics.splits import SplitPlan
    data=sample();splits=plan(data);splits=replace(splits,folds=splits.folds[:1])
    result=FootballExperiment('Checkpoint proxy',output_dir=tmp_path/'experiments').run(PreparedExperiment(data,splits),
        model=Candidate('Ridge',lambda:EstimatorAdapter(make_pipeline(StandardScaler(),Ridge()))),checkpoint_policy=CheckpointPolicy())
    path=save_model(result.training,tmp_path/'prediction-model');loaded=load_model(path)
    assert isinstance(loaded.model,EstimatorAdapter)
    fold=result.training.folds[0]
    pd.testing.assert_frame_equal(loaded.predict(data,positions=fold.test_positions)['predict'],fold.predictions['predict'])
