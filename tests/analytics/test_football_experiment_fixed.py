from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from football_experiment_samples import prepared,ridge,post
from xdiyo_analytics import FootballExperiment
from xdiyo_analytics.training import EstimatorAdapter

@pytest.mark.parametrize('layout',['match','team_match'])
def test_real_estimator_fixed_run_reuse_and_load_without_computation(tmp_path,monkeypatch,layout):
    data=prepared(layout=layout);calls={'prepare':0,'fit':0,'predict':0}
    original_fit=EstimatorAdapter.fit;original_predict=EstimatorAdapter.predict
    def fit(self,context):calls['fit']+=1;return original_fit(self,context)
    def predict(self,context):calls['predict']+=1;return original_predict(self,context)
    monkeypatch.setattr(EstimatorAdapter,'fit',fit);monkeypatch.setattr(EstimatorAdapter,'predict',predict)
    def prepare():calls['prepare']+=1;return deepcopy(data)
    experiment=FootballExperiment('Fixed actual estimator',output_dir=tmp_path,prepare=prepare)
    first=experiment.run(model=ridge(),post_analysis=post(),name_fields=['alpha'])
    assert not first.reused and first.record['name']=='Ridge · alpha=0.1'
    assert calls=={'prepare':1,'fit':2,'predict':2}
    again=experiment.run(model=ridge(),post_analysis=post(),name_fields=['alpha'])
    assert again.reused and again.record['run_id']==first.record['run_id']
    assert calls=={'prepare':2,'fit':2,'predict':2}
    def forbidden():raise AssertionError('load must not prepare')
    reopened=FootballExperiment('Fixed actual estimator',output_dir=tmp_path,prepare=forbidden)
    loaded=reopened.load(first.record['run_id'])
    assert calls=={'prepare':2,'fit':2,'predict':2}
    assert len(experiment.store.read_runs())==1 and loaded.reused
    for a,b in zip(first.training.folds,loaded.training.folds):
        assert b.model is None
        pd.testing.assert_frame_equal(a.predictions['predict'],b.predictions['predict'])
        pd.testing.assert_frame_equal(a.metadata,b.metadata)
        np.testing.assert_array_equal(a.fit_positions,b.fit_positions)
    pd.testing.assert_frame_equal(loaded.dataset.X,first.dataset.X)
    pd.testing.assert_frame_equal(loaded.prepared.outputs['observed_history'],data.outputs['observed_history'])
    assert '<iframe' in loaded._repr_html_()
    assert calls=={'prepare':2,'fit':2,'predict':2}

def test_reuse_false_creates_new_execution_group_and_one_final_each(tmp_path):
    experiment=FootballExperiment('fresh executions',output_dir=tmp_path)
    one=experiment.run(prepared(),model=ridge(),post_analysis=post())
    two=experiment.run(prepared(),model=ridge(),post_analysis=post(),reuse=False)
    assert one.record['run_id']!=two.record['run_id'] and one.record['run_group']!=two.record['run_group']
    assert len(experiment.store.read_runs(role='final'))==2
    assert experiment.run(prepared(),model=ridge(),post_analysis=post()).record['run_id']==two.record['run_id']

@pytest.mark.parametrize('change',['X','y','metadata','order','definitions','split','prepared_config','experiment_config','run_config','model_alpha','name'])
def test_changed_execution_inputs_do_not_reuse_completed_result(tmp_path,change):
    data=prepared();experiment=FootballExperiment('identity',output_dir=tmp_path)
    one=experiment.run(data,model=ridge(),post_analysis=post())
    data=deepcopy(data);kwargs={}
    if change=='X':data.dataset.X.iloc[0,0]+=1
    elif change=='y':data.dataset.y.iloc[0,0]+=1
    elif change=='metadata':data.dataset.metadata.loc[0,'case']+=100
    elif change=='order':
        for name in ('X','y','metadata'):setattr(data.dataset,name,getattr(data.dataset,name).iloc[::-1].reset_index(drop=True))
    elif change=='definitions':data.dataset.definitions['revision']='new'
    elif change=='split':data.split_plan.folds[0].score=data.split_plan.folds[0].test
    elif change=='prepared_config':data.config['revision']=2
    elif change=='experiment_config':experiment.config['revision']=2
    elif change=='run_config':kwargs['config']={'revision':2}
    elif change=='name':kwargs['name']='another display name'
    model=ridge(.2 if change=='model_alpha' else .1)
    two=experiment.run(data,model=model,post_analysis=post(),**kwargs)
    assert not two.reused and two.record['run_id']!=one.record['run_id']
