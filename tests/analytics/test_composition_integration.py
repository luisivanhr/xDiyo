from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from xdiyo_analytics.composition import *
from xdiyo_analytics.training import EstimatorAdapter,TrainingRunner
from xdiyo_analytics.ui import run_recipe,prepare_recipe
from xdiyo_analytics.ui.recipe import node,export_python,export_notebook,ModelFactory,catalog_for_ui
from xdiyo_analytics.evaluation import AllCombinations,compose_bets,DecisionLayer
from test_ui_workflow import ui_recipe
from test_bet_tickets import ledger
from test_model_composition import setup


def test_native_composite_recipe_run_export_and_reuse(ui_recipe,tmp_path,monkeypatch):
    recipe=deepcopy(ui_recipe)
    child=lambda alpha:node('composition.Estimator',estimator=node('sklearn.linear_model.Ridge',alpha=alpha),
        preprocessors=[node('sklearn.impute.SimpleImputer'),node('sklearn.preprocessing.StandardScaler')])
    recipe['model']=node('composition.ModelEnsemble',models={'one':child(.5),'two':child(2)},schema=node('composition.OutputSchema',target='corners'),reducer=node('composition.Mean'))
    recipe['preprocessors']=[]
    recipe['artifact_export']=node('experiments.ArtifactExport')
    recipe['post_reporters']={}
    prepared=prepare_recipe(json.loads(json.dumps(recipe)))
    assert len(prepared.dataset.X)>0
    compile(export_python(recipe),'recipe.py','exec')
    notebook=export_notebook(recipe)
    assert notebook['cells']
    result=run_recipe(recipe)
    expected=result.training.folds[0].predictions['predict'].copy()
    monkeypatch.setattr(Ridge,'fit',lambda *a,**k:pytest.fail('reused run must not fit'))
    again=run_recipe(recipe)
    pd.testing.assert_frame_equal(expected,again.training.folds[0].predictions['predict'])
    assert list(result.path.glob('exports/*/model_000/model/composition.json'))


def test_complete_ticket_and_uses_each_joint_probability_and_audits():
    data=ledger(('win',)*3).assign(stage='main',decision_at=pd.Timestamp('2024-12-31',tz='UTC'),quote_at=pd.Timestamp('2024-12-30',tz='UTC'))
    data['quote_id']=['q1','q2','q3']
    data['a::probability']=[.9,.9,.9]
    data['b::probability']=[.4,.8,.8]
    for name in ('a','b'):
        data[name+'::issued_at']=pd.Timestamp('2024-12-31',tz='UTC')
        data[name+'::artifact_vintage']=pd.Timestamp('2024-12-29',tz='UTC')
        data[name+'::trained_through']=pd.Timestamp('2024-12-28',tz='UTC')
    policy=AllCombinations(legs=2,stage_column='stage',probability_mode='independent',payoff='binary',
                           probability_columns={'a':'a::probability','b':'b::probability'},ticket_gate=DecisionLayer(('a','b'),threshold=.4))
    tickets,members,metrics=compose_bets(data,policy)
    assert len(tickets)==1  # .4*.8*4-1=.28 fails; .8*.8*4-1=1.56 passes
    assert len(tickets.attrs['decision_policy_audit'])==6
    # Blending leg probabilities is not mixing joint ticket probabilities.
    assert ((.9+.4)/2)*((.9+.8)/2) != pytest.approx((.9*.9+.4*.8)/2)
    changed=compose_bets(data.assign(settlement='loss'),policy)
    assert tickets.ticket_id.tolist()==changed[0].ticket_id.tolist()
    assert tickets.attrs['decision_policy_audit']==changed[0].attrs['decision_policy_audit']


def test_chronological_feature_availability_and_late_release_purge():
    data,plan=setup()
    from xdiyo_analytics.training.contracts import FitContext
    metadata=data.metadata.copy()
    metadata['feature_at']=metadata.kickoff_at
    metadata['label_at']=metadata.kickoff_at+pd.Timedelta('3h')
    metadata.loc[0,'label_at']=pd.Timestamp('2030-01-01',tz='UTC')
    context=FitContext(data.X,metadata,'match',data.match_columns,0,y=data.y)
    training=TrainingPlan(mode='chronological',min_train_groups=4,label_availability_column='label_at',feature_availability_column='feature_at')
    for train,test,boundary in training.splits(context):
        assert 0 not in train
        assert (metadata.label_at.iloc[train]<=boundary).all()
    with pytest.raises(ValueError,match='Features were unavailable'):
        list(training.splits(replace(context,metadata=metadata.assign(feature_at=metadata.kickoff_at+pd.Timedelta('1h')))))


def test_meta_schema_mismatch_fails_before_any_fit(monkeypatch):
    data,plan=setup()
    monkeypatch.setattr(Ridge,'fit',lambda *a,**k:pytest.fail('schema error must precede fit'))
    nodes={'a':ModelNode(EstimatorAdapter(Ridge()),{'predict':OutputSchema('target')}),
           'b':ModelNode(EstimatorAdapter(Ridge()),{'predict':OutputSchema('target',units='metres')}),
           'mean':ReducerNode(Mean(),(OutputRef('a'),OutputRef('b')),OutputSchema('target'))}
    with pytest.raises(ValueError,match='identical schemas'):
        TrainingRunner(lambda:CompositeModelAdapter(PredictionGraphSpec(nodes,{'predict':OutputRef('mean')}))).run(data,plan)
