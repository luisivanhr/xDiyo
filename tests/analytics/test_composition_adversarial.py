from dataclasses import replace
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from xdiyo_analytics.composition import *
from xdiyo_analytics.training import EstimatorAdapter,TrainingRunner,ProbabilityCalibrator
from xdiyo_analytics.training.contracts import FitContext,PredictionContext
from xdiyo_analytics.evaluation import *
from xdiyo_analytics.evaluation.stake_policy import AllocationResult
from test_stake_bankroll import batch
from test_model_composition import setup
from model_selection_samples import sample
from split_samples import rows_for


@pytest.mark.parametrize('kind',['missing','duplicate','extra','negative','nan','infinite'])
def test_bad_allocations_never_reach_ledger(kind):
    from xdiyo_analytics.evaluation.stake_policy import allocate_batch
    class Bad:
        def allocate(self,tickets,context):
            values=pd.Series([1.,2.],index=['a','b'])
            if kind=='missing':values=values.iloc[:1]
            if kind=='duplicate':values.index=['a','a']
            if kind=='extra':values.loc['c']=3.
            if kind=='negative':values.iloc[0]=-1
            if kind=='nan':values.iloc[0]=np.nan
            if kind=='infinite':values.iloc[0]=np.inf
            return AllocationResult(values,pd.DataFrame(index=values.index))
    with pytest.raises(ValueError):
        allocate_batch(FrozenTable(batch()),Bad(),StakeContext(100,100,'u','2025-01-01'))


@pytest.mark.parametrize('value',[-1,np.nan,np.inf,1.1])
def test_invalid_kelly_fraction(value):
    with pytest.raises(ValueError):
        FractionalKelly('u',alpha=value).allocate(FrozenTable(batch()),StakeContext(100,100,'u','2025-01-01'))


@pytest.mark.parametrize('layout',['match','team_match'])
def test_inner_groups_preserve_partners_and_unavailable_outer_features(layout):
    data=sample(layout=layout,shuffle=True)
    # Adjacent fixtures share a round and a kickoff batch.
    data.metadata['round']=(data.metadata.case//2).astype(int)
    data.metadata['kickoff_at']=pd.Timestamp('2024-01-01',tz='UTC')+pd.to_timedelta(data.metadata.case//2,unit='d')
    data.metadata['label_at']=data.metadata.kickoff_at+pd.Timedelta('3h')
    context=FitContext(data.X,data.metadata,layout,data.match_columns,0,y=data.y)
    plan=TrainingPlan(mode='chronological',min_train_groups=2,group_by=('competition_id','season_id','round'),label_availability_column='label_at',features_as_of_issue=True)
    for train,test,boundary in plan.splits(context):
        for rows in (train,test):
            cases=set(data.metadata.iloc[rows].case)
            assert set(rows)==set(rows_for(data,cases))
        assert data.metadata.kickoff_at.iloc[train].max()<boundary
        assert set(data.metadata.iloc[train]['round']).isdisjoint(data.metadata.iloc[test]['round'])


def test_prediction_schema_missing_keys_classes_units_and_voting():
    index=pd.Index([10,20],name='row_position')
    meta=pd.DataFrame({'event_id':[1,2],'kickoff_at':['2025-01-01']*2},index=index)
    context=PredictionContext(pd.DataFrame({'x':[1,2]},index=index),meta,'match',('event_id',),0)
    schema=OutputSchema('y',kind='probability',classes=('home','away'),units='probability')
    p=pd.DataFrame([[.2,.8],[.7,.3]],index=index,columns=pd.MultiIndex.from_tuples([('y','home'),('y','away')]))
    schema.validate(p,context)
    for bad in (p.iloc[::-1],p.iloc[:,::-1],p.assign(**{})*2):
        with pytest.raises(ValueError):schema.validate(bad,context)
    with pytest.raises(ValueError,match='identities'):
        schema.validate(p,replace(context,metadata=meta.assign(event_id=1)))
    labels=replace(schema,kind='labels')
    f=pd.DataFrame({'y':['home','away']},index=index)
    votes=HardVote(True).reduce([f,f],[labels,labels])
    assert votes.values.tolist()==[[1.,0.],[0.,1.]]
    node=ModelNode(None,{'predict':labels})
    graph=PredictionGraphSpec({'a':node,'v':ReducerNode(HardVote(True),(OutputRef('a'),),replace(schema,kind='vote_fraction'))},{'predict_proba':OutputRef('v')})
    with pytest.raises(ValueError,match='vote'):
        graph.order()


def test_frozen_model_requires_matching_artifact_cutoffs_and_never_fits():
    from model_selection_samples import FixedAdapter
    data,plan=setup()
    class Frozen(FixedAdapter):
        def fit(self,context):raise AssertionError('frozen model cannot fit')
    model=Frozen(3.)
    model.targets=['target']
    evidence=dict(artifact_id='verified-local-hash',artifact_vintage='2023-12-01',trained_through='2023-11-30')
    model.training_summary_={'frozen_provenance':evidence}
    node=ModelNode(model,{'predict':OutputSchema('target')},frozen=True,**evidence)
    graph=PredictionGraphSpec({'saved':node},{'predict':OutputRef('saved')})
    result=TrainingRunner(lambda:CompositeModelAdapter(graph)).run(data,plan)
    assert result.folds[0].predictions['predict'].iloc[:,0].eq(3).all()
    late=replace(node,artifact_vintage='2030-01-01')
    with pytest.raises(ValueError,match='provenance'):
        TrainingRunner(lambda:CompositeModelAdapter(PredictionGraphSpec({'saved':late},graph.outputs))).run(data,plan)


def test_margin_calibration_is_inside_child_fits_and_keeps_original_keys():
    from sklearn.svm import SVC
    from xdiyo_analytics.composition.training import fit_node
    data,plan=setup(True)
    context=FitContext(data.X,data.metadata,'match',data.match_columns,0,{'fit_at':'2024-02-01'},y=data.y)
    policy=ProbabilityCalibrator(method='sigmoid',response_method='decision_function',fraction=.4,
        availability_delay='3h',min_calibration_rows=4,min_calibration_per_class=2)
    schema=OutputSchema('target',kind='probability',classes=('a','b'),units='probability')
    node=ModelNode(Estimator(SVC(probability=False),prediction_methods=('predict','decision_function')),{'predict_proba':schema},calibration=policy)
    model=fit_node(node,context)
    audit=model.calibrator.split_audit_ if hasattr(model,'calibrator') else model.calibration.split_audit_
    assert set(audit['original_fit_keys']).isdisjoint(audit['original_calibration_keys'])
    changed=deepcopy(context)
    changed.y.loc[audit['original_calibration_keys']]=changed.y.loc[audit['original_calibration_keys']].iloc[::-1].to_numpy()
    second=fit_node(node,changed)
    np.testing.assert_array_equal(model.estimator.estimator.support_vectors_,second.estimator.estimator.support_vectors_)
