"""Remaining native boundary and nonempty exposure identity review cases."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from xdiyo_analytics.composition import (
    ModelNode, Estimator, ModelStack, ModelEnsemble, Mean, OutputSchema,
    OutputFeatures, TrainingPlan, ResidualModel, TargetSpec,
)
from xdiyo_analytics.composition.training import prediction_context
from xdiyo_analytics.training import TrainingRunner, save_model, load_model
from xdiyo_analytics.training.contracts import PredictionContext
from xdiyo_analytics.evaluation import FrozenTable, FixedStake, StakeContext, RiskLimits, allocate_batch, compose_bets, Parlay
from test_composition_review_staking import offers


@pytest.mark.parametrize('blank',['',' ','\t\n','\u00a0'])
@pytest.mark.parametrize('nested',[False,True])
@pytest.mark.parametrize('where',['selected','open'])
def test_blank_exposure_identities_reject_recursively(blank,nested,where):
    identity=(('league-A',blank),) if nested else blank
    selected=identity if where=='selected' else ('league-A',)
    outstanding=identity if where=='open' else 'league-A'
    batch=FrozenTable(pd.DataFrame({'nominal_stake':[1.],'decision_at':['2025-01-01'],
                                   'league_keys':[selected]},index=['ticket']))
    context=StakeContext(100,100,'u','2025-01-01',open_exposure=(('league_keys',outstanding,10.),))
    with pytest.raises(ValueError,match='complete nonempty'):
        allocate_batch(batch,FixedStake(10,'u'),context,RiskLimits(exposure_caps=(('league_keys',10.),)))


@pytest.mark.parametrize('blank',['',' \t '])
@pytest.mark.parametrize('where',['selected','open'])
@pytest.mark.parametrize('dimension',['league_keys','round_keys'])
def test_native_source_league_blank_cannot_disable_cap(blank,where,dimension):
    data=offers(1).drop(columns=['competition_id','season_id']).assign(source_league=blank if where=='selected' else 'league-A')
    key=blank if where=='open' else 'league-A'
    if dimension=='round_keys':
        key=(key,'24_25',1)
    context=StakeContext(100,100,'u','2025-01-01',open_exposure=((dimension,key,10.),))
    with pytest.raises(ValueError,match='complete nonempty'):
        compose_bets(data,Parlay(size=1),stake_policy=FixedStake(10,'u'),stake_context=context,
                     risk_limits=RiskLimits(exposure_caps=((dimension,10.),)))


def test_nonblank_exposure_ids_are_preserved_without_normalizing_or_rejecting_zero():
    for identity in (0,'league-A',' league-A '):
        batch=FrozenTable(pd.DataFrame({'nominal_stake':[1.],'decision_at':['2025-01-01'],
                                       'fixture':[identity]},index=['ticket']))
        context=StakeContext(100,100,'u','2025-01-01',open_exposure=(('fixture',identity,8.),))
        result=allocate_batch(batch,FixedStake(10,'u'),context,RiskLimits(exposure_caps=(('fixture',10.),)))
        assert result.amounts.iloc[0]==2.


def native_data():
    from xdiyo_analytics.labels import create_labels,BetOption,Outcome
    from xdiyo_analytics.datasets import assemble_dataset
    from xdiyo_analytics.splits import Fold,SplitPlan
    rows=[]
    for i in range(16):
        for side,team,other,result in [('home',1,2,'L'),('away',2,1,'W')]:
            rows.append(dict(event_id=i,team_id=team,opponent_id=other,side=side,result=result,
                status='cancelled' if i%2 else 'finished',is_awarded=False,
                kickoff_at=pd.Timestamp('2024-01-01',tz='UTC')+pd.Timedelta(days=i)))
    history=pd.DataFrame(rows)
    label=create_labels(history,{'target':BetOption(Outcome(perspective='home'),selection='win',
        void_statuses=('cancelled',),void_value=1.)})['target']
    features=history[['event_id','team_id','side']].copy()
    features['safe_feature']=1.
    data=assemble_dataset(features,label,layout='match')
    data.metadata['feature_at']=data.metadata.kickoff_at
    plan=SplitPlan([Fold(np.arange(12),np.arange(12,16),np.arange(12,16),
        {'fit_at':data.metadata.kickoff_at.iloc[12]})],16,np.arange(16))
    return data,plan


class NativeStatusBoundaryProbe:
    def fit(self,context): pass
    def predict(self,context):
        assert 'settlement::target' not in context.metadata
        assert 'label' not in context.definitions
        assert 'status' not in context.metadata
        assert 'is_awarded' not in context.metadata
        return {'predict':pd.DataFrame({'target':0.},index=context.X.index)}


@pytest.mark.parametrize('kind',['stack','residual','ensemble'])
def test_native_label_status_removed_for_oof_outer_and_restored_predictions(kind,tmp_path,monkeypatch):
    data,plan=native_data()
    original=deepcopy(data.metadata)
    # Confirms this is a native label-construction leak, not future labels in X.
    np.testing.assert_array_equal(data.metadata.status.eq('cancelled').astype(float),data.y.target)
    schema=OutputSchema('target')
    base=ModelNode(NativeStatusBoundaryProbe(),{'predict':schema})
    training=dict(min_train_groups=4,availability_delay='3h',feature_availability_column='feature_at')
    if kind=='stack':
        spec=ModelStack({'base':base},Estimator(Ridge()),OutputFeatures(('base',)),schema,TrainingPlan(mode='chronological',**training))
        factory=spec.build
    elif kind=='ensemble':
        factory=ModelEnsemble({'base':base},schema,Mean()).build
    else:
        factory=lambda:ResidualModel(base,Estimator(Ridge()),TrainingPlan(mode='honest_error_meta',**training),TargetSpec('oof_error'))
    result=TrainingRunner(factory).run(data,plan)
    fitted=result.folds[0].model
    if kind=='stack':
        assert fitted.oof_[('base','predict')].eq(0).all().all()
    elif kind=='residual':
        assert fitted.oof_predictions_.eq(0).all().all()
    path=save_model(result,tmp_path/'status-boundary')
    monkeypatch.setattr(NativeStatusBoundaryProbe,'fit',lambda *a,**k:pytest.fail('restore must not fit'))
    monkeypatch.setattr(Ridge,'fit',lambda *a,**k:pytest.fail('restore must not fit'))
    outputs=load_model(path).predict(data,positions=np.arange(12,16))
    for key,frame in outputs.items():
        pd.testing.assert_frame_equal(frame,result.folds[0].predictions[key])
    pd.testing.assert_frame_equal(data.metadata,original)  # reporting context remains intact


def test_status_boundary_removes_named_nested_metadata_but_preserves_audited_feature_input():
    data,_=native_data()
    X=data.X.copy()
    X['as_of_status_feature']=0.
    metadata=data.metadata.assign(as_of_status='scheduled')
    metadata.attrs={'status':'finished','nested':[{'is_awarded':True,'keep':1}]}
    context=PredictionContext(X,metadata,'match',data.match_columns,0,
        {'status':'finished','keep':1},{'is_awarded':True,'keep':2})
    safe=prediction_context(context)
    assert not {'status','is_awarded'} & set(safe.metadata)
    assert safe.metadata.attrs=={'nested':[{'keep':1}]}
    assert safe.fold_metadata=={'keep':1} and safe.definitions=={'keep':2}
    pd.testing.assert_frame_equal(safe.X,X)
    assert safe.metadata.as_of_status.eq('scheduled').all()
