"""Run with PYTHONPATH=src: python examples/composition_and_staking.py.

Synthetic, offline examples. Capital and fractions are illustrative research
settings. No real dataset, frozen experiment, network or financial action is used.
"""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from xdiyo_analytics.composition import (
    Estimator, ModelNode, OutputSchema, Mean, ModelEnsemble, ModelStack,
    OutputFeatures, TrainingPlan, TargetSpec, ResidualModel, UtilityTarget,
    LearnedTargetAdapter, SavedUtilityModel,
)
from xdiyo_analytics.datasets import ModelDataset
from xdiyo_analytics.splits import Fold, SplitPlan
from xdiyo_analytics.training import TrainingRunner, save_model, load_model
from xdiyo_analytics.evaluation import (
    FrozenTable, StakeContext, FractionalKelly, FixedFraction, RiskLimits,
    allocate_batch, replay_bankroll, settlements_from_legs, Parlay,
    AllCombinations, DecisionLayer, compose_bets, LearnedAllocation,
)
from xdiyo_analytics.ui.recipe import catalog_for_ui


def population(classifier=False):
    n = 40
    x = np.arange(n, dtype=float)
    metadata = pd.DataFrame({'event_id':np.arange(n), 'kickoff_at':pd.date_range('2024-01-01',periods=n,tz='UTC')})
    X = pd.DataFrame({'signal':np.sin(x), 'trend':x/40})
    y = pd.DataFrame({'target':(x.astype(int)%2) if classifier else 2+np.sin(x)+x/40})
    data = ModelDataset(X,y,metadata,'match',('event_id',),('event_id',),'match')
    plan = SplitPlan([Fold(np.arange(30),np.arange(30,40),np.arange(30,40),
                          {'fit_at':metadata.kickoff_at.iloc[30]})],n,np.arange(n))
    return data,plan


def models():
    results = {}
    for classifier in (False,True):
        data,plan = population(classifier)
        output = 'predict_proba' if classifier else 'predict'
        schema = OutputSchema('target',kind='probability',classes=(0,1),units='probability') if classifier else OutputSchema('target')
        child = Estimator(LogisticRegression() if classifier else Ridge(),
                          preprocessors=(StandardScaler(),),prediction_methods=(output,))
        base = ModelNode(child,{output:schema})
        training = TrainingPlan(mode='chronological',n_splits=3,min_train_groups=8,
                                availability_delay='3h',features_as_of_issue=True)
        ensemble = ModelEnsemble({'first':child,'second':deepcopy(child)},schema,
                                 Mean((1,2),schema.kind),output=output)
        blend = TrainingRunner(ensemble.build).run(data,plan)
        results['probability_blend' if classifier else 'regression_mean'] = blend
        stack = ModelStack({'first':base},child,
            OutputFeatures(('probability_0','probability_1') if classifier else ('base_prediction',)),
            schema,training,output=output)
        results['classification_stack' if classifier else 'regression_stack'] = TrainingRunner(stack.build).run(data,plan)
        for mode,kind in [('algorithmic_residual','residual'),('honest_error_meta','oof_error')]:
            correction = ResidualModel(base,Estimator(Ridge()),replace(training,mode=mode),
                TargetSpec(kind,'log_loss' if classifier else 'squared_error','logit' if classifier else 'identity'),learning_rate=.2)
            results[mode+('_classification' if classifier else '_regression')] = TrainingRunner(lambda:deepcopy(correction)).run(data,plan)
        # The same typed specs can be used as recipe['model']; child preparation
        # replaces top-level preprocessing. Encode/build is a native JSON round trip.
        spec = catalog_for_ui().encode(ensemble)
        assert isinstance(catalog_for_ui().build(spec),ModelEnsemble)
        with TemporaryDirectory(prefix='xdiyo-composition-') as folder:
            path = save_model(blend,Path(folder)/'blend')
            restored = load_model(path).predict(data,positions=np.arange(30,40))
            pd.testing.assert_frame_equal(restored[output],blend.folds[0].predictions[output])
    return results


def tickets():
    legs = pd.DataFrame(dict(fold_id=[0]*3,row_position=range(3),event_id=range(3),competition_id=[1]*3,
        season_id=[1]*3,source_season=['24_25']*3,round=[1]*3,
        kickoff_at=pd.date_range('2025-01-02',periods=3,freq='h',tz='UTC'),
        bet=['over']*3,take=[True]*3,odds=[2.]*3,stake=[20.]*3,
        settlement=['win','win','loss'],p_win=[.8]*3,p_push=[0.]*3,p_loss=[.2]*3))
    # Unchanged nominal benchmark: no capital/context/policy is required.
    fixed,_,_ = compose_bets(legs,Parlay(size=1,stake=1.))
    assert fixed.stake.tolist()==[1.,1.,1.]
    legs = legs.assign(stage='main',decision_at=pd.Timestamp('2025-01-01',tz='UTC'),
                       quote_at=pd.Timestamp('2024-12-31',tz='UTC'),quote_id=['q1','q2','q3'])
    for name,values in [('a',[.9,.9,.9]),('b',[.4,.8,.8])]:
        legs[name+'::probability']=values
        for field,time in [('issued_at','2025-01-01'),('artifact_vintage','2024-12-30'),('trained_through','2024-12-29')]:
            legs[name+'::'+field]=pd.Timestamp(time,tz='UTC')
    selection = AllCombinations(legs=2,stage_column='stage',probability_mode='independent',payoff='binary',
        probability_columns={'a':'a::probability','b':'b::probability'},ticket_gate=DecisionLayer(('a','b'),threshold=.4))
    gated,_,_ = compose_bets(legs,selection)
    assert len(gated)==1
    batch = pd.DataFrame(dict(nominal_stake=[1.,1.],odds=[2.,4.],probability=[.6,.4],
        payoff=['binary']*2,probability_provenance=['heldout-synthetic-v1']*2,
        probability_issued_at=['2024-12-31']*2,decision_at=['2025-01-01']*2,
        fixture_keys=[(1,),(2,)]),index=pd.Index(['a','b'],name='ticket_id'))
    policy = FractionalKelly('research_units',alpha=.5,max_fraction=.2)
    allocation = allocate_batch(FrozenTable(batch),policy,StakeContext(100,100,'research_units','2025-01-01'),RiskLimits(per_ticket=15))
    membership = pd.DataFrame({'ticket_id':['a','b'],'leg_id':[1,2]})
    outcomes = pd.DataFrame({'available_at':['2025-01-03','2025-01-04'],'return_multiplier':[2.,0.]},index=[1,2])
    settlements = settlements_from_legs(membership,outcomes)
    later = batch.copy()
    later.loc['b','decision_at']='2025-01-02'
    replay = replay_bankroll(FrozenTable(later),settlements,FixedFraction(.75,'research_units'),
                             initial_capital=100,currency='research_units')
    assert replay.summary['ending_wealth']==150.
    assert len(replay.report().artifacts)==3
    return fixed,gated,allocation,replay


def learned_allocation():
    data,plan = population()
    # Quotes are explicit decision/allocation inputs, not base forecasting inputs.
    data.X = pd.DataFrame({'odds':2+np.arange(40)%3})
    data.metadata['selection_id']='synthetic-rule-v1'
    data.metadata['prediction_origin']='fixed_rule'
    data.metadata['issued_at']=data.metadata.kickoff_at
    data.metadata['trained_through']=pd.Timestamp('2023-12-01',tz='UTC')
    data.metadata['available_at']=data.metadata.kickoff_at+pd.Timedelta('3h')
    outcomes = pd.DataFrame({'available_at':data.metadata.available_at,
                            'net_return_per_unit':np.where(np.arange(40)%2,1.,-1.)})
    # Label creation is retrospective; adapter fit rechecks availability on every fold.
    data.y = UtilityTarget('allocation',scale=.02,max_fraction=.05).build(outcomes,cutoff='2024-03-01')
    adapter = LearnedTargetAdapter(Estimator(DecisionTreeRegressor(max_depth=1,random_state=1)),
                                  TargetSpec('allocation'),'available_at','synthetic-rule-v1')
    result = TrainingRunner(lambda:deepcopy(adapter)).run(data,plan)
    batch = pd.DataFrame({'odds':[2.,3.],'nominal_stake':[1.,1.],'decision_at':['2025-01-01']*2},index=['a','b'])
    with TemporaryDirectory(prefix='xdiyo-utility-') as folder:
        path = save_model(result,Path(folder)/'allocation')
        pinned = SavedUtilityModel(str(path),sha256((path/'model.json').read_bytes()).hexdigest())
        policy = LearnedAllocation(pinned,('odds',),'research_units')
        result = allocate_batch(FrozenTable(batch),policy,StakeContext(100,100,'research_units','2025-01-01'))
        assert result.amounts.between(0,5).all()
    return result


def main():
    fitted = models()
    _,gated,allocation,replay = tickets()
    learned = learned_allocation()
    print('Synthetic model examples:', ', '.join(fitted))
    print('Ticket AND accepted:',len(gated),'; Kelly amounts:',allocation.amounts.tolist())
    print('Closed replay:',replay.summary)
    print('Declared learned allocation:',learned.amounts.tolist())


if __name__ == '__main__':
    main()
