import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from model_selection_samples import sample,plan,development,outer,candidate,FixedAdapter
from test_model_selection_nested import outer_plan,inner_plan
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.reporting import StudyResult,PerformanceReporter,ExperimentLeaderboardReporter
from xdiyo_analytics.selection import Candidate,ModelSelection,MetricSelection,SelectionDecision
from xdiyo_analytics.experiments import ExperimentStore

def test_recorded_failed_trials_saved_ids_and_separate_final_selection_link(tmp_path):
    data=sample();store=ExperimentStore(tmp_path,'selection')
    bad=Candidate('failed',lambda:FixedAdapter(fail=True),config={'deliberate_failure':True})
    result=ModelSelection([bad,candidate('good')],metrics='mse',on_error='record').run(data,plan(data),development_positions=development(data),experiment=store)
    records=store.read_runs();assert {r['role'] for r in records}=={'trial'}
    assert {r['status'] for r in records}=={'failed','complete'}
    assert all(r['run_group']==result.run_group for r in records)
    assert result.winner.saved_run_id in {r['run_id'] for r in records}
    assert result.winner.saved_run_id!=result.winner.trial_id
    assert result.comparison.set_index('run_id').loc[result.trials[0].trial_id,'error'].startswith('RuntimeError')
    board=PostTrainingAnalysis({'runs':ExperimentLeaderboardReporter(type='overall',weights={'mse':1.})}).run(experiment=store)
    assert board.studies[0].result.tables['leaderboard'].empty
    evaluated=result.evaluate(data,outer(data));report=PostTrainingAnalysis({'error':PerformanceReporter(type='overall',partition='score',metrics=['mse'])}).run(evaluated)
    saved=store.save_run(evaluated,report,name='Independent holdout',config=result.winner.candidate.config,
        role='final',run_group=result.run_group,selected_trial_id=result.winner.saved_run_id)
    assert saved['role']=='final' and saved['selected_trial_id']==result.winner.saved_run_id
    board=PostTrainingAnalysis({'runs':ExperimentLeaderboardReporter(type='overall',weights={'mse':1.})}).run(experiment=store)
    assert board.studies[0].result.tables['leaderboard'].run_id.tolist()==[saved['run_id']]

@pytest.mark.parametrize('failure',['save_run','save_failure'])
def test_storage_errors_propagate_even_when_candidate_errors_are_recorded(failure):
    class BrokenStore:
        def start_run(self,*args,**kwargs):return 'group'
        def save_run(self,*args,**kwargs):raise OSError('storage write failed')
        def save_failure(self,*args,**kwargs):raise OSError('storage failure write failed')
    data=sample();item=Candidate('candidate',lambda:FixedAdapter(fail=failure=='save_failure'))
    with pytest.raises(OSError,match='storage'):
        ModelSelection([item],metrics='mse',on_error='record').run(data,plan(data),development_positions=development(data),experiment=BrokenStore())

def test_raise_policy_preserves_failed_trial_then_propagates(tmp_path):
    data=sample();store=ExperimentStore(tmp_path,'failed search')
    with pytest.raises(RuntimeError,match='deliberate'):
        ModelSelection([Candidate('broken',lambda:FixedAdapter(fail=True))],metrics='mse').run(data,plan(data),development_positions=development(data),experiment=store)
    records=store.read_runs();assert len(records)==1 and records[0]['role']=='trial' and records[0]['status']=='failed'

def test_custom_numerical_betting_evidence_is_explicit_and_development_score_only(monkeypatch):
    data=sample();seen=[]
    class NetReturn:
        type='overall';partition='score';pooling=None;supported_types=('overall',)
        def run(self,context):
            cases=context.metadata.case.tolist();seen.append(cases)
            y=context.y.target.to_numpy();pred=context.predictions['predict'].target.to_numpy()
            # Declared toy payout: bet above 20, +1.5 for a win, -1 for a loss.
            value=float(np.where(pred>20,np.where(y>20,1.5,-1.),0.).mean())
            digest=hashlib.sha256(json.dumps([cases,y.tolist()]).encode()).hexdigest()
            table=pd.DataFrame([dict(metric='net_return',calculation='toy_fixed_payout',target='target',output='predict',
                value=value,direction='maximize',n=len(y),n_total=len(y),n_missing=0,status='ok',
                parameters='threshold=20;payout=1.5;unit=one',sample_hash=digest)])
            return StudyResult('Numerical evidence',tables={'metrics':table})
    def no_plots(*args,**kwargs):raise AssertionError('default search must not create figures')
    monkeypatch.setattr('xdiyo_analytics.reporting.studies._plotting',no_plots)
    result=ModelSelection([candidate('no bet',0,0),candidate('threshold forecast')],decision=MetricSelection('net_return'),
        evidence_reporters={'toy_return':NetReturn()}).run(data,plan(data),development_positions=development(data))
    assert result.winner.candidate.name=='threshold forecast'
    assert all(cases==[7,10,11] for cases in seen)
    assert result.comparison.set_index('name').loc['threshold forecast','raw::net_return']==1.

def test_nested_store_keeps_separate_trial_groups_and_one_explicit_final_record(tmp_path):
    data=sample();data.y['target']=np.where(data.metadata.case<8,0.,10.)
    store=ExperimentStore(tmp_path,'nested')
    candidates=[Candidate(str(v),lambda v=v:FixedAdapter(v),config={'constant':v}) for v in (0.,10.)]
    result=ModelSelection(candidates,metrics='mse').run_nested(data,outer_plan(data),inner_plan,experiment=store)
    records=store.read_runs();assert len(records)==4 and all(r['role']=='trial' for r in records)
    assert len({r['run_group'] for r in records})==2
    report=PostTrainingAnalysis({'outer error':PerformanceReporter(type='overall',partition='score',metrics=['mse'])}).run(result.training)
    saved=store.save_run(result.training,report,name='Nested evaluation',role='final',config={'outer_winners':{str(i):s.winner.saved_run_id for i,s in result.selections.items()}})
    assert saved['selected_trial_id'] is None
    assert sum(r['role']=='final' for r in store.read_runs())==1

@pytest.mark.parametrize('problem',['empty_source','duplicate_names','not_candidate','multiple_metrics','ic_without_rule','bad_error_policy','group_without_store','bad_decision_winner','bad_decision_table','all_failed'])
def test_invalid_search_or_decision_contracts_propagate(problem):
    data=sample();items=[candidate('good')];kwargs={'metrics':'mse'};run_kwargs={}
    if problem=='empty_source':items=[]
    elif problem=='duplicate_names':items=[candidate('same'),candidate('same')]
    elif problem=='not_candidate':items=[object()]
    elif problem=='multiple_metrics':kwargs['metrics']=['mse','mae']
    elif problem=='ic_without_rule':kwargs=dict(information_criteria=True)
    elif problem=='bad_error_policy':kwargs['on_error']='skip'
    elif problem=='group_without_store':run_kwargs['run_group']='missing'
    elif problem=='all_failed':items=[Candidate('bad',lambda:FixedAdapter(fail=True))];kwargs['on_error']='record'
    else:
        class BrokenDecision:
            def decide(self,trials):return SelectionDecision('absent' if problem=='bad_decision_winner' else trials[0].trial_id,pd.DataFrame({'wrong':[]}))
        kwargs['decision']=BrokenDecision()
    with pytest.raises((ValueError,TypeError)):
        ModelSelection(items,**kwargs).run(data,plan(data),development_positions=development(data),**run_kwargs)
