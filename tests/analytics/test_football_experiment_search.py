from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from football_experiment_samples import prepared,ridge,post
from model_selection_samples import plan,sample,FixedAdapter
from test_model_selection_nested import outer_plan,inner_plan
from xdiyo_analytics.experiments import FootballExperiment,PreparedExperiment,SelectionSummary
from xdiyo_analytics.selection import Candidate,ModelSelection,MetricSelection,WeightedSelection
from xdiyo_analytics.training import EstimatorAdapter
from sklearn.linear_model import Ridge
from functools import partial
from xdiyo_analytics.experiments import ExperimentStore

class CountingFactory:
    def __init__(self,alpha,events,gate=None):self.alpha=alpha;self.events=events;self.gate=gate
    def cache_key(self):return {'alpha':self.alpha,'adapter':'counting Ridge'}
    def __call__(self):
        owner=self
        class Model(EstimatorAdapter):
            def fit(self,context):
                owner.events.append((owner.alpha,tuple(context.metadata.case)))
                if owner.gate is not None and not owner.gate.exists():
                    raise RuntimeError('injected process interruption before fit')
                return super().fit(context)
        return Model(Ridge(alpha=self.alpha))

def test_interrupted_search_recovers_only_completed_trials_then_fresh_holdout(tmp_path):
    data=prepared(holdout=True);events=[];gate=tmp_path/'continue.flag'
    candidates=[Candidate('completed',CountingFactory(.01,events),config={'alpha':.01}),
                Candidate('interrupted',CountingFactory(1.,events,gate),config={'alpha':1.})]
    search=ModelSelection(candidates,metrics='mse')
    experiment=FootballExperiment('Interrupted grid',output_dir=tmp_path)
    with pytest.raises(RuntimeError,match='injected process interruption'):
        experiment.run(data,model_selection=search,selection_plan=plan(data.dataset),post_analysis=post())
    first=experiment.store.read_runs();completed=next(r for r in first if r['status']=='complete')
    assert len(first)==2 and completed['role']=='trial'
    assert len(events)==3 and sum(e[0]==.01 for e in events)==2
    gate.write_text('resume')
    resumed=experiment.run(data,model_selection=search,selection_plan=plan(data.dataset),post_analysis=post())
    assert not resumed.reused and resumed.selection.trials[0].saved_run_id==completed['run_id']
    assert all(f.model is None for f in resumed.selection.trials[0].training.folds)
    assert len(events)==6
    records=experiment.store.read_runs()
    assert sum(r['role']=='final' for r in records)==1
    assert sum(r['role']=='trial' and r['status']=='complete' for r in records)==2
    assert resumed.record['selected_trial_id']==resumed.selection.winner.saved_run_id
    assert resumed.record['run_group']==completed['run_group']
    assert any(s.name=='selection/comparison' for s in resumed.report.studies)
    assert 'Candidate comparison' in (resumed.path/'report.html').read_text(encoding='utf-8')
    assert {m['study'] for m in resumed.record['metrics']}=={'Errors'}
    result=experiment.run(data,model_selection=search,selection_plan=plan(data.dataset),post_analysis=post())
    assert result.reused and len(events)==6 and isinstance(result.selection,SelectionSummary)
    assert result.selection.winners['holdout']['saved_trial_id']==resumed.selection.winner.saved_run_id
    pd.testing.assert_frame_equal(result.training.folds[0].predictions['predict'],resumed.training.folds[0].predictions['predict'])

@pytest.mark.parametrize('layout',['match','team_match'])
@pytest.mark.parametrize('source_kind',['list','iterator'])
def test_nested_orchestration_has_separate_winners_only_outer_predictions_and_reloads(tmp_path,layout,source_kind):
    data=sample(layout=layout,shuffle=True);data.y['target']=np.where(data.metadata.case<8,0.,10.)
    scoped=[]
    def inner(population):
        scoped.append(set(population.metadata.case))
        return inner_plan(population)
    # Keep diagnostic state outside the identity; the declared factory is fixed.
    class Inner:
        def cache_key(self):return {'split':'two local folds'}
        def __call__(self,population):return inner(population)
    candidates=[Candidate(str(v),lambda v=v:FixedAdapter(v),config={'constant':v}) for v in (0.,10.)]
    preparation=PreparedExperiment(data,outer_plan(data))
    experiment=FootballExperiment('Nested',output_dir=tmp_path)
    source=iter(candidates) if source_kind=='iterator' else candidates
    result=experiment.run(preparation,model_selection=ModelSelection(source,metrics='mse'),
        inner_plan_factory=Inner(),post_analysis=post())
    assert scoped==[set(range(6)),set(range(8,14))]
    assert [s.winner.candidate.name for s in result.selection.selections.values()]==['0.0','10.0']
    assert result.record['name']=='Tuned per fold' and result.record['selected_trial_id'] is None
    records=experiment.store.read_runs();assert len(records)==5 and sum(r['role']=='final' for r in records)==1
    assert len({r['run_group'] for r in records if r['role']=='trial'})==2
    loaded=experiment.load(result.record['run_id'])
    assert set(loaded.selection.winners)=={'0','1'}
    for stored,original in zip(loaded.training.folds,preparation.split_plan.folds):
        assert stored.model is None
        np.testing.assert_array_equal(stored.test_positions,original.test)
        np.testing.assert_array_equal(stored.score_positions,original.score)
        assert set(stored.metadata.case)==set(data.metadata.iloc[original.test].case)
    assert len(loaded.selection.comparison)==4 and len(loaded.report.studies)==2


class CountingFixedFactory:
    def __init__(self,value,models):self.value=value;self.models=models
    def cache_key(self):return {'constant':self.value}
    def __call__(self):
        model=FixedAdapter(self.value)
        self.models.append(model)
        return model


class SequentialCandidates:
    """Fresh candidate declarations for each outer fold, with stable identity."""
    def __init__(self):self.calls=0;self.created=[];self.models=[]
    def cache_key(self):return {'constants_by_outer_fold':[7.,14.]}
    def __call__(self):
        self.calls+=1
        value=float(self.calls*7)
        candidate=Candidate(f'constant {value}',CountingFixedFactory(value,self.models),config={'constant':value})
        self.created.append(candidate)
        return [candidate]


@pytest.mark.parametrize('layout',['match','team_match'])
def test_nested_callable_orchestration_preserves_fresh_sources_and_trial_recovery(tmp_path,layout):
    data=sample(layout=layout,shuffle=True)
    preparation=PreparedExperiment(data,outer_plan(data))
    experiment=FootballExperiment('Callable nested orchestration',output_dir=tmp_path)
    first_source=SequentialCandidates()
    first=experiment.run(preparation,model_selection=ModelSelection(first_source,metrics='mse'),
                         inner_plan_factory=inner_plan,config={'report_view':1})
    assert first_source.calls==2 and len({id(c) for c in first_source.created})==2
    assert len(first_source.models)==6 and len({id(m) for m in first_source.models})==6
    saved=[]
    for outer_id,selection in first.selection.selections.items():
        value=float((outer_id+1)*7)
        winner=selection.winner
        saved.append(winner.saved_run_id)
        assert winner.candidate.config=={'constant':value}
        record=experiment.store.load_run(winner.saved_run_id)['record']
        assert record['config']=={'constant':value}
        for fold in winner.training.folds:
            np.testing.assert_array_equal(fold.predictions['predict'].target,value)
        np.testing.assert_array_equal(first.training.folds[outer_id].predictions['predict'].target,value)
    # A changed final view still recovers each fold's original fitting evidence.
    second_source=SequentialCandidates()
    second=experiment.run(preparation,model_selection=ModelSelection(second_source,metrics='mse'),
                          inner_plan_factory=inner_plan,config={'report_view':2})
    assert second_source.calls==2
    assert len(second_source.models)==2  # Only fresh outer evaluations are fitted.
    assert all(new is not old for new in second_source.models for old in first_source.models)
    assert [part.winner.saved_run_id for part in second.selection.selections.values()]==saved
    assert all(f.model is None for part in second.selection.selections.values() for f in part.winner.training.folds)
    assert len(experiment.store.read_runs(role='trial'))==2
    assert len(experiment.store.read_runs(role='final'))==2
    for old,new in zip(first.training.folds,second.training.folds):
        pd.testing.assert_frame_equal(old.predictions['predict'],new.predictions['predict'])
    cached_source=SequentialCandidates()
    cached=experiment.run(preparation,model_selection=ModelSelection(cached_source,metrics='mse'),
                          inner_plan_factory=inner_plan,config={'report_view':2})
    assert cached.reused and cached.record['run_id']==second.record['run_id']
    assert cached_source.calls==0 and cached_source.models==[]


@pytest.mark.parametrize('change',['metrics','decision'])
def test_holdout_rescoring_reuses_foreign_group_trials_and_publishes_reloadable_final(tmp_path,change):
    preparation=prepared(holdout=True)
    events=[]
    candidates=[Candidate(f'Ridge {alpha}',CountingFactory(alpha,events),config={'alpha':alpha})
                for alpha in (.01,1.)]
    search=ModelSelection(candidates,metrics=['mse','mae'],decision=MetricSelection('mse'))
    experiment=FootballExperiment('Rescored holdout',output_dir=tmp_path)
    first=experiment.run(preparation,model_selection=search,selection_plan=plan(preparation.dataset),post_analysis=post())
    assert len(events)==5
    assert first.record['selected_trial_id']==first.selection.winner.saved_run_id
    original={trial.saved_run_id:(experiment.store.path/'runs'/trial.saved_run_id/'run.json').read_bytes()
              for trial in first.selection.trials}
    revised=ModelSelection(candidates,metrics='mae',decision=MetricSelection('mae')) if change=='metrics' else \
        ModelSelection(candidates,metrics=['mse','mae'],decision=WeightedSelection({'mse':.25,'mae':.75}))
    result=experiment.run(preparation,model_selection=revised,selection_plan=plan(preparation.dataset),post_analysis=post())
    assert len(events)==6  # Retained inner predictions are rescored; only the holdout fit is new.
    assert result.record['run_group']!=first.record['run_group']
    assert [trial.saved_run_id for trial in result.selection.trials]==list(original)
    assert all(f.model is None for trial in result.selection.trials for f in trial.training.folds)
    assert result.record['selected_trial_id'] is None
    winner=result.selection.winner.saved_run_id
    assert result.record['config']['selection']['holdout']['saved_trial_id']==winner
    expected_metrics={'mae'} if change=='metrics' else {'mse','mae'}
    assert {column[5:] for column in result.selection.comparison if column.startswith('raw::')}==expected_metrics
    for trial in result.selection.trials:
        assert trial.record['run_group']==first.record['run_group']
        assert {metric['metric'] for metric in trial.record['metrics']}==expected_metrics
        assert original[trial.saved_run_id]==(experiment.store.path/'runs'/trial.saved_run_id/'run.json').read_bytes()
    assert len(experiment.store.read_runs(role='trial'))==2
    assert len(experiment.store.read_runs(role='final'))==2
    loaded=experiment.load(result.record['run_id'])
    assert loaded.selection.winners['holdout']['saved_trial_id']==winner
    pd.testing.assert_frame_equal(loaded.training.prediction_frame(),result.training.prediction_frame())
    cached=experiment.run(preparation,model_selection=revised,selection_plan=plan(preparation.dataset),post_analysis=post())
    assert cached.reused and cached.record['run_id']==result.record['run_id'] and len(events)==6

def test_stateful_nested_source_recovery_identity_matches_the_fitted_candidates(tmp_path):
    data=sample();store=ExperimentStore(tmp_path,'Stateful nested source')
    class Source:
        def __init__(self):self.calls=0
        def __call__(self):
            self.calls+=1
            value=float(self.calls*7)
            return [Candidate(f'constant {value}',partial(FixedAdapter,value),config={'constant':value})]
    first_source=Source()
    first=ModelSelection(first_source,metrics='mse').run_nested(data,outer_plan(data),inner_plan,
        experiment=store,resume=True,recovery_namespace='same-execution')
    assert first_source.calls==2
    saved=[]
    for i,selection in first.selections.items():
        value=float((i+1)*7);trial=selection.winner;saved.append(trial.saved_run_id)
        assert trial.candidate.config=={'constant':value}
        assert trial.record['config']=={'constant':value}
        for fold in trial.training.folds:
            np.testing.assert_array_equal(fold.predictions['predict'].target,value)
    second_source=Source()
    second=ModelSelection(second_source,metrics='mse').run_nested(data,outer_plan(data),inner_plan,
        experiment=store,resume=True,recovery_namespace='same-execution')
    assert second_source.calls==2 and len(store.read_runs())==2
    assert [part.winner.saved_run_id for part in second.selections.values()]==saved
    assert all(f.model is None for part in second.selections.values() for f in part.winner.training.folds)
    for old,new in zip(first.training.folds,second.training.folds):
        pd.testing.assert_frame_equal(old.predictions['predict'],new.predictions['predict'])
        assert new.model is not None  # Outer evaluation still fits afresh.
