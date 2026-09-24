from dataclasses import replace
import json
import pandas as pd
import pytest
from football_experiment_samples import prepared,ridge,post
from model_selection_samples import plan
from xdiyo_analytics.experiments import FootballExperiment
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.reporting import ExperimentLeaderboardReporter
from xdiyo_analytics.selection import ModelSelection

def with_leaderboard():
    report=post();report.reporters['Board']=ExperimentLeaderboardReporter(weights={'mse':1.})
    return report

def test_current_leaderboard_refreshes_after_save_and_reuse_without_recursive_metrics(tmp_path):
    preparation=prepared(holdout=True);experiment=FootballExperiment('Live leaderboard',output_dir=tmp_path)
    one=experiment.run(preparation,model=ridge(.1),post_analysis=with_leaderboard())
    board=one.post_report.studies[-1].result.tables['leaderboard']
    assert board.run_id.tolist()==[one.record['run_id']]
    two=experiment.run(preparation,model=ridge(1.),post_analysis=with_leaderboard())
    assert len(two.post_report.studies[-1].result.tables['leaderboard'])==2
    cached=experiment.run(preparation,model=ridge(.1),post_analysis=with_leaderboard())
    assert cached.reused and cached.record['run_id']==one.record['run_id']
    assert len(cached.post_report.studies[-1].result.tables['leaderboard'])==2
    assert sum(s.name=='Board' for s in cached.post_report.studies)==1
    assert {m['study'] for record in experiment.store.read_runs() for m in record['metrics']}=={'Errors'}

def test_generated_names_explicit_nested_fields_and_escaped_config_details(tmp_path):
    preparation=prepared(holdout=True);experiment=FootballExperiment('Names',output_dir=tmp_path)
    candidate=replace(ridge(),name='Ridge <config>',config={'model':{'alpha':.25},'text':'<script>unsafe</script>'})
    result=experiment.run(preparation,model=candidate,name_fields=['model.alpha'],post_analysis=post())
    assert result.record['name']=='Ridge <config> · model.alpha=0.25'
    board=experiment.leaderboard({'mse':1.})
    artifact=board.studies[0].result.artifacts[1]
    assert '<details>' in artifact.data and '&lt;script&gt;' in artifact.data
    assert '<script>unsafe</script>' not in artifact.data
    assert '&lt;config&gt;' in artifact.data
    with pytest.raises(KeyError,match='absent'):
        experiment.run(preparation,model=candidate,name_fields=['missing'],post_analysis=post())
    assert len(experiment.store.read_runs())==1

def test_final_only_leaderboard_and_opt_in_internal_trials(tmp_path):
    preparation=prepared(holdout=True);experiment=FootballExperiment('Final versus trials',output_dir=tmp_path)
    result=experiment.run(preparation,model_selection=ModelSelection([ridge(.1),replace(ridge(1.),name='Ridge 1')],metrics='mse'),
        selection_plan=plan(preparation.dataset),post_analysis=post())
    assert experiment.leaderboard({'mse':1.}).studies[0].result.tables['leaderboard'].run_id.tolist()==[result.record['run_id']]
    detailed=experiment.leaderboard({'mse':1.},include_trials=True).studies[0].result.tables['leaderboard']
    assert len(detailed)==3 and set(detailed.role)=={'trial','final'}
    assert len(result.report.studies)==2 and result.report.studies[0].name=='selection/comparison'
    assert {m['study'] for m in result.record['metrics']}=={'Errors'}
