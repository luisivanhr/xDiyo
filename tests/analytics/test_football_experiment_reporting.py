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
from xdiyo_analytics.experiments.recovery import load_bundle

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

def test_leaderboard_snapshot_is_published_with_html_and_load_never_recomputes(tmp_path,monkeypatch):
    from xdiyo_analytics.experiments import store as store_module
    experiment=FootballExperiment('Saved leaderboard',output_dir=tmp_path)
    publications=[]
    publish=store_module._publish
    def inspect_publication(stage,destination):
        bundle=load_bundle(stage/'recovery.json')
        board=next(s for s in bundle['report'].studies if s.partition=='experiment')
        assert board.result.tables['leaderboard'].run_id.tolist()==[destination.name]
        html=(stage/'report.html').read_text(encoding='utf-8')
        assert 'Experiment leaderboard' in html and destination.name in html
        assert experiment.store.read_runs()==[]
        publications.append(destination.name)
        publish(stage,destination)
    monkeypatch.setattr(store_module,'_publish',inspect_publication)
    result=experiment.run(prepared(holdout=True),model=ridge(),post_analysis=with_leaderboard())
    assert publications==[result.record['run_id']]
    def unexpected_run(*args,**kwargs):
        pytest.fail('Loading a snapshot must not run a reporter')
    monkeypatch.setattr(ExperimentLeaderboardReporter,'run',unexpected_run)
    loaded=experiment.load(result.record['run_id'])
    assert [s.name for s in loaded.report.studies]==[s.name for s in result.report.studies]
    actual=next(s for s in loaded.post_report.studies if s.partition=='experiment')
    expected=next(s for s in result.post_report.studies if s.partition=='experiment')
    pd.testing.assert_frame_equal(actual.result.tables['leaderboard'],expected.result.tables['leaderboard'])
    assert actual.result.artifacts[1].data==expected.result.artifacts[1].data
    tables=json.loads((result.path/'tables.json').read_text(encoding='utf-8'))
    assert all(table['study']!='Board' for table in tables)

def test_reuse_replaces_saved_leaderboard_without_refitting_or_rewriting_snapshot(tmp_path):
    from test_football_experiment_search import CountingFactory
    from xdiyo_analytics.selection import Candidate
    events=[]
    experiment=FootballExperiment('Refreshed leaderboard',output_dir=tmp_path)
    preparation=prepared(holdout=True)
    candidate=Candidate('counted',CountingFactory(.1,events),config={'alpha':.1})
    first=experiment.run(preparation,model=candidate,post_analysis=with_leaderboard())
    second=experiment.run(preparation,model=ridge(1.),post_analysis=with_leaderboard())
    for _ in range(2):
        reused=experiment.run(preparation,model=candidate,post_analysis=with_leaderboard())
        boards=[s for s in reused.post_report.studies if s.partition=='experiment']
        assert reused.reused and reused.record['run_id']==first.record['run_id']
        assert len(boards)==1
        assert set(boards[0].result.tables['leaderboard'].run_id)=={first.record['run_id'],second.record['run_id']}
    assert len(events)==1
    saved=experiment.load(first.record['run_id'])
    boards=[s for s in saved.post_report.studies if s.partition=='experiment']
    assert len(boards)==1 and boards[0].result.tables['leaderboard'].run_id.tolist()==[first.record['run_id']]

def test_failed_experiment_reporter_does_not_publish_an_incomplete_final(tmp_path):
    experiment=FootballExperiment('Failed leaderboard',output_dir=tmp_path)
    report=post()
    report.reporters['Board']=ExperimentLeaderboardReporter(weights={})
    with pytest.raises(ValueError,match='weight'):
        experiment.run(prepared(holdout=True),model=ridge(),post_analysis=report)
    assert experiment.store.read_runs()==[]
    assert list((experiment.store.path/'runs').glob('*/run.json'))==[]

def test_experiment_reporters_receive_isolated_sorted_and_filtered_record_view(tmp_path):
    experiment=FootballExperiment('Record snapshot',output_dir=tmp_path)
    inspected=[]
    class InspectingLeaderboard(ExperimentLeaderboardReporter):
        def cache_key(self):
            return {'reporter':'snapshot inspection','weights':self.weights}
        def run(self,context):
            view=context.experiment
            records=view.read_runs()
            assert records==sorted(records,key=lambda item:(item['created_at'],item['run_id']))
            finals=view.read_runs(role='final')
            assert len(finals)==1 and len(view.read_runs(role='trial'))==2
            current=finals[0]
            assert view.read_runs(run_group=current['run_group'])==records
            assert view.read_runs(role='final',run_group=current['run_group'])==finals
            assert view.read_runs(run_group='absent')==[]
            with pytest.raises(ValueError,match='role'):
                view.read_runs(role='absent')
            assert view.path==experiment.store.path and view.manifest==experiment.store.manifest
            assert all(not hasattr(view,name) for name in ('save_run','save_failure','start_run','open_run','load_run'))
            assert all(record['run_id']!=current['run_id'] for record in experiment.store.read_runs())
            assert not (view.path/'runs'/current['run_id']).exists()
            current['config']['tampered']=True
            current['metrics'].clear()
            view.manifest['name']='tampered'
            assert view.read_runs(role='final')[0]['metrics']
            assert 'tampered' not in view.read_runs(role='final')[0]['config']
            assert view.manifest==experiment.store.manifest
            inspected.append(current['run_id'])
            return super().run(context)
    report=post();report.reporters['Board']=InspectingLeaderboard(weights={'mse':1.})
    preparation=prepared(holdout=True)
    result=experiment.run(preparation,model_selection=ModelSelection(
        [ridge(.1),replace(ridge(1.),name='Ridge 1')],metrics='mse'),
        selection_plan=plan(preparation.dataset),post_analysis=report)
    assert inspected==[result.record['run_id']]
    saved=experiment.store.read_runs(role='final')[0]
    assert saved['metrics'] and 'tampered' not in saved['config']
    assert result.post_report.studies[-1].result.tables['leaderboard'].run_id.tolist()==inspected

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
