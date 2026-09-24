import json
import numpy as np
import pandas as pd
import pytest
from football_experiment_samples import prepared,ridge,post
from xdiyo_analytics.experiments import FootballExperiment,ExperimentStore
from xdiyo_analytics.training import TrainingRunner

@pytest.mark.parametrize('save_predictions',[False,True])
@pytest.mark.parametrize('save_html',[False,True])
def test_legacy_load_retains_only_actual_artifacts_without_inventing_scopes(tmp_path,save_predictions,save_html):
    preparation=prepared(holdout=True)
    training=TrainingRunner(ridge().model_factory).run(preparation.dataset,preparation.split_plan)
    report=post().run(training);store=ExperimentStore(tmp_path,'Old artifacts')
    record=store.save_run(training,report,name='Legacy',config={},save_predictions=save_predictions,save_html=save_html)
    folder=store.path/'runs'/record['run_id']
    # Older versions did not retain match identities; do not guess from event IDs.
    document=json.loads((folder/'run.json').read_text());document.pop('match_columns')
    (folder/'run.json').write_text(json.dumps(document))
    def forbidden():raise AssertionError('load must not prepare')
    result=FootballExperiment('Old artifacts',output_dir=tmp_path,prepare=forbidden).load(record['run_id'])
    assert result.reused and result.prepared is None and result.dataset is None and result.splits is None
    if save_predictions:
        assert result.training.match_columns==() and result.training.folds[0].model is None
        pd.testing.assert_frame_equal(result.training.folds[0].predictions['predict'],training.folds[0].predictions['predict'])
    else:assert result.training is None
    assert result.post_report.studies
    assert all(study.partition=='saved' and len(study.row_positions)==0 and study.n_matches==0 for study in result.post_report.studies)
    assert all('not retained' in study.result.notes[0] for study in result.post_report.studies)
    if save_html:assert result.to_html()==(folder/'report.html').read_text(encoding='utf-8')
    assert store.find_completed('unknown') is None

@pytest.mark.parametrize('modern',[False,True])
def test_stored_artifact_must_stay_within_its_run_directory(tmp_path,modern):
    experiment=FootballExperiment('Contained artifacts',output_dir=tmp_path);preparation=prepared(holdout=True)
    if modern:record=experiment.run(preparation,model=ridge(),post_analysis=post()).record
    else:
        training=TrainingRunner(ridge().model_factory).run(preparation.dataset,preparation.split_plan)
        record=experiment.store.save_run(training,post().run(training),name='Legacy',config={})
    path=experiment.store.path/'runs'/record['run_id']/'run.json';record=json.loads(path.read_text())
    record['artifacts']['recovery' if modern else 'tables']='../../outside.json';path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='escapes'):experiment.load(record['run_id'])

def test_loading_unknown_failed_and_selection_trial_records_is_explicit(tmp_path):
    experiment=FootballExperiment('Invalid records',output_dir=tmp_path)
    with pytest.raises(KeyError,match='Unknown'):experiment.load('unknown')
    failed=experiment.store.save_failure(name='failed',config={},error='expected')
    with pytest.raises(ValueError,match='not complete'):experiment.load(failed['run_id'])
