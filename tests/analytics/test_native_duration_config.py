"""Native duration settings publish canonically without changing artifact types."""

from dataclasses import replace
from datetime import timedelta
from functools import partial
import json

import pandas as pd
import pytest

from football_experiment_samples import prepared, post
from model_selection_samples import FixedAdapter, plan
from test_football_experiment_search import CountingFactory
from xdiyo_analytics.experiments import FootballExperiment
from xdiyo_analytics.experiments.store import _json, configuration_hash
from xdiyo_analytics.selection import Candidate, ModelSelection


def descriptor(value):
    return {'__native_timedelta__': {'days': value.days, 'seconds': value.seconds,
                                     'microseconds': value.microseconds}}


@pytest.mark.parametrize('value', [timedelta.min, timedelta.max, timedelta(0), timedelta(microseconds=-1),
                                 timedelta(days=4, seconds=17, microseconds=23)])
def test_native_duration_configuration_has_exact_stable_distinct_identity(value):
    equivalent = timedelta(days=value.days, seconds=value.seconds, microseconds=value.microseconds)
    assert _json({'nested': [value]}) == {'nested': [descriptor(value)]}
    assert configuration_hash({'duration': value}) == configuration_hash({'duration': equivalent})
    assert configuration_hash({'duration': value}) != configuration_hash({'duration': str(value)})
    assert configuration_hash({'duration': value}) != configuration_hash({'duration': value.days})
    with pytest.raises(TypeError, match='data-only'):
        _json({'nested': [value]}, missing=True, temporal_descriptors=False)


def test_equivalent_normalized_durations_hash_identically_and_changes_do_not():
    assert configuration_hash({'duration': timedelta(seconds=-1)}) == configuration_hash(
        {'duration': timedelta(days=-1, seconds=86399)})
    assert configuration_hash({'duration': timedelta(hours=48)}) == configuration_hash(
        {'duration': timedelta(days=2)})
    assert configuration_hash({'duration': timedelta(microseconds=1)}) != configuration_hash(
        {'duration': timedelta(microseconds=2)})


def test_native_duration_subclasses_and_other_unsupported_config_values_still_fail():
    class CustomDuration(timedelta):
        pass
    for value in (CustomDuration(seconds=1), pd.Timedelta(seconds=1), {1: timedelta(seconds=1)}):
        with pytest.raises(TypeError):
            configuration_hash({'value': value})


@pytest.mark.parametrize('candidate_duration', [False, True])
def test_duration_config_scopes_publish_reload_and_reuse_without_refitting(tmp_path, candidate_duration):
    events = []
    experiment = FootballExperiment('Duration settings', output_dir=tmp_path,
                                     config={'window': timedelta.max})
    preparation = prepared(holdout=True)
    preparation.config = {'window': timedelta.min}
    candidate = Candidate('Counted durations', CountingFactory(.1, events),
                          config={'window': timedelta(hours=48)} if candidate_duration else {})
    run_config = {'delay': timedelta(microseconds=-1)}
    result = experiment.run(preparation, model=candidate, config=run_config, post_analysis=post())
    config = result.record['config']
    assert config['experiment']['window'] == descriptor(timedelta.max)
    assert config['preparation']['window'] == descriptor(timedelta.min)
    assert config['model'] == ({'window': descriptor(timedelta(days=2))} if candidate_duration else {})
    assert config['run']['delay'] == descriptor(timedelta(microseconds=-1))
    raw = json.loads((result.path / 'run.json').read_text(encoding='utf-8'))
    assert raw['config'] == config
    loaded = experiment.load(result.record['run_id'])
    assert type(loaded.prepared.config['window']) is timedelta
    assert loaded.prepared.config['window'] == timedelta.min
    assert loaded.record['config'] == config
    equivalent = replace(candidate, config={'window': timedelta(days=2)} if candidate_duration else {})
    reused = experiment.run(preparation, model=equivalent, config=run_config, post_analysis=post())
    assert reused.reused and reused.record['run_id'] == result.record['run_id'] and len(events) == 1
    changed = experiment.run(preparation, model=equivalent,
                             config={'delay': timedelta(microseconds=-2)}, post_analysis=post())
    assert not changed.reused and changed.record['config_hash'] != result.record['config_hash'] and len(events) == 2


def test_candidate_duration_settings_survive_selection_trial_and_final_publication(tmp_path):
    events = []
    preparation = prepared(holdout=True)
    candidates = [Candidate(str(alpha), CountingFactory(alpha, events),
                            config={'alpha': alpha, 'window': timedelta(days=2)}) for alpha in (.1, 1.)]
    experiment = FootballExperiment('Selection duration settings', output_dir=tmp_path)
    search = ModelSelection(candidates, metrics='mse')
    result = experiment.run(preparation, model_selection=search, selection_plan=plan(preparation.dataset),
                             post_analysis=post())
    assert len(events) == 5 and len(experiment.store.read_runs(role='trial')) == 2
    records = {item['run_id']: item for item in experiment.store.read_runs(role='trial')}
    for trial in result.selection.trials:
        assert type(trial.record['config']['window']) is timedelta
        assert records[trial.saved_run_id]['config']['window'] == descriptor(timedelta(days=2))
    loaded = experiment.load(result.record['run_id'])
    assert type(loaded.selection.winners['holdout']['config']['window']) is timedelta
    assert loaded.selection.winners['holdout']['config']['window'] == timedelta(days=2)
    assert loaded.record['config']['model']['window'] == descriptor(timedelta(days=2))
    reused = experiment.run(preparation, model_selection=search, selection_plan=plan(preparation.dataset),
                             post_analysis=post())
    assert reused.reused and reused.record['run_id'] == result.record['run_id'] and len(events) == 5


class SummaryAdapter(FixedAdapter):
    def __init__(self, summary):
        super().__init__()
        self.summary = summary

    def fit(self, context):
        super().fit(context)
        self.training_summary_ = self.summary
        return self


@pytest.mark.parametrize('native', [False, True])
def test_summary_duration_fallback_preserves_values_and_literal_descriptor_mappings(tmp_path, native):
    summary = {'literal': descriptor(timedelta(microseconds=-1)), 'ordinary': None}
    if native:
        summary.update(elapsed=timedelta.max, nested=[{'delay': timedelta(microseconds=-1)}])
    experiment = FootballExperiment('Summary durations', output_dir=tmp_path)
    result = experiment.run(prepared(holdout=True), model=Candidate('Summary', partial(SummaryAdapter, summary)))
    raw = json.loads((result.path / 'training.json').read_text(encoding='utf-8'))[0]
    if native:
        assert raw['summary_encoding'] == 'xdiyo.data-only.v1'
    else:
        assert 'summary_encoding' not in raw and {key: raw['summary'][key] for key in summary} == summary
    record = experiment.store.save_run(result.training, result.post_report, name='Numerical summary', config={})
    for training in (experiment.load(result.record['run_id']).training,
                     experiment.store.load_run(record['run_id'])['training']):
        actual = training.folds[0].training_summary
        assert {key: actual[key] for key in summary} == summary
        if native:
            assert type(actual['elapsed']) is timedelta and type(actual['nested'][0]['delay']) is timedelta
