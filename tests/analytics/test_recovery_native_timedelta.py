"""Native durations retain their type and full Python range in saved artifacts."""

from dataclasses import dataclass
from datetime import timedelta
import json

import pandas as pd
import pytest

from football_experiment_samples import prepared
from model_selection_samples import FixedAdapter
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.experiments import FootballExperiment
from xdiyo_analytics.experiments.recovery import pack, unpack
from xdiyo_analytics.reporting import Artifact, PredictionReporter, StudyResult
from xdiyo_analytics.selection import Candidate


NATIVE_DURATIONS = [timedelta.min, timedelta.max, timedelta(0), timedelta(microseconds=-1),
                    timedelta(days=-3, seconds=17, microseconds=29),
                    timedelta(days=4, seconds=86399, microseconds=999999)]


@pytest.mark.parametrize('value', NATIVE_DURATIONS)
def test_native_timedelta_scalar_round_trips_exact_components_and_type(value):
    encoded = pack(value)
    assert encoded == {'@': 'native_timedelta', 'days': value.days,
                       'seconds': value.seconds, 'microseconds': value.microseconds}
    actual = unpack(json.loads(json.dumps(encoded, allow_nan=False)))
    assert type(actual) is timedelta and actual == value
    assert (actual.days, actual.seconds, actual.microseconds) == (value.days, value.seconds, value.microseconds)


@pytest.mark.parametrize('value', [pd.Timedelta.min, pd.Timedelta.max, pd.Timedelta(0),
                                 pd.Timedelta(-1, unit='ns'), pd.Timedelta(1, unit='ns')])
def test_existing_pandas_timedelta_tag_keeps_nanosecond_semantics(value):
    encoded = {'@': 'timedelta', 'value': value.value}
    assert pack(value) == encoded
    actual = unpack(encoded)
    assert type(actual) is pd.Timedelta and actual.value == value.value


def duration_table():
    table = pd.DataFrame({'duration': pd.Series(NATIVE_DURATIONS, dtype=object),
                          'nested': [{'delay': value, 'pandas': pd.Timedelta(1, unit='ns')}
                                     for value in NATIVE_DURATIONS]})
    table.attrs = {'window': timedelta.max, 'nested': [timedelta(microseconds=-1)]}
    return table


def assert_duration_table(actual):
    expected = duration_table()
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.attrs == expected.attrs
    assert type(actual.attrs['window']) is timedelta
    assert type(actual.attrs['nested'][0]) is timedelta
    for actual_value, expected_value in zip(actual['duration'], NATIVE_DURATIONS):
        assert type(actual_value) is timedelta and actual_value == expected_value
    for item, expected_value in zip(actual['nested'], NATIVE_DURATIONS):
        assert type(item['delay']) is timedelta and item['delay'] == expected_value
        assert type(item['pandas']) is pd.Timedelta and item['pandas'].value == 1


def test_native_timedelta_in_library_dataclasses_series_attrs_and_mapping_keys():
    frame = duration_table()
    series = frame['duration'].copy()
    series.attrs = {'delay': timedelta.min}
    artifact = Artifact('table', frame, 'Native durations', {'delay': timedelta.max})
    restored = unpack(json.loads(json.dumps(pack({timedelta(microseconds=-1): (artifact, series)}))))
    key, (actual_artifact, actual_series) = next(iter(restored.items()))
    assert type(key) is timedelta and key == timedelta(microseconds=-1)
    assert type(actual_artifact) is Artifact
    assert type(actual_artifact.options['delay']) is timedelta
    assert actual_artifact.options['delay'] == timedelta.max
    assert_duration_table(actual_artifact.data)
    pd.testing.assert_series_equal(actual_series, series, check_exact=True)
    assert type(actual_series.attrs['delay']) is timedelta and actual_series.attrs['delay'] == timedelta.min


@dataclass(kw_only=True)
class DurationReporter(PredictionReporter):
    def run(self, context):
        return StudyResult('Native durations', tables={'durations': duration_table()})


class DurationAdapter(FixedAdapter):
    def fit(self, context):
        super().fit(context)
        self.training_history_ = duration_table()
        self.training_summary_ = {'elapsed': timedelta.max, 'attempts': {1: timedelta(microseconds=-1)},
                                  'pandas': pd.Timedelta(1, unit='ns')}
        return self


@pytest.mark.parametrize('metadata', [False, True])
def test_native_durations_publish_and_load_full_and_numerical_experiments(tmp_path, metadata):
    preparation = prepared(holdout=True)
    preparation.outputs['retained_duration'] = timedelta.max
    if metadata:
        preparation.split_plan.folds[0].metadata['duration'] = timedelta.min
    experiment = FootballExperiment('Native duration artifacts', output_dir=tmp_path)
    report = PostTrainingAnalysis({'durations': DurationReporter(type='overall', partition='score')})
    result = experiment.run(preparation, model=Candidate('Duration model', DurationAdapter), post_analysis=report)
    full = experiment.load(result.record['run_id'])
    assert type(full.prepared.outputs['retained_duration']) is timedelta
    assert full.prepared.outputs['retained_duration'] == timedelta.max
    record = experiment.store.save_run(result.training, result.post_report, name='Numerical durations', config={})
    numerical = experiment.store.load_run(record['run_id'])
    assert numerical['extra']['kind'] == 'legacy'
    for training, restored_report in ((full.training, full.post_report), (numerical['training'], numerical['report'])):
        assert_duration_table(restored_report.studies[0].result.tables['durations'])
        fold = training.folds[0]
        assert_duration_table(fold.training_history)
        if metadata:
            assert type(fold.fold_metadata['duration']) is timedelta and fold.fold_metadata['duration'] == timedelta.min
        assert type(fold.training_summary['elapsed']) is timedelta and fold.training_summary['elapsed'] == timedelta.max
        assert type(fold.training_summary['attempts'][1]) is timedelta
        assert fold.training_summary['attempts'][1] == timedelta(microseconds=-1)
        assert type(fold.training_summary['pandas']) is pd.Timedelta and fold.training_summary['pandas'].value == 1


@pytest.mark.parametrize('field,value', [('days', True), ('days', 1.5), ('days', 1000000000),
                                        ('seconds', -1), ('seconds', 86400), ('microseconds', 1000000)])
def test_native_timedelta_decoder_rejects_noncanonical_components(field, value):
    encoded = {'@': 'native_timedelta', 'days': 0, 'seconds': 0, 'microseconds': 0, field: value}
    with pytest.raises(ValueError, match='normalized integer'):
        unpack(encoded)


def test_arbitrary_timedelta_subclasses_are_not_silently_coerced():
    class AnnotatedDuration(timedelta):
        pass
    value = AnnotatedDuration(seconds=3)
    value.annotation = 'additional behavior'
    with pytest.raises(TypeError, match='AnnotatedDuration'):
        pack(value)
