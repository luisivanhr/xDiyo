"""Typed configuration values are distinct from user-written descriptor mappings."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from football_experiment_samples import prepared
from test_football_experiment_search import CountingFactory
from test_post_training_experiments import computed_run
from xdiyo_analytics.experiments import ExperimentStore, FootballExperiment
from xdiyo_analytics.experiments.store import _json, configuration_hash
from xdiyo_analytics.selection import Candidate


@dataclass
class Configuration:
    scale: int = 2


@pytest.mark.parametrize('value', [timedelta(days=1), np.timedelta64(3, 'ns'),
                                 np.array([1, 2], dtype='datetime64[D]'),
                                 datetime(2024, 1, 1, tzinfo=timezone.utc), Configuration()])
def test_typed_configuration_and_literal_descriptor_publish_distinctly(tmp_path, value):
    literal = _json(value)
    typed_config, literal_config = {'setting': value}, {'setting': literal}
    assert configuration_hash(typed_config) != configuration_hash(literal_config)
    expected = _json(typed_config)
    escaped = _json(literal_config)
    assert escaped == {'setting': {'__literal_config__': literal}}
    training, report = computed_run()
    store = ExperimentStore(tmp_path, 'Descriptor distinctions')
    finalized = []
    def finalize(record):
        finalized.append(record['config'])
        return report
    typed = store.save_run(training, report, name='Typed', config=typed_config, _finalize_report=finalize)
    mapped = store.save_run(training, report, name='Literal', config=literal_config, _finalize_report=finalize)
    assert finalized == [expected, escaped]
    for record, original, normalized in ((typed, typed_config, expected), (mapped, literal_config, escaped)):
        assert record['config'] == normalized and record['config_hash'] == configuration_hash(original)
        raw = json.loads((store.path / 'runs' / record['run_id'] / 'run.json').read_text(encoding='utf-8'))
        assert raw == record
        assert store.load_run(record['run_id'])['record'] == record
    assert typed['config_hash'] != mapped['config_hash']
    # Stored representations are inspectable JSON, not reconstructed raw input.
    assert configuration_hash(typed['config']) == configuration_hash(literal_config)
    assert configuration_hash(typed['config']) != typed['config_hash']


def test_escape_marker_and_nested_descriptor_mappings_are_unambiguous_and_order_independent(tmp_path):
    literal = {'__native_timedelta__': {'days': 1, 'seconds': 0, 'microseconds': 0}}
    wrapped = {'__literal_config__': literal}
    nested = {'left': [literal, wrapped], '__datetime__': 'user literal with other fields'}
    expected_literal = {'__literal_config__': literal}
    expected_wrapped = {'__literal_config__': {'__literal_config__': expected_literal}}
    expected_nested = {'__literal_config__': {'left': [expected_literal, expected_wrapped],
                                             '__datetime__': 'user literal with other fields'}}
    assert _json(wrapped) == expected_wrapped
    assert _json(nested) == expected_nested
    assert len({configuration_hash(value) for value in (timedelta(days=1), literal, wrapped, nested)}) == 4
    assert configuration_hash(nested) == configuration_hash(dict(reversed(list(nested.items()))))
    assert _json(nested, temporal_descriptors=False) == nested
    store = ExperimentStore(tmp_path, 'Nested escapes')
    group = store.start_run('Escaped group', config=nested)
    saved_group = json.loads((store.path / 'run-groups' / f'{group}.json').read_text(encoding='utf-8'))
    assert saved_group['config'] == expected_nested
    failed = store.save_failure(name='Escaped failure', config=nested, error='test', run_group=group)
    assert failed['config'] == expected_nested and failed['config_hash'] == configuration_hash(nested)
    assert store.read_runs()[0] == failed


def test_football_candidate_typed_and_literal_configs_keep_distinct_provenance_on_reuse(tmp_path):
    events = []
    experiment = FootballExperiment('Candidate descriptor identity', output_dir=tmp_path)
    data = prepared(holdout=True)
    native = Candidate('Duration', CountingFactory(.1, events), config={'delay': timedelta(days=1)})
    literal = replace(native, config={'delay': {'__native_timedelta__': {'days': 1, 'seconds': 0, 'microseconds': 0}}})
    typed_result = experiment.run(data, model=native)
    literal_result = experiment.run(data, model=literal)
    assert len(events) == 2
    assert typed_result.record['config_hash'] != literal_result.record['config_hash']
    assert typed_result.record['config']['model'] == _json(native.config)
    assert literal_result.record['config']['model'] == _json(literal.config)
    for candidate, original in ((native, typed_result), (literal, literal_result)):
        reused = experiment.run(data, model=candidate)
        assert reused.reused and reused.record['run_id'] == original.record['run_id']
        assert reused.record == original.record
    assert len(events) == 2


def test_ordinary_configuration_shape_and_hash_remain_unchanged():
    ordinary = {'model': 'Ridge', 'nested': [{'alpha': .1, 'seed': 2}], 'optional': None}
    assert _json(ordinary) == ordinary
    assert configuration_hash(ordinary) == configuration_hash(dict(reversed(list(ordinary.items()))))
