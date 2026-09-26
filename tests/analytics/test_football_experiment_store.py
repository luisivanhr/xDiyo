"""Rich experiment artifacts stay typed through publication and fallback readers."""

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import pytest

from football_experiment_samples import prepared, ridge
from model_selection_samples import sample
from test_football_experiment_search import CountingFactory
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.experiments import FootballExperiment, PreparedExperiment
from xdiyo_analytics.experiments.store import configuration_hash
from xdiyo_analytics.reporting import PredictionReporter, StudyResult
from xdiyo_analytics.selection import Candidate
from xdiyo_analytics.splits import CPCV, create_split_plan


def assert_cpcv_metadata(actual, expected):
    assert type(actual['embargo']) is pd.Timedelta
    assert actual['embargo'] == expected['embargo']
    assert actual['test_blocks'] == expected['test_blocks']
    assert isinstance(actual['test_blocks'], tuple)
    assert set(actual['block_rows']) == set(expected['block_rows'])
    assert all(isinstance(key, int) for key in actual['block_rows'])
    for key in expected['block_rows']:
        np.testing.assert_array_equal(actual['block_rows'][key], expected['block_rows'][key])
        assert isinstance(actual['block_rows'][key], np.ndarray)
    for name in ('purged_train', 'embargoed_train'):
        np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_real_cpcv_experiment_publishes_typed_metadata_and_legacy_fallback(tmp_path, layout):
    data = sample(layout=layout, shuffle=True)
    plan = create_split_plan(data, CPCV(n_blocks=3, n_test_blocks=1, embargo='1h'))
    experiment = FootballExperiment('CPCV artifacts', output_dir=tmp_path)
    fits = []
    result = experiment.run(PreparedExperiment(data, plan),
                            model=Candidate('Ridge', CountingFactory(.1, fits), config={'alpha': .1}))
    assert len(fits) == len(plan.folds)
    loaded = experiment.load(result.record['run_id'])
    for actual, expected in zip(loaded.training.folds, plan.folds):
        assert_cpcv_metadata(actual.fold_metadata, expected.metadata)
    raw = json.loads((result.path / 'training.json').read_text(encoding='utf-8'))
    assert all(item['fold_metadata_encoding'] == 'xdiyo.data-only.v1' for item in raw)
    # Exercise the public reader used when no modern recovery bundle is retained.
    record = experiment.store.save_run(result.training, result.post_report, name='Numerical CPCV', config={})
    restored = experiment.store.load_run(record['run_id'])
    assert restored['extra']['kind'] == 'legacy'
    for actual, expected in zip(restored['training'].folds, plan.folds):
        assert_cpcv_metadata(actual.fold_metadata, expected.metadata)


@dataclass(kw_only=True)
class ProbabilityTableReporter(PredictionReporter):
    empty: bool = False

    def run(self, context):
        columns = pd.MultiIndex(levels=[pd.Index(['outcome'], dtype='string', name='target'),
                                       pd.Index([0, 1, 2, 3], dtype='Int64', name='class')],
                                codes=[[0, 0, 0], [0, 1, 2]], names=['target', 'class'])
        table = pd.DataFrame(np.tile([.2, .3, .5], (len(context.y), 1)),
                             index=context.y.index, columns=columns)
        if self.empty:
            table = table.iloc[:0]
        return StudyResult('Probability tables', tables={
            'ordinary': pd.DataFrame({'count': [len(table)]}), 'probabilities': table})


@pytest.mark.parametrize('empty', [False, True])
def test_mixed_report_tables_publish_exact_multiindex_columns_with_or_without_recovery(tmp_path, empty):
    experiment = FootballExperiment('Probability artifacts', output_dir=tmp_path)
    report = PostTrainingAnalysis({'probabilities': ProbabilityTableReporter(type='overall', partition='score', empty=empty)})
    result = experiment.run(prepared(holdout=True), model=ridge(), post_analysis=report)
    raw = json.loads((result.path / 'tables.json').read_text(encoding='utf-8'))
    by_name = {item['name']: item['table'] for item in raw}
    assert by_name['ordinary']['data'] == [{'index': 0, 'count': 0 if empty else 2}]
    assert by_name['probabilities']['encoding'] == 'xdiyo.data-only.v1'
    record = experiment.store.save_run(result.training, result.post_report, name='Numerical probabilities', config={})
    for loaded in (experiment.load(result.record['run_id']).post_report,
                   experiment.store.load_run(record['run_id'])['report']):
        for name, expected in result.post_report.studies[0].result.tables.items():
            actual = loaded.studies[0].result.tables[name]
            pd.testing.assert_frame_equal(actual, expected)
            if name == 'probabilities':
                for actual_level, expected_level in zip(actual.columns.levels, expected.columns.levels):
                    pd.testing.assert_index_equal(actual_level, expected_level)


def test_plain_fold_metadata_that_resembles_tagged_data_stays_literal(tmp_path):
    preparation = prepared(holdout=True)
    literal = {'@': 'dict', 'encoding': 'xdiyo.data-only.v1', 'value': [1, 2],
               'fold_metadata_encoding': 'user data'}
    preparation.split_plan.folds[0].metadata = literal
    experiment = FootballExperiment('Literal metadata', output_dir=tmp_path)
    result = experiment.run(preparation, model=ridge())
    record = experiment.store.save_run(result.training, result.post_report, name='Numerical literal', config={})
    raw = json.loads((experiment.store.path / 'runs' / record['run_id'] / 'training.json').read_text())
    assert 'fold_metadata_encoding' not in raw[0]
    assert raw[0]['fold_metadata'] == literal
    assert experiment.store.load_run(record['run_id'])['training'].folds[0].fold_metadata == literal


@pytest.mark.parametrize('artifact', ['metadata', 'table'])
def test_unknown_artifact_encoding_fails_visibly(tmp_path, artifact):
    experiment = FootballExperiment('Unknown artifact encoding', output_dir=tmp_path)
    report = PostTrainingAnalysis({'probabilities': ProbabilityTableReporter(type='overall', partition='score')})
    result = experiment.run(prepared(holdout=True), model=ridge(), post_analysis=report)
    record = experiment.store.save_run(result.training, result.post_report, name='Numerical', config={})
    folder = experiment.store.path / 'runs' / record['run_id']
    path = folder / ('training.json' if artifact == 'metadata' else 'tables.json')
    raw = json.loads(path.read_text())
    if artifact == 'metadata':
        raw[0]['fold_metadata_encoding'] = 'unknown.v99'
    else:
        raw[1]['table']['encoding'] = 'unknown.v99'
    path.write_text(json.dumps(raw), encoding='utf-8')
    with pytest.raises(ValueError, match='encoding'):
        experiment.store.load_run(record['run_id'])


@pytest.mark.parametrize('config', [{1: 'not a string key'}, {'duration': pd.Timedelta('1h')}])
def test_rich_artifacts_do_not_relax_configuration_validation(config):
    with pytest.raises(TypeError):
        configuration_hash(config)
