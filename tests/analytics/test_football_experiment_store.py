"""Rich experiment artifacts stay typed through publication and fallback readers."""

from dataclasses import dataclass
from datetime import date
from functools import partial
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_experiment_samples import prepared, ridge
from model_selection_samples import FixedAdapter, sample
from test_football_experiment_search import CountingFactory
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.experiments import FootballExperiment, PreparedExperiment
from xdiyo_analytics.experiments.store import configuration_hash
from xdiyo_analytics.experiments.recovery import pack
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


def axis_table(problem, empty=False):
    table = pd.DataFrame([[.2, .8], [.4, .6]], columns=['home', 'away'])
    if problem == 'tuple_columns':
        table.columns = pd.Index([('outcome', 'home'), ('outcome', 'away')], tupleize_cols=False)
    elif problem == 'duplicate_columns':
        table.columns = ['probability', 'probability']
    elif problem == 'integer_columns':
        table.columns = [0, 1]
    elif problem == 'mixed_columns':
        table.columns = ['home', 1]
    elif problem == 'named_columns':
        table.columns.name = 'outcome'
    elif problem == 'categorical_columns':
        table.columns = pd.CategoricalIndex(['home', 'away'], categories=['home', 'draw', 'away'], ordered=True)
    elif problem == 'column_index_collision':
        table.index.name = 'home'
    elif problem == 'unnamed_index_collision':
        table.columns = ['index', 'away']
    elif problem == 'reserved_index_name':
        table.index.name = 'index'
    elif problem == 'integer_index_name':
        table.index.name = 7
    elif problem == 'duplicate_index':
        table.index = [5, 5]
    elif problem == 'tuple_index':
        table.index = pd.Index([('match', 5), ('match', 8)], tupleize_cols=False)
    elif problem == 'date_index':
        table.index = pd.Index([date(2024, 1, 1), date(2024, 1, 2)], dtype=object)
    elif problem == 'timedelta_index':
        table.index = pd.to_timedelta([1, 2], unit='h')
    elif problem == 'numpy_datetime_index':
        table.index = pd.Index([np.datetime64('2024-01-01T00:00:00.000000001', 'ns'),
                                np.datetime64('2024-01-01T00:00:00.000000002', 'ns')], dtype=object)
    elif problem == 'numpy_timedelta_index':
        table.index = pd.Index([np.timedelta64(1, 'ns'), np.timedelta64(2, 'ns')], dtype=object)
    elif problem == 'nanosecond_datetime_index':
        table.index = pd.DatetimeIndex([pd.Timestamp('2024-01-01T00:00:00.000000001'),
                                        pd.Timestamp('2024-01-01T00:00:00.000000002')])
    elif problem == 'multiindex':
        table.index = pd.MultiIndex.from_tuples([('match', 5), ('match', 8)], names=['event', 'event'])
    else:
        raise AssertionError(problem)
    return table.iloc[:0] if empty else table


@dataclass(kw_only=True)
class AxisTableReporter(PredictionReporter):
    problem: str
    empty: bool = False

    def run(self, context):
        return StudyResult('Axis tables', tables={
            'ordinary': pd.DataFrame({'count': [len(context.y)]}),
            'axes': axis_table(self.problem, self.empty)})


@pytest.mark.parametrize(('problem', 'empty'), [
    (problem, empty)
    for problem in ('tuple_columns', 'duplicate_columns', 'integer_columns', 'mixed_columns', 'named_columns',
                    'categorical_columns')
    for empty in (False, True)
] + [(problem, False) for problem in (
    'column_index_collision', 'unnamed_index_collision', 'reserved_index_name',
    'integer_index_name', 'duplicate_index', 'tuple_index', 'date_index', 'timedelta_index',
    'numpy_datetime_index', 'numpy_timedelta_index', 'nanosecond_datetime_index', 'multiindex')])
def test_report_axes_round_trip_through_experiment_and_numerical_artifacts(tmp_path, problem, empty):
    experiment = FootballExperiment('Table axes', output_dir=tmp_path)
    report = PostTrainingAnalysis({'axes': AxisTableReporter(
        type='overall', partition='score', problem=problem, empty=empty)})
    result = experiment.run(prepared(holdout=True), model=ridge(), post_analysis=report)
    raw = json.loads((result.path / 'tables.json').read_text(encoding='utf-8'))
    by_name = {item['name']: item['table'] for item in raw}
    assert by_name['ordinary']['data'] == [{'index': 0, 'count': 2}]
    record = experiment.store.save_run(result.training, result.post_report, name='Numerical axes', config={})
    expected = axis_table(problem, empty)
    for loaded in (experiment.load(result.record['run_id']).post_report,
                   experiment.store.load_run(record['run_id'])['report']):
        actual = loaded.studies[0].result.tables['axes']
        pd.testing.assert_frame_equal(actual, expected)
        assert [type(label) for label in actual.columns] == [type(label) for label in expected.columns]
        assert [type(label) for label in actual.index] == [type(label) for label in expected.index]
    assert by_name['axes']['encoding'] == 'xdiyo.data-only.v1'


def cell_table(kind, empty=False):
    if kind == 'duration':
        values = pd.Series([pd.Timedelta('1h'), pd.NaT], dtype='timedelta64[ns]')
    elif kind == 'tuple':
        values = pd.Series([(1, 2), (3, 4)], dtype=object)
    elif kind == 'date':
        values = pd.Series([date(2024, 1, 1), None], dtype=object)
    elif kind == 'numpy_datetime':
        values = pd.Series([np.datetime64('2024-01-01T00:00:00.000000001', 'ns'),
                            np.datetime64('NaT', 'D')], dtype=object)
    elif kind == 'numpy_timedelta':
        values = pd.Series([np.timedelta64(1, 'ns'), np.timedelta64('NaT', 'h')], dtype=object)
    elif kind == 'nested':
        values = pd.Series([{'attempts': [(1, date(2024, 1, 1))]},
                            {2: {'elapsed': pd.Timedelta('1s'), 'path': Path('result.json')}}], dtype=object)
    elif kind == 'array':
        values = pd.Series([np.array([1, 3], dtype='int16'), np.array([], dtype='int16')], dtype=object)
    elif kind == 'object_nulls':
        values = pd.Series([None, pd.NA, pd.NaT, np.nan], dtype=object)
    elif kind == 'object_none':
        values = pd.Series([True, None], dtype=object)
    elif kind == 'object_integer':
        values = pd.Series([1, 2], dtype=object)
    elif kind == 'object_boolean':
        values = pd.Series([True, False], dtype=object)
    elif kind in ('infinite_float', 'infinite_nullable_float'):
        values = pd.Series([np.inf, -np.inf], dtype='Float64' if kind == 'infinite_nullable_float' else 'float64')
    elif kind == 'tuple_category':
        values = pd.Series(pd.Categorical([(1, 2), None], categories=[(1, 2), (3, 4)], ordered=True))
    elif kind == 'date_category':
        values = pd.Series(pd.Categorical([date(2024, 1, 1), None],
                           categories=[date(2024, 1, 1), date(2024, 1, 2)], ordered=True))
    elif kind in ('nullable_category', 'object_category', 'string_category'):
        categories = (pd.Index([1, 2], dtype='Int64') if kind == 'nullable_category' else
                      pd.Index(['home', 'away'], dtype='object' if kind == 'object_category' else 'string'))
        values = pd.Series(pd.Categorical([categories[0], None], categories=categories, ordered=True))
    elif kind in ('nondefault_string', 'nondefault_nullable_string'):
        storage = 'python' if pd.api.types.pandas_dtype('string').storage == 'pyarrow' else 'pyarrow'
        values = pd.Series(['home', pd.NA if kind == 'nondefault_nullable_string' else 'away'],
                           dtype=pd.StringDtype(storage=storage))
    elif kind == 'float32':
        values = pd.Series([.25, .75], dtype='float32')
    elif kind == 'int16':
        values = pd.Series([1, 3], dtype='int16')
    elif kind == 'uint64':
        values = pd.Series([1, 2**63 + 1], dtype='uint64')
    elif kind == 'nanosecond_timestamp':
        values = pd.Series([pd.Timestamp('2024-01-01T00:00:00.000000001Z'), pd.NaT])
    elif kind == 'datetime_unit':
        values = pd.Series([pd.Timestamp('2024-01-01'), pd.NaT], dtype='datetime64[us]')
    else:
        raise AssertionError(kind)
    table = values.to_frame('value')
    return table.iloc[:0] if empty else table


def ordinary_cell_table():
    return pd.DataFrame({
        'count': [1, 2], 'value': [.123456789123, np.nan], 'label': ['home', 'away'],
        'nullable_count': pd.Series([1, pd.NA], dtype='Int64'),
        'nullable_value': pd.Series([.25, pd.NA], dtype='Float64'),
        'nullable_flag': pd.Series([True, pd.NA], dtype='boolean'),
        'nullable_label': pd.Series(['home', pd.NA], dtype='string'),
        'category': pd.Categorical(['home', None], categories=['home', 'draw'], ordered=True),
        'mixed': pd.Series([1, 'home'], dtype=object),
        'json': pd.Series([{'attempts': [1, None, np.int64(3)]}, ['home', .25]], dtype=object)})


@dataclass(kw_only=True)
class CellTableReporter(PredictionReporter):
    kind: str
    empty: bool = False

    def run(self, context):
        return StudyResult('Cell tables', tables={
            'ordinary': ordinary_cell_table(), 'cells': cell_table(self.kind, self.empty)})


class CellHistoryAdapter(FixedAdapter):
    def __init__(self, kind, empty):
        super().__init__()
        self.kind, self.empty = kind, empty

    def fit(self, context):
        super().fit(context)
        self.training_history_ = cell_table(self.kind, self.empty)
        return self


@pytest.mark.parametrize(('kind', 'empty'), [(kind, False) for kind in (
    'duration', 'tuple', 'date', 'numpy_datetime', 'numpy_timedelta', 'nested', 'array', 'object_nulls',
    'object_none', 'object_integer', 'object_boolean', 'infinite_float', 'infinite_nullable_float',
    'tuple_category', 'date_category',
    'nullable_category', 'object_category', 'string_category', 'nondefault_string', 'nondefault_nullable_string',
    'float32', 'int16', 'uint64', 'nanosecond_timestamp', 'datetime_unit')]
    + [('duration', True)])
def test_typed_report_cells_and_history_survive_both_public_loaders(tmp_path, kind, empty):
    experiment = FootballExperiment('Cell artifacts', output_dir=tmp_path)
    model = Candidate('History', partial(CellHistoryAdapter, kind, empty), config={'kind': kind, 'empty': empty})
    report = PostTrainingAnalysis({'cells': CellTableReporter(type='overall', partition='score', kind=kind, empty=empty)})
    result = experiment.run(prepared(holdout=True), model=model, post_analysis=report)
    record = experiment.store.save_run(result.training, result.post_report, name='Numerical cells', config={})
    modern, numerical = experiment.load(result.record['run_id']), experiment.store.load_run(record['run_id'])
    expected = cell_table(kind, empty)
    for actual in (modern.post_report.studies[0].result.tables['cells'], modern.training.folds[0].training_history,
                   numerical['report'].studies[0].result.tables['cells'], numerical['training'].folds[0].training_history):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        # pandas equality treats tuple/list cells as equivalent; the data-only
        # representation also checks the actual nested container/scalar types.
        assert pack(actual) == pack(expected)
    raw = json.loads((result.path / 'tables.json').read_text())
    tables = {item['name']: item['table'] for item in raw}
    # Object string categories are already the inferred ordinary dtype on
    # pandas 2, whereas pandas 3 needs the tag to retain them as object.
    if kind != 'object_category':
        assert tables['cells']['encoding'] == 'xdiyo.data-only.v1'
    assert 'encoding' not in tables['ordinary']
    assert tables['ordinary']['data'][0]['value'] == .1234567891
    assert tables['ordinary']['data'][1]['value'] is None
    assert tables['ordinary']['data'][0]['json'] == {'attempts': [1, None, 3]}
    original = ordinary_cell_table()
    original.loc[0, 'value'] = .1234567891
    pd.testing.assert_frame_equal(numerical['report'].studies[0].result.tables['ordinary'], original, check_exact=True)


def rich_summary(kind):
    summary = {'n_iter': 3, 'termination_reason': 'complete', 'missing': None,
               'missing_float': float('nan'), 'missing_pandas': pd.NA, 'missing_datetime': pd.NaT,
               'steps': np.array([1, 3], dtype='int32')}
    if kind == 'attempts':
        summary['attempts'] = {1: {'best_step': 3}, 2: {'best_step': None}}
    elif kind == 'duration':
        summary['elapsed'] = pd.Timedelta('1h 2min')
    else:
        raise AssertionError(kind)
    return summary


class SummaryAdapter(FixedAdapter):
    def __init__(self, kind='attempts'):
        super().__init__()
        self.kind = kind

    def fit(self, context):
        super().fit(context)
        self.training_summary_ = rich_summary(self.kind)
        self.training_history_ = pd.DataFrame([[.4, .3]], columns=['loss', 'loss'])
        return self


@pytest.mark.parametrize('kind', ['attempts', 'duration'])
def test_rich_training_summary_and_history_publish_with_or_without_recovery(tmp_path, kind):
    from functools import partial
    experiment = FootballExperiment('Summary artifacts', output_dir=tmp_path)
    model = Candidate('Summary', partial(SummaryAdapter, kind), config={'summary': kind})
    result = experiment.run(prepared(holdout=True), model=model)
    raw = json.loads((result.path / 'training.json').read_text(encoding='utf-8'))
    assert raw[0]['summary_encoding'] == 'xdiyo.data-only.v1'
    assert raw[0]['history']['encoding'] == 'xdiyo.data-only.v1'
    record = experiment.store.save_run(result.training, result.post_report, name='Numerical summary', config={})
    for training in (experiment.load(result.record['run_id']).training,
                     experiment.store.load_run(record['run_id'])['training']):
        summary = training.folds[0].training_summary
        assert summary['n_iter'] == 3 and summary['termination_reason'] == 'complete'
        assert summary['missing'] is None and np.isnan(summary['missing_float'])
        assert summary['missing_pandas'] is pd.NA and summary['missing_datetime'] is pd.NaT
        np.testing.assert_array_equal(summary['steps'], np.array([1, 3], dtype='int32'), strict=True)
        if kind == 'attempts':
            assert summary['attempts'] == {1: {'best_step': 3}, 2: {'best_step': None}}
            assert all(type(key) is int for key in summary['attempts'])
        else:
            assert type(summary['elapsed']) is pd.Timedelta and summary['elapsed'] == pd.Timedelta('1h 2min')
        pd.testing.assert_frame_equal(training.folds[0].training_history, result.training.folds[0].training_history)


def test_plain_summary_keeps_legacy_shape_nulls_and_literal_encoding_keys(tmp_path):
    experiment = FootballExperiment('Literal summary', output_dir=tmp_path)
    result = experiment.run(prepared(holdout=True), model=ridge())
    summary = {'n_iter': np.int64(3), 'termination_reason': 'complete', 'value': np.nan,
               'missing': pd.NA, 'steps': np.array([1, 3]), '@': 'dict',
               'encoding': 'xdiyo.data-only.v1', 'summary_encoding': 'user data'}
    result.training.folds[0].training_summary = summary
    record = experiment.store.save_run(result.training, result.post_report, name='Numerical literal', config={})
    raw = json.loads((experiment.store.path / 'runs' / record['run_id'] / 'training.json').read_text())
    expected = dict(summary, n_iter=3, value=None, missing=None, steps=[1, 3])
    assert 'summary_encoding' not in raw[0]
    assert raw[0]['summary'] == expected
    assert experiment.store.load_run(record['run_id'])['training'].folds[0].training_summary == expected


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


@pytest.mark.parametrize('artifact', ['metadata', 'table', 'summary'])
def test_unknown_artifact_encoding_fails_visibly(tmp_path, artifact):
    experiment = FootballExperiment('Unknown artifact encoding', output_dir=tmp_path)
    report = PostTrainingAnalysis({'probabilities': ProbabilityTableReporter(type='overall', partition='score')})
    result = experiment.run(prepared(holdout=True), model=ridge(), post_analysis=report)
    record = experiment.store.save_run(result.training, result.post_report, name='Numerical', config={})
    folder = experiment.store.path / 'runs' / record['run_id']
    path = folder / ('tables.json' if artifact == 'table' else 'training.json')
    raw = json.loads(path.read_text())
    if artifact == 'metadata':
        raw[0]['fold_metadata_encoding'] = 'unknown.v99'
    elif artifact == 'summary':
        raw[0]['summary_encoding'] = 'unknown.v99'
    else:
        raw[1]['table']['encoding'] = 'unknown.v99'
    path.write_text(json.dumps(raw), encoding='utf-8')
    with pytest.raises(ValueError, match='encoding'):
        experiment.store.load_run(record['run_id'])


@pytest.mark.parametrize('artifact', ['metadata', 'table', 'summary'])
def test_encoded_artifact_with_wrong_payload_type_fails_visibly(tmp_path, artifact):
    experiment = FootballExperiment('Malformed artifact', output_dir=tmp_path)
    result = experiment.run(prepared(holdout=True), model=ridge())
    record = experiment.store.save_run(result.training, result.post_report, name='Numerical', config={})
    folder = experiment.store.path / 'runs' / record['run_id']
    path = folder / ('tables.json' if artifact == 'table' else 'training.json')
    raw = json.loads(path.read_text())
    if artifact == 'table':
        raw.append({'study': 'Invalid', 'fold_id': None, 'name': 'bad',
                    'table': {'encoding': 'xdiyo.data-only.v1', 'value': pack([])}})
    else:
        field = 'fold_metadata' if artifact == 'metadata' else 'summary'
        raw[0][field] = pack([])
        raw[0][field + '_encoding'] = 'xdiyo.data-only.v1'
    path.write_text(json.dumps(raw), encoding='utf-8')
    with pytest.raises(ValueError, match='mapping|DataFrame'):
        experiment.store.load_run(record['run_id'])


@pytest.mark.parametrize('config', [{1: 'not a string key'}, {'duration': pd.Timedelta('1h')}])
def test_rich_artifacts_do_not_relax_configuration_validation(config):
    with pytest.raises(TypeError):
        configuration_hash(config)
