"""Persisted numerical artifacts and independently calculated weighted rankings."""
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import json
from pathlib import Path
import re
from uuid import UUID
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from post_training_samples import run_sample
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.experiments import ExperimentStore, configuration_hash, rank_runs
from xdiyo_analytics.reporting import PerformanceReporter, ExperimentLeaderboardReporter


def saved_metric(name, value, *, target='count', sample='a'*64, direction=None, **fields):
    metric = dict(metric=name, calculation=name, target=target, output='predict', value=value,
                  direction=direction or ('minimize' if name == 'mse' else 'maximize'),
                  n=10, n_total=10, n_missing=0, status='ok', parameters='{}', sample_hash=sample,
                  study='performance', type='overall', partition='score', fold_id=None,
                  layout='match', scope_label='prediction pooling: first')
    metric.update(fields)
    return metric


def record(name, mse, accuracy, **fields):
    value = dict(run_id='run-'+name, name=name, config_hash=configuration_hash({'candidate': name}),
                 status='complete', metrics=[saved_metric('mse', mse), saved_metric('accuracy', accuracy)])
    value.update(fields)
    return value


def computed_run():
    training = run_sample()
    # Save an additional class-labeled output, independent of the regression study.
    for fold in training.folds:
        fold.predictions['proba/special'] = pd.DataFrame([[.7, .3]]*len(fold.y_true), index=fold.y_true.index,
                                                        columns=pd.MultiIndex.from_product([['count'], [1, 0]], names=['target', 'class']))
    report = PostTrainingAnalysis({'performance': PerformanceReporter(type='overall', partition='test', pooling='first', metrics=['mse'])}).run(training)
    return training, report


def test_percentile_weights_directions_and_contributions_hand_calculation():
    records = [record('a', 1., 0.), record('b', 2., .5), record('c', 3., 1.)]
    ranked = rank_runs(records, {'mse': 2., 'accuracy': 1., 'ignored': 0.}).set_index('name')
    assert ranked.index.tolist() == ['a', 'b', 'c']
    np.testing.assert_allclose(ranked['utility::mse'], [1., .5, 0.])
    np.testing.assert_allclose(ranked['utility::accuracy'], [0., .5, 1.])
    np.testing.assert_allclose(ranked['contribution::mse'], [2/3, 1/3, 0.])
    np.testing.assert_allclose(ranked.score, [2/3, .5, 1/3])
    assert ranked['rank'].tolist() == [1., 2., 3.]
    assert ranked.attrs['weights'] == {'mse': 2/3, 'accuracy': 1/3}
    assert 'raw::ignored' not in ranked


def test_fixed_scales_clip_and_do_not_change_existing_scores_when_run_added():
    records = [record('a', -2., .2), record('b', 5., 2.), record('c', 12., -.4)]
    kwargs = dict(scaling='fixed', reference_scales={'mse': (0, 10), 'accuracy': (0, 1)})
    ranks = rank_runs(records, {'mse': 3., 'accuracy': 1.}, **kwargs).set_index('name')
    expected = {'a': .8, 'b': .625, 'c': 0.}
    assert ranks.score.to_dict() == expected
    expanded = rank_runs(records+[record('d', 1., .9)], {'mse': 3., 'accuracy': 1.}, **kwargs).set_index('name')
    assert expanded.loc[list(expected), 'score'].to_dict() == expected


def test_ties_singletons_and_large_weights():
    single = rank_runs([record('a', 1., .8)], {'mse': 1e308, 'accuracy': 1e308})
    assert single.iloc[0].score == .5 and single.iloc[0]['rank'] == 1
    ties = rank_runs([record('a', 1., .8), record('b', 1., .8)], {'mse': 1})
    assert ties.score.tolist() == [.5, .5] and ties['rank'].tolist() == [1., 1.]
    partial = rank_runs([record('a', 1., 0), record('b', 1., 0), record('c', 3., 0)], {'mse': 1}).set_index('name')
    assert partial.score.to_dict() == {'a': .75, 'b': .75, 'c': 0.}


@pytest.mark.parametrize('field,value', [('sample_hash', 'b'*64), ('parameters', '{"choice": 2}'),
                                       ('target', 'other'), ('output', 'another'), ('partition', 'test'),
                                       ('layout', 'team_match'), ('scope_label', 'prediction pooling: occurrences'),
                                       ('direction', 'maximize'), ('calculation', 'custom_mse')])
def test_incompatible_definitions_and_cohorts_rank_in_separate_groups(field, value):
    first, second = record('a', 1., 1.), record('b', 9., 0.)
    second['metrics'][0][field] = value
    result = rank_runs([first, second], {'mse': 1})
    assert result.comparison_group.nunique() == 2
    assert result.score.eq(.5).all() and result['rank'].eq(1).all()


def test_missing_partial_failed_and_empty_records_remain_unranked():
    good = record('good', 1., .8)
    missing = record('missing', 2., .9, metrics=[])
    partial = record('partial', 2., .9)
    partial['metrics'][0]['status'] = 'partial'
    undefined = record('undefined', None, .9)
    no_hash = record('no_hash', 1., .9)
    no_hash['metrics'][0]['sample_hash'] = None
    failed = record('failed', 0., 1., status='failed')
    result = rank_runs([good, missing, partial, undefined, no_hash, failed], {'mse': 2., 'accuracy': 1.}).set_index('name')
    assert result.loc['good', 'status'] == 'ranked' and result.loc['good', 'score'] == .5
    assert result.loc['failed', 'status'] == 'failed'
    assert result.loc[['missing', 'partial', 'undefined', 'no_hash', 'failed'], 'score'].isna().all()
    assert rank_runs([], {'mse': 1}).empty


def test_metric_selectors_resolve_ambiguity_and_require_explicit_fold_choice():
    candidate = record('a', 1., .8)
    candidate['metrics'].append(saved_metric('mse', 4., target='other'))
    with pytest.raises(ValueError, match='ambiguous'):
        rank_runs([candidate], {'mse': 1})
    selected = rank_runs([candidate], {'loss': 1}, selectors={'loss': {'metric': 'mse', 'target': 'other'}})
    assert selected.iloc[0]['raw::loss'] == 4
    per_fold = record('fold', 2., .7)
    per_fold['metrics'][0].update(type='per_fold', fold_id=9)
    assert rank_runs([per_fold], {'mse': 1}).iloc[0].status == 'unranked_missing_or_undefined'
    assert rank_runs([per_fold], {'mse': 1}, selectors={'mse': {'fold_id': 9}}).iloc[0].status == 'ranked'


def test_descriptive_metric_requires_user_direction():
    candidate = record('a', 1., .8)
    candidate['metrics'][0]['direction'] = None
    with pytest.raises(ValueError, match='direction'):
        rank_runs([candidate], {'mse': 1})
    result = rank_runs([candidate], {'mse': 1}, directions={'mse': 'maximize'})
    assert result.iloc[0].status == 'ranked'


@pytest.mark.parametrize('weights,kwargs', [({}, {}), ({'mse': 0}, {}), ({'mse': -1}, {}),
    ({'mse': np.inf}, {}), ({'mse': np.nan}, {}), ({'': 1}, {}),
    ({'mse': 1}, {'scaling': 'automatic'}), ({'mse': 1}, {'scaling': 'fixed'}),
    ({'mse': 1}, {'scaling': 'fixed', 'reference_scales': {'mse': (1, 1)}}),
    ({'mse': 1}, {'scaling': 'fixed', 'reference_scales': {'mse': (2, 1)}})])
def test_invalid_weight_or_scale_configuration(weights, kwargs):
    with pytest.raises(ValueError):
        rank_runs([record('a', 1., .8)], weights, **kwargs)


def test_configuration_hash_canonical_values_and_no_implicit_object_repr():
    @dataclass
    class Settings:
        penalty: float
    first = {'model': Settings(1.), 'columns': ('a', 'b'), 'seed': np.int64(4),
             'time': datetime(2025, 1, 1, tzinfo=timezone.utc), 'path': Path('relative')}
    second = {key: first[key] for key in reversed(first)}
    second['seed'], second['columns'] = 4, ['a', 'b']
    assert configuration_hash(first) == configuration_hash(second)
    second['seed'] = 5
    assert configuration_hash(first) != configuration_hash(second)
    for bad in ({'model': object()}, {4: 'integer key'}, {'number': np.inf}, {'number': np.nan}):
        with pytest.raises((TypeError, ValueError)):
            configuration_hash(bad)


def test_experiment_path_names_and_uuid_identity(tmp_path):
    first = ExperimentStore(tmp_path, '../../Example / model')
    again = ExperimentStore(tmp_path, '../../Example / model')
    other = ExperimentStore(tmp_path, 'example model')
    assert first.path.resolve().is_relative_to(tmp_path.resolve())
    assert first.manifest == again.manifest
    assert first.path != other.path and first.manifest['experiment_id'] != other.manifest['experiment_id']
    UUID(first.manifest['experiment_id'])
    for name in ['', '   ', None, 5]:
        with pytest.raises(ValueError):
            ExperimentStore(tmp_path, name)


def test_persisted_json_parquet_html_and_repeat_config_roundtrip(tmp_path):
    training, report = computed_run()
    store = ExperimentStore(tmp_path, 'synthetic verification')
    config = {'model': 'recorded synthetic', 'seed': 19, 'columns': ['synthetic']}
    first = store.save_run(training, report, name='first', config=config, save_html=True)
    second = store.save_run(training, report, name='repeat', config=dict(reversed(list(config.items()))), save_predictions=False)
    assert first['run_id'] != second['run_id'] and first['config_hash'] == second['config_hash']
    UUID(first['run_id'])
    assert len(store.read_runs()) == 2
    folder = store.path / 'runs' / first['run_id']
    saved = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    assert saved == first
    assert 'folds' not in second['artifacts'] and 'report' not in second['artifacts']
    for fold, record_ in zip(training.folds, first['artifacts']['folds']):
        for output, path in record_['outputs'].items():
            assert_frame_equal(pd.read_parquet(folder / path), fold.predictions[output])
        assert_frame_equal(pd.read_parquet(folder / record_['targets']), fold.y_true)
        assert_frame_equal(pd.read_parquet(folder / record_['metadata']), fold.metadata)
        assert record_['score_positions'] == fold.score_positions.tolist()
    tables = json.loads((folder / 'tables.json').read_text(encoding='utf-8'))
    assert tables[0]['table']['data'][0]['value'] == pytest.approx(report.studies[0].result.tables['metrics'].iloc[0].value)
    assert first['metrics'][0]['sample_hash'] == report.studies[0].result.tables['metrics'].iloc[0].sample_hash
    html = (folder / 'report.html').read_text(encoding='utf-8')
    csvs = [base64.b64decode(value).decode('utf-8-sig') for value in re.findall(r'href="data:text/csv;base64,([^"]+)"', html)]
    assert 'prediction pooling: first' in html and any('fold_id,row_position' in csv for csv in csvs)
    assert not list(folder.rglob('*.pkl')) and not list(folder.rglob('*.pickle'))


def test_atomic_run_publication_and_explicit_failure(tmp_path, monkeypatch):
    training, report = computed_run()
    store = ExperimentStore(tmp_path, 'interrupted')
    def failure(*args, **kwargs):
        raise OSError('synthetic write failure')
    with monkeypatch.context() as context_:
        context_.setattr(pd.DataFrame, 'to_parquet', failure)
        with pytest.raises(OSError, match='synthetic write failure'):
            store.save_run(training, report, name='incomplete', config={'seed': 1})
    assert store.read_runs() == []
    assert len(list((store.path / 'runs').glob('.pending-*'))) == 1
    failed = store.save_failure(name='failed', config={'seed': 1}, error='explicit failure')
    assert failed['status'] == 'failed' and store.read_runs()[0]['error'] == 'explicit failure'
    complete = store.save_run(training, report, name='complete', config={'seed': 1})
    assert complete['config_hash'] == failed['config_hash']
    assert [record_['status'] for record_ in store.read_runs()] == ['failed', 'complete']


def test_read_corrupt_or_incompatible_published_records_raises(tmp_path):
    store = ExperimentStore(tmp_path, 'corruption')
    run = store.save_failure(name='failure', config={}, error='explicit')
    path = store.path / 'runs' / run['run_id'] / 'run.json'
    record_ = json.loads(path.read_text())
    record_['schema'] = 999
    path.write_text(json.dumps(record_), encoding='utf-8')
    with pytest.raises(ValueError, match='Incompatible'):
        store.read_runs()
    path.write_text('{broken', encoding='utf-8')
    with pytest.raises(json.JSONDecodeError):
        store.read_runs()


def test_leaderboard_runs_without_training_and_is_not_resaved_as_metrics(tmp_path):
    training, report = computed_run()
    store = ExperimentStore(tmp_path, 'leaderboard')
    store.save_run(training, report, name='first', config={'candidate': 1})
    reporter = ExperimentLeaderboardReporter(weights={'error': 1}, selectors={'error': {'metric': 'mse', 'target': 'count'}})
    board = PostTrainingAnalysis({'leaderboard': reporter}).run(experiment=store)
    study = board.studies[0]
    assert study.partition == 'experiment' and study.layout == 'experiment'
    assert study.scope.empty and study.scope_label == 'saved experiment runs'
    table = study.result.tables['leaderboard']
    assert table.iloc[0].score == .5 and table.iloc[0].status == 'ranked'
    assert 'Higher score is better' in study.result.artifacts[0].options['legend']
    assert 'coefficients' not in study.result.artifacts[0].options['legend']
    report.studies.extend(board.studies)
    saved = store.save_run(training, report, name='with display board', config={'candidate': 2}, save_predictions=False)
    assert len(saved['metrics']) == 2
    tables = json.loads((store.path / 'runs' / saved['run_id'] / 'tables.json').read_text())
    assert not any(table['study'] == 'leaderboard' for table in tables)
    assert PostTrainingAnalysis({'empty': reporter}).run(experiment=ExperimentStore(tmp_path, 'empty')).studies[0].result.tables['leaderboard'].empty
