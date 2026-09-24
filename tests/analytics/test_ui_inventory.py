"""Persistent presentation inventory and source-scoped UI discovery."""

import base64
import inspect
import json
from pathlib import Path
import tomllib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_ui_workflow import ui_recipe
from xdiyo_analytics.ui.inventory import inventory
from xdiyo_analytics.ui.recipe import catalog_for_ui, node, prepare_recipe, run_recipe
from xdiyo_analytics.ui.schema import stage_schema
from xdiyo_analytics.ui.server import BuilderState


def test_saved_inventory_has_widgets_help_and_packaged_file():
    saved = inventory()
    assert saved['version'] == 1
    json.dumps(saved, allow_nan=False)
    assert {'components', 'stages', 'metrics'} <= set(saved)
    for owner in [*saved['components'].values(), *saved['stages'].values()]:
        names = []
        for field in owner['fields']:
            names.append(field['name'])
            assert field['kind'], (owner.get('id'), field)
            assert field['help'].strip(), (owner.get('id'), field)
            assert 'annotation' not in field
        assert len(names) == len(set(names)), owner.get('id')
    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))
    assert 'inventory.json' in config['tool']['setuptools']['package-data']['xdiyo_analytics.ui']


def test_known_form_schemas_use_saved_inventory_without_signature_inspection(monkeypatch):
    catalog_for_ui()  # Import optional libraries before instrumenting schema creation.
    monkeypatch.setattr(inspect, 'signature', lambda *args, **kwargs: pytest.fail('Runtime signature inspection'))
    catalog = catalog_for_ui()
    schema = catalog.schema()
    assert len(schema) == len(catalog.entries)
    assert set(catalog.entries) <= set(inventory()['components'])
    assert stage_schema()['assembly']['fields']


def test_schema_reads_are_isolated_from_persistent_defaults():
    first = catalog_for_ui().schema()
    first[0]['fields'][0]['help'] = 'modified by consumer'
    assert catalog_for_ui().schema()[0]['fields'][0]['help'] != 'modified by consumer'
    stages = stage_schema()
    stages['data']['fields'].clear()
    assert stage_schema()['data']['fields']


def test_mcc_fields_are_conditional_and_metric_options_are_specific():
    schema = inventory()
    fields = {f['name']: f for f in schema['components']['reporting.CorrelationAnalysis']['fields']}
    assert fields['methods']['kind'] == 'multiselect'
    assert 'mcc' in fields['methods']['choices']
    for name in ('categorical_features', 'categorical_targets', 'feature_thresholds', 'target_thresholds'):
        assert fields[name]['visible_when'] == {'methods': ['mcc']}
    metrics = {m['name']: m for m in schema['metrics']}
    assert metrics['mse']['fields'] == []
    f1 = {f['name']: f for f in metrics['f1']['fields']}
    assert f1['pos_label']['visible_when'] == {'average': ['binary']}
    assert 'average' not in {f['name'] for f in metrics['mse']['fields']}


def test_round_range_control_preserves_inclusive_scoring_semantics(ui_recipe):
    fields = {f['name']: f for f in inventory()['components']['splits.TemporalSplit']['fields']}
    assert fields['score_rounds']['kind'] == 'range'
    ui_recipe['split']['params']['score_rounds'] = [2, 3]
    prepared = prepare_recipe(ui_recipe)
    fold = prepared.split_plan.folds[0]
    rounds = prepared.dataset.metadata.iloc[fold.score]['round']
    assert rounds.tolist() == [2, 3]
    assert len(fold.test) > len(fold.score)


def test_rating_transition_widget_constructs_context_and_retains_named_stream(ui_recipe):
    stages = stage_schema()
    context_field = next(f for f in stages['rating_options']['fields'] if f['name'] == 'transition_context')
    assert context_field['initial_component'] == 'features.TransitionContext'
    descriptor = inventory()['components'][context_field['initial_component']]
    history_field = next(f for f in descriptor['fields'] if f['name'] == 'history')
    assert history_field['initial'] == {'ref': 'history'}
    ui_recipe['ratings']['strength'] = {
        'transition': node('ratings.GlickoTransition', phi_scale=1.1),
        'transition_context': node(context_field['initial_component'], history=history_field['initial']),
    }
    ui_recipe['features']['rating'] = node('features.Rating', name='strength')
    prepared = prepare_recipe(ui_recipe)
    stream = prepared.outputs['ratings']['strength']
    assert not stream.snapshots.empty
    assert 'transition' in set(stream.snapshots.snapshot_kind)
    assert any('rating' in str(column) for column in prepared.dataset.X.columns)


def test_auxiliary_table_widget_can_preserve_league_season_index(tmp_path):
    fields = {f['name']: f for f in inventory()['components']['input.Table']['fields']}
    assert fields['index']['kind'] == 'multiselect'
    assert {'competition_id', 'season_id'} <= set(fields['index']['choices'])
    path = tmp_path / 'counts.csv'
    pd.DataFrame({'competition_id': [17, 17], 'season_id': [202223, 202324], 'count': [20, 18]}).to_csv(path, index=False)
    counts = catalog_for_ui().build(node('input.Table', path=str(path),
                                        index=['competition_id', 'season_id'], column='count'))
    assert counts.index.names == ['competition_id', 'season_id']
    assert counts.loc[(17, 202324)] == 18


def test_iterative_inventory_factory_keeps_preprocessing_and_restart_seed(ui_recipe):
    from xdiyo_analytics.ui.recipe import ModelFactory

    field = next(f for f in inventory()['components']['training.IterativeAdapter']['fields']
                 if f['name'] == 'backend_factory')
    ui_recipe['model'] = node('sklearn.linear_model.SGDRegressor', learning_rate='constant', eta0=0.01)
    ui_recipe['adapter'] = node('training.IterativeAdapter', backend_factory=field['initial'])
    ui_recipe['candidate'].update(control=node('training.TrainingControl', max_steps=3, restarts=1, seed=19),
                                  validation=node('training.ValidationTail', fraction=0.25))
    adapter = ModelFactory(ui_recipe, catalog_for_ui())()
    first, second = adapter.backend_factory(seed=19), adapter.backend_factory(seed=20)
    assert first.estimator.random_state == 19
    assert second.estimator.random_state == 20
    assert first.estimator is not second.estimator
    assert first.preprocessor is not second.preprocessor
    result = run_recipe(ui_recipe)
    fold = result.training.folds[0]
    backend = fold.model.backend_
    assert backend.estimator.random_state in (19, 20)
    assert backend.preprocessor.named_steps['prepare_1'].n_samples_seen_ == len(backend.context_.X)
    assert len(backend.validation_.X) > 0
    assert {'train_loss', 'validation_loss'} <= set(fold.training_history.metric)
    assert fold.training_history.step.max() == 3


def test_inspection_exposes_only_main_tables_and_short_summary(export):
    export.replace('coverage', pa.table({'source': ['test']}))
    state = BuilderState(export.root, catalog_for_ui())
    try:
        response = state.dispatch('inspect', {'root': str(export.root), 'stem': 'Premier_League_24_25'})
        summary = response['tables']
        assert summary['columns'] == ['table', 'rows', 'description']
        assert {row[0] for row in summary['rows']} == {'matches', 'statistics', 'pregame', 'shots'}
        assert all(isinstance(row[2], str) and row[2] for row in summary['rows'])
    finally:
        state.executor.shutdown(wait=True)


def test_discovery_choices_follow_selected_seasons_and_leagues(ui_recipe, tmp_path):
    root = Path(ui_recipe['data']['data_root'])
    publication = json.loads((root / 'Alpha_22_23.manifest.json').read_text())
    manifest_path = root / publication['manifest']
    manifest = json.loads(manifest_path.read_text())
    stats_path = manifest_path.parent / manifest['tables']['statistics']['file']
    stats = pd.read_parquet(stats_path)
    stats['key'] = 'seasonSpecificStat'
    pq.write_table(pa.Table.from_pandas(stats, preserve_index=False), stats_path)
    # Discovery reads metadata columns; normal loading is not used by this check.
    state = BuilderState(tmp_path, catalog_for_ui())
    try:
        first = state.dispatch('choices', {'root': str(root), 'seasons': ['22_23'], 'leagues': ['Alpha']})
        second = state.dispatch('choices', {'root': str(root), 'seasons': ['24_25'], 'leagues': ['Alpha']})
        assert {s['key'] for s in first['stats']} == {'seasonSpecificStat'}
        assert {s['key'] for s in second['stats']} == {'cornerKicks'}
        assert [p['stem'] for p in second['publications']] == ['Alpha_24_25']
        assert {t['label'] for t in second['teams']} == {'Alpha Home', 'Alpha Away'}
        assert all(isinstance(t['value'], str) for t in second['teams'])
        assert set(second['rounds']) == set(range(1, 9))
        empty = state.dispatch('choices', {'root': str(root), 'seasons': ['24_25'], 'leagues': ['Other']})
        assert empty == dict(stats=[], teams=[], rounds=[], publications=[])
    finally:
        state.executor.shutdown(wait=True)


def test_timeline_accepts_exact_team_identity_returned_by_discovery(ui_recipe, tmp_path):
    from xdiyo_analytics.analysis import PreTrainingAnalysis

    state = BuilderState(tmp_path, catalog_for_ui())
    try:
        choices = state.dispatch('choices', {'root': ui_recipe['data']['data_root'],
                                             'seasons': ['24_25'], 'leagues': ['Alpha']})
        selected = next(team['value'] for team in choices['teams'] if team['label'] == 'Alpha Home')
        assert selected == str(2**53 + 3)
        prepared = prepare_recipe(ui_recipe)
        reporter = catalog_for_ui().build(node('reporting.FeatureTimeline', type='overall', partition='train',
                                               features=['home::is_home'], teams=[selected]))
        report = PreTrainingAnalysis({'timeline': reporter}).run(prepared.dataset, split_plan=prepared.split_plan)
        points = report.studies[0].result.tables['points']
        assert not points.empty
        assert set(points.team_id) == {2**53 + 3}
    finally:
        state.executor.shutdown(wait=True)


@pytest.mark.parametrize('relative', [False, True])
def test_local_badges_merge_into_source_names_and_offline_report(ui_recipe, tmp_path, relative):
    image = tmp_path / 'badge.png'
    image.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jfN8AAAAASUVORK5CYII='))
    catalog_path = tmp_path / 'badge_catalog.json'
    catalog_path.write_text(json.dumps({str(2**53 + 3): {'name': 'Older name', 'badge_path': 'badge.png'}}), encoding='utf-8')
    ui_recipe['team_badges'] = catalog_path.name if relative else str(catalog_path)
    state = BuilderState(tmp_path, catalog_for_ui())
    try:
        recipe = state.recipe_paths(ui_recipe)
        prepared = prepare_recipe(recipe)
        name, badge, note = prepared.outputs['team_catalog'].display(2**53 + 3, badges=True)
        assert name == 'Alpha Home'  # Current source names win over old badge catalog spellings.
        assert badge and badge.startswith('data:image/png;base64,')
        assert note is None
        report = run_recipe(recipe, prepared=prepared).to_html()
        assert 'data:image/png;base64,' in report
    finally:
        state.executor.shutdown(wait=True)
