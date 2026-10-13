"""Graph construction, native labels, reporting, and persisted UI workflows."""
from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')
from test_temporal_graph import sample, config, context, deep_state
from test_ui_workflow import ui_recipe
from xdiyo_analytics.graphs import TemporalGraphAdapter, TemporalGraphModel
from xdiyo_analytics.training import TrainingControl, TrainingRunner, save_model, load_model
from xdiyo_analytics.datasets import ModelDataset
from xdiyo_analytics.splits import Fold, SplitPlan
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.reporting import GATLossReporter, EmbeddingReporter, GraphAttentionWeightReporter, TeamCatalog
from xdiyo_analytics.ui.recipe import catalog_for_ui, node, prepare_recipe, run_recipe, export_python


@pytest.fixture
def trained():
    store = sample()
    ctx = context(store, list(range(9)))
    frames = [x.reset_index(drop=True) for x in (ctx.X, ctx.y, ctx.metadata)]
    data = ModelDataset(*frames, 'match', ctx.match_columns, ctx.match_columns, 'home')
    folds = [Fold(np.array([1,2,3]), np.array([6,7,8]), np.array([6,7,8]))]
    plan = SplitPlan(folds, 9, np.arange(9))
    result = TrainingRunner(lambda: TemporalGraphAdapter(config(), store), control=TrainingControl(max_steps=2)).run(data, plan)
    return data, result


def test_native_outcome_probability_contract_and_reject_text(trained):
    _, training = trained
    fold = training.folds[0]
    p = fold.predictions['predict_proba']
    assert list(p.columns) == [('result',1),('result',0),('result',-1)]
    assert p.columns.names == ['target','class']
    assert fold.predictions['predict'].result.isin([1,0,-1]).all()
    from xdiyo_analytics.evaluation.metrics import evaluate_metrics
    scores = evaluate_metrics(fold.y_true, fold.predictions, ['log_loss','accuracy'])
    assert not scores.empty
    ctx = context(sample(), [1,2,3]); ctx.y = ctx.y.astype(object); ctx.y.iloc[:,0] = ['home','draw','away']
    with pytest.raises(ValueError, match='1/0/-1'):
        TemporalGraphAdapter(config(), sample()).fit(ctx)


def test_auto_native_save_load_history_embeddings_attention(trained, tmp_path):
    data, training = trained
    original = training.folds[0].model
    import torch
    rng = torch.get_rng_state().clone()
    path = save_model(training, tmp_path/'saved')
    loaded = load_model(path)
    assert torch.equal(rng, torch.get_rng_state())
    pd.testing.assert_frame_equal(loaded.predict(data, positions=np.array([6,7,8]))['predict_proba'], training.folds[0].predictions['predict_proba'])
    pd.testing.assert_frame_equal(original.training_history_, loaded.model.training_history_)
    queries = pd.DataFrame({'team_id':[1,2,3], 'issuance_at':pd.Timestamp('2020-03-01',tz='UTC')})
    pd.testing.assert_frame_equal(original.encode_teams(queries), loaded.model.encode_teams(queries))
    before = deep_state(original.backend_.model)
    original.backend_.model.train()
    original.encode_teams(queries)
    assert original.backend_.model.training
    a = original.attention_weights(queries)
    assert original.backend_.model.training
    pd.testing.assert_frame_equal(a, loaded.model.attention_weights(queries))
    np.testing.assert_allclose(a.groupby(['team_id','layer','head']).attention.sum(), 1, atol=1e-6)
    assert before == deep_state(original.backend_.model)
    # Future event mutations cannot change earlier inspection.
    assert (a.kickoff_at < a.history_cutoff).all()


@pytest.mark.parametrize('mode', ['PCA','tSNE'])
def test_native_reporters_render_badges_and_recover(trained, tmp_path, mode):
    data, training = trained
    badge = tmp_path/'badge.svg'; badge.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><circle cx="10" cy="10" r="8" fill="red"/></svg>')
    catalog = TeamCatalog({1:dict(name='One',badge_path=str(badge)),2:'Two'})
    report = PostTrainingAnalysis({
        'loss':GATLossReporter(type='overall'),
        'embedding':EmbeddingReporter(type='overall', partition='score', mode=mode, max_iter=250, catalog=catalog),
        'attention':GraphAttentionWeightReporter(type='overall', partition='score', catalog=catalog),
    }).run(training)
    html = report.to_html()
    # Plotly escapes slashes in its embedded JSON for safe HTML serialization.
    assert 'svg+xml;base64,' in html
    assert 'Incoming attention' in html
    assert all(s.result.tables for s in report.studies)
    attention = report.studies[2].result.artifacts[0].data
    assert attention.layout.legend.y > 1
    assert not attention.layout.updatemenus
    assert attention.layout.meta['xdiyo_attention']
    assert {r['team'] for r in attention.layout.meta['roster']} == {'1', '2'}
    assert 'initializeAttention(el,spec)' in html
    assert report.studies[1].result.artifacts[0].options['embedding_fold'] == '0'
    from xdiyo_analytics.experiments.recovery import pack, unpack
    restored = unpack(pack(report))
    assert 'Team embeddings' in restored.to_html()
    raw = report.studies[1].result.tables['embeddings']
    assert raw.model_version.nunique() == 1 and raw.issuance_at.nunique() == 1
    assert set(raw.team_id) == {1,2}
    if mode == 'tSNE':
        assert report.studies[1].result.tables['projection_summary'].effective_perplexity.iloc[0] == 1


def graph_recipe(recipe, tmp_path):
    recipe = deepcopy(recipe)
    from xdiyo_analytics.data import load_seasons
    from xdiyo_analytics.histories import build_team_history
    data = load_seasons(**recipe['data'])
    h = build_team_history(data)
    memberships = h[['team_id','competition_id','season_id']].drop_duplicates()
    memberships['movement'] = 'retained'
    path = tmp_path/'memberships.parquet'; memberships.to_parquet(path,index=False)
    recipe.update(model=node('graphs.TemporalGraphModel', config=node('graphs.TemporalGraphConfig', timezone='UTC',
        embedding_dim=4, layers=1, heads=1, hidden_width=8), control=node('training.TrainingControl', max_steps=2), memberships_path=str(path)),
        preprocessors=[], target_transformer=None, adapter=None, prediction_methods=['predict','predict_proba'],
        labels={'result':node('labels.Outcome',perspective='home')}, target='result', cutoff_hours=None,
        features={'is_home':node('features.IsHome')},
        post_reporters={'loss':node('reporting.GATLossReporter',type='overall'),
                        'embedding':node('reporting.EmbeddingReporter',type='per_fold',partition='score',mode='PCA'),
                        'attention':node('reporting.GraphAttentionWeightReporter',type='per_fold',partition='score')})
    return recipe


def test_attention_keeps_selectable_roster_when_no_edges_exist(trained, monkeypatch):
    _, training = trained
    monkeypatch.setattr(training.folds[0].model, 'attention_weights', lambda queries: pd.DataFrame())
    report = PostTrainingAnalysis({'attention': GraphAttentionWeightReporter(type='overall', partition='score')}).run(training)
    result = report.studies[0].result
    assert result.tables['attention'].empty
    assert len(result.artifacts) == 1
    assert {row['team'] for row in result.artifacts[0].data.layout.meta['roster']} == {'1', '2'}
    assert 'initializeAttention' in report.to_html()


@pytest.mark.parametrize('deterministic', [False, True])
def test_recipe_prepare_fit_reuse_export_and_saved_prediction(ui_recipe, tmp_path, deterministic):
    recipe = graph_recipe(ui_recipe,tmp_path)
    recipe['model']['params']['config']['params']['deterministic'] = deterministic
    prepared = prepare_recipe(recipe)
    assert prepared.outputs['graph_inputs'] is not None
    assert prepared.dataset.y.iloc[:,0].eq(1).all()
    assert prepared.dataset.metadata.feature_cutoff_at.notna().all()
    result = run_recipe(recipe, prepared=prepared)
    assert len(result.training.folds) > 0
    restored = run_recipe(recipe, prepared=prepared)
    for a,b in zip(result.training.folds, restored.training.folds):
        assert b.model is not None
        pd.testing.assert_frame_equal(a.predictions['predict_proba'],b.predictions['predict_proba'])
    exported = export_python(recipe)
    compile(exported,'graph_recipe.py','exec')
    namespace = {}
    exec(exported.split('prepared =')[0], namespace)
    assert namespace['recipe']['model']['params']['config']['params']['deterministic'] is deterministic
    assert catalog_for_ui().build(json.loads(json.dumps(recipe['model']))).config.deterministic is deterministic
    assert catalog_for_ui().build(json.loads(json.dumps(recipe['model']))).config.embedding_dim == 4
    from xdiyo_analytics.ui.recipe import predict_recipe
    model_path = save_model(result.training, tmp_path/'ui_model', fold_id=0)
    assert load_model(model_path).model.config.deterministic is deterministic
    future = deepcopy(recipe)
    future['prediction'].update(model_path=str(model_path), rounds={'Alpha':[8]})
    aligned, predictions = predict_recipe(future)
    assert len(aligned.X) > 0
    assert predictions['predict_proba'].columns.tolist() == [('result',1),('result',0),('result',-1)]


def test_validation_calibration_and_inspection_conventions(trained, tmp_path):
    from xdiyo_analytics.training import ValidationTail, ProbabilityCalibrator
    from xdiyo_analytics.training.calibration import CalibratedAdapter
    from xdiyo_analytics.graphs import TemporalGraphSerializer
    data, _ = trained
    # Held-out tail is strictly later than fitting labels under the graph policy.
    plan = SplitPlan([Fold(np.array([1,2,3,4,5]), np.array([7,8]), np.array([7,8]))], 9, np.arange(9))
    runner = TrainingRunner(lambda: TemporalGraphAdapter(config(), sample()),
                            control=TrainingControl(max_steps=2), validation=ValidationTail(.4))
    result = runner.run(data, plan)
    assert set(result.folds[0].training_history.metric) >= {'train_loss','validation_loss'}
    report = PostTrainingAnalysis({'loss':GATLossReporter(type='overall')}).run(result)
    assert 'validation_loss' in set(report.studies[0].result.tables['history'].metric)
    # The same (target, class) schema feeds the existing reusable calibrator.
    base = result.folds[0].model
    p = result.folds[0].predictions['predict_proba']
    calibrator = ProbabilityCalibrator().fit(p, result.folds[0].y_true)
    result.folds[0].model = CalibratedAdapter(base, calibrator, np.array([7,8]))
    path = save_model(result, tmp_path/'calibrated')
    restored = load_model(path)
    assert isinstance(restored.model, CalibratedAdapter)
    np.testing.assert_allclose(restored.predict(data,positions=np.array([7,8]))['predict_proba'].sum(axis=1), 1)


def test_schedule_and_result_release_identity():
    from xdiyo_analytics.graphs import TemporalEventStore
    store = sample()
    history = store.history
    history['result_available_at'] = pd.NaT
    history['result_available_at'] = pd.to_datetime(history.result_available_at, utc=True)
    history.loc[0,'result_available_at'] = pd.Timestamp('2020-02-01',tz='UTC')
    with pytest.raises(ValueError,match='paired perspectives'):
        TemporalEventStore(history,store.team_seasons)
    schedule = store.schedule
    schedule.loc[0,'home_id'] = 3
    with pytest.raises(ValueError,match='Schedule disagrees'):
        TemporalEventStore(store.history,store.team_seasons,schedule=schedule)
    for policy in ('by_round','one_day_ahead'):
        with pytest.raises(ValueError,match='home_id disagrees'):
            store.issue_times(schedule,config(forecast_policy=policy))


def test_catalog_fields_and_defaults():
    catalog = catalog_for_ui()
    schema = {s['id']:s for s in catalog.schema()}
    from xdiyo_analytics.graphs import TemporalGraphConfig
    from dataclasses import fields
    assert {f.name for f in fields(TemporalGraphConfig)} == {f['name'] for f in schema['graphs.TemporalGraphConfig']['fields']}
    mode = next(f for f in schema['reporting.EmbeddingReporter']['fields'] if f['name']=='mode')
    assert mode['choices'] == ['tSNE','PCA'] and mode['default']=='tSNE'
    assert schema['reporting.GATLossReporter']['category']=='post_reporter'


def test_input_scope_and_graph_owned_scaling(ui_recipe,tmp_path):
    recipe = graph_recipe(ui_recipe,tmp_path)
    recipe['labels']['result'] = node('labels.MatchGoals')
    with pytest.raises(ValueError,match='Outcome'):
        prepare_recipe(recipe)
    recipe = graph_recipe(ui_recipe,tmp_path)
    recipe['cutoff_hours']=3
    with pytest.raises(ValueError,match='cutoff'):
        prepare_recipe(recipe)


def test_graph_grid_uses_native_chronological_search(ui_recipe, tmp_path):
    recipe = graph_recipe(ui_recipe, tmp_path)
    recipe['post_reporters'] = {}
    recipe['search'] = dict(nested=False,
        grid={'model.params.config.params.learning_rate':[.001,.002]},
        split=node('splits.TemporalSplit',train_size=1,test_size=1,unit='seasons'),
        options={'metrics':['log_loss']})
    prepared = prepare_recipe(recipe)
    result = run_recipe(recipe,prepared=prepared)
    assert result.selection is not None
    assert len(result.training.folds)==1
    np.testing.assert_array_equal(result.training.folds[0].test_positions,prepared.split_plan.folds[0].test)
    # A grid cannot silently reuse another candidate's membership/cutoff inputs.
    from xdiyo_analytics.ui.recipe import _grid_candidates
    recipe['search']['grid']={'model.params.config.params.timezone':['Asia/Tokyo']}
    candidates = _grid_candidates(recipe,recipe['search'],catalog_for_ui(),prepared.outputs)
    with pytest.raises(ValueError,match='change preparation'):
        candidates[0].model_factory()

