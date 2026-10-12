"""Native CNN contracts on synthetic data; no predictive-improvement claims."""
from dataclasses import replace
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
from test_temporal_graph import sample, config, context
from xdiyo_analytics.neural import HeatmapCNNConfig, Reconstruction, HeatmapCNNModel, HeatmapCNNSerializer
from xdiyo_analytics.neural.inputs import neural_maps
from xdiyo_analytics.neural.network import HeatmapEncoder
from xdiyo_analytics.graphs import GraphHeatmaps, TemporalGraphModel, TemporalGraphAdapter, TemporalGraphSerializer
from xdiyo_analytics.graphs.spatial import prepare_heatmaps, tensors
from xdiyo_analytics.features import Heatmap
from xdiyo_analytics.training import TrainingControl, TrainingRunner, save_model, load_model
from xdiyo_analytics.training.contracts import FitContext
from xdiyo_analytics.datasets import ModelDataset
from xdiyo_analytics.splits import Fold, SplitPlan
from test_ui_workflow import ui_recipe


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def points_for(history):
    keys = [c for c in ('source_league', 'source_season', 'event_id', 'team_id') if c in history]
    p = history[keys].copy()
    p['kind'], p['x'], p['y'], p['point_order'] = 'player', 15., 25., 0
    p.loc[history.side.eq('away'), ['x', 'y']] = [65., 75.]
    return p.reset_index(drop=True)


def graph_setup(reconstruction=None, **kwargs):
    original = sample()
    spatial = GraphHeatmaps(encoder=HeatmapCNNConfig(reconstruction=reconstruction), batch_size=3)
    cfg = config(heatmaps=spatial, **kwargs)
    spec = TemporalGraphModel(config=cfg, membership_policy='explicit', control=TrainingControl(max_steps=2))
    store = spec.event_store(original.history, original.team_seasons, points_for(original.history))
    return spec, store


def test_default_encoder_masks_and_reconstruction_gradients():
    cfg = HeatmapCNNConfig()
    encoder = HeatmapEncoder(cfg, 2, 10, 10)
    assert sum(p.numel() for p in encoder.parameters()) == 4096
    assert encoder.decoder is None
    raw = np.zeros((4, 2, 10, 10))
    raw[:, 0, 2, 3], raw[:, 1, 7, 6] = 1., 1.
    raw[0, 1], raw[1, 0], raw[2] = np.nan, np.nan, np.nan
    maps, flags = neural_maps(raw)
    np.testing.assert_array_equal(flags, [[1, 0], [0, 1], [0, 0], [1, 1]])
    np.testing.assert_array_equal(maps[0, 0], raw[0, 0])
    np.testing.assert_array_equal(maps[1, 1], raw[1, 1])
    assert maps[0, 0, 0, 0] == 0 and flags[0, 0] == 1
    m, a = torch.tensor(maps), torch.tensor(flags)
    assert encoder(m, a)[2].count_nonzero() == 0
    encoder = HeatmapEncoder(replace(cfg, reconstruction=Reconstruction()), 2, 10, 10)
    z = encoder(m, a)
    errors = encoder.reconstruction_errors(z, m, a)
    assert errors[2] == 0 and torch.isfinite(errors).all()
    # The absent counterpart never becomes a zero-valued reconstruction target.
    changed = m.clone(); changed[0, 1] = 100
    torch.testing.assert_close(errors, encoder.reconstruction_errors(z, changed, a))
    errors.sum().backward()
    for block in (encoder.encoder, encoder.decoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters())


@pytest.mark.parametrize('bad', ['partial', 'infinite', 'overflow'])
def test_corrupt_maps_raise(bad):
    raw = np.zeros((1, 2, 10, 10))
    raw[0, 0, 0, 0] = {'partial':np.nan, 'infinite':np.inf, 'overflow':1e300}[bad]
    with pytest.raises(ValueError):
        neural_maps(raw)


def test_native_orientation_independent_missingness_and_validation():
    h = sample().history
    p = points_for(h)
    cfg = GraphHeatmaps()
    prepared = prepare_heatmaps(h, p, cfg)
    maps, flags = tensors(prepared, cfg, 'cpu')
    torch.testing.assert_close(maps[0, 0], maps[1, 1].flip((0, 1)))
    torch.testing.assert_close(maps[1, 0], maps[0, 1].flip((0, 1)))
    assert maps[0, 0, 2, 1] == maps[0, 1, 2, 3] == 1
    torch.testing.assert_close(flags, torch.ones_like(flags))
    # Empty source and goalkeeper-only source use native unavailable semantics.
    missing = p.loc[~((p.event_id == 1) & (p.team_id == 2))].copy()
    missing.loc[missing.event_id == 2, 'kind'] = 'goalkeeper'
    m, a = tensors(prepare_heatmaps(h, missing, cfg), cfg, 'cpu')
    torch.testing.assert_close(a[:4], torch.tensor([[1.,0.],[0.,1.],[0.,0.],[0.,0.]]))
    torch.testing.assert_close(m[0,0], maps[0,0])
    weighted = p.assign(weight=0.)
    weighted_cfg = replace(cfg, grid=replace(cfg.grid, use_weights=True))
    assert tensors(prepare_heatmaps(h, weighted, weighted_cfg), weighted_cfg, 'cpu')[1].sum() == 0
    for bad in (p.assign(x=-1), pd.concat([p,p.iloc[[0]]]), p.assign(team_id=np.nan)):
        with pytest.raises(ValueError):
            prepare_heatmaps(h, bad, cfg)
    bad = pd.concat([p.assign(raw_hash='a'), p.iloc[[0]].assign(raw_hash='b', point_order=1)])
    with pytest.raises(ValueError, match='Conflicting'):
        prepare_heatmaps(h, bad, cfg)


@pytest.mark.parametrize('policy', ['by_round','one_day_ahead'])
def test_graph_maps_never_include_forecast_or_future_evidence(policy):
    spec, store = graph_setup(forecast_policy=policy)
    query = store.schedule.iloc[[4]]
    issue = store.issue_times(query, spec.config).iloc[0]
    snapshot = store.snapshot(issue, spec.config)
    assert 5 not in set(snapshot.edges.event_id)
    p = points_for(sample().history)
    p.loc[p.event_id >= 5, ['x','y']] = [99., 99.]
    mutated = spec.event_store(sample().history, sample().team_seasons, p)
    pd.testing.assert_frame_equal(snapshot.edges, mutated.snapshot(issue,spec.config).edges)
    # A completed but late-released result cannot contribute its maps early.
    h = sample().history
    h['result_available_at'] = pd.NaT
    h['result_available_at'] = pd.to_datetime(h.result_available_at, utc=True)
    h.loc[h.event_id == 1,'result_available_at'] = pd.Timestamp('2020-03-01',tz='UTC')
    late = spec.event_store(h, sample().team_seasons, points_for(h))
    assert 1 not in set(late.snapshot(issue,spec.config).edges.event_id)


@pytest.mark.parametrize('reconstruction', [None, Reconstruction(weight=.25), Reconstruction(weight=.25,loss='kl')])
def test_gat_end_to_end_loss_and_native_persistence(reconstruction,tmp_path):
    spec, store = graph_setup(reconstruction)
    adapter = spec.build(store)
    fit = context(store,[1,2,3]); fit.validation = context(store,[5,6])
    adapter.fit(fit)
    b = adapter.backend_
    assert len(b.model.layers) == 2
    assert b.model.layers[0].e.in_features == b.model.layers[0].u.in_features
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in b.model.spatial_encoder.encoder.parameters())
    assert (b.model.spatial_encoder.decoder is None) == (reconstruction is None)
    metrics = set(adapter.training_history_.metric)
    assert {'train_loss','validation_loss'} <= metrics
    if reconstruction:
        assert {'train_reconstruction_loss','validation_reconstruction_loss'} <= metrics
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in b.model.spatial_encoder.decoder.parameters())
    serializer = TemporalGraphSerializer(embed_event_store=True)
    serializer.save(adapter,tmp_path)
    restored = serializer.load(tmp_path)
    query = context(store,[7,8],fit=False)
    pd.testing.assert_frame_equal(adapter.predict(query)['predict_proba'],restored.predict(query)['predict_proba'])
    assert restored.config == adapter.config
    for key,value in b.model.state_dict().items():
        assert torch.equal(value,restored.backend_.model.state_dict()[key])


def standalone_context(task='classification'):
    rng = np.random.default_rng(13)
    maps = rng.random((10,2,4,4))
    maps /= maps.sum(axis=(2,3),keepdims=True)
    maps[0,0] = np.nan; maps[1,1] = np.nan; maps[2] = np.nan
    names = [f'{side}::recent::heatmap_for_mass_4_y{y:02}_x{x:02}' for side in ('home','away') for y in range(4) for x in range(4)]
    X = pd.DataFrame(maps.reshape(10,-1),columns=names,index=pd.Index(np.arange(10),name='row_position'))
    X['strength'] = np.arange(10,dtype=float)
    meta = pd.DataFrame({'event_id':np.arange(10),'kickoff_at':pd.date_range('2020-01-01',periods=10,tz='UTC')}, index=X.index)
    y = pd.DataFrame({'result':np.arange(10)%2 if task == 'classification' else np.arange(10)/3},index=X.index)
    return FitContext(X,meta,'match',('event_id',),0,y=y)


@pytest.mark.parametrize('task', ['classification','regression'])
@pytest.mark.parametrize('reconstruction', [None, Reconstruction(), Reconstruction(loss='kl')])
def test_standalone_runner_weighting_validation_export_and_reload(task,reconstruction,tmp_path):
    c = standalone_context(task)
    spec = HeatmapCNNModel(task=task,encoder=HeatmapCNNConfig(channels=(3,5),embedding_dim=7,reconstruction=reconstruction),
                          numeric_features=('strength',),batch_size=2, control=TrainingControl(max_steps=2))
    data = ModelDataset(c.X,c.y,c.metadata,'match',('event_id',),('event_id',),'home')
    plan = SplitPlan([Fold(np.arange(8),np.array([8,9]),np.array([8,9]))],10,np.arange(10))
    from xdiyo_analytics.training import ValidationTail
    result = TrainingRunner(spec.build,validation=ValidationTail(.25)).run(data,plan)
    adapter = result.folds[0].model
    assert {'train_loss','validation_loss'} <= set(adapter.training_history_.metric)
    assert adapter.backend_.inputs.numeric_scale.mean[0] == 2.5
    saved = save_model(result,tmp_path/'saved')
    loaded = load_model(saved)
    for method,expected in result.folds[0].predictions.items():
        pd.testing.assert_frame_equal(loaded.predict(data,positions=np.array([8,9]))[method],expected)
    np.testing.assert_array_equal(adapter.encode(c).iloc[2],np.zeros(7))
    # Equal seeds recreate the same network, including the optional decoder.
    a, b = spec.build(), spec.build()
    c.sample_weight = pd.Series([0.,1.,2.,3.,1.,2.,3.,1.,2.,1.],index=c.X.index)
    a.fit(c); b.fit(c)
    pd.testing.assert_frame_equal(a.predict(c)['predict'],b.predict(c)['predict'])
    assert all(torch.isfinite(p).all() for p in a.backend_.model.parameters())
    if task == 'classification':
        from xdiyo_analytics.training import ProbabilityCalibrator
        from xdiyo_analytics.training.calibration import CalibratedAdapter
        p = adapter.predict(c)['predict_proba']
        calibrator = ProbabilityCalibrator().fit(p,c.y)
        result.folds[0].model = CalibratedAdapter(adapter,calibrator,np.array([8,9]))
        fitted = load_model(save_model(result,tmp_path/'calibrated'))
        np.testing.assert_allclose(fitted.predict(data)['predict_proba'].sum(axis=1),1)


def test_native_catalog_roundtrip_and_no_eager_torch_import():
    from xdiyo_analytics.ui.recipe import catalog_for_ui
    catalog = catalog_for_ui()
    for spec in (HeatmapCNNModel(encoder=HeatmapCNNConfig(reconstruction=Reconstruction(loss='kl'))),graph_setup(Reconstruction(loss='kl'))[0]):
        assert catalog.build(json.loads(json.dumps(catalog.encode(spec)))) == spec
    code = 'import sys; import xdiyo_analytics.neural; import xdiyo_analytics.graphs; assert "torch" not in sys.modules'
    subprocess.run([sys.executable,'-c',code],check=True)


@pytest.mark.parametrize('graph', [True,False])
@pytest.mark.parametrize('loss', ['mse','kl'])
def test_recipe_preparation_training_export_and_reload(ui_recipe,tmp_path,graph,loss):
    from pathlib import Path
    import hashlib
    from xdiyo_analytics.ui.recipe import node, prepare_recipe, run_recipe, export_python, predict_recipe
    from test_graph_integration import graph_recipe
    # Install a native heatmap table into each synthetic season export.
    for path in Path(ui_recipe['data']['data_root']).rglob('versions/v1/manifest.json'):
        manifest = json.loads(path.read_text(encoding='utf-8'))
        matches = pd.read_parquet(path.parent/'matches.parquet')
        points = pd.concat([matches[['event_id',side+'_id']].rename(columns={side+'_id':'team_id'}).assign(
            kind='player', x=15. if side=='home' else 65., y=25.,point_order=0) for side in ('home','away')],ignore_index=True)
        target = path.parent/'heatmap_points.parquet'
        points.to_parquet(target,index=False,compression='zstd')
        manifest['tables']['heatmap_points'] = dict(file=target.name,rows=len(points),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        path.write_text(json.dumps(manifest),encoding='utf-8')
    if graph:
        recipe = graph_recipe(ui_recipe,tmp_path)
        recipe['model']['params']['config']['params']['heatmaps'] = node('graphs.GraphHeatmaps',encoder=node('neural.HeatmapCNNConfig',reconstruction=node('neural.Reconstruction',loss=loss)))
    else:
        recipe = ui_recipe
        recipe.update(model=node('neural.HeatmapCNNModel',task='regression',control=node('training.TrainingControl',max_steps=2),
                                 encoder=node('neural.HeatmapCNNConfig',reconstruction=node('neural.Reconstruction',loss=loss))),
                      preprocessors=[],post_reporters={},features={'recent':node('features.RollingMean',
                          source=node('features.Heatmap',grid_size=4,kinds=['player'],orientation='team'),window=3,min_periods=1)})
    prepared = prepare_recipe(recipe)
    result = run_recipe(recipe,prepared=prepared)
    assert len(result.training.folds) > 0
    assert 'train_reconstruction_loss' in set(result.training.folds[0].training_history.metric)
    from xdiyo_analytics.analysis import PostTrainingAnalysis
    from xdiyo_analytics.reporting import GATLossReporter, LearningCurveReporter
    reporter = GATLossReporter if graph else LearningCurveReporter
    report = PostTrainingAnalysis({'loss':reporter(type='overall',metrics=['train_loss','train_reconstruction_loss','train_objective'])}).run(result.training)
    assert set(report.studies[0].result.tables['history'].metric) == {'train_loss','train_reconstruction_loss','train_objective'}
    assert 'reconstruction' in report.to_html()
    compile(export_python(recipe),'cnn_recipe.py','exec')
    recovered = run_recipe(json.loads(json.dumps(recipe)),prepared=prepared)
    for a,b in zip(result.training.folds,recovered.training.folds):
        pd.testing.assert_frame_equal(a.predictions['predict'],b.predictions['predict'])
    model_path = save_model(result.training,tmp_path/'recipe_model',fold_id=0)
    future = json.loads(json.dumps(recipe))
    future['prediction']['model_path'] = str(model_path)
    predictions = predict_recipe(future)
    assert predictions is not None


def test_standalone_input_identity_and_partial_map_rejection():
    from xdiyo_analytics.neural.adapter import CNNInputs
    ctx = standalone_context()
    for bad in (ctx.X.drop(columns=ctx.X.columns[0]),pd.concat([ctx.X,ctx.X.iloc[:,[0]]],axis=1)):
        with pytest.raises(ValueError):
            CNNInputs(HeatmapCNNModel()).fit(bad)
    bad = ctx.X.copy(); bad.iloc[5,0] = np.nan
    with pytest.raises(ValueError,match='Partially missing'):
        CNNInputs(HeatmapCNNModel()).fit(bad)


def test_legacy_scalar_artifact_and_nondefault_cnn_grid(tmp_path):
    adapter = TemporalGraphAdapter(config(),sample(),control=TrainingControl(max_steps=1))
    adapter.fit(context(sample(),[1,2,3]))
    serializer = TemporalGraphSerializer(embed_event_store=True)
    serializer.save(adapter,tmp_path)
    # An old v2 artifact has no heatmaps field, including in its fingerprint.
    path = tmp_path/'graph.json'
    metadata = json.loads(path.read_text(encoding='utf-8'))
    del metadata['config']['heatmaps']
    path.write_text(json.dumps(metadata),encoding='utf-8')
    loaded = serializer.load(tmp_path)
    assert loaded.backend_.model_version == adapter.backend_.model_version
    pd.testing.assert_frame_equal(loaded.predict(context(sample(),[7],fit=False))['predict_proba'],adapter.predict(context(sample(),[7],fit=False))['predict_proba'])
    spatial = GraphHeatmaps(grid=Heatmap(grid_size=6,kinds=('player',),side='both',orientation='team'),
                           encoder=HeatmapCNNConfig(channels=(4,),kernel_size=5,pool_size=3,embedding_dim=5,activation='gelu',reconstruction=Reconstruction(hidden_width=9)))
    cfg = config(heatmaps=spatial)
    spec = TemporalGraphModel(config=cfg,control=TrainingControl(max_steps=1))
    h = sample().history
    store = spec.event_store(h,sample().team_seasons,points_for(h).iloc[:0])
    model = spec.build(store); model.fit(context(store,[1,2,3]))
    assert model.training_history_.loc[lambda t:t.metric.eq('train_reconstruction_loss'),'value'].eq(0).all()
    assert np.isfinite(model.predict(context(store,[7],fit=False))['predict_proba'].to_numpy()).all()
    changed = replace(spec,config=replace(cfg,heatmaps=replace(spatial,grid=replace(spatial.grid,grid_size=9))))
    with pytest.raises(ValueError,match='grid settings change preparation'):
        changed.build(store)
