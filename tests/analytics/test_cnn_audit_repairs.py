"""Synthetic regressions adapted from Ayre's 5eded399 audit, no research fits."""
from dataclasses import asdict, replace
import json
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
from xdiyo_analytics.neural import HeatmapCNNConfig, HeatmapCNNModel, HeatmapCNNSerializer, Reconstruction
from xdiyo_analytics.neural.adapter import CNNBackend
from xdiyo_analytics.training.contracts import FitContext
from xdiyo_analytics.training.control import EarlyStopping, ReduceOnPlateau, TrainingControl, run_iterations
from xdiyo_analytics.graphs import GraphHeatmaps, TemporalGraphAdapter, TemporalGraphConfig, TemporalGraphModel, TemporalGraphSerializer
from xdiyo_analytics.graphs.spatial import prepare_heatmaps, tensors
from xdiyo_analytics.features import Heatmap
from xdiyo_analytics.features.spatial import heatmap_values
from xdiyo_analytics.features.point_geometry import point_summary_source
from test_heatmap_cnn import one_thread

def cnn_context(weight=None, target=1.0):
    index = pd.Index(range(4), name="row_position")
    names = [f"home::recent::heatmap_for_mass_2_y{y:02}_x{x:02}" for y in range(2) for x in range(2)]
    X = pd.DataFrame(np.full((4, 4), 0.25), columns=names, index=index)
    meta = pd.DataFrame({"event_id": range(4), "kickoff_at": pd.date_range("2020-01-01", periods=4, tz="UTC")}, index=index)
    c = FitContext(X, meta, "match", ("event_id",), 0, y=pd.DataFrame({"target": target}, index=index))
    if weight is not None:
        c.sample_weight = pd.Series(weight, index=index)
    return c


class ZeroBackend(CNNBackend):
    def initialize(self, c):
        super().initialize(c)
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()


def fixture():
    rows = []
    for event in range(1, 7):
        kick = pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=3*(event-1))
        gf, ga = event % 3, (event+1) % 3
        for team, opponent, side, a, b in [
            (1, 2, "home", gf, ga), (2, 1, "away", ga, gf)
        ]:
            rows.append(dict(
                event_id=event, competition_id=10, season_id=900, round=event,
                team_id=team, opponent_id=opponent, side=side, kickoff_at=kick,
                status="finished", goals_for=a, goals_against=b,
            ))
    history = pd.DataFrame(rows)
    memberships = pd.DataFrame([
        dict(team_id=t, competition_id=10, season_id=900,
             entry_at=pd.Timestamp("2019-07-01", tz="UTC"), movement="retained")
        for t in (1, 2)
    ])
    points = history[["event_id", "team_id"]].assign(
        kind="player", x=15., y=25., point_order=0
    )
    points = pd.concat([
        points, points.loc[points.event_id.gt(4)].assign(point_order=1)
    ], ignore_index=True)
    spatial = GraphHeatmaps(grid=Heatmap(
        grid_size=4, side="both", kinds=("player",),
        orientation="team", normalization="mass",
    ))
    spec = TemporalGraphModel(
        config=TemporalGraphConfig(
            "UTC", embedding_dim=4, hidden_width=8, heatmaps=spatial
        ),
        membership_policy="explicit",
        control=TrainingControl(max_steps=1, seed=17),
    )
    return history, memberships, points, spatial, spec


def graph_context(store):
    metadata = store.schedule.iloc[[1, 2]].copy()
    X = pd.DataFrame(index=metadata.index)
    target = (
        (metadata.goals_for > metadata.goals_against).astype(int)
        - (metadata.goals_for < metadata.goals_against).astype(int)
    )
    y = pd.DataFrame({"result": target}, index=metadata.index)
    return FitContext(
        X=X, y=y, metadata=metadata, layout="match",
        match_columns=("competition_id", "season_id", "event_id"), fold_id=0,
    )


def test_grid_identity_at_every_entry_point(tmp_path):
    h, m, p, spatial, spec = fixture()
    count = replace(spec, config=replace(spec.config, heatmaps=replace(
        spatial, grid=replace(spatial.grid, normalization='count'))))
    old = spec.event_store(h[h.event_id.le(4)], m, p[p.event_id.le(4)])
    wrong = count.event_store(h[h.event_id.le(4)], m, p[p.event_id.le(4)])
    extended_wrong = count.event_store(h, m, p)
    assert old.version == wrong.version  # Coincidentally identical old cells.
    for build in (lambda: spec.build(wrong), lambda: TemporalGraphAdapter(spec.config, wrong)):
        with pytest.raises(ValueError, match='grid settings'):
            build()
    model = spec.build(old)
    model.fit(graph_context(old))
    serializer = TemporalGraphSerializer(old)
    serializer.save(model, tmp_path)
    with pytest.raises(ValueError, match='grid settings'):
        TemporalGraphSerializer(wrong).load(tmp_path)
    with pytest.raises(ValueError, match='grid settings'):
        model.attach_event_store(extended_wrong)
    assert model.event_store is old and model.backend_.store is old
    model.event_store = wrong
    with pytest.raises(ValueError, match='grid settings'):
        model.fit(graph_context(old))
    model.event_store = old
    loaded = serializer.load(tmp_path)
    later = spec.event_store(h, m, p)
    loaded.attach_event_store(later)
    assert loaded.event_store is later
    # Missing semantic metadata is not an implicit match.
    later.provenance.pop('heatmap_grid')
    with pytest.raises(ValueError, match='grid settings'):
        TemporalGraphAdapter(spec.config, later)


@pytest.mark.parametrize('monitor', ['train_loss', 'train_objective', 'validation_loss', 'validation_objective'])
@pytest.mark.parametrize('restore', [True, False])
def test_cnn_selected_metric_matches_retained_parameters(monitor, restore):
    spec = HeatmapCNNModel(task='regression', learning_rate=10., weight_decay=0.,
                          encoder=HeatmapCNNConfig(reconstruction=Reconstruction()))
    c = cnn_context()
    c.validation = cnn_context()
    control = TrainingControl(max_steps=3, restarts=1, seed=17, monitor=monitor,
        restore_best=restore, early_stopping=EarlyStopping(2), scheduler=ReduceOnPlateau(1))
    b, history, summary = run_iterations(lambda seed: ZeroBackend(spec, seed, 'cpu'), c, control)
    with torch.no_grad():
        primary, aux = b.loss(b.validation if monitor.startswith('validation') else b.train)
    actual = primary + (spec.encoder.reconstruction.weight*aux if monitor.endswith('objective') else 0)
    assert actual == pytest.approx(summary['monitored_value'], abs=1e-9)
    recorded = history[(history.attempt == summary['selected_attempt']) &
                       (history.step == summary['retained_step']) & (history.metric == monitor)]
    assert recorded.value.iloc[0] == pytest.approx(actual, abs=1e-9)


@pytest.mark.parametrize('monitor', ['train_loss', 'train_objective', 'validation_loss'])
def test_gat_selected_metric_matches_retained_parameters(monitor):
    h, m, p, spatial, spec = fixture()
    spatial = replace(spatial, encoder=replace(spatial.encoder, reconstruction=Reconstruction()))
    spec = replace(spec, config=replace(spec.config, heatmaps=spatial))
    store = spec.event_store(h, m, p)
    c = graph_context(store)
    # Separate fitting/validation batches with native increasing cutoffs.
    from test_temporal_graph import context
    c.validation = context(store, [3], fit=True)
    model = spec.build(store)
    model.fit_controlled(c, TrainingControl(max_steps=2, monitor=monitor, restarts=1))
    b = model.backend_
    with torch.no_grad():
        if monitor == 'validation_loss':
            actual = b._loss(b.validation_groups, b.validation_targets, b.validation_weights)
        else:
            actual = b._loss(b.train_groups, b.train_targets, b.train_weights)
            if monitor == 'train_objective':
                actual += spatial.encoder.reconstruction.weight*b._reconstruction_loss
                actual += float(spec.config.prior_penalty*(b.model.league.weight.square().sum()+b.model.entry.weight.square().sum()))
    assert actual == pytest.approx(model.training_summary_['monitored_value'], abs=1e-8)


def test_effective_controls_persist_separately_and_legacy_unknown(tmp_path):
    spec = HeatmapCNNModel(task='regression', control=TrainingControl(max_steps=100, seed=0))
    effective = TrainingControl(max_steps=2, seed=17, early_stopping=EarlyStopping(1, .2),
        scheduler=ReduceOnPlateau(1, .3, 1e-5), restarts=1, monitor='train_loss', restore_best=False)
    model = spec.build()
    c = cnn_context()
    model.fit_controlled(c, effective)
    serializer = HeatmapCNNSerializer()
    serializer.save(model, tmp_path)
    saved = json.loads((tmp_path/'cnn.json').read_text(encoding='utf-8'))
    assert saved['effective_control'] == asdict(effective)
    assert saved['spec']['control'] == asdict(spec.control)
    loaded = serializer.load(tmp_path)
    assert loaded.effective_control_ == effective and loaded.spec.control == spec.control
    pd.testing.assert_frame_equal(model.predict(c)['predict'], loaded.predict(c)['predict'])
    del saved['effective_control']
    (tmp_path/'cnn.json').write_text(json.dumps(saved), encoding='utf-8')
    legacy = serializer.load(tmp_path)
    assert legacy.effective_control_ is None
    pd.testing.assert_frame_equal(model.predict(c)['predict'], legacy.predict(c)['predict'])
    serializer.save(legacy, tmp_path)
    assert serializer.load(tmp_path).effective_control_ is None


@pytest.mark.parametrize('domain', ['history', 'points'])
@pytest.mark.parametrize('column', ['event_id', 'team_id'])
def test_rounded_float_id_rejected_before_lookup(domain, column):
    h, _, p, spatial, _ = fixture()
    h[column] = h[column].astype('uint64') + 2**53
    p[column] = p[column].astype('uint64') + 2**53
    (h if domain == 'history' else p)[column] = (h if domain == 'history' else p)[column].astype(float)
    for read in (lambda: prepare_heatmaps(h, p, spatial), lambda: heatmap_values(h, p, spatial.grid),
                 lambda: point_summary_source(h, p, ('player',), 1)):
        with pytest.raises(ValueError, match='exact integer'):
            read()


def test_exact_large_ids_and_intentional_missing_maps():
    h, _, p, spatial, _ = fixture()
    for frame in (h, p):
        frame['event_id'] = frame.event_id.astype('uint64') + 2**63
    prepared = prepare_heatmaps(h, p, spatial)
    _, flags = tensors(prepared, spatial, 'cpu')
    assert flags.eq(1).all()
    _, absent = tensors(prepare_heatmaps(h, p.iloc[:0], spatial), spatial, 'cpu')
    assert absent.eq(0).all()


@pytest.mark.parametrize('loss', ['mse', 'kl'])
@pytest.mark.parametrize('scale', [1., 1e-32, 1e-50, 5e37, 1e307])
def test_weight_scale_and_batch_invariance(loss, scale):
    spec = HeatmapCNNModel(task='regression', encoder=HeatmapCNNConfig(reconstruction=Reconstruction(loss=loss)))
    def evaluate(weights, batch):
        c = cnn_context(weights, target=2.)
        c.X.iloc[0] = np.nan
        c.X.iloc[1] = [1., 0., 0., 0.]
        b = ZeroBackend(replace(spec, batch_size=batch), 0, 'cpu')
        b.initialize(c)
        values = b.loss(b.train, backward=True)
        grads = [v.grad.clone() for v in b.model.parameters() if v.grad is not None]
        assert all(torch.isfinite(v).all() for v in grads)
        return values, grads
    expected, expected_grad = evaluate(np.array([1., 2., 0., 4.]), 4)
    actual, actual_grad = evaluate(scale*np.array([1., 2., 0., 4.]), 2)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual[0] == 4.
    for a, e in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, e)


def test_zero_observed_weight_and_separate_aux_normalization():
    spec = HeatmapCNNModel(task='regression', encoder=HeatmapCNNConfig(reconstruction=Reconstruction()))
    for tiny, expected_aux in [(0., 0.), (1e-300, .0625)]:
        c = cnn_context([1e300, tiny, 0., 0.])
        c.X.iloc[0] = np.nan
        b = ZeroBackend(spec, 0, 'cpu'); b.initialize(c)
        primary, aux = b.loss(b.train, backward=True)
        assert primary == 1. and aux == expected_aux


def test_validation_monitor_does_not_add_training_replay(monkeypatch):
    spec = HeatmapCNNModel(task='regression')
    c = cnn_context(); c.validation = cnn_context()
    b = CNNBackend(spec, 0, 'cpu'); b.initialize(c)
    calls = []
    original = b.loss
    def loss(population, backward=False):
        calls.append((population is b.train, backward))
        return original(population, backward)
    monkeypatch.setattr(b, 'loss', loss)
    values = b.step()
    assert b.checkpoint_metrics('validation_loss', values) is values
    assert calls == [(True, True), (False, False)]


@pytest.mark.parametrize('kind', ['mse', 'kl'])
def test_batch_accumulation_preserves_values_gradients_without_scalar_reads(monkeypatch, kind):
    spec = HeatmapCNNModel(task='regression', batch_size=1,
        encoder=HeatmapCNNConfig(reconstruction=Reconstruction(loss=kind)))
    c = cnn_context([1., 3., 0., 2.]); c.X.iloc[0] = [1., 0., 0., 0.]
    a, b = [CNNBackend(spec, 17, 'cpu') for _ in range(2)]
    a.initialize(c); b.initialize(c)
    expected = np.zeros(2)
    maps, masks, numeric, y, weights = b.train
    for i in range(len(y)):
        m, av, n = [torch.as_tensor(v[i:i+1]) for v in (maps, masks, numeric)]
        prediction, z = b.model(m, av, n)
        primary = ((prediction-torch.as_tensor(y[i:i+1], dtype=torch.float32)).square().mean(1)
                   *torch.as_tensor(weights[0, i:i+1])).sum()
        aux = (b.model.encoder.reconstruction_errors(z, m, av)*torch.as_tensor(weights[1, i:i+1])).sum()
        expected += [float(primary.detach()), float(aux.detach())]
        (primary+spec.encoder.reconstruction.weight*aux).backward()
    scalar_reads = []
    original = torch.Tensor.__float__
    def read(tensor):
        scalar_reads.append(tensor.shape)
        return original(tensor)
    monkeypatch.setattr(torch.Tensor, '__float__', read)
    actual = a.loss(a.train, backward=True)
    assert len(scalar_reads) == 2  # Once per returned aggregate, not per batch.
    np.testing.assert_array_equal(actual, expected)
    for left, right in zip(a.model.parameters(), b.model.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


def test_typed_identifier_check_does_not_iterate_rows(monkeypatch):
    from xdiyo_analytics.features.point_geometry import validate_exact_point_ids
    ids = pd.DataFrame({'event_id': np.arange(100000, dtype=np.uint64), 'team_id': 17})
    def iterate(*args):
        raise AssertionError('Typed identifier validation must not loop through rows in Python.')
    monkeypatch.setattr(pd.Series, '__iter__', iterate)
    validate_exact_point_ids(ids, ids)


