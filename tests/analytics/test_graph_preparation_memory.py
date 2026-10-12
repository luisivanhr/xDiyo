"""Bounded execution and bit-exact parity with the frozen f8962af adapter."""
from copy import deepcopy
import weakref
from pathlib import Path
import types

import numpy as np
import pandas as pd
import pytest

from test_temporal_graph import config, context, sample, neural
from xdiyo_analytics.graphs import TemporalEventStore, TemporalGraphAdapter, TemporalGraphSerializer
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.training import TrainingControl


def reference_module(name):
    module = types.ModuleType("xdiyo_analytics.graphs._eager_reference")
    module.__package__ = "xdiyo_analytics.graphs"
    path = Path(__file__).parent / "reference" / f"gat_{name}_f8962af.txt"
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    # The frozen adapter predates optional CNN and goal-selection fields. Preserve its original
    # serialized config when testing old model fingerprints with today's config
    # instance; otherwise asdict would inject a field absent at f8962af.
    if name == 'adapter':
        original_asdict = module.asdict
        def legacy_asdict(value):
            result = original_asdict(value)
            from xdiyo_analytics.graphs import TemporalGraphConfig
            if isinstance(value, TemporalGraphConfig):
                assert value.heatmaps is None
                result.pop('heatmaps', None)
                assert value.include_goal_features is True
                result.pop('include_goal_features', None)
            return result
        module.asdict = legacy_asdict
    return module


def assert_exact(a, b):
    import torch
    assert type(a) is type(b)
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.detach().cpu().numpy().tobytes() == b.detach().cpu().numpy().tobytes()
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    elif isinstance(a, dict):
        assert list(a) == list(b)
        for key in a:
            assert_exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_exact(x, y)
    else:
        assert a == b


@neural
@pytest.mark.parametrize("policy", ["by_round", "one_day_ahead"])
@pytest.mark.parametrize("statistics", [False, True])
@pytest.mark.parametrize("restore_best", [False, True])
def test_streaming_bit_exact_training_and_exports(monkeypatch, tmp_path, policy, statistics, restore_best):
    store = sample()
    h, schedule, members = store.history, store.schedule, store.team_seasons
    # Multiple fixtures per issuance, deliberately interleaved input order.
    schedule.loc[[2, 3], "round"] = 3
    if policy == "one_day_ahead":
        schedule["issuance_at"] = schedule.kickoff_at - pd.Timedelta(days=1)
        schedule.loc[3, "issuance_at"] = schedule.loc[2, "issuance_at"]
    h["result_available_at"] = pd.to_datetime(h.kickoff_at + pd.Timedelta(hours=6), utc=True)
    h.loc[h.event_id.eq(1), "result_available_at"] = pd.Timestamp("2020-01-07T12:00Z")
    extra = members.iloc[[0]].copy()
    extra["entry_at"] = pd.Timestamp("2020-01-08T00:00Z")
    extra["season_id"], extra["movement"] = 901, "promoted"
    extra["competition_id"] = 11  # Category introduced by a snapshot, not its edges.
    members = pd.concat([members, extra], ignore_index=True)
    stat = "team::ALL::Shots::shotsOnGoal::value"
    other = stat.replace("team::", "opponent::")
    if statistics:
        h[stat] = np.where(h.side.eq("home"), 0., np.nan)
        h[other] = np.where(h.side.eq("away"), 0., np.nan)
    store = TemporalEventStore(h, members, schedule=schedule)
    cfg = config(forecast_policy=policy, statistics=(stat, other) if statistics else (),
                 fixture_features=("unused",), half_life_days=41., seasonal_retention=.7)
    train = context(store, [4, 3, 1, 2])
    train.validation = context(store, [6, 5])
    predict = context(store, [8, 7], fit=False)
    for ctx in (train, train.validation, predict):
        ctx.X["unused"] = np.arange(len(ctx.X), dtype=float)
        ctx.X.iloc[0, 0] = np.nan
        ctx.metadata["feature_cutoff_at"] = [cfg.history_cutoff(t) for t in store.issue_times(ctx.metadata, cfg)]
    train.sample_weight = pd.Series([2., 0., 1., .5], index=train.X.index)
    train.validation.sample_weight = pd.Series([.7, 1.3], index=train.validation.X.index)
    control = TrainingControl(max_steps=4, restore_best=restore_best, seed=72)
    import torch
    reference = reference_module("adapter")
    steps, gradients = [], []
    adam_step = torch.optim.AdamW.step

    def recorded_adam(optimizer, *args, **kwargs):
        gradients.append([p.grad.detach().clone() if p.grad is not None else None
                          for group in optimizer.param_groups for p in group["params"]])
        return adam_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", recorded_adam)
    for cls in (_Backend, reference._Backend):
        original_step = cls.step
        def recorded_step(backend, step=original_step):
            values = step(backend)
            steps.append(deepcopy((values, backend.model.state_dict(), backend.optimizer.state_dict())))
            return values
        monkeypatch.setattr(cls, "step", recorded_step)
    native = TemporalGraphAdapter(cfg, store, control=control)
    native.fit(train)
    original = reference.TemporalGraphAdapter(cfg, store, control=control)
    original.fit(train)
    reference_prediction = original.predict(predict)
    a, b = native.backend_, original.backend_
    assert len(steps) == 8
    assert_exact(steps[:4], steps[4:])
    assert_exact(gradients[:4], gradients[4:])
    assert_exact(list(a.train_groups), b.train_groups)
    assert_exact(list(a.validation_groups), b.validation_groups)
    assert [p.tolist() for p, *_ in a.train_groups] == [[2], [1, 3], [0]]
    assert [p.tolist() for p, *_ in a.validation_groups] == [[1], [0]]
    assert_exact(a.preprocessor.state(), b.preprocessor.state())
    assert_exact(a.model.state_dict(), b.model.state_dict())
    assert_exact(a.optimizer.state_dict(), b.optimizer.state_dict())
    # Native weights are now safely binary-rescaled before conversion. The
    # frozen eager reference keeps raw weights; their proportions are exact
    # here, as are all gradients, optimizer states and outputs checked above.
    assert_exact(a.train_weights/a.train_weights.sum(), b.train_weights/b.train_weights.sum())
    assert_exact(a.validation_weights/a.validation_weights.sum(), b.validation_weights/b.validation_weights.sum())
    pd.testing.assert_frame_equal(native.training_history_, original.training_history_, check_exact=True)
    assert native.training_summary_ == original.training_summary_
    assert a.model_version == b.model_version
    frozen = deepcopy((a.model.state_dict(), a.optimizer.state_dict(), a.preprocessor.state(),
                       a.model_version, a.model.training))
    for name, frame in native.predict(predict).items():
        pd.testing.assert_frame_equal(frame, reference_prediction[name], check_exact=True)
    queries = pd.DataFrame({"team_id": [2, 999, 1], "issuance_at": pd.to_datetime(["2020-03-01T12:00Z"] * 3)})
    expected = original.encode_teams(queries)
    pd.testing.assert_frame_equal(native.encode_teams(queries), expected, check_exact=True)
    pd.testing.assert_frame_equal(native.attention_weights(queries),
        reference_module("inspection").attention_weights(original, queries), check_exact=True)
    serializer = TemporalGraphSerializer(embed_event_store=True)
    for name, model in (("native", native), ("original", original)):
        serializer.save(model, tmp_path / name)
        restored = serializer.load(tmp_path / name)
        pd.testing.assert_frame_equal(restored.encode_teams(queries), expected, check_exact=True)
        for key, frame in restored.predict(predict).items():
            pd.testing.assert_frame_equal(frame, reference_prediction[key], check_exact=True)
    assert_exact(frozen, (a.model.state_dict(), a.optimizer.state_dict(), a.preprocessor.state(),
                         a.model_version, a.model.training))


def watch_lifetimes(monkeypatch, store):
    from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
    refs = []
    snapshot, tensors = store.snapshot, GraphPreprocessor.tensors

    def no_live_graph():
        assert all(ref() is None for ref in refs)

    def tracked_snapshot(*args, **kwargs):
        no_live_graph()
        snap = snapshot(*args, **kwargs)
        refs.extend((weakref.ref(snap), weakref.ref(snap.edges)))
        return snap

    def tracked_tensors(self, snap, device):
        result = tensors(self, snap, device)
        refs.extend(weakref.ref(tensor) for tensor in result.values())
        return result

    monkeypatch.setattr(store, "snapshot", tracked_snapshot)
    monkeypatch.setattr(GraphPreprocessor, "tensors", tracked_tensors)
    return no_live_graph


@neural
def test_only_one_live_graph_across_all_operations(monkeypatch):
    store = sample()
    no_live_graph = watch_lifetimes(monkeypatch, store)
    train = context(store, [3, 1, 2])
    train.validation = context(store, [6, 5])
    model = TemporalGraphAdapter(config(), store, control=TrainingControl(max_steps=2, restarts=1))
    model.fit(train)
    no_live_graph()
    model.predict(context(store, [8, 7], fit=False))
    no_live_graph()
    queries = pd.DataFrame({"team_id": [1, 2, 999],
                           "issuance_at": pd.to_datetime(["2020-03-02", "2020-03-01", "2020-03-02"], utc=True)})
    model.encode_teams(queries)
    no_live_graph()
    model.attention_weights(queries)
    no_live_graph()
    # Iterators themselves do not retain a most recently returned tensor group.
    iterator = iter(model.backend_.train_groups)
    first = next(iterator)
    del first
    no_live_graph()
    second = next(iterator)
    del second
    no_live_graph()
    del iterator


@neural
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("operation", ["initialize", "step", "predict", "encode", "attention"])
def test_exception_and_cancellation_release_graphs(monkeypatch, error_type, operation):
    store = sample()
    model = TemporalGraphAdapter(config(), store, control=TrainingControl(max_steps=1))
    model.fit(context(store, [1, 2, 3]))
    no_live_graph = watch_lifetimes(monkeypatch, store)
    mode = model.backend_.model.training
    prediction = context(store, [7, 8], fit=False)
    queries = pd.DataFrame({"team_id": [1], "issuance_at": pd.to_datetime(["2020-03-01"], utc=True)})

    if operation == "initialize":
        from xdiyo_analytics.graphs.streaming import GroupPlan
        def fail_summary(plan, index, numeric):
            raw = plan.raw(index)
            raise error_type("interrupted with a live snapshot")
        monkeypatch.setattr(GroupPlan, "_summarize", fail_summary)
        with pytest.raises(error_type) as caught:
            model.backend_.initialize(context(store, [1, 2, 3]))
    else:
        def fail(*args, **kwargs):
            raise error_type("interrupted with live tensors")
        monkeypatch.setattr(model.backend_.model, "encode", fail)
        with pytest.raises(error_type) as caught:
            if operation == "step":
                model.backend_.step()
            elif operation == "predict":
                model.predict(prediction)
            elif operation == "encode":
                model.encode_teams(queries)
            else:
                model.attention_weights(queries)
    # Retain the exception traceback: it must not keep raw frames/tensors alive.
    assert caught.value.__traceback__ is not None
    no_live_graph()
    if operation in ("predict", "encode", "attention"):
        assert model.backend_.model.training == mode


@neural
def test_restartable_stream_freezes_fixture_inputs(monkeypatch):
    store = sample()
    train = context(store, [3, 1, 2])
    cfg = config(fixture_features=("unused",))
    train.metadata["feature_cutoff_at"] = [cfg.history_cutoff(t) for t in store.issue_times(train.metadata, cfg)]
    backend = _Backend(cfg, store, 17, "cpu")
    backend.initialize(train)
    first = list(backend.train_groups)
    train.X["unused"] = 999
    train.metadata["home_id"] = 999
    assert_exact(first, list(backend.train_groups))
    assert_exact(first, list(backend.train_groups))


@neural
def test_frozen_round_with_long_postponement_and_future_mutation():
    store = sample()
    model = TemporalGraphAdapter(config(), store, control=TrainingControl(max_steps=2))
    model.fit(context(store, [1, 2, 3]))
    pred = context(store, [8, 7], fit=False)
    before = model.predict(pred)
    schedule = store.schedule
    schedule["round_issue_at"] = schedule.kickoff_at
    schedule.loc[8, "kickoff_at"] += pd.Timedelta(days=100)
    h = store.history
    # Results released after both test issue times cannot enter either graph.
    h.loc[h.event_id.eq(9), ["goals_for", "goals_against"]] = 99
    changed = TemporalEventStore(h, store.team_seasons, schedule=schedule)
    model.backend_.store = changed
    model.event_store = changed
    postponed = context(changed, [8, 7], fit=False)
    for key, frame in model.predict(postponed).items():
        pd.testing.assert_frame_equal(frame, before[key], check_exact=True)


@neural
def test_streaming_cold_only_training_and_control_selection():
    from xdiyo_analytics.training.control import EarlyStopping, ReduceOnPlateau
    store = sample()
    cold = TemporalEventStore(store.history.iloc[:0], store.team_seasons, schedule=store.schedule)
    train = context(store, [3, 1, 2])
    train.validation = context(store, [6, 5])
    control = TrainingControl(max_steps=5, seed=17, restarts=1,
        early_stopping=EarlyStopping(patience=2, min_delta=100.),
        scheduler=ReduceOnPlateau(patience=1), restore_best=True)
    models = []
    for cls in (TemporalGraphAdapter, reference_module("adapter").TemporalGraphAdapter):
        model = cls(config(), cold, control=control)
        model.fit(train)
        models.append(model)
    a, b = models
    assert a.training_summary_ == b.training_summary_
    pd.testing.assert_frame_equal(a.training_history_, b.training_history_, check_exact=True)
    assert_exact(a.backend_.optimizer.state_dict(), b.backend_.optimizer.state_dict())
    assert_exact(a.backend_.model.state_dict(), b.backend_.model.state_dict())
    assert_exact(list(a.backend_.train_groups), b.backend_.train_groups)


@neural
def test_stream_failure_direct_consumer_does_not_pin_raw_arguments(monkeypatch):
    from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
    store = sample()
    backend = _Backend(config(), store, 17, "cpu")
    backend.initialize(context(store, [1, 2, 3]))
    no_live_graph = watch_lifetimes(monkeypatch, store)
    def failed_transform(preprocessor, snap, device):
        raise RuntimeError("transform error")
    monkeypatch.setattr(GraphPreprocessor, "tensors", failed_transform)
    with pytest.raises(RuntimeError) as caught:
        backend.train_groups[0]
    assert caught.value.__traceback__
    no_live_graph()
