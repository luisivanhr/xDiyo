"""Lifetime and bit-exact parity with the aca3e31 preparation loop."""
from copy import deepcopy
import weakref

import numpy as np
import pandas as pd
import pytest

from test_temporal_graph import config, context, sample, neural
from xdiyo_analytics.graphs import TemporalEventStore, TemporalGraphAdapter, TemporalGraphSerializer
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.training import TrainingControl


def reference_prepare(self, raw):
    """Frozen aca3e31 loop: intentionally retains all raw snapshots."""
    import torch
    groups = []
    for positions, snap, home, away, meta, numeric in raw:
        groups.append((positions, self.preprocessor.tensors(snap, self.device),
                       torch.as_tensor(home, device=self.device),
                       torch.as_tensor(away, device=self.device),
                       torch.as_tensor(self.preprocessor.fixture(meta, numeric), dtype=torch.float32, device=self.device)))
    return groups


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
def test_consuming_preparation_bit_exact_training_and_exports(monkeypatch, tmp_path, policy, statistics):
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
    control = TrainingControl(max_steps=4, restore_best=False, seed=72)
    steps = []
    original_step = _Backend.step

    def recorded_step(backend):
        values = original_step(backend)
        steps.append(deepcopy((values, backend.model.state_dict(), backend.optimizer.state_dict())))
        return values

    monkeypatch.setattr(_Backend, "step", recorded_step)
    native = TemporalGraphAdapter(cfg, store, control=control)
    native.fit(train)
    with monkeypatch.context() as patch:
        patch.setattr(_Backend, "_prepare", reference_prepare)
        original = TemporalGraphAdapter(cfg, store, control=control)
        original.fit(train)
        reference_prediction = original.predict(predict)
    a, b = native.backend_, original.backend_
    assert len(steps) == 8
    assert_exact(steps[:4], steps[4:])
    assert_exact(a.train_groups, b.train_groups)
    assert_exact(a.validation_groups, b.validation_groups)
    assert [p.tolist() for p, *_ in a.train_groups] == [[2], [1, 3], [0]]
    assert [p.tolist() for p, *_ in a.validation_groups] == [[1], [0]]
    assert_exact(a.preprocessor.state(), b.preprocessor.state())
    assert_exact(a.model.state_dict(), b.model.state_dict())
    assert_exact(a.optimizer.state_dict(), b.optimizer.state_dict())
    assert_exact(a.train_weights, b.train_weights)
    assert_exact(a.validation_weights, b.validation_weights)
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
    serializer = TemporalGraphSerializer(embed_event_store=True)
    for name, model in (("native", native), ("original", original)):
        serializer.save(model, tmp_path / name)
        restored = serializer.load(tmp_path / name)
        pd.testing.assert_frame_equal(restored.encode_teams(queries), expected, check_exact=True)
        for key, frame in restored.predict(predict).items():
            pd.testing.assert_frame_equal(frame, reference_prediction[key], check_exact=True)
    assert_exact(frozen, (a.model.state_dict(), a.optimizer.state_dict(), a.preprocessor.state(),
                         a.model_version, a.model.training))


@neural
def test_each_snapshot_released_before_next_tensor_preparation(monkeypatch):
    store = sample()
    backend = _Backend(config(), store, 1, "cpu")
    backend.initialize(context(store, [1, 2, 3]))
    raw, _ = backend._raw_groups(context(store, [1, 2, 3]))
    refs = [(weakref.ref(g[1]), weakref.ref(g[1].edges), weakref.ref(g[4])) for g in raw]
    expected = reference_prepare(backend, raw)
    original = backend.preprocessor.tensors
    calls = []

    def tensors(snapshot, device):
        for previous in refs[:len(calls)]:
            assert all(ref() is None for ref in previous)
        calls.append(snapshot.issuance)
        return original(snapshot, device)

    monkeypatch.setattr(backend.preprocessor, "tensors", tensors)
    actual = backend._prepare(raw)
    assert raw == []
    assert all(ref() is None for group in refs for ref in group)
    # Shared NumPy backing buffers must survive raw-object destruction.
    assert_exact(actual, expected)
    assert backend._prepare([]) == []
