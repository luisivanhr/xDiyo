"""Tiny CPU checks only; no experimental population is fitted or selected."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch
import run
from xdiyo_analytics.graphs import TemporalEventStore, TemporalGraphAdapter, TemporalGraphSerializer
from xdiyo_analytics.training import TrainingControl, EarlyStopping
from xdiyo_analytics.training.resume_loop import run_resumable_iterations
from xdiyo_analytics.graphs.research_training import digest_state
from temporal_team_gat import make_context


def sample():
    rows = []
    for i in range(12):
        for home in (True, False):
            row = dict(event_id=i+1, kickoff_at=pd.Timestamp('2020-01-01', tz='UTC')+pd.Timedelta(days=7*i),
                team_id=1 if home else 2, opponent_id=2 if home else 1, side='home' if home else 'away',
                goals_for=(i % 3) if home else ((i+1) % 3), goals_against=((i+1) % 3) if home else i % 3,
                competition_id=10, season_id=900, round=i+1, round_stage='regular', status='finished')
            for j, col in enumerate(run.STATISTICS):
                row[col] = float(2 + i % 7)
            rows.append(row)
    memberships = pd.DataFrame([dict(team_id=t, competition_id=10, season_id=900,
        entry_at=pd.Timestamp('2019-07-01', tz='UTC'), movement='retained') for t in (1, 2)])
    return TemporalEventStore(pd.DataFrame(rows), memberships)


def main():
    torch.set_num_threads(1)
    store = sample()
    cfg = run.config('time_B/N', store)
    train = make_context(store.schedule.iloc[2:6].reset_index(drop=True))
    train.validation = make_context(store.schedule.iloc[7:9].reset_index(drop=True))
    outer = make_context(store.schedule.iloc[10:].reset_index(drop=True), fitting=False)
    control = TrainingControl(max_steps=3, early_stopping=EarlyStopping(15,0), seed=17,
        monitor=run.MONITOR, restore_best=True)
    Backend = run.backend_class()
    factory = lambda seed: Backend(cfg, store, seed, 'cpu')
    baseline, h, summary = run_resumable_iterations(factory, train, control, save=lambda *a, **k: None)
    assert 1 <= summary['retained_step'] <= 3
    assert summary['monitored_value'] == float(h.loc[h.metric.eq(run.MONITOR),'value'].min())
    state = None
    class Interrupted(Exception):
        pass
    def save(capture, *, complete):
        nonlocal state
        state = capture()
        if not complete:
            raise Interrupted()
    try:
        run_resumable_iterations(factory, train, control, save=save)
    except Interrupted:
        pass
    assert state is not None
    resumed, rh, rs = run_resumable_iterations(factory, train, control, state=state, save=lambda *a, **k: None)
    assert digest_state(baseline.model.state_dict()) == digest_state(resumed.model.state_dict())
    assert summary == rs
    # Variable timings differ; numerical histories must match exactly.
    pd.testing.assert_frame_equal(h.loc[h.metric.ne('update_seconds')].reset_index(drop=True), rh.loc[rh.metric.ne('update_seconds')].reset_index(drop=True))
    p, rates = run.predict(baseline, baseline.validation_groups, train.validation.metadata)
    assert np.isclose(run.table(train.validation.metadata,p,rates).ce.mean(), summary['monitored_value'], atol=2e-7)
    adapter = TemporalGraphAdapter(cfg, store, control=control, device='cpu')
    adapter.backend_, adapter.training_history_, adapter.training_summary_ = baseline, h, summary
    baseline._boundary(baseline._validate(outer))
    p0, r0 = run.predict(baseline, baseline._stream(outer), outer.metadata)
    with tempfile.TemporaryDirectory() as d:
        TemporalGraphSerializer(store).save(adapter, d)
        loaded = TemporalGraphSerializer(store).load(d).backend_
        p1, r1 = run.predict(loaded, loaded._stream(outer), outer.metadata)
        assert np.array_equal(p0,p1) and np.array_equal(r0,r1)
    # Fixed-budget refit must retain final state, never retrospectively choose a
    # training-loss minimum as if it were a validation checkpoint.
    train.validation = None
    fixed = TrainingControl(max_steps=2, seed=17, monitor='train_loss', restore_best=False)
    refit, fh, fs = run_resumable_iterations(factory, train, fixed, save=lambda *a, **k: None)
    assert fs['retained_step'] == 2 and fs['restore_best'] is False
    result = dict(status='PASS', device='cpu', fixtures=12, selection_steps=3,
        fixed_refit_steps=2, interrupted_resume_bit_exact=True, serializer_predictions_bit_exact=True,
        raw_validation_ce_matches_export=True, summary_keys=list(summary),
        source_sha256=run.sha(run.ROOT/'run.py'))
    run.write(run.OUT/'RUNNER_UNIT_TEST.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
