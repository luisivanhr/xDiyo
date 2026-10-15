"""Independent output and lifetime checks for the library performance branch."""
import gc
import json
import weakref
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.features.history import _eligible_history_index, eligible_history_rows
from xdiyo_analytics.features import EMA, RollingMean, Lag, evaluate_features
from xdiyo_analytics.splits.cpcv import _closed_overlap
from transition_samples import history_from_games, STAT, OWN


@pytest.mark.parametrize('h2h', [False, True])
@pytest.mark.parametrize('venue', [False, True])
def test_lazy_history_matches_independent_eligibility_oracle(h2h, venue):
    rng = np.random.default_rng(701)
    history = history_from_games([dict(day=i//2, release=i//2+int(rng.integers(0, 5)),
        status='notstarted' if i % 13 == 0 else 'finished') for i in range(90)])
    history = history.sample(frac=1, random_state=3)
    keys = ('team_id', 'competition_id', *(['side'] if venue else []))
    index = _eligible_history_index(history, group_by=keys, head_to_head=h2h, available_at='available_at')
    public = eligible_history_rows(history, group_by=keys, head_to_head=h2h, available_at='available_at')
    for row in range(len(history)):
        target = history.iloc[row]
        candidates = [i for i, item in enumerate(history.itertuples())
            if item.status == 'finished' and item.kickoff_at < target.kickoff_at
            and item.available_at <= target.kickoff_at and item.event_id != target.event_id
            and all(getattr(item, k) == target[k] for k in keys)
            and (not h2h or item.opponent_id == target.opponent_id)]
        candidates.sort(key=lambda i: history.kickoff_at.iloc[i])
        np.testing.assert_array_equal(public[row], candidates)
        for size in (1, 5, 20, 500):
            np.testing.assert_array_equal(index.window(row, size), candidates[-size:])
    assert sum(len(c) for c, _ in index.groups) <= len(history)
    public[-1][:] = -1
    assert not (index[-1] == -1).any()


@pytest.mark.parametrize('shuffle', [False, True])
def test_incremental_ema_exact_recurrence_and_missing_slots(shuffle):
    history = history_from_games([dict(day=i, release=i+(6 if i % 7 == 0 else 0),
        values=(np.nan if i%5 == 0 else (-1.)**i*10**(i%9), float(i))) for i in range(80)])
    if shuffle:
        history = history.sample(frac=1, random_state=8)
    expressions = {'ema': EMA(STAT, span=7, min_periods=3),
                   'mean': RollingMean(STAT, window=4, min_periods=1), 'lag': Lag(STAT, periods=2)}
    actual = evaluate_features(history, expressions, available_at='available_at')
    indices = eligible_history_rows(history, available_at='available_at')
    raw = history[OWN].to_numpy(dtype=float, na_value=np.nan)
    expected = np.full((len(history), 3), np.nan)
    for row, positions in enumerate(indices):
        values = raw[positions]; observed = values[np.isfinite(values)]
        if len(observed) >= 3:
            mean = observed[0]
            for value in observed[1:]: mean = .25*value+.75*mean
            expected[row, 0] = mean
        recent = values[-4:]; recent = recent[np.isfinite(recent)]
        if len(recent): expected[row, 1] = recent.mean()
        if len(values) >= 2: expected[row, 2] = values[-2]
    np.testing.assert_array_equal(actual.to_numpy(), expected)


def test_closed_interval_randomized_oracle():
    rng = np.random.default_rng(341)
    origin = pd.Timestamp('2024-01-01', tz='UTC')
    for _ in range(80):
        raw = sorted((int(x), int(x+y)) for x,y in rng.integers(0, 100, size=(20, 2)))
        merged = []
        for start,end in raw:
            if merged and start <= merged[-1][1]: merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else: merged.append((start,end))
        starts = rng.integers(-20, 200, size=150); ends = starts+rng.integers(0, 80, size=150)
        actual = _closed_overlap(origin+pd.to_timedelta(starts,unit='s'), origin+pd.to_timedelta(ends,unit='s'),
            [(origin+pd.Timedelta(seconds=a),origin+pd.Timedelta(seconds=b)) for a,b in merged])
        expected = [any(a <= end and b >= start for a,b in raw) for start,end in zip(starts,ends)]
        np.testing.assert_array_equal(actual, expected)
    assert not _closed_overlap([], [], []).any()


def test_bayesian_index_reused_only_inside_one_invocation(monkeypatch):
    from xdiyo_analytics.ratings import build_bayesian_ratings
    from xdiyo_analytics.ratings.snapshots import RatingRun
    from test_bayesian_training import synthetic_history
    history=synthetic_history(leagues=(10,20),seasons=1,matches=4)
    run=build_bayesian_ratings(history)
    calls=[];original=RatingRun._snapshot_groups
    def index(self,fields):calls.append(1);return original(self,fields)
    monkeypatch.setattr(RatingRun,'_snapshot_groups',index)
    before=run.features(history,fields=('attack_mean',))
    assert len(calls)==1
    run.snapshots['attack_mean'] += 1
    after=run.features(history,fields=('attack_mean',))
    assert len(calls)==2 and not before.equals(after)
    run.snapshots=pd.concat([run.snapshots,run.snapshots.iloc[:1]],ignore_index=True)
    with pytest.raises(ValueError,match='Duplicate snapshots'):run.features(history)
