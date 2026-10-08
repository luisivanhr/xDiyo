"""Fixed-parameter replay benchmark; never launches an optimizer or model study.

PYTHONPATH=src python examples/benchmark_bayesian_suffix_replay.py EVIDENCE.zip OUTPUT.json
Reads the supplied longer-calibration preflight archive without extracting or
executing its scripts. Input bytes and output timings are recorded for review.
"""
import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np
import pandas as pd

from xdiyo_analytics.ratings import BayesianModel, BayesianConfig, build_bayesian_ratings
from xdiyo_analytics.ratings.bayesian_core import BayesianParameters
from xdiyo_analytics.ratings.bayesian_calibration import CalibrationReplay, _Schedule, _States
from xdiyo_analytics.ratings.bayesian_training import _restrict_transitions
from xdiyo_analytics.ratings import bayesian_replay as replay


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('archive', type=Path)
    args.add_argument('output', type=Path)
    opt = args.parse_args()
    with zipfile.ZipFile(opt.archive) as archive:
        prefit = json.loads(archive.read('bayesian_calibration_reference/ORIGINAL_PREFIT_CALIBRATION_CONFIG.json'))
        raw = archive.read('long_calibration_native_8207e29/history_calibration.parquet')
        history = pd.read_parquet(BytesIO(raw))
        movements = pd.read_parquet(BytesIO(archive.read('long_calibration_native_8207e29/team_seasons_calibration.parquet')))
    movements, _ = _restrict_transitions(history, movements, None)
    model = BayesianModel(parameters=BayesianParameters(**prefit['initial_parameters']), config=BayesianConfig(**prefit['config']))
    start = perf_counter()
    plan = CalibrationReplay(history, config=model.config, cutoff=prefit['original_cutoff'], available_at='available_at', team_seasons=movements)
    prepared = perf_counter()-start
    print(json.dumps(dict(phase='plan_ready', seconds=prepared, matches=len(plan.events))), flush=True)
    work = [0]
    original = replay.update_matches
    def counted(*args, **kwargs):
        work[0] += len(args[2])
        return original(*args, **kwargs)
    replay.update_matches = counted
    try:
        start = perf_counter()
        evidence = plan.evaluate(model)
        objective_time = perf_counter()-start
        objective_work = work[0]
        print(json.dumps(dict(phase='objective_done', seconds=objective_time, assimilations=objective_work, loss=evidence.loss)), flush=True)
        work[0] = 0
        start = perf_counter()
        native = build_bayesian_ratings(history, model=model, available_at='available_at', team_seasons=movements)
        native_time = perf_counter()-start
        native_work = work[0]
        print(json.dumps(dict(phase='native_done', seconds=native_time, assimilations=native_work)), flush=True)
    finally:
        replay.update_matches = original
    p = native.predictions
    np.testing.assert_array_equal(evidence.home, p.expected_home_goals)
    np.testing.assert_array_equal(evidence.away, p.expected_away_goals)
    np.testing.assert_array_equal(evidence.probabilities, p[['p_home_win', 'p_draw', 'p_away_win']])
    np.testing.assert_array_equal(evidence.score_log, p.score_log_probability)
    loss = -np.log(p[['p_home_win', 'p_draw', 'p_away_win']].to_numpy()[np.arange(len(p)), plan.outcomes]).mean()
    assert loss == evidence.loss
    # Independent full chronological final-state replay, once, without resuming
    # internal suffix checkpoints or constructing reporting frames.
    canonical = _Schedule([dict(e, available_at=e['kickoff_at']) for e in plan.events], plan.schedule.entries)
    start = perf_counter()
    final = replay._replay_in_order(plan.history, model=model, _plan=canonical, _predict=False, _sink=_States(history=False))
    final_time = perf_counter()-start
    for key, items in final.teams.items():
        assert evidence.states.teams[key][-1][1:] == items[-1][1:]
    for key, items in final.leagues.items():
        assert evidence.states.leagues[key][-1][1:] == items[-1][1:]
    releases = np.sort(np.array([e['available_at'].value for e in plan.events], dtype=np.int64))
    boundaries = np.array([t.value for t, _ in plan.versions], dtype=np.int64)
    previous_work = int(np.searchsorted(releases, boundaries, side='right').sum())
    result = dict(optimizer_run=False, parameters=prefit['initial_parameters'], config=prefit['config'],
        history_sha256=hashlib.sha256(raw).hexdigest(), archive_sha256=hashlib.file_digest(opt.archive.open('rb'), 'sha256').hexdigest(),
        matches=len(plan.events), boundaries=len(plan.versions), plan_seconds=prepared,
        objective_seconds=objective_time, native_seconds=native_time, canonical_seconds=final_time,
        objective_assimilations=objective_work, native_assimilations=native_work,
        previous_global_prefix_assimilations=previous_work, work_reduction=previous_work/objective_work,
        loss=evidence.loss, exact_native_predictions=True, exact_canonical_final_states=True,
        note='Fixed original pre-fit parameters only; original full-prefix runtime not measured. Work ratio is not wall-time speedup.')
    opt.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
