"""Compare fixed-parameter calibration evidence with the frozen b0ecb580 oracle.

Run with PYTHONPATH=src. No fitting, dataset writes, or held-out evaluation.
The default is the small committed real-data fixture. --full-english-seasons
reads complete 2019/20 and 2020/21 Premier League/Championship publications.
"""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from xdiyo_analytics.ratings.bayesian import BayesianModel
from xdiyo_analytics.ratings.bayesian_calibration import CalibrationReplay
from xdiyo_analytics.ratings.bayesian_training import _restrict_transitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-english-seasons", action="store_true")
    parser.add_argument("--data-root", default="data/xDiyo_data")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fixtures = root / "tests/analytics/fixtures"
    spec = spec_from_file_location("xdiyo_analytics.ratings._reference_benchmark", fixtures / "bayesian_replay_b0ecb580.py")
    reference = module_from_spec(spec)
    spec.loader.exec_module(reference)
    fixture = json.loads((fixtures / "bayesian_real_boundary.json").read_text())
    if args.full_english_seasons:
        from xdiyo_analytics.data.loading import load_season
        from xdiyo_analytics.histories.team import build_team_history
        frames, movements = [], []
        for stem in fixture["sources"]:
            data = load_season(args.data_root, stem, tables=("matches", "team_seasons"))
            h = build_team_history(data)
            h["source_season"], h["source_league"] = stem[-5:], stem[:-6]
            frames.append(h.loc[h.status.eq("finished")].copy())
            movements.extend(data["team_seasons"].to_dict("records"))
        history = pd.concat(frames, ignore_index=True)
    else:
        history, movements = pd.DataFrame(fixture["history"]), fixture["team_seasons"]
    history["kickoff_at"] = pd.to_datetime(history.kickoff_at, utc=True)
    history["release"] = history.kickoff_at + pd.Timedelta(hours=2)
    movements, _ = _restrict_transitions(history, movements, None)
    model = BayesianModel()
    start = perf_counter()
    plan = CalibrationReplay(history, config=model.config, cutoff=history.release.max() + pd.Timedelta(days=1),
                             available_at="release", team_seasons=movements)
    prepare_seconds = perf_counter() - start
    start = perf_counter()
    actual = plan.evaluate(model)
    trial_seconds = perf_counter() - start
    start = perf_counter()
    expected = reference.replay(history, model=model, available_at="release", team_seasons=movements)
    original_seconds = perf_counter() - start
    np.testing.assert_array_equal(actual.home, expected.predictions.expected_home_goals)
    np.testing.assert_array_equal(actual.away, expected.predictions.expected_away_goals)
    np.testing.assert_array_equal(actual.probabilities, expected.predictions[["p_home_win", "p_draw", "p_away_win"]])
    np.testing.assert_array_equal(actual.score_log, expected.predictions.score_log_probability)
    for item in expected.checkpoint["teams"]:
        s = actual.states.teams[(item["competition_id"], item["team_id"])][-1][2]
        assert [s.attack.shape, s.attack.rate, s.defence.shape, s.defence.rate] == item["state"]
    summary = dict(matches=len(plan.events), sources=fixture["sources"],
                   availability="simulated kickoff plus two hours", parameters=model.parameters.__dict__,
                   refilter=plan.refilter, prepare_seconds=prepare_seconds, trial_seconds=trial_seconds,
                   original_seconds=original_seconds, trial_speedup=original_seconds / trial_seconds,
                   forecasts_and_posteriors_exact=True,
                   note="One fixed-parameter evaluation; not a full-fit speedup or an accuracy study.")
    payload = json.dumps(summary, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
