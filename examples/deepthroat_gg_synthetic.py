"""Bounded synthetic checkpoint, never reads the football dataset or starts GAT.

PYTHONPATH=src python examples/deepthroat_gg_synthetic.py --output <new-folder>
The output folder must not already exist. Defaults exercise all n=9..100 and
k=3..350 candidates. This is an implementation benchmark, not a model result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from threading import Event, Thread
from time import perf_counter

import numpy as np
import pandas as pd
import psutil

from xdiyo_analytics.data import SeasonData
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.research.deepthroat_features import prepare_bundles
from xdiyo_analytics.research.deepthroat_gg import DeepthroatGGRecipe, fit_bundles, save_research


def synthetic_data():
    rng = np.random.default_rng(29)
    matches = []
    for round_ in range(48):
        teams = np.roll(np.arange(6), round_ % 5)
        for i in range(3):
            home, away = int(teams[i]), int(teams[5-i])
            if round_ % 2:
                home, away = away, home
            kickoff = pd.Timestamp('2020-01-01', tz='UTC') + pd.Timedelta(days=round_*3, hours=i)
            matches.append(dict(event_id=len(matches)+1, competition_id=17, tournament_id=50,
                season_id=1+round_//24, round=1+round_%24, home_id=home, away_id=away,
                kickoff_utc=kickoff.timestamp(), status='finished',
                home_score_current=int(rng.poisson(1.6)), away_score_current=int(rng.poisson(1.1))))
    history = build_team_history(SeasonData({'matches':pd.DataFrame(matches)}, {}))
    history['released_at'] = history.kickoff_at + pd.Timedelta(hours=3)
    roster = history[['competition_id','season_id','team_id']].drop_duplicates().reset_index(drop=True)
    roster['known_at'] = pd.Timestamp('2019-12-01', tz='UTC')
    return history, roster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error('Use a new output directory.')
    history, roster = synthetic_data()
    recipe = DeepthroatGGRecipe('2020-04-15', '2020-06-01', available_at='released_at')
    process = psutil.Process()
    peak = [process.memory_info().rss]
    baseline = peak[0]
    done = Event()
    def sample():
        while not done.wait(.01):
            peak[0] = max(peak[0], process.memory_info().rss)
    thread = Thread(target=sample, daemon=True); thread.start()
    start = perf_counter()
    try:
        bundles = prepare_bundles(history, roster, n_values=recipe.n_values, available_at=recipe.available_at)
        prepared = perf_counter()
        results = fit_bundles(bundles, development_end=recipe.development_end, holdout_end=recipe.holdout_end, k_values=recipe.k_values)
        fitted = perf_counter()
        manifest = save_research(results,recipe,history,roster,args.output)
        saved = perf_counter()
    finally:
        done.set(); thread.join()
    receipt = dict(kind='synthetic implementation checkpoint; not predictive evidence',
        command=[sys.executable,*sys.argv], python=sys.version, fixtures=len(history)//2,
        n_candidates=len(recipe.n_values), k_candidates=len(recipe.k_values),
        prepare_seconds=prepared-start, selection_fit_score_seconds=fitted-prepared,
        persistence_seconds=saved-fitted, total_seconds=saved-start,
        peak_rss_bytes=peak[0], baseline_rss_bytes=baseline, incremental_peak_rss_bytes=peak[0]-baseline,
        memory_method='process RSS sampled every 10 ms; includes native allocations; no GPU',
        competitions=manifest['competitions'], versions=manifest['versions'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library_source_sha256=manifest['library_source_sha256'])
    (Path(args.output)/'benchmark.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'library_source_sha256'},indent=2))


if __name__ == '__main__':
    main()
