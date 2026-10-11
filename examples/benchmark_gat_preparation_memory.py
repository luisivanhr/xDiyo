"""Fresh-process native preparation RSS benchmark; no experiment/model fitting.

Run from a Git checkout with PYTHONPATH=src. Both modes use actual native
snapshots. The reference adapter is read verbatim from the requested commit.
RSS includes imports, event store, raw groups, model/optimizer and tensors.
Requires psutil for process high-water counters (Windows/Linux).
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
import types


def worker(args):
    import numpy as np
    import pandas as pd
    import psutil
    import torch
    from xdiyo_analytics.graphs import TemporalEventStore, TemporalGraphConfig
    from xdiyo_analytics.graphs.adapter import _Backend
    from xdiyo_analytics.training.contracts import FitContext

    if args.mode == "reference":
        source = subprocess.check_output([
            "git", "show", f"{args.reference}:src/xdiyo_analytics/graphs/adapter.py"
        ], text=True)
        module = types.ModuleType("xdiyo_analytics.graphs._memory_reference")
        module.__package__ = "xdiyo_analytics.graphs"
        exec(compile(source, "reference_adapter.py", "exec"), module.__dict__)
        _Backend = module._Backend
    torch.set_num_threads(1)
    rows = []
    for i in range(args.history_matches):
        home, away = 1 + 2 * (i % 10), 2 + 2 * (i % 10)
        kickoff = pd.Timestamp("2018-01-01", tz="UTC") + pd.Timedelta(hours=i)
        for team, opponent, side, gf, ga in [(home, away, "home", i % 4, i % 3), (away, home, "away", i % 3, i % 4)]:
            rows.append(dict(event_id=i + 1, kickoff_at=kickoff, team_id=team,
                             opponent_id=opponent, side=side, goals_for=gf, goals_against=ga,
                             competition_id=10, season_id=900, round=i + 1, status="finished"))
    history = pd.DataFrame(rows)
    # Eight paired statistic channels, including missing and observed zeros.
    stats = []
    for n in range(4):
        col = f"team::ALL::Synthetic::stat{n}::value"
        other = col.replace("team::", "opponent::")
        history[col] = np.where(history.side.eq("home"), n, np.nan)
        history[other] = np.where(history.side.eq("away"), n, np.nan)
        stats.extend((col, other))
    start = history.kickoff_at.max().normalize() + pd.Timedelta(days=3)
    schedule = pd.DataFrame([dict(event_id=args.history_matches + i + 1,
        kickoff_at=start + pd.Timedelta(days=i), home_id=1 + 2 * (i % 10),
        away_id=2 + 2 * (i % 10), competition_id=10, season_id=900, round=args.history_matches + i + 1)
        for i in range(args.issuances + args.validation_issuances)])
    members = pd.DataFrame([dict(team_id=t, competition_id=10, season_id=900,
        entry_at=pd.Timestamp("2017-07-01", tz="UTC"), movement="retained") for t in range(1, 21)])
    historical_schedule = history[history.side.eq("home")].rename(columns={"team_id": "home_id", "opponent_id": "away_id"})
    store = TemporalEventStore(history, members, schedule=pd.concat([historical_schedule, schedule], ignore_index=True))
    cfg = TemporalGraphConfig("UTC", statistics=tuple(stats), embedding_dim=4, hidden_width=8)

    def context(meta):
        return FitContext(X=pd.DataFrame(index=meta.index),
            y=pd.DataFrame({"result": [1, 0, -1] * (len(meta) // 3) + [1] * (len(meta) % 3)}, index=meta.index),
            metadata=meta, layout="match", match_columns=("competition_id", "season_id", "event_id"), fold_id=0)

    train = context(schedule.iloc[:args.issuances])
    if args.validation_issuances:
        train.validation = context(schedule.iloc[args.issuances:])
    backend = _Backend(cfg, store, 72, "cpu")
    process = psutil.Process()
    baseline = process.memory_info().rss
    started = time.perf_counter()
    backend.initialize(train)
    elapsed = time.perf_counter() - started
    memory = process.memory_info()
    if sys.platform == "win32":
        peak = memory.peak_wset
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    digest, edges, tensor_bytes = sha256(), 0, 0
    for group in backend.train_groups + (backend.validation_groups or []):
        positions, data, home, away, fixture = group
        digest.update(positions.tobytes())
        edges += len(data["receiver"])
        for tensor in (*data.values(), home, away, fixture):
            digest.update(tensor.numpy().tobytes())
            tensor_bytes += tensor.numel() * tensor.element_size()
    digest.update(json.dumps(backend.preprocessor.state(), sort_keys=True).encode())
    print(json.dumps(dict(mode=args.mode, reference=args.reference, issuances=args.issuances,
        validation_issuances=args.validation_issuances, history_matches=args.history_matches,
        directed_edges=edges, tensor_bytes=tensor_bytes, baseline_rss_bytes=baseline,
        peak_rss_bytes=peak, final_rss_bytes=memory.rss, seconds=elapsed,
        prepared_sha256=digest.hexdigest(), model_version=backend.model_version,
        python=sys.version, pandas=pd.__version__, torch=torch.__version__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reference", "native"))
    parser.add_argument("--reference", default="aca3e31")
    parser.add_argument("--history-matches", type=int, default=4000)
    parser.add_argument("--issuances", type=int, default=240)
    parser.add_argument("--validation-issuances", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode:
        worker(args)
        return
    results = []
    for mode in ("reference", "native"):
        command = [sys.executable, __file__, "--mode", mode, "--reference", args.reference,
            "--history-matches", str(args.history_matches), "--issuances", str(args.issuances),
            "--validation-issuances", str(args.validation_issuances)]
        result = json.loads(subprocess.check_output(command, text=True))
        results.append(result)
        print(json.dumps(result), flush=True)
    assert results[0]["prepared_sha256"] == results[1]["prepared_sha256"]
    assert results[0]["model_version"] == results[1]["model_version"]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
