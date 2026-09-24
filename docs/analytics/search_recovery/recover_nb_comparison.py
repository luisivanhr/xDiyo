"""Rescore the completed September 19 NB search; never fit or alter stored runs.

Run from the repository root. Outputs are a comparison CSV and winner configs.
The source search belongs to outer fold 0; these are inner CV scores only.
"""
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd

from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.experiments import ExperimentStore
from xdiyo_analytics.reporting import PerformanceReporter
from xdiyo_analytics.selection import MetricSelection, TrialResult
from xdiyo_analytics.selection.core import _metric_records


def main():
    root = Path(__file__).resolve().parents[3]
    folder = root / "experiments/corners_nb_first_grid_search--8767f003"
    manifest = json.loads((folder / "experiment.json").read_text(encoding="utf-8"))
    store = ExperimentStore(folder.parent, manifest["name"])
    group = "933d5702-69f3-478c-9278-7a8ed7cdb281"
    records = store.read_runs(role="trial", run_group=group)
    trials, rows = [], []
    for saved in records:
        if saved["status"] != "complete":
            continue
        loaded = store.load_run(saved["run_id"])
        report = PostTrainingAnalysis({"selection_metrics": PerformanceReporter(
            type="overall", partition="score", pooling="occurrences", metrics=["mse", "mae"])
        }).run(loaded["training"])
        record = deepcopy(saved)
        record["metrics"] = _metric_records(report)
        metrics = {item["metric"]: item["value"] for item in record["metrics"]}
        old_mse = next(item["value"] for item in saved["metrics"]
                       if item["metric"] == "mse" and item["type"] == "overall")
        if abs(metrics["mse"] - old_mse) > 1e-10:
            raise ValueError(f"Rescoring changed the original MSE for {saved['run_id']}")
        trials.append(TrialResult(saved["run_id"], None, record))
        rows.append({"run_id": saved["run_id"], **saved["config"]["model"]["params"], **metrics})
    if len(trials) != 90:
        raise ValueError(f"Expected 90 completed trials, found {len(trials)}")
    winners = {}
    for metric in ("mse", "mae"):
        choice = MetricSelection(metric).decide(trials)
        winner = next(trial for trial in trials if trial.trial_id == choice.winner_id)
        winners[metric] = {"run_id": winner.trial_id, "config": winner.record["config"],
                           "metrics": {item["metric"]: item["value"] for item in winner.record["metrics"]}}
    output = Path(__file__).parent
    pd.DataFrame(rows).sort_values(["mse", "mae"]).to_csv(output / "nb_90_candidates.csv", index=False)
    summary = {"source_run_group": group, "completed_candidates": len(trials), "outer_fold": 0,
               "scope": "Inner cross-validation evidence, not outer-test performance",
               "models_fitted_by_recovery": 0, "winners": winners}
    (output / "nb_recovered_selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
