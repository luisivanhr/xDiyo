"""Read older numerical run artifacts without inventing unavailable state."""

from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..reporting import AnalysisReport, Artifact, StudyResult, StudyRun
from ..training import FoldResult, TrainingResult


class StoredHTMLReport(AnalysisReport):
    """A retained HTML view with independently accessible numerical study tables."""
    def __init__(self, studies, title, document):
        super().__init__(studies, title)
        self.document = document

    def to_html(self, path=None, *, renderers=None):
        if path is not None:
            Path(path).write_text(self.document, encoding="utf-8")
        return self.document


def load_legacy(folder, record):
    def path(relative):
        resolved = (folder / relative).resolve()
        if not resolved.is_relative_to(folder.resolve()):
            raise ValueError("Artifact escapes its run directory.")
        return resolved

    def frame(document):
        return pd.read_json(StringIO(json.dumps(document)), orient="table")

    artifacts = record.get("artifacts", {})
    diagnostics = {item["fold_id"]: item for item in
                   json.loads(path(artifacts["training"]).read_text(encoding="utf-8"))} if "training" in artifacts else {}
    folds = []
    for item in artifacts.get("folds", []):
        info = diagnostics.get(item["fold_id"], {})
        folds.append(FoldResult(
            item["fold_id"], None, np.asarray(item["train_positions"], dtype=int),
            np.asarray(item["test_positions"], dtype=int), np.asarray(item["score_positions"], dtype=int),
            tuple(item["feature_columns"]), tuple(item["target_columns"]),
            {name: pd.read_parquet(path(file)) for name, file in item["outputs"].items()},
            pd.read_parquet(path(item["targets"])), pd.read_parquet(path(item["metadata"])),
            info.get("fold_metadata", {}), fit_positions=np.asarray(info.get("fit_positions", item["train_positions"]), dtype=int),
            validation_positions=np.asarray(info.get("validation_positions", []), dtype=int),
            training_history=frame(info["history"]) if "history" in info else pd.DataFrame(),
            training_summary=info.get("summary", {})))
    # Old records did not always retain match_columns; do not guess them from IDs.
    training = TrainingResult(folds, record.get("layout", "unknown"), tuple(record.get("identity_columns", [])),
                              tuple(record.get("match_columns", [])), record.get("target_perspective", "unknown")) if folds else None
    grouped = {}
    if "tables" in artifacts:
        for item in json.loads(path(artifacts["tables"]).read_text(encoding="utf-8")):
            grouped.setdefault((item["study"], item["fold_id"]), {})[item["name"]] = frame(item["table"])
    studies = [StudyRun(name, "overall" if fold is None else "per_fold", "saved", fold,
                        record.get("layout", "unknown"), np.array([], dtype=int), 0,
                        StudyResult(name, [Artifact("table", table, key) for key, table in tables.items()], tables,
                                    ["Legacy artifacts: original report scopes and fitted models were not retained."]))
               for (name, fold), tables in grouped.items()]
    report = AnalysisReport(studies, record["name"])
    if "report" in artifacts:
        report = StoredHTMLReport(studies, record["name"], path(artifacts["report"]).read_text(encoding="utf-8"))
    return dict(training=training, report=report, extra={"kind": "legacy"})
