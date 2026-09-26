"""Small local experiment/run store; JSON metadata and Parquet predictions."""

from dataclasses import is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from uuid import uuid4

import numpy as np
import pandas as pd


def _json(value, *, missing=False):
    """Canonical supported configuration values; never unstable object reprs."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _json(value.item(), missing=missing)
    if isinstance(value, np.ndarray):
        return _json(value.tolist(), missing=missing)
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        if missing:
            return None
        raise ValueError("Configuration numbers must be finite.")
    if isinstance(value, (pd.Timestamp, datetime)):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {"__type__": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": {key: _json(getattr(value, key), missing=missing) for key in value.__dataclass_fields__}}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Configuration dictionary keys must be strings.")
        return {key: _json(item, missing=missing) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item, missing=missing) for item in value]
    raise TypeError(f"Cannot record {type(value).__name__}; provide a descriptive configuration mapping.")


def configuration_hash(config):
    """Automatic stable digest of explicit JSON-compatible configuration values."""
    payload = json.dumps(_json(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_json(value, missing=True), stream, ensure_ascii=False, indent=2, allow_nan=False)


def _publish(stage, destination):
    """Publish by directory rename, retrying only Windows access/sharing errors.

    Six attempts allow brief locks to clear, with 1.55 seconds total backoff.
    Persistent errors retain the original exception and the unpublished stage;
    there is no copy fallback that could expose a partially published run.
    """
    for delay in (0.05, 0.1, 0.2, 0.4, 0.8, None):
        try:
            stage.rename(destination)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in {5, 32, 33} or delay is None:
                error.add_note(f"Run publication failed; unpublished artifacts remain at {stage}")
                raise
            time.sleep(delay)


class ExperimentStore:
    """Named local experiment with distinct run IDs and automatic config hashes.

    root is the parent experiments directory. Names resolve to a safe slug plus
    a short name hash; opening an existing name reuses its experiment UUID. No
    model pickle is loaded or saved. save_run requires the caller's configuration
    mapping (including model/feature/seed choices as applicable); arbitrary model
    internals cannot be inferred reliably. The digest is a configuration identity,
    not a proof that data, software or stochastic results are identical.
    """

    def __init__(self, root, name):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Experiments need a nonempty display name.")
        slug = re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-")[:70] or "experiment"
        self.path = Path(root) / f"{slug}--{hashlib.sha256(name.encode()).hexdigest()[:8]}"
        self.path.mkdir(parents=True, exist_ok=True)
        manifest = self.path / "experiment.json"
        if not manifest.exists():
            try:
                _write(manifest, dict(schema=1, experiment_id=str(uuid4()), name=name))
            except FileExistsError:
                pass
        self.manifest = json.loads(manifest.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != 1 or self.manifest.get("name") != name:
            raise ValueError("Experiment manifest does not match this name/schema.")
        (self.path / "runs").mkdir(exist_ok=True)
        (self.path / "run-groups").mkdir(exist_ok=True)

    def start_run(self, name, *, config=None):
        """Create an experiment-run group before storing optional search trials.

        Returns an automatically generated ID. Pass it as run_group to trial
        saves and to the single final result. This records identity only; it
        neither launches search nor decides which trial wins.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Run groups need a nonempty display name.")
        group = str(uuid4())
        record = dict(run_group=group, experiment_id=self.manifest["experiment_id"], name=name,
                      config=_json(config), created_at=datetime.now(timezone.utc).isoformat())
        _write(self.path / "run-groups" / f"{group}.json", record)
        return group

    def _start(self, name, config, status, *, role="final", run_group=None, selected_trial_id=None):
        if role not in {"final", "trial"}:
            raise ValueError("Run role must be final or trial.")
        if role == "trial" and run_group is None:
            raise ValueError("Search trials require a run_group created by start_run().")
        records = self.read_runs()
        if run_group is not None:
            # Resolve only an existing manifest ID; never interpolate an unchecked path.
            known = {path.stem for path in (self.path / "run-groups").glob("*.json")}
            if run_group not in known:
                raise ValueError("Unknown run_group; call start_run() first.")
            if role == "final" and any(item.get("run_group") == run_group and item.get("role", "final") == "final" for item in records):
                raise ValueError("This experiment run already has a final result.")
        if selected_trial_id is not None:
            trial = next((item for item in records if item["run_id"] == selected_trial_id), None)
            if (role != "final" or run_group is None or trial is None or trial.get("role") != "trial" or
                    trial.get("run_group") != run_group or trial["status"] != "complete"):
                raise ValueError("selected_trial_id must identify a successful trial in the same run_group.")
        run_id = str(uuid4())
        record = dict(schema=1, experiment_id=self.manifest["experiment_id"], run_id=run_id,
                      name=name, config=_json(config), config_hash=configuration_hash(config),
                      created_at=datetime.now(timezone.utc).isoformat(), status=status, metrics=[],
                      role=role, run_group=run_group or run_id, selected_trial_id=selected_trial_id)
        stage = self.path / "runs" / f".pending-{run_id}"
        stage.mkdir()
        return stage, record

    def save_run(self, training, report, *, name, config, save_predictions=True, save_html=False,
                 role="final", run_group=None, selected_trial_id=None, recovery=None, recovery_key=None,
                 display_report=None):
        """Persist computed numerical studies and optional predictions/HTML.

        A successful run is published only after its artifacts are written.
        Incomplete .pending directories are ignored by read_runs. Metrics retain
        study/fold/target/output/settings, evaluated-sample fingerprint and scope.
        Tables are stored as JSON table documents; predictions/y/metadata are
        Parquet with original indexes. No reporter or model is rerun to save.
        Publication retries brief Windows access/sharing failures up to six
        attempts (1.55 seconds backoff); persistent errors propagate and leave
        artifacts in the unpublished .pending directory.

        role='final' is the default: one displayed result per experiment run.
        Optional search trials use role='trial' and a start_run() group. Save the
        final held-out evaluation separately, linking selected_trial_id if useful;
        a link does not copy tuning metrics or establish evaluation independence.
        Writes to a given run group must be serialized by the caller.
        """
        stage, record = self._start(name, config, "complete", role=role, run_group=run_group,
                                    selected_trial_id=selected_trial_id)
        record.update(layout=training.layout, target_perspective=training.target_perspective,
                      identity_columns=list(training.identity_columns), match_columns=list(training.match_columns), artifacts={})
        tables = []
        for study in report.studies:
            if study.partition == "experiment":
                continue  # Do not recursively treat an earlier leaderboard as this run's metrics.
            for table_name, table in study.result.tables.items():
                tables.append(dict(study=study.name, fold_id=study.fold_id, name=table_name,
                                   table=json.loads(table.to_json(orient="table", date_format="iso"))))
            if "metrics" in study.result.tables:
                for metric in study.result.tables["metrics"].to_dict("records"):
                    record["metrics"].append(dict(metric, study=study.name, type=study.type,
                                                  partition=study.partition, fold_id=study.fold_id,
                                                  scope_label=study.scope_label, layout=study.layout))
        _write(stage / "tables.json", tables)
        record["artifacts"]["tables"] = "tables.json"
        diagnostics = []
        for fold in training.folds:
            fit = fold.fit_positions if fold.fit_positions is not None else fold.train_positions
            validation = fold.validation_positions if fold.validation_positions is not None else []
            diagnostics.append(dict(fold_id=fold.fold_id, train_positions=list(fold.train_positions),
                                    fit_positions=list(fit), validation_positions=list(validation),
                                    feature_columns=list(fold.feature_columns), target_columns=list(fold.target_columns),
                                    fold_metadata=fold.fold_metadata,
                                    summary=fold.training_summary,
                                    history=json.loads(fold.training_history.to_json(orient="table", date_format="iso"))))
        _write(stage / "training.json", diagnostics)
        record["artifacts"]["training"] = "training.json"
        if save_predictions:
            saved = []
            for fold in training.folds:
                folder = stage / f"fold-{fold.fold_id}"
                folder.mkdir()
                files = {}
                for i, (output, frame) in enumerate(fold.predictions.items()):
                    path = folder / f"output-{i}.parquet"
                    frame.to_parquet(path)
                    files[output] = str(path.relative_to(stage)).replace("\\", "/")
                fold.y_true.to_parquet(folder / "targets.parquet")
                fold.metadata.to_parquet(folder / "metadata.parquet")
                saved.append(dict(fold_id=fold.fold_id, outputs=files,
                                  targets=f"fold-{fold.fold_id}/targets.parquet",
                                  metadata=f"fold-{fold.fold_id}/metadata.parquet",
                                  train_positions=fold.train_positions.tolist(), score_positions=fold.score_positions.tolist(),
                                  test_positions=fold.test_positions.tolist(), feature_columns=list(fold.feature_columns),
                                  target_columns=list(fold.target_columns)))
            record["artifacts"]["folds"] = saved
        if save_html:
            (display_report if display_report is not None else report).to_html(stage / "report.html")
            record["artifacts"]["report"] = "report.html"
        if recovery is not None:
            from .recovery import dump_bundle
            dump_bundle(stage / "recovery.json", dict(training=training, report=report, extra=recovery))
            record["artifacts"]["recovery"] = "recovery.json"
            record["recovery_key"] = recovery_key
        _write(stage / "run.json", record)
        _publish(stage, self.path / "runs" / record["run_id"])
        return _json(record, missing=True)

    def open_run(self, name, recovery_key, *, reuse=True):
        """Reopen a matching execution group, or allocate a new one.

        Names alone never identify reusable computations. Calls/writes to the
        same experiment must be serialized, as with save_run.
        """
        if reuse:
            groups = [json.loads(path.read_text(encoding="utf-8"))
                      for path in (self.path / "run-groups").glob("*.json")]
            matches = [group for group in groups if group["experiment_id"] == self.manifest["experiment_id"]
                       and (group.get("config") or {}).get("recovery_key") == recovery_key]
            if matches:
                return max(matches, key=lambda group: group["created_at"])["run_group"]
        return self.start_run(name, config={"recovery_key": recovery_key})

    def find_completed(self, recovery_key, *, role="final", run_group=None):
        """Return the latest exact completed recovery record; ignore failed/partial runs."""
        matches = [record for record in self.read_runs(role=role, run_group=run_group)
                   if record["status"] == "complete" and record.get("recovery_key") == recovery_key
                   and "recovery" in record.get("artifacts", {})]
        return matches[-1] if matches else None

    def load_run(self, run_id):
        """Load retained results without fitting, predicting, or loading model code.

        Requires a recovery bundle saved by FootballExperiment or resumable
        ModelSelection. Legacy numerical records remain readable with read_runs.
        Fold models are None; predictions, histories and rendered report artifacts
        are retained. Native model checkpoints have a separate adapter contract.
        """
        from .recovery import load_bundle
        record = next((item for item in self.read_runs() if item["run_id"] == run_id), None)
        if record is None:
            raise KeyError(f"Unknown run {run_id!r}.")
        if record["status"] != "complete":
            raise ValueError("This run is not complete.")
        # Resolve paths from the manifest only within this published run.
        folder = (self.path / "runs" / record["run_id"]).resolve()
        if not folder.is_relative_to((self.path / "runs").resolve()):
            raise ValueError("Run escapes its experiment directory.")
        if "recovery" not in record.get("artifacts", {}):
            from .legacy import load_legacy
            return dict(record=record, **load_legacy(folder, record))
        path = (folder / record["artifacts"]["recovery"]).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("Recovery artifact escapes its run directory.")
        return dict(record=record, **load_bundle(path))

    def save_failure(self, *, name, config, error, role="final", run_group=None):
        """Record a failed candidate explicitly; no exception swallowing in runners."""
        stage, record = self._start(name, config, "failed", role=role, run_group=run_group)
        record["error"] = str(error)
        _write(stage / "run.json", record)
        _publish(stage, self.path / "runs" / record["run_id"])
        return record

    def read_runs(self, *, role=None, run_group=None):
        """Read published records, optionally filtered by role/run group.

        No filters returns all records, including trials. Legacy records without
        roles are final results; their run_id serves as their run group.
        Corrupt/incompatible records raise visibly.
        """
        if role not in {None, "final", "trial"}:
            raise ValueError("role must be final, trial or None.")
        records = []
        for path in sorted((self.path / "runs").glob("*/run.json")):
            if path.parent.name.startswith(".pending-"):
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if (record.get("schema") != 1 or record.get("experiment_id") != self.manifest["experiment_id"]
                    or record.get("run_id") != path.parent.name):
                raise ValueError(f"Incompatible experiment run record: {path}")
            if role is not None and record.get("role", "final") != role:
                continue
            if run_group is not None and record.get("run_group", record["run_id"]) != run_group:
                continue
            records.append(record)
        return sorted(records, key=lambda item: (item["created_at"], item["run_id"]))
