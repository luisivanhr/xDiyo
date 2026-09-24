"""Explicit fitted-model artifacts, separate from data-only experiment recovery."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
from uuid import uuid4

import numpy as np

from .contracts import TrainingResult
from .refit import FittedModel


@dataclass(frozen=True)
class JoblibSerializer:
    """Persist an EstimatorAdapter, including its fitted preprocessing pipeline.

    Joblib/pickle artifacts can execute Python when loaded: load only trusted
    model artifacts you created or obtained from a trusted producer, in a compatible
    Python/library environment. No model artifact is loaded by experiment recovery.
    Custom adapter classes/callables must remain importable when using joblib.
    """
    format_id: str = "xdiyo.joblib.v1"

    def save(self, adapter, directory):
        import joblib
        joblib.dump(adapter, Path(directory) / "adapter.joblib")

    def load(self, directory):
        import joblib
        return joblib.load(Path(directory) / "adapter.joblib")


def _fitted_model(value, fold_id):
    if isinstance(value, FittedModel):
        if fold_id is not None:
            raise ValueError("fold_id applies to TrainingResult, not a final refitted model.")
        return value
    if not isinstance(value, TrainingResult):
        raise TypeError("save_model needs FittedModel or TrainingResult.")
    if fold_id is None:
        if len(value.folds) != 1:
            raise ValueError("Choose fold_id explicitly when saving one of several fitted fold models.")
        fold = value.folds[0]
    else:
        fold = next((fold for fold in value.folds if fold.fold_id == fold_id), None)
        if fold is None:
            raise KeyError(f"No fitted fold {fold_id!r}.")
    return FittedModel(fold.model, fold.train_positions.copy(),
                       np.asarray(fold.fit_positions if fold.fit_positions is not None else fold.train_positions).copy(),
                       np.asarray(fold.validation_positions if fold.validation_positions is not None else [], dtype=int).copy(),
                       tuple(fold.feature_columns), tuple(fold.target_columns), value.layout,
                       tuple(value.identity_columns), tuple(value.match_columns), value.target_perspective, value.definitions,
                       dict(fold.training_summary.get("execution", {})))


def save_model(model, path, *, fold_id=None, serializer=None):
    """Save a fitted prediction model and its feature/target/layout contract.

    model is FittedModel (e.g. result.refit) or TrainingResult; for several folds
    select fold_id explicitly. It never chooses a fold by its evaluation score.
    The default serializer supports EstimatorAdapter/sklearn pipelines and their
    TargetTransformAdapter wrapper, including the fitted target scaler. Other
    frameworks supply serializer.format_id, save(adapter, directory), and
    load(directory). Pass that same serializer explicitly when loading.

    path is a NEW directory. Files publish together after successful serialization;
    existing artifacts are never overwritten. This is a prediction artifact, not
    a promise of optimizer restart; mid-fit resume uses CheckpointPolicy instead.
    """
    from ..experiments.recovery import dump_bundle
    from ..experiments.store import _publish
    from .estimators import EstimatorAdapter
    from .targets import TargetTransformAdapter
    fitted = _fitted_model(model, fold_id)
    if fitted.model is None:
        raise ValueError("Recovered predictions contain no fitted model to save.")
    adapter = fitted.model
    # A checkpoint wrapper forwards to the underlying prediction adapter; its
    # run directory and resume callback are not part of a deployment artifact.
    from .checkpoints import _CheckpointAdapter
    if isinstance(adapter, _CheckpointAdapter):
        adapter = adapter.estimator
    if serializer is None:
        if not isinstance(adapter, (EstimatorAdapter, TargetTransformAdapter)):
            raise TypeError("Supply a native serializer for this custom model adapter.")
        serializer = JoblibSerializer()
    format_id = getattr(serializer, "format_id", None)
    if not isinstance(format_id, str) or not format_id:
        raise TypeError("Model serializers need a nonempty format_id.")
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"Model artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".pending-model-{uuid4()}"
    stage.mkdir()
    payload = stage / "model"
    payload.mkdir()
    serializer.save(adapter, payload)
    dump_bundle(stage / "metadata.json", fitted)
    files = {}
    for item in sorted(stage.rglob("*")):
        if item.is_symlink():
            raise ValueError("Model artifacts must contain local files, not symlinks.")
        if item.is_file():
            files[item.relative_to(stage).as_posix()] = hashlib.sha256(item.read_bytes()).hexdigest()
    if not any(name.startswith("model/") for name in files):
        raise ValueError("Serializer did not write model state.")
    manifest = {"schema": 1, "format_id": format_id, "python": platform.python_version(), "files": files}
    (stage / "model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _publish(stage, destination)
    return destination


def load_model(path, *, serializer=None):
    """Load a trusted fitted model, without fitting or accessing source datasets.

    Returns FittedModel with predict(dataset). Its pipeline, feature order, target
    names, class outputs and layout are retained. Joblib is the built-in sklearn
    format; it is executable deserialization, so only load trusted artifacts.
    Hashes detect accidental corruption; they do not authenticate an untrusted
    producer. Other formats require an explicit matching native serializer.
    """
    from ..experiments.recovery import load_bundle
    folder = Path(path).resolve()
    manifest = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 1:
        raise ValueError("Unsupported model artifact schema.")
    if serializer is None:
        if manifest.get("format_id") != JoblibSerializer.format_id:
            raise ValueError("Pass the native serializer used to save this model.")
        serializer = JoblibSerializer()
    if getattr(serializer, "format_id", None) != manifest.get("format_id"):
        raise ValueError("Serializer format does not match the saved model.")
    files = manifest.get("files", {})
    if "metadata.json" not in files or not any(name.startswith("model/") for name in files):
        raise ValueError("Model artifact is missing required files.")
    for relative, expected in files.items():
        item = (folder / relative).resolve()
        if not item.is_relative_to(folder) or not item.is_file():
            raise ValueError("Invalid model artifact file path.")
        if hashlib.sha256(item.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Model artifact file has changed: {relative}")
    fitted = load_bundle(folder / "metadata.json")
    if not isinstance(fitted, FittedModel):
        raise ValueError("Model artifact has no fitted prediction contract.")
    adapter = serializer.load(folder / "model")
    if not callable(getattr(adapter, "predict", None)):
        raise TypeError("Model serializer must restore an adapter with predict(context).")
    fitted.model = adapter
    return fitted
