"""Opt-in native checkpoint handoff for adapters that can resume their own fit."""

from dataclasses import dataclass
import json
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class CheckpointPolicy:
    """Publish every nth checkpoint offered by a capable adapter.

    fit_resumable(context, *, control, checkpoint, save_checkpoint) owns native
    model/optimizer/preprocessor/RNG state and the training-loop cursor/history.
    checkpoint is an immutable directory or None. To publish a new state call
    save_checkpoint(writer), where writer(directory) writes a complete native
    checkpoint. The directory becomes visible only after writer returns.

    unsupported='skip' uses ordinary training for adapters without this capability;
    'raise' requires support. This does not make a plain fit() resumable. Existing
    checkpoints are trusted local adapter files, never generic model pickles loaded
    by this library. A namespace must have only one active writer.
    """
    every: int = 1
    unsupported: str = "skip"

    def __post_init__(self):
        if isinstance(self.every, bool) or not isinstance(self.every, int) or self.every < 1:
            raise ValueError("Checkpoint every must be a positive integer.")
        if self.unsupported not in {"skip", "raise"}:
            raise ValueError("unsupported must be skip or raise.")

    def wrap(self, factory, directory, namespace):
        return lambda: _CheckpointAdapter(factory(), Path(directory), namespace, self)


class _CheckpointAdapter:
    def __init__(self, adapter, directory, namespace, policy):
        self.estimator = adapter
        self.directory, self.namespace, self.policy = directory, namespace, policy

    def __getattr__(self, name):
        adapter = self.__dict__.get("estimator")
        if adapter is None:
            raise AttributeError(name)
        return getattr(adapter, name)

    def predict(self, context):
        return self.estimator.predict(context)

    def configure_device(self, requested):
        configure = getattr(self.estimator, "configure_device", None)
        if callable(configure):
            device = configure(requested)
        elif requested in {"cpu", "auto"}:
            device = "cpu"
        else:
            raise ValueError("This checkpoint adapter's model does not support CUDA selection.")
        self._execution_device = device
        return device

    def configure_threads(self, threads):
        self._execution_threads = threads
        configure = getattr(self.estimator, "configure_threads", None)
        if callable(configure):
            configure(threads)

    def fit(self, context):
        self.fit_controlled(context, None)

    def fit_controlled(self, context, control):
        from ..experiments.recovery import execution_key
        from ..experiments.store import _publish
        adapter = self.estimator
        resume = getattr(adapter, "fit_resumable", None)
        if not callable(resume):
            if self.policy.unsupported == "raise":
                raise TypeError("This adapter does not implement fit_resumable().")
            if control is not None or context.validation is not None:
                adapter.fit_controlled(context, control)
            else:
                adapter.fit(context)
            return
        key = execution_key(self.namespace, context, control, self.__dict__.get("_execution_device"),
                            self.__dict__.get("_execution_threads"))
        folder = self.directory / key
        folder.mkdir(parents=True, exist_ok=True)
        pointer = folder / "latest.json"
        checkpoint = None
        if pointer.exists():
            state = json.loads(pointer.read_text(encoding="utf-8"))
            checkpoint = (folder / state["directory"]).resolve()
            if not checkpoint.is_relative_to(folder.resolve()) or not checkpoint.is_dir():
                raise ValueError("Invalid native checkpoint pointer.")
        offered = 0

        def save_checkpoint(writer):
            nonlocal offered
            offered += 1
            if offered % self.policy.every:
                return
            identifier = str(uuid4())
            stage, final = folder / f".pending-{identifier}", folder / identifier
            stage.mkdir()
            writer(stage)
            _publish(stage, final)
            temporary = folder / f".pointer-{identifier}.json"
            temporary.write_text(json.dumps({"directory": identifier}), encoding="utf-8")
            temporary.replace(pointer)

        resume(context, control=control, checkpoint=checkpoint, save_checkpoint=save_checkpoint)
