"""Independent fold jobs and adapter-owned device selection."""

from dataclasses import dataclass
import os
import re


@dataclass(frozen=True)
class ExecutionPolicy:
    """Configure local fold execution; candidates/outer search remain sequential.

    n_jobs=1 is sequential, -1 uses available CPUs, and positive values cap
    concurrent fold processes. inner_threads limits supported native thread pools
    and adapter-owned estimator threads per fit. device is cpu, auto, cuda or
    cuda:N. CUDA is resolved by the adapter, not assumed from an estimator name.

    cuda/auto requests are capped by gpu_jobs (default 1), even when auto eventually
    resolves to CPU, because capability discovery occurs inside fitting jobs.
    Set device='cpu' to use the full CPU worker budget. gpu_jobs>1 explicitly
    permits concurrent fits on the selected device; it does not distribute models
    across GPUs or implement distributed training. ExecutionPolicy(None) is not
    needed: execution=None on runners preserves existing serial/model settings.
    """
    n_jobs: int = 1
    device: str = "cpu"
    inner_threads: int = 1
    gpu_jobs: int = 1

    def __post_init__(self):
        if isinstance(self.n_jobs, bool) or not isinstance(self.n_jobs, int) or self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("n_jobs must be -1 or a positive integer.")
        if not isinstance(self.device, str) or not re.fullmatch(r"cpu|auto|cuda(?::[0-9]+)?", self.device):
            raise ValueError("device must be cpu, auto, cuda or cuda:N.")
        for name in ("inner_threads", "gpu_jobs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

    def workers(self, count):
        if self.n_jobs == -1:
            from joblib import cpu_count
            requested = cpu_count()
        else:
            requested = self.n_jobs
        if self.device != "cpu":
            requested = min(requested, self.gpu_jobs)
        return max(1, min(requested, count))


def configure_model(model, execution):
    """Apply an explicit policy before fit; unsupported explicit CUDA raises."""
    if execution is None:
        return {}
    if not isinstance(execution, ExecutionPolicy):
        raise TypeError("execution must be ExecutionPolicy or None.")
    configure = getattr(model, "configure_device", None)
    if callable(configure):
        device = configure(execution.device)
    elif execution.device in {"cpu", "auto"}:
        device = "cpu"
    else:
        raise ValueError("This adapter does not support explicit CUDA selection; provide configure_device().")
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device):
        raise ValueError("configure_device must return the resolved cpu/cuda device.")
    if execution.device == "cpu" and device != "cpu":
        raise ValueError("Adapter did not honor the requested CPU device.")
    if execution.device.startswith("cuda") and (not device.startswith("cuda") or
            (":" in execution.device and device != execution.device)):
        raise ValueError("Adapter did not honor the requested CUDA device.")
    threads = getattr(model, "configure_threads", None)
    if callable(threads):
        threads(execution.inner_threads)
    return {"requested_device": execution.device, "device": device, "worker_pid": os.getpid(),
            "inner_threads": execution.inner_threads}


@dataclass
class DeviceAdapter:
    """Attach native device behavior to an existing model adapter.

    devices() returns actually usable devices, e.g. ['cpu', 'cuda:0']; configure
    is called as configure(adapter, resolved_device) before fit. The callback owns
    moving/creating model state and configuring tensor conversion inside the
    wrapped adapter. This never pretends pandas feature construction runs on CUDA.
    Methods/results such as fit_controlled, fit_resumable and coefficient_table
    are forwarded. Native serializers may unwrap .estimator as appropriate.
    """
    estimator: object
    devices: object
    configure: object

    def __getattr__(self, name):
        # Safe during worker pickle reconstruction before fields are restored.
        adapter = self.__dict__.get("estimator")
        if adapter is None:
            raise AttributeError(name)
        return getattr(adapter, name)

    def configure_device(self, requested):
        available = tuple(self.devices())
        if any(not isinstance(item, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", item) for item in available):
            raise ValueError("devices() must return usable cpu/cuda device names.")
        gpu = next((item for item in available if item.startswith("cuda")), None)
        selected = (gpu or "cpu") if requested == "auto" else gpu if requested == "cuda" else requested
        if selected is None or selected not in available:
            raise ValueError(f"Requested device {requested!r} is unavailable for this adapter; available: {available}.")
        self.configure(self.estimator, selected)
        self.device_ = selected
        return selected

    def fit(self, context):
        return self.estimator.fit(context)

    def predict(self, context):
        return self.estimator.predict(context)


def torch_devices():
    """Optional device discovery for a PyTorch-based custom adapter; no install."""
    import torch
    return ["cpu", *[f"cuda:{i}" for i in range(torch.cuda.device_count())]] if torch.cuda.is_available() else ["cpu"]


def run_fold_jobs(function, jobs, execution, observer=None):
    """Collect fold jobs in input order; replay worker events in the parent.

    Processes return completed fits in completion order, but the public fold list
    retains input order. Parallel loss observers receive each job's retained events
    on completion, not live per iteration. Serial observers remain live. Only the
    parent runs downstream report/storage code. Worker arguments/results must be
    serializable by joblib/loky, including any custom fitted model state.
    """
    if execution is not None and not isinstance(execution, ExecutionPolicy):
        raise TypeError("execution must be ExecutionPolicy or None.")
    workers = 1 if execution is None else execution.workers(len(jobs))
    if workers == 1:
        return [function(job, observer) for job in jobs]
    from joblib import Parallel, delayed, parallel_config
    collect_events = observer is not None

    def compute(number, job):
        events = []
        result = function(job, events.append if collect_events else None)
        return number, result, events

    results = {}
    with parallel_config(backend="loky", inner_max_num_threads=execution.inner_threads):
        with Parallel(n_jobs=workers, return_as="generator_unordered", batch_size=1, pre_dispatch=workers) as parallel:
            completed = parallel(delayed(compute)(number, job) for number, job in enumerate(jobs))
            try:
                for number, result, events in completed:
                    results[number] = result
                    if observer is not None:
                        for event in events:
                            observer(event)
            finally:
                completed.close()
    return [results[number] for number in range(len(jobs))]
