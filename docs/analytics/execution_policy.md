# Run one fold per job and choose the model device

`ExecutionPolicy` is a small set of execution settings. It controls how many
independent folds can run at once, each fit's thread budget, and the device
requested from a capable model adapter. Every fold still fits its own fresh model.

This increment keeps candidate configurations and outer nested-CV folds sequential.
Within one candidate, inner folds can run in separate local processes. Final
results retain the requested fold order even when jobs finish in another order.
One job does not imply a new process for every fold: joblib can reuse workers.

The examples below use sixteen synthetic rows. They neither read collector exports
nor measure forecasting quality or speed. [Notebook 16](../../notebooks/16_execution_policy_quickstart.ipynb)
contains the two CPU examples. Earlier notebooks remain unchanged.

## 1. Set an explicit CPU budget and verify prediction parity

```python
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from notebooks.helpers.execution_demo import prepared_demo
from xdiyo_analytics.training import ExecutionPolicy, EstimatorAdapter, TrainingRunner

root = Path.cwd()
output = root / "experiment/execution_policy_demo/guide_revised"
data, splits = prepared_demo()

def model_factory():
    return EstimatorAdapter(make_pipeline(StandardScaler(), Ridge(alpha=.1)))

serial = TrainingRunner(model_factory).run(data, splits)
policy = ExecutionPolicy(n_jobs=2, device="cpu", inner_threads=1)
parallel = TrainingRunner(model_factory, execution=policy).run(data, splits)
assert [fold.fold_id for fold in parallel.folds] == [0, 1]
for expected, actual in zip(serial.folds, parallel.folds):
    pd.testing.assert_frame_equal(expected.predictions["predict"], actual.predictions["predict"])
    assert actual.training_summary["execution"]["worker_pid"] != os.getpid()
execution_table = pd.DataFrame([
    {"fold": fold.fold_id, **fold.training_summary["execution"]} for fold in parallel.folds
])
print(execution_table)
```

Omitting execution, or passing `execution=None`, keeps the previous serial path
and the estimator's own settings. An explicit policy, even with n_jobs=1, applies
device selection and thread limits. No policy changes feature construction,
train/test membership, validation scopes or missing-label handling.

| Option | Default | Meaning |
| --- | --- | --- |
| `n_jobs` | `1` | `1` runs serially; `-1` uses joblib's available CPU count; a positive integer caps workers; actual count cannot exceed the selected folds |
| `device` | `"cpu"` | `cpu`, `auto`, `cuda`, or `cuda:N`; the adapter resolves and configures the backend |
| `inner_threads` | `1` | Positive integer; limits supported native BLAS/OpenMP pools during fitting and calls the adapter's configure_threads hook |
| `gpu_jobs` | `1` | Positive integer cap for every auto/CUDA request, including auto that resolves to CPU |

Booleans, zero and invalid types are rejected for worker/thread counts. GPU jobs
are capped conservatively because device discovery occurs inside a fitting job.
Use device="cpu" for the full CPU worker budget. A larger gpu_jobs permits
concurrent fits on the selected GPU; it does not distribute a model across devices.
With one effective job, fitting happens in the caller process.

EstimatorAdapter declares CPU support. It updates every exposed sklearn n_jobs
parameter, including nested pipeline parameters, when an explicit policy is used.
A CPU-only Lasso/Ridge does not become a CUDA model. Native libraries outside
threadpoolctl's support need an adapter-owned configure_threads implementation.
The explicit threadpoolctl context surrounds fit; arbitrary prediction methods
are outside that context, while worker environment limits still apply in loky.

## 2. Apply the same policy to experiment fitting and recovery

```python
from xdiyo_analytics.experiments import FootballExperiment, PreparedExperiment, RefitPolicy
from xdiyo_analytics.selection import Candidate
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.reporting import PerformanceReporter, MatchResultReporter

experiment = FootballExperiment("Synthetic fold execution", output_dir=output)
prepared = PreparedExperiment(data, splits, config={"synthetic": True})
candidate = Candidate("Ridge", model_factory, config={"alpha": .1})
reports = PostTrainingAnalysis({
    "Errors": PerformanceReporter(type="overall", partition="score", metrics=["mse", "mae"]),
    "Predictions": MatchResultReporter(type="per_fold", partition="test", target="target", tolerance=1.),
})
result = experiment.run(prepared, model=candidate, execution=policy, post_analysis=reports,
    refit_policy=RefitPolicy(train_positions=np.arange(len(data.X))))
cached = experiment.run(prepared, model=candidate, execution=policy, post_analysis=reports,
    refit_policy=RefitPolicy(train_positions=np.arange(len(data.X))))
assert cached.reused and cached.refit.model is None
assert result.refit.execution["worker_pid"] == os.getpid()
assert all(fold.model is None for fold in cached.training.folds)
result.to_html(output / "report.html")
print(result.record["run_id"], "reused:", cached.reused)
```

The additional refit is explicitly requested on all sixteen synthetic rows and
stays in the caller process. It does not alter the held-out evaluation report.
The requested policy enters experiment/trial recovery identity. Checkpoint keys
also include resolved device and thread budget. Reuse loads retained numbers and
reports with model=None; executable model save/load remains a separate API.

Selection receives the policy through `ModelSelection.run(..., execution=policy)`
or `run_nested(..., execution=policy)`. `SelectionResult.evaluate` inherits it;
an explicit replacement overrides it and execution=None restores the legacy path.
Candidate feature selectors remain parent-side and train-scoped, excluding
monitoring validation. Fits and predictions use the worker processes. Candidate
and outer-fold loops wait for their current fold jobs, avoiding nested pools.

Reports, trial publication and final result coordination run in the parent.
For native resumable adapters, checkpoint writers run with the fitting job in its
distinct checkpoint directory. Resume still requires complete optimizer, RNG,
cursor, preprocessing and history state; process parallelism does not provide it.

## 3. Let a capable adapter own CUDA placement

DeviceAdapter takes an existing adapter, a `devices()` callback that lists actually
usable device names, and `configure(adapter, resolved_device)` called before fit.
`auto` chooses the first reported GPU, otherwise CPU. `cuda` chooses the first
reported GPU; `cuda:N` requires that exact available device. Unsupported explicit
CUDA raises. The callback owns native configuration; the wrapper cannot establish
capability just from a model name or a GPU-looking string.

This optional example uses the [small Torch adapter](../../notebooks/helpers/execution_torch.py)
that was tested on this machine. Its `to` method moves model parameters with
`model.to(device)`. Its fit and predict methods also create input/target tensors
on that device and return predictions to a pandas DataFrame on the CPU. Moving
only model parameters would be incomplete. The example requires an existing
compatible PyTorch installation; no package or driver installation is performed.

```python
from notebooks.helpers.execution_torch import TorchLinear, move_torch_model
from xdiyo_analytics.training import DeviceAdapter, torch_devices

def torch_factory():
    return DeviceAdapter(TorchLinear(), devices=torch_devices, configure=move_torch_model)

torch_result = TrainingRunner(torch_factory,
    execution=ExecutionPolicy(n_jobs=2, device="auto", gpu_jobs=1)).run(data, splits)
print([fold.training_summary["execution"] for fold in torch_result.folds])
```

For a GPU-capable native estimator with a verified device parameter, a callback
can use its own `adapter.estimator.set_params(device=resolved_device)` API.
Parameter names and valid values are framework-specific; do not apply this
blindly to a CPU-only sklearn estimator. A custom Torch adapter can similarly
move its model and own tensor conversion. These hooks do not move pandas feature
construction to the GPU or implement distributed training.

## Observers, serialization and resource limits

Serial observers receive training events during fitting. Parallel workers retain
their events and the parent replays them when that fold completes; this is not
live per-step progress across workers. Fold event order is preserved, while
completion order between folds can vary. Unpicklable notebook display observers
remain in the parent. Very long histories therefore also consume worker memory.

Custom factories, job inputs and returned fitted objects must be serializable
when multiple workers are used. Fresh models and learned preprocessing belong
inside the factory for every fold. Do not rely on worker mutations reaching
parent lists or notebook variables. Single-job execution avoids transport of the
fitted object; it does not make save_model serialization requirements disappear.
Process startup, data transfer and memory duplication can outweigh parallelism
on small fits. No speedup is claimed by these correctness checks.

An adapter/observer failure propagates and closes unfinished job delivery. Worker
cancellation and a subsequent successful pool are covered by synthetic tests.
External side effects inside arbitrary adapters cannot be rolled back. Earlier
completed search trials and published native checkpoints retain their existing
recovery behavior.

The process implementation uses joblib's loky backend and unordered completion,
then restores public fold order ([joblib documentation](https://joblib.readthedocs.io/en/latest/user_guide/parallel.html)).
Native thread limits use [threadpoolctl](https://github.com/joblib/threadpoolctl).
Those dependencies are declared in the training extra; verification used existing
installed versions. See the [reference](execution_policy_reference.md),
[coverage](execution_policy_documentation_checklist.md), and
[frozen verification record](execution_policy_check.json).

## Known notebook shutdown limitation

On the verified Windows environment (Python 3.14.0, joblib 1.5.2), notebook cells
complete correctly, but kernel shutdown emits loky resource_tracker KeyError
tracebacks for temporary joblib_memmapping_folder paths. A one-cell plain-joblib
notebook without xDiyo imports reproduces this. max_nbytes=None also reproduces
it, so disabling automatic memmapping did not resolve the symptom.

The captured notebook run has four cleanup tracebacks; each plain-joblib control
has two. All return exit status 0 with correct numerical results. This evidence
does not establish clean notebook shutdown. Standalone guide, test and isolated
wheel computations pass. The library and installed packages were left unchanged;
no workaround or dependency upgrade is claimed. See the runtime and cleanup
control logs linked from the verification record.
