# Execution policy reference

## Imports and option contract

Public execution imports are `ExecutionPolicy`, `DeviceAdapter`, and
`torch_devices` from `xdiyo_analytics.training`. Their new module is
`training.execution`. The optional Torch import happens only when its discovery
helper is called. Ordinary training-module imports do not load Torch/joblib.

ExecutionPolicy is immutable. n_jobs accepts Python int -1 or a positive int;
inner_threads and gpu_jobs accept positive Python integers. Booleans and floats
are rejected. device exactly matches cpu, auto, cuda or cuda followed by a colon
and nonnegative integer digits. The [guide option table](execution_policy.md)
records every default and its effective-worker rule.

DeviceAdapter wraps `estimator`, `devices`, `configure`. devices is a no-argument
callable returning available cpu/cuda names; configure receives the wrapped
adapter and resolved string. It must perform native placement before fitting.
The wrapper forwards fit/predict and optional fit_controlled, fit_resumable,
configure_threads and diagnostics. device_ retains the resolved name.
An unavailable explicit request or malformed reported name raises. Adapter
configure_device implementations must honor explicit CPU/exact CUDA requests and
return a valid resolved name. torch_devices reports CPU and available CUDA indexes
from the installed torch runtime; it does not install or configure that runtime.

## Where the policy applies

| Entry point | Policy behavior |
| --- | --- |
| TrainingRunner(..., execution=...) | One selected fold per job; stable supplied fold_ids order; no implicit refit |
| fit_predict(..., execution=...) | Configures one fit in the current process |
| ModelSelection.run / run_nested | Forwards to each candidate's folds; candidates and outer loops remain sequential |
| SelectionResult.evaluate | Omitted _UNSET inherits selection.execution; None explicitly disables policy |
| FootballExperiment.run | Forwards through fixed/search/evaluation/refit; includes policy in recovery identity |
| refit_model / RefitPolicy.run | One explicit current-process fit; n_jobs does not split one estimator |
| FittedModel.execution | Data-only resolved execution metadata retained by model save/load and numerical recovery |

`configure_model` is an internal pre-fit helper. It calls configure_device if
present, falls back to CPU for plain cpu/auto adapters, rejects unsupported
explicit CUDA, verifies the returned name, and optionally calls configure_threads.
`run_fold_jobs` is the internal scheduler: None/one worker runs serially; multiple
workers use loky, batch_size=1, pre_dispatch=workers and generator_unordered.
It buffers worker events, coordinates in the parent, closes the generator on
failure and reconstructs input order. Parent-only reporting/store publication
does not prohibit adapter-owned native checkpoint writes inside a worker.

EstimatorAdapter's thread hook recursively replaces exposed n_jobs parameters.
Native threadpoolctl limits surround fitting and restore caller state on exit,
including errors. Loky also receives inner_max_num_threads. Framework-specific
thread pools and device transfer remain adapter responsibilities.

## Retained execution evidence

fold.training_summary["execution"] and fitted_model.execution contain
requested_device, resolved device, worker_pid and inner_threads. They describe
the producing fit, so a reused result retains its original PID. They do not
promise that an old PID still exists. execution=None leaves no new fold execution
entry and an empty final-refit execution dictionary.

Execution settings participate in run/trial recovery identity. Native checkpoint
keys also contain resolved device and thread count. No complete model, estimator,
or GPU resume state is inferred from these fields. Custom fitted models returned
from process jobs must serialize; native state completeness is adapter-owned.

## Exact verified signatures

```text
ExecutionPolicy(n_jobs: int = 1, device: str = 'cpu', inner_threads: int = 1, gpu_jobs: int = 1) -> None
```

```text
DeviceAdapter(estimator: object, devices: object, configure: object) -> None
```

```text
torch_devices()
```

```text
TrainingRunner(model_factory: object, feature_columns: object = None, target_columns: object = None, control: object = None, validation: object = None, observer: object = None, execution: object = None) -> None
```

```text
fit_predict(dataset, fold, model_factory, *, fold_id=0, feature_columns=None, target_columns=None, validation=None, control=None, observer=None, execution=None)
```

```text
refit_model(dataset, model_factory, *, train_positions, feature_columns=None, target_columns=None, validation=None, control=None, observer=None, execution=None)
```

```text
FittedModel(model: object, train_positions: numpy.ndarray, fit_positions: numpy.ndarray, validation_positions: numpy.ndarray, feature_columns: tuple, target_columns: tuple, layout: str, identity_columns: tuple, match_columns: tuple, target_perspective: str, definitions: dict, execution: dict = <factory>) -> None
```

```text
configure_model(model, execution)
```

```text
run_fold_jobs(function, jobs, execution, observer=None)
```

```text
ExecutionPolicy.workers(self, count)
```

```text
DeviceAdapter.configure_device(self, requested)
```

```text
EstimatorAdapter.configure_device(self, device)
```

```text
EstimatorAdapter.configure_threads(self, threads)
```

```text
FootballExperiment.run(self, prepared=None, *, model=None, model_selection=None, selection_plan=None, development_positions=None, inner_plan_factory=None, pre_analysis=None, post_analysis=None, refit_policy=None, checkpoint_policy=None, name=None, name_fields=None, config=None, reuse=True, execution=None)
```

```text
ModelSelection.run(self, dataset, split_plan, *, development_positions, experiment=None, run_group=None, name='Model selection', resume=False, checkpoint_policy=None, _identity_candidates=None, recovery_namespace=None, execution=None)
```

```text
ModelSelection.run_nested(self, dataset, outer_plan, inner_plan_factory, *, experiment=None, name='Nested selection', resume=False, checkpoint_policy=None, recovery_namespace=None, execution=None)
```

```text
SelectionResult.evaluate(self, dataset, fold, *, validation=_UNSET, control=_UNSET, checkpoint_policy=None, checkpoint_directory=None, checkpoint_namespace=None, execution=_UNSET)
```

```text
RefitPolicy.run(self, dataset, candidate, *, execution=None)
```

```text
FittedModel.save(self, path, *, serializer=None)
```

```text
FittedModel.predict(self, dataset, *, positions=None)
```

_UNSET is the internal omitted-value sentinel. Every new option is optional.
For unchanged training, selection, persistence and checkpoint parameters, see
the preserved [experiment reference](football_experiment_reference.md) and
[model persistence reference](prediction_persistence_reference.md).

## Verified limits

All tests and notebook fits use synthetic inputs. Real CUDA evidence is limited
to the installed Torch runtime and a tiny explicit tensor-transfer adapter.
Simulated routing cases establish dispatch rules only. Multi-GPU/distributed
training, concurrent CUDA jobs, production forecasting and live Jupyter display
are not verified. Saved offline notebook/report interactions are checked.
No package/driver installation or source-export modification occurred.

[Guide](execution_policy.md), [notebook 16](../../notebooks/16_execution_policy_quickstart.ipynb),
[coverage](execution_policy_documentation_checklist.md), [evidence](execution_policy_check.json).

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
