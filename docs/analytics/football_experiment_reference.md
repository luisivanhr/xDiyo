# FootballExperiment API reference

This reference describes the preserved FootballExperiment revision. The separate
prediction-fixture and explicit model-persistence increment is outside this
verification batch. See the [guide](football_experiment.md) for executable examples.

## Imports and ownership

| Namespace | Public names used here | Responsibility |
| --- | --- | --- |
| `xdiyo_analytics.experiments` | `FootballExperiment`, `PreparedExperiment`, `ExperimentResult`, `RefitPolicy`, `SelectionSummary`, `ExperimentStore` | Orchestration, numerical persistence and reopening |
| `xdiyo_analytics.training` | `CheckpointPolicy`, `EstimatorAdapter`, `refit_model`, `FittedModel` | Fresh fitting, prediction and native resume handoff |
| `xdiyo_analytics.selection` | `Candidate`, `ModelSelection`, `GridCandidates` | Executable candidate configuration and finite search |
| `xdiyo_analytics.analysis` | `PreTrainingAnalysis`, `PostTrainingAnalysis` | Compute scoped studies |
| `xdiyo_analytics.reporting` | `ExperimentLeaderboardReporter` | Read and compare saved run records |

The package root lazily exposes `FootballExperiment`; the other experiment
symbols use the `experiments` namespace. Importing that namespace
namespace does not construct an estimator or fit data. Existing preparation
functions remain in data, histories, ratings, features, labels, datasets and splits.

## Preparation, results and refit

`PreparedExperiment.dataset` and `.split_plan` must be a ModelDataset and a
nonempty SplitPlan for its exact row count. `outputs={}` retains inspectable
data-only intermediates; `config={}` records preparation choices and relevant
external revisions. Frames and folds should remain aligned and unchanged while
a call runs. Preparation does not itself train a model.

`ExperimentResult` stores `prepared`, `training`, `pre_report`, `post_report`,
optional `selection` and `refit`, saved `record`/`path`, and `reused`. `dataset`
and `splits` expose preparation; both are None for legacy records. `report`
constructs the combined pre/selection/post view. `show`, `to_html`, `to_notebook`
and `_repr_html_` delegate presentation only. Keyword arguments follow
AnalysisReport's existing viewer API. Loaded selection is `SelectionSummary`
containing `comparison` and per-holdout/per-outer-fold `winners` data, without
executable factories. Live selection is the corresponding SelectionResult or
NestedSelectionResult.

`RefitPolicy.train_positions` explicitly declares deployment training rows.
`candidate=None` uses a fixed candidate or single holdout winner; nested selection
requires an explicit Candidate. `validation=None` and `control=None` deliberately
start a new refit decision. A candidate's consumed feature selector is recomputed
on actual fitting rows before `refit_model`; preprocessing is fresh. The candidate
observer and selected target columns carry through. Refitting does not change
stored evaluation predictions. Restored FittedModel metadata has no live model
and `predict` raises until a separate explicit restoration/refit supplies one.

## FootballExperiment options

| Argument | Meaning |
| --- | --- |
| `name`, `output_dir="experiments"` | Named experiment in a saved output folder |
| `prepare=None` | Callable returning PreparedExperiment; required only when run has no explicit prepared argument |
| constructor `config=None` | Experiment configuration included in identity |
| run `prepared=None` | Explicit preparation, otherwise call prepare() |
| `model=None`, `model_selection=None` | Exactly one Candidate or ModelSelection |
| `selection_plan=None` | Inner holdout-search splits on original positions; outer plan must contain one fold |
| `development_positions=None` | Holdout selection population; defaults to outer train; excludes outer test |
| `inner_plan_factory=None` | Nested selection; receives local outer-training dataset, returns local SplitPlan; incompatible with holdout selection arguments |
| `pre_analysis=None`, `post_analysis=None` | Empty analyses by default; descriptive pre-analysis is separate from candidate feature selection |
| `refit_policy=None` | Optional explicit additional deployment fit |
| `checkpoint_policy=None` | Optional capable-adapter native resume; disabled by default |
| run `name=None` | Explicit final display name; otherwise generated |
| `name_fields=None` | Config field paths to append to candidate name; defaults to first six scalar/None entries |
| run `config=None` | Run configuration included alongside experiment/prepared config |
| `reuse=True` | Restore exact complete final or reuse completed search trials; false allocates new execution group |

Generated names start with candidate.name and append `key=value` parts separated
by a middle dot. Top-level `grid_parameters` overlay candidate config for naming;
explicit paths such as `model.alpha` follow nested dictionaries. An absent path
raises KeyError. A nested result defaults to `Tuned per fold`. Configuration
details in leaderboards are escaped HTML, so names and values remain text.

`load(run_id)` finds an exact completed record and never calls preparation,
fitting or prediction. It rejects failed/unknown records and modern internal
trial bundles that are not FootballExperiment final results. `leaderboard(weights,
**kwargs)` forwards selectors, scaling, reference_scales, directions,
include_trials and run_group to ExperimentLeaderboardReporter; see the existing
post-training reference for metric comparison groups and normalization.

## Completed-run and completed-trial recovery

The identity includes X/y/metadata and order, definitions, folds, configuration,
candidate factory/defaults/closures/globals, reports, refit/checkpoint policies,
name/name_fields, local analytics source, Python/NumPy/pandas and identified
framework versions. It does not infer hidden service/file changes. Custom opaque
objects implement cache_key() with stable configuration data. Observer state is
excluded. Unchanged notebook code locations are normalized; changed constants
remain relevant. Prepared outputs are stored but are not identity inputs.

`ExperimentStore.open_run(name, recovery_key, reuse=True)` reopens the newest
matching execution group or creates one. `find_completed` uses recovery_key,
role (default final), and optional run_group; only complete records with recovery
artifacts qualify. `load_run` returns record/training/report/extra; native models
are not deserialized. `save_run` adds optional recovery, recovery_key and
display_report arguments. The combined display is saved as HTML while numerical
metrics come from the supplied final report. Existing save_predictions/save_html,
role/run_group/selected_trial_id behavior remains in force. One final publication
per group; serialize calls. Incomplete .pending directories are not published.

`ModelSelection.run(..., resume=False, checkpoint_policy=None,
recovery_namespace=None)` requires an ExperimentStore when recovery/checkpoints
are enabled. Completed trials load their saved evidence/model=None. Failed or
incomplete trials run again. `run_nested` supports the same options and materializes
each callable candidate source once per outer fold, using that same configuration
list for both recovery identity and guarded fitting. `_identity_candidates` is
an internal bridge, not an application option. `SelectionResult.evaluate` accepts
checkpoint_policy, checkpoint_directory and checkpoint_namespace for its fresh
outer fit. Recovery never makes internal selection scores independent evaluation.
Its `_UNSET` defaults preserve the winning candidate's validation/control;
explicit `None` clears either option for the outer fit.

## Native checkpoint protocol

`CheckpointPolicy(every=1, unsupported="skip")` validates a positive integer and
skip/raise behavior. `wrap(factory, directory, namespace)` builds a fresh proxy;
the ordinary runner calls its fit or fit_controlled. The capable adapter implements
`fit_resumable(context, *, control, checkpoint, save_checkpoint)`. It receives a
FitContext containing fitting rows and a separate optional validation context,
the requested control, and an immutable prior directory or None. It owns all
fitted preprocessing, model/optimizer/scheduler/mixed-precision scaler, RNG,
cursor/restart and history state relevant to equivalent continuation. A writer
callback must write a complete checkpoint before returning.

`every` counts checkpoint offers per invocation. Publication writes an isolated
pending directory, publishes it, then atomically replaces latest.json. A failed
writer does not replace the previous pointer. Missing/escaping pointers raise.
There is no shared-worker locking, cleanup daemon or model-independent checkpoint
format. A plain EstimatorAdapter fit runs normally under skip; raise rejects it.
The wrapper forwards prediction and model diagnostics to its adapter.

## Data-only bundle and legacy helpers

| Helper/module | Purpose and limits |
| --- | --- |
| `recovery.pack`, `unpack` | Tagged frames/Series/indexes/arrays, timestamps, nullable values, tuples/dicts, Plotly and library dataclasses; omit FoldResult/FittedModel.model; reject arbitrary objects |
| `_pack_dtype`, `_unpack_dtype` | StringDtype storage/missing sentinel and categorical metadata; accept previous dtype strings |
| `dump_bundle`, `load_bundle` | Exclusive schema-1 JSON writer/reader; no executable model pickle |
| `signature`, nested `semantic_code` | Executable configuration identity with normalized source locations, explicit custom keys and bounded recursion |
| `execution_key` | Combines input signatures, runtime versions and every local analytics Python source file |
| `_display_name`, `_summary` | Generated final names and serializable selection evidence |
| `FootballExperiment._experiment_reports` | Refresh experiment reporters after save and on reuse |
| `_CheckpointAdapter` | Fresh fit proxy, contained latest pointer, offered-checkpoint publication and passthrough prediction |
| `legacy.load_legacy`, internal `path`/`frame` | Read actual saved tables/diagnostics/parquet with path containment; absent scope is not guessed |
| `StoredHTMLReport.to_html` | Preserve exact saved legacy HTML when it exists |

MultiIndex levels/codes/names/sortorder and nullable class dtypes are retained by
new bundles. Previous tuple tags remain readable, but omitted old metadata cannot
be recovered. A legacy result has prepared=None, possible training=None, saved
tables with partition='saved'/empty row positions, and original HTML only when
that artifact was actually saved. Original unknown match_columns stay empty.

## Exact signatures of the verified revision

```text
FootballExperiment(name, *, output_dir='experiments', prepare=None, config=None)
```

```text
PreparedExperiment(dataset: xdiyo_analytics.datasets.assembly.ModelDataset, split_plan: xdiyo_analytics.splits.core.SplitPlan, outputs: dict = <factory>, config: dict = <factory>) -> None
```

```text
ExperimentResult(prepared: xdiyo_analytics.experiments.football.PreparedExperiment, training: object, pre_report: xdiyo_analytics.reporting.contracts.AnalysisReport, post_report: xdiyo_analytics.reporting.contracts.AnalysisReport, selection: object = None, refit: object = None, record: dict = <factory>, path: object = None, reused: bool = False) -> None
```

```text
RefitPolicy(train_positions: object, candidate: object = None, validation: object = None, control: object = None) -> None
```

```text
SelectionSummary(comparison: pandas.core.frame.DataFrame, winners: dict) -> None
```

```text
CheckpointPolicy(every: int = 1, unsupported: str = 'skip') -> None
```

```text
FootballExperiment.prepare(self)
```

```text
FootballExperiment.run(self, prepared=None, *, model=None, model_selection=None, selection_plan=None, development_positions=None, inner_plan_factory=None, pre_analysis=None, post_analysis=None, refit_policy=None, checkpoint_policy=None, name=None, name_fields=None, config=None, reuse=True)
```

```text
FootballExperiment.load(self, run_id)
```

```text
FootballExperiment.leaderboard(self, weights, **kwargs)
```

```text
ExperimentResult.show(self, **kwargs)
```

```text
ExperimentResult.to_html(self, path=None, **kwargs)
```

```text
ExperimentResult.to_notebook(self, **kwargs)
```

```text
ExperimentResult._repr_html_(self)
```

```text
RefitPolicy.run(self, dataset, candidate)
```

```text
CheckpointPolicy.wrap(self, factory, directory, namespace)
```

```text
ExperimentStore.open_run(self, name, recovery_key, *, reuse=True)
```

```text
ExperimentStore.find_completed(self, recovery_key, *, role='final', run_group=None)
```

```text
ExperimentStore.load_run(self, run_id)
```

```text
ExperimentStore.save_run(self, training, report, *, name, config, save_predictions=True, save_html=False, role='final', run_group=None, selected_trial_id=None, recovery=None, recovery_key=None, display_report=None)
```

```text
ModelSelection.run(self, dataset, split_plan, *, development_positions, experiment=None, run_group=None, name='Model selection', resume=False, checkpoint_policy=None, _identity_candidates=None, recovery_namespace=None)
```

```text
ModelSelection.run_nested(self, dataset, outer_plan, inner_plan_factory, *, experiment=None, name='Nested selection', resume=False, checkpoint_policy=None, recovery_namespace=None)
```

```text
SelectionResult.evaluate(self, dataset, fold, *, validation=_UNSET, control=_UNSET, checkpoint_policy=None, checkpoint_directory=None, checkpoint_namespace=None)
```

## Related material

[Recovery equations](football_experiment_equations.md),
[coverage](football_experiment_documentation_checklist.md),
[verification](football_experiment_check.json),
[model selection](model_selection_reference.md),
[notebook 14](../../notebooks/14_football_experiment_quickstart.ipynb).
