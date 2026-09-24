# Run, reuse and reopen a football experiment

`FootballExperiment` joins already prepared data and splits, an optional model
search, fresh training, reports and saved numerical results. Give it a descriptive
experiment name; each execution produces one named final run. Preparation remains
an ordinary function that you can inspect independently.

The examples run sequentially from the repository root in the existing Python
environment. All fixtures and outcomes are synthetic. They establish behavior,
not forecasting quality. [Notebook 14](../../notebooks/14_football_experiment_quickstart.ipynb)
is the shorter fixed-model workflow.

## 1. Make preparation explicit

In a real project, the preparation function calls the existing loading,
history/rating, feature, label, assembly and split APIs in that order. Return the
assembled `ModelDataset` and its `SplitPlan`; optionally keep data-only intermediate
outputs such as history frames. No model is fitted during this step.

```python
from pathlib import Path
from functools import partial
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xdiyo_analytics.datasets import ModelDataset
from xdiyo_analytics.splits import Fold, SplitPlan
from xdiyo_analytics.analysis import PreTrainingAnalysis, PostTrainingAnalysis
from xdiyo_analytics.reporting import (
    CorrelationAnalysis, PerformanceReporter, ExperimentLeaderboardReporter,
)
from xdiyo_analytics.selection import Candidate, ModelSelection
from xdiyo_analytics.training import EstimatorAdapter, CheckpointPolicy
from xdiyo_analytics.experiments import FootballExperiment, PreparedExperiment, RefitPolicy

root = Path.cwd()
output = root / "experiment/football_experiment_demo/guide"
rows = np.arange(30)
X = pd.DataFrame({"signal": np.sin(rows / 4) + rows / 20, "wave": np.cos(rows)})
y = pd.DataFrame({"total": 7 + 2 * X.signal + .1 * X.wave})
metadata = pd.DataFrame({"competition_id": 1, "season_id": 2026,
    "event_id": rows + 1001, "home_id": 1 + rows % 3, "away_id": 4 + rows % 3})
keys = ("competition_id", "season_id", "event_id")
dataset = ModelDataset(X, y, metadata, "match", keys, keys, "total")
outer = SplitPlan([Fold(rows[:20], rows[20:], rows[22:])], len(rows), rows)
inner = SplitPlan([Fold(rows[:8], rows[8:12], rows[8:12]),
                   Fold(rows[:12], rows[12:20], rows[12:20])], len(rows), rows)

def prepare():
    return PreparedExperiment(dataset, outer,
        outputs={"input_metadata": metadata.copy()}, config={"synthetic": True})

def adapter(alpha):
    return EstimatorAdapter(make_pipeline(StandardScaler(), Ridge(alpha=alpha)))

def candidate(alpha):
    return Candidate(f"Ridge {alpha}", partial(adapter, alpha),
        config={"alpha": alpha, "preprocessing": "StandardScaler"})

before = PreTrainingAnalysis({
    "Training association": CorrelationAnalysis(type="overall", partition="train"),
})
after = PostTrainingAnalysis({
    "Errors": PerformanceReporter(type="overall", partition="score", metrics=["mse", "mae"]),
    "Saved runs": ExperimentLeaderboardReporter(weights={"mse": 1.}),
})
experiment = FootballExperiment("Synthetic workflow", output_dir=output, prepare=prepare)
```

The plan uses positions in the original dataset. Predictions cover rows 20–29;
only rows 22–29 contribute to the final score. Descriptive `pre_analysis` does
not select model inputs. Put learned feature selection in the `Candidate` using
its train-scoped `pre_analysis` and `features_from` fields.

## 2. Run a fixed configuration, reuse it, then reopen it

```python
fixed = experiment.run(model=candidate(.1), pre_analysis=before, post_analysis=after,
                       name_fields=["alpha"])
same = experiment.run(model=candidate(.1), pre_analysis=before, post_analysis=after,
                      name_fields=["alpha"])
assert same.reused and same.record["run_id"] == fixed.record["run_id"]
assert same.training.folds[0].model is None

reopened = FootballExperiment("Synthetic workflow", output_dir=output).load(fixed.record["run_id"])
assert reopened.reused
pd.testing.assert_frame_equal(reopened.dataset.X, dataset.X)
reopened.to_html(output / "fixed.html")
print(fixed.record["name"], fixed.record["run_id"])
```

`run()` calls preparation each time to discover the current inputs. Exact reuse
skips model fitting and prediction. `load(run_id)` reads saved inputs and results
directly and skips preparation as well. Restored results contain predictions,
labels, row identities, report artifacts and history; fitted model objects are
`None`. Rendering a result never fits or predicts.

The automatic identity includes the prepared data and row order, definitions,
folds, candidate factories and configuration, analysis/refit/checkpoint settings,
local library source and runtime versions. Factory code, defaults, closures and
referenced globals are included. Moving unchanged code to another notebook cell
does not invalidate it. Declare external files/services or opaque callable
objects through configuration or `cache_key()` returning data. Keep diagnostic
counters outside that key. `PreparedExperiment.outputs` is retained data, not an
identity input: put a relevant external revision in `config` when it changes
without changing model inputs.

Use `reuse=False` for a new execution group even when everything matches.
Names alone never identify reusable work. Only one writer may use an experiment
or checkpoint namespace at a time.

## 3. Add a search and inspect its final evaluation

```python
selected = experiment.run(
    model_selection=ModelSelection([candidate(.01), candidate(.2), candidate(2.)], metrics="mse"),
    selection_plan=inner, pre_analysis=before, post_analysis=after,
    name_fields=["alpha"],
)
selected.to_html(output / "selected.html")
leaderboard = experiment.leaderboard({"mse": 1.})
leaderboard.to_html(output / "leaderboard.html")
assert len(leaderboard.studies[0].result.tables["leaderboard"]) == 2
```

For a holdout search, preparation contains exactly one outer fold. The inner
`selection_plan` uses original dataset positions; development defaults to the
outer training rows. Selection is confined to that population, then the winner
is fitted afresh for the outer evaluation. Completed search trials are recovered
after interruption. An incomplete trial is fitted again unless its adapter also
implements native checkpoints.

### Reuse completed fits when changing the comparison

Candidate fitting and candidate ranking have separate identities. With normal
reuse enabled, a completed trial with matching fitting inputs can be reused even
when you change the built-in decision rule, requested standard prediction metrics,
metric pooling or the surrounding run/report configuration. Standard numerical
evidence is recomputed from its retained predictions and labels; this does not
fit or predict again. Saved model-only evidence is preserved. The original trial
record and report remain unchanged; the new comparison uses its own in-memory
evidence and references the saved trial.

Matching is based on the actual dataset, row order, splits/development population,
candidate fitting configuration, implementation identity and relevant execution
settings. Changing model parameters, preprocessing, target transforms, fitted
feature selection or fitting data prevents reuse. A UI model factory's identity
includes its model, X preprocessors, adapter, target transformer and prediction
methods; changing an exploratory reporter or the search decision does not itself
change that factory. Custom model-only evidence and information-criterion settings
remain part of the recoverability contract.

`reuse=False` explicitly skips candidate-cache lookup as well as final-result
reuse. The new fits are still saved as recoverable completed trials. Likewise,
reusing inner-search candidates does not eliminate the separate fit of the chosen
configuration on the outer training population. Its untouched holdout evaluation
is a distinct step.

In nested evaluation, this applies separately to each outer fold. Recovering a
completed inner grid for the first outer fold does not complete later outer
folds' searches or evaluations. A recovered candidate comparison is not a final
whole-experiment result.

There are therefore three separate operations:

| Operation | Retained material | What runs again? |
|---|---|---|
| Rescore completed candidates | Fold predictions, labels, row identities and numerical evidence | Requested metrics/ranking; no candidate fits. |
| Finish outer evaluation | Chosen configuration and declared outer split | Fresh outer training fit and predictions unless that complete final run matches. |
| Resume an interrupted optimizer | Adapter-supported checkpoint state | Remaining fitting iterations according to the checkpoint contract. |

A partial optimizer fit is not a completed candidate. A saved trial alone is not
a deployment model: data-only trial recovery does not restore fitted estimator
objects. The same artifact directory can retain multiple comparison views without
overwriting its prior results. Older bundles lacking a compatible fitting cache
identity may require explicit recovery from their saved predictions rather than
automatic reuse; changing identity rules does not silently rewrite old artifacts.

Before fitting candidates, built-in metric, parsimony and weighted decisions
check that their required metric names are requested. A configured `Metric.key`
is the evidence name. AIC/BIC require the explicit information-criteria option;
custom evidence reporters/decision rules retain their own runtime contract.
This catches, for example, selecting by MAE when only MSE was requested before an
expensive search starts.

Recovery verification on synthetic fixtures passed **58 tests in 11.57 seconds**.
The focused checks confirm zero extra candidate fits after changing decisions,
standard metrics and pooling; unchanged saved trial files; preservation of
model-only AIC/BIC evidence; invalidation for changed fitting data/splits/model
parameters and UI preprocessing identity; forced-fresh fitting; validation before
pre-analysis; interruption recovery; and existing checkpoint behavior. These
checks do not fit or modify production experiments.

The saved NB search was separately recovered as an explicit legacy comparison:
[90-candidate table](search_recovery/nb_90_candidates.csv),
[selection summary](search_recovery/nb_recovered_selection.json), and
[read-only reproducer](search_recovery/recover_nb_comparison.py).
This is the **first outer fold's inner search**, not the full nested evaluation.
All 90 recomputed MSE values matched retained values within 1e-10; MAE was added
without fitting. MSE and MAE selected the same saved candidate, with values
11.4728329366 and 2.6498389581 respectively. Further outer folds still require
their own searches/evaluations if not already completed.

The combined report orders descriptive pre-training studies, internal candidate
comparison, and final post-training studies. Only final post-training numerical
metrics enter the final run record. Default leaderboards hide trials and show
expandable configuration details. Experiment reporters refresh after publication
and on reuse; an explicitly loaded result retains its saved report snapshot.

For nested CV, pass `inner_plan_factory` instead of `selection_plan`. It receives
each outer training population with local positions. It returns inner splits in
that local order. Each outer fold selects its own winner; the single final run
contains all outer predictions and is named `Tuned per fold` unless overridden.
See the [selection guide](model_selection.md) for a complete nested example.

## 4. Request an additional deployment refit explicitly

```python
refitted = experiment.run(model=candidate(.1), post_analysis=after,
    refit_policy=RefitPolicy(train_positions=rows), name="Ridge deployment refit")
assert refitted.refit.model is not refitted.training.folds[0].model
assert len(refitted.refit.fit_positions) == 30
```

The evaluation remains unchanged; the extra model uses the rows you requested.
Refit validation/control default to `None` independently of evaluation controls.
Train-scoped feature selection and preprocessing are learned again on actual
refit fitting rows, excluding any new validation subset. After nested selection,
supply `RefitPolicy(..., candidate=...)` because there is no single overall winner.
Reloading numerical results restores refit metadata with `model=None`; calling
that metadata object's `predict()` raises a clear error. This workflow does not
schedule future retraining.

## 5. Native checkpoints for a capable adapter

There are three recovery levels: a completed final run; completed search trials;
and an interrupted fit whose adapter can resume. Plain sklearn/Ridge/Lasso `fit`
does not gain mid-fit recovery from setting `CheckpointPolicy`.

The following small SGD adapter offers a complete native checkpoint after every
epoch. It preserves fitted scaling, weights, momentum, RNG state, completed epoch
and loss history. The same responsibilities apply to a neural framework's native
serializer: optimizer/scheduler/scaler and device state belong to that adapter.
Do not save only model weights and claim an equivalent resumed training path.

```python
class NativeSGD:
    def __init__(self, steps=6, seed=23):
        self.steps, self.seed = steps, seed

    def fit_resumable(self, context, *, control, checkpoint, save_checkpoint):
        if control is not None:
            raise ValueError("This example implements a fixed epoch budget only.")
        self.columns, self.targets = tuple(context.X), tuple(context.y)
        X, y = context.X.to_numpy(), context.y.iloc[:, 0].to_numpy()
        rng = np.random.default_rng(self.seed)
        if checkpoint is None:
            self.mean, self.scale = X.mean(axis=0), X.std(axis=0)
            self.scale[self.scale == 0] = 1
            self.weights = np.zeros(X.shape[1] + 1)
            velocity, cursor, history = np.zeros_like(self.weights), 0, []
        else:
            with np.load(checkpoint / "state.npz", allow_pickle=False) as state:
                self.mean, self.scale = state["mean"], state["scale"]
                self.weights, velocity = state["weights"], state["velocity"]
            saved = json.loads((checkpoint / "metadata.json").read_text())
            cursor, history = saved["cursor"], saved["history"]
            rng.bit_generator.state = saved["rng"]
            assert saved["columns"] == list(self.columns) and saved["targets"] == list(self.targets)
        design = np.c_[(X - self.mean) / self.scale, np.ones(len(X))]
        for epoch in range(cursor + 1, self.steps + 1):
            for row in rng.permutation(len(X)):
                gradient = 2 * (design[row] @ self.weights - y[row]) * design[row]
                velocity = .4 * velocity + gradient
                self.weights -= .01 * velocity
            history.append({"step": epoch, "loss": float(np.mean((design @ self.weights - y) ** 2))})
            def writer(directory):
                np.savez(directory / "state.npz", weights=self.weights, velocity=velocity,
                         mean=self.mean, scale=self.scale)
                (directory / "metadata.json").write_text(json.dumps({
                    "cursor": epoch, "history": history, "rng": rng.bit_generator.state,
                    "columns": list(self.columns), "targets": list(self.targets),
                }))
            save_checkpoint(writer)
        self.training_history_ = pd.DataFrame(history)
        self.training_summary_ = {"steps": self.steps}

    def predict(self, context):
        design = np.c_[(context.X.to_numpy() - self.mean) / self.scale, np.ones(len(context.X))]
        return {"predict": pd.DataFrame(design @ self.weights, index=context.X.index, columns=self.targets)}

native = experiment.run(model=Candidate("Native SGD", partial(NativeSGD, steps=6), config={"steps": 6}),
    checkpoint_policy=CheckpointPolicy(every=1, unsupported="raise"), post_analysis=after)
assert native.training.folds[0].training_summary["steps"] == 6
```

`checkpoint` is `None` on a fresh fit or an immutable directory on resume.
`save_checkpoint(writer)` publishes a new directory only after the writer
returns, then replaces the latest pointer. A failed write leaves the prior
published checkpoint usable. `every=2` publishes every second offered checkpoint,
not every second model epoch unless those offers coincide. The offer counter
starts again on each fit invocation. Default `unsupported="skip"` runs ordinary
training for an incapable adapter; `"raise"` requires support. Reuse false creates
a fresh namespace. Serialize writers; there is no distributed lock or automatic
cleanup of incomplete directories.

## Reading older runs and limits

Legacy runs expose only actual saved tables, optional prediction files and
optional original HTML. They cannot reconstruct absent input datasets, original
report scopes, missing match-key definitions or fitted models. A legacy report
with no saved predictions has `training=None`. Numerical bundle recovery is
data-only and does not unpickle models. It is distinct from explicit model
serialization added in a separate increment.

New bundles retain nullable labels/IDs, StringDtype storage and MultiIndex
levels/codes. Old tuple-based bundles remain readable; metadata omitted when
they were written cannot be reconstructed. Arbitrary custom objects in retained
outputs or report artifacts are rejected; use data-only frames, arrays, figures,
configuration and library result dataclasses.

The [reference](football_experiment_reference.md) lists all APIs and helpers.
The [recovery equations](football_experiment_equations.md) describe scope and
checkpoint equivalence. See the [coverage checklist](football_experiment_documentation_checklist.md)
and [verification record](football_experiment_check.json) for tested boundaries.
Fresh-kernel execution and saved iframe interaction are checked separately;
live Jupyter frontend trust/display remains unverified.
