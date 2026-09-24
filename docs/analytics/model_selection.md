# Select a model, then evaluate it independently

`ModelSelection` compares explicitly supplied finite candidates using only a
declared development population. After choosing a configuration, fit it freshly
on the outer training rows and evaluate its untouched test rows.

This guide runs from the repository root in the existing Python environment.
All numbers and fixtures below are synthetic. They demonstrate the workflow,
not football forecasting performance. The library does not run a search until
`run()` or `run_nested()` is called.

Built-in metric, weighted and parsimony decision rules validate their required
evidence before any candidate fit. Request the decision's metric explicitly (or
its configured `Metric.key` alias); enable information criteria for AIC/BIC.
Custom score reporters can supply their own evidence, which is checked through
the existing runtime comparison contract.

With an experiment store and `resume=True`, matching completed candidate fits
can be reused across changes to standard scoring metrics, pooling and decision
rules. New numerical evidence is calculated from retained predictions without
overwriting the saved trial. Fitting data/configuration changes require new fits.
`FootballExperiment` manages this recovery under its `reuse` setting; `reuse=False`
forces fresh candidate fitting while still saving completed trials. The selected
configuration's outer evaluation remains a separate fit. See
[candidate recovery](football_experiment.md#reuse-completed-fits-when-changing-the-comparison)
for the distinction from final-result reuse and optimizer checkpoints.

## 1. Declare development rows and an untouched holdout

The frames share one row order. A `Fold` uses integer positions in that original
order. Its `score` rows are a subset of `test`: predictions are retained for
every test row, but only score rows contribute to the selection metric.

```python
from pathlib import Path
import sys
import numpy as np
import pandas as pd

root = Path.cwd()
sys.path.insert(0, str(root / "src"))
from xdiyo_analytics.datasets import ModelDataset
from xdiyo_analytics.splits import Fold, SplitPlan
from xdiyo_analytics.analysis import PreTrainingAnalysis, PostTrainingAnalysis
from xdiyo_analytics.reporting import (
    TopKCorrelationSelector, PerformanceReporter, MatchResultReporter,
    ExperimentLeaderboardReporter,
)
from xdiyo_analytics.training import EstimatorAdapter
from xdiyo_analytics.experiments import ExperimentStore
from xdiyo_analytics.selection import (
    Candidate, GridCandidates, ModelSelection, MetricSelection,
    WeightedSelection, ParsimonySelection, FitStatistics, information_criteria,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

row = np.arange(36)
X = pd.DataFrame({
    "signal": np.sin(row / 3) + row / 30,
    "wave": np.cos(row / 5),
    "nuisance": np.sin(row * 1.7),
})
y = pd.DataFrame({"total": 8 + 2 * X.signal - X.wave + .1 * np.cos(row * 2)})
metadata = pd.DataFrame({
    "competition_id": 1, "season_id": 2026, "event_id": row + 1001,
    "kickoff_at": pd.date_range("2026-01-01", periods=36, tz="UTC"),
    "round": row // 4 + 1, "home_id": 1 + row % 3, "away_id": 4 + row % 3,
    "home_name": ["Home " + str(i % 3 + 1) for i in row],
    "away_name": ["Away " + str(i % 3 + 4) for i in row],
})
keys = ("competition_id", "season_id", "event_id")
dataset = ModelDataset(X, y, metadata, "match", keys, keys, "total")
development = np.arange(28)
inner_plan = SplitPlan([
    Fold(np.arange(12), np.arange(12, 20), np.arange(14, 20)),
    Fold(np.arange(20), np.arange(20, 28), np.arange(22, 28)),
], len(X), row)
holdout = Fold(development, np.arange(28, 36), np.arange(30, 36))
output_dir = root / "experiment/model_selection_demo/guide"
output_dir.mkdir(parents=True, exist_ok=True)
```

Selection receives a copied development-only dataset. Outer rows do not reach
candidate fitting, feature selection, validation selectors or metric reporters.
Custom callables remain trusted application code: a callback must not read a
global copy of the holdout. Dataset alignment and original row order remain the
caller's responsibility.

## 2. Build fresh models and a conditional grid

Each factory constructs a new adapter, estimator and scaler. A train-scoped
selector learns its columns separately for every fit. The selector is one
implementation of the common `FeatureSelector` contract; other selectors can
be used in the same position.

```python
preparation = PreTrainingAnalysis({
    "columns": TopKCorrelationSelector(
        type="per_fold", partition="train", k=2, method="pearson"),
})

def build_candidate(parameters):
    family = parameters["family"]
    alpha = parameters["alpha"]
    ratio = parameters.get("l1_ratio")

    def fresh_model():
        if family == "ridge":
            estimator = Ridge(alpha=alpha, tol=1e-10, max_iter=10000)
        elif family == "lasso":
            estimator = Lasso(alpha=alpha, tol=1e-10, max_iter=10000)
        else:
            estimator = ElasticNet(
                alpha=alpha, l1_ratio=ratio, tol=1e-10, max_iter=10000)
        return EstimatorAdapter(make_pipeline(StandardScaler(), estimator))

    return Candidate(
        family, fresh_model,
        config={"model": parameters, "scaling": "StandardScaler",
                "features": {"selector": "absolute Pearson", "k": 2},
                "solver": {"tol": 1e-10, "max_iter": 10000}},
        pre_analysis=preparation, features_from="columns",
        complexity=lambda model: {
            "active_coefficients": int(np.count_nonzero(
                model.estimator.steps[-1][1].coef_))
        },
    )

grid = GridCandidates([
    {"family": ["ridge", "lasso"], "alpha": [.01, .2]},
    {"family": ["elastic_net"], "alpha": [.01, .2], "l1_ratio": [.3, .8]},
], build_candidate)
assert len(list(grid)) == 8
```

`GridCandidates` produces the Cartesian product within each mapping, in input
order, and concatenates conditional mappings. Names receive stable one-based
suffixes; parameters are copied into `config.grid_parameters`. A list of
hand-built `Candidate` objects works too. The explicit configuration is the
recorded description; the library does not reconstruct callable behavior from
Python objects.

## 3. Search, inspect the comparison, and retain trial records

```python
store = ExperimentStore(output_dir / "experiments", "Synthetic selection")
search = ModelSelection(grid, metrics=["mse", "mae"], decision=MetricSelection("mse"))
selection = search.run(
    dataset, inner_plan, development_positions=development,
    experiment=store, name="Inner search",
)
comparison_report = selection.to_report()
comparison_report.to_html(output_dir / "comparison.html")
print(selection.comparison[["name", "raw::mse", "selected"]])
print("Selected configuration:", selection.winner.candidate.config)
```

The decision table exposes the metrics used by its rule; every requested metric
remains in each trial's numerical report and record. For example, choosing by
MSE does not add MAE to that decision table. `selection.winner.training` retains
the inner fits. It is selection evidence and is not the final evaluation.

The overall metric is calculated once from retained score predictions after the
declared pooling policy. It is not an unweighted average of fold metrics.
Repeated score rows require `pooling="occurrences"`, `"first"`, `"last"` or
`"mean"`; see the [equations](model_selection_equations.md).
Ordinary search creates numerical evidence, without requesting diagnostic plots.

## 4. Fit the winner freshly and save the independent result

```python
evaluation = selection.evaluate(dataset, holdout)
evaluation_report = PostTrainingAnalysis({
    "Holdout errors": PerformanceReporter(
        type="overall", partition="score", metrics=["mse", "mae"]),
    "Holdout fixtures": MatchResultReporter(
        type="per_fold", partition="test", target="total", tolerance=.5),
}, title="Synthetic untouched holdout").run(evaluation)
evaluation_report.to_html(output_dir / "holdout.html")
final_record = store.save_run(
    evaluation, evaluation_report, name="Untouched holdout",
    config=selection.winner.candidate.config, role="final",
    run_group=selection.run_group,
    selected_trial_id=selection.winner.saved_run_id,
)
leaderboard = PostTrainingAnalysis({
    "Final evaluations": ExperimentLeaderboardReporter(weights={"mse": 1}),
}).run(experiment=store)
leaderboard.to_html(output_dir / "leaderboard.html")
assert all(f.model is not evaluation.folds[0].model
           for trial in selection.trials for f in trial.training.folds)
```

The selected feature rule and scaler are fitted again on the outer training
population. `evaluate(..., validation=None, control=None)` explicitly removes
candidate monitoring validation or controls; omitting these arguments retains
them. Inner-fold validation dictionaries may need an appropriate outer override.
This is the training needed for holdout evaluation. No additional deployment
fit is performed.

The store receives trial-role records during selection. The final record above
is a separate explicit action, linked to the winner's **saved run ID**, not its
in-memory trial ID. The default leaderboard shows final records. Set
`include_trials=True` to inspect internal searches.

## 5. Reuse numerical evidence with another decision rule

```python
weighted = WeightedSelection({"mse": .75, "mae": .25}).decide(selection.trials)
parsimonious = ParsimonySelection(
    "mse", complexity="active_coefficients", tolerance=.05,
).decide(selection.trials)
print(weighted.table[["name", "raw::mse", "raw::mae", "score", "selected"]])
print(parsimonious.table[[
    "name", "raw::mse", "rank", "complexity", "within_tolerance", "selected",
]])
```

These calls compare retained evidence and do not fit models. Percentile utility
is relative to the candidate set. Use `scaling="fixed"` with
`reference_scales={"mse": (0, 1), "mae": (0, 1)}` for declared fixed bounds.
Missing required metrics leave a candidate unranked; weights are not redistributed.
All eligible candidates must describe compatible observations and metric definitions.

Parsimony chooses the smallest declared complexity within an **absolute metric
tolerance** of the best candidate. Its `selected` flag can differ from the raw
metric's rank 1. Active coefficient count is only the illustrative complexity
measure here; it is not automatically a likelihood parameter count or effective
degrees of freedom. Choosing a decision rule after looking at holdout outcomes
would invalidate the holdout's role.

## 6. Optional nested cross-validation

Each outer fold runs its own search and selects its own winner. The inner-plan
factory receives only that outer training population and returns **local**
positions. Outer results are combined only after the separate decisions.

```python
outer_plan = SplitPlan([
    Fold(np.arange(24), np.arange(24, 28), np.arange(24, 28)),
    Fold(np.arange(28), np.arange(28, 36), np.arange(30, 36)),
], len(X), row)

def make_inner_plan(development_dataset):
    n = len(development_dataset.X)
    a, b = n // 2, 3 * n // 4
    return SplitPlan([
        Fold(np.arange(a), np.arange(a, b), np.arange(a, b)),
        Fold(np.arange(b), np.arange(b, n), np.arange(b, n)),
    ], n, np.arange(n))

nested = search.run_nested(dataset, outer_plan, make_inner_plan)
nested_report = PostTrainingAnalysis({
    "Outer errors": PerformanceReporter(
        type="per_fold", partition="score", metrics=["mse", "mae"]),
    "Pooled outer errors": PerformanceReporter(
        type="overall", partition="score", metrics=["mse", "mae"]),
}, title="Synthetic nested evaluation").run(nested.training)
nested_report.to_html(output_dir / "nested.html")
print(nested.comparison[["outer_fold", "name", "selected"]])
```

This example is an alternative evaluation design on synthetic data. In a real
study, choose the holdout or nested protocol before inspecting its outcomes.
There is no universal winner chosen from outer scores. For stored nested runs,
pass `experiment=store` to retain separate search groups, then explicitly save
one final outer-evaluation record with each outer fold's winner ID in its config.
Do not claim that one selected-trial link represents all outer winners.

## 7. Optional fitted likelihood evidence

AIC/BIC require a justified fitted data likelihood and parameter convention.
This example uses ordinary least squares under a Gaussian model with **known
variance 4**. It includes the likelihood constants and counts the fitted
intercept and slope. It does not infer likelihood from a regularized model's
prediction MSE.

```python
class GaussianLine:
    def __init__(self, linear):
        self.linear = linear

    def design(self, X):
        return (np.c_[np.ones(len(X)), X.signal.to_numpy()]
                if self.linear else np.ones((len(X), 1)))

    def fit(self, context):
        self.targets = context.y.columns
        design = self.design(context.X)
        observed = context.y.iloc[:, 0].to_numpy()
        self.beta = np.linalg.lstsq(design, observed, rcond=None)[0]
        self.residual = observed - design @ self.beta

    def predict(self, context):
        return {"predict": pd.DataFrame(
            self.design(context.X) @ self.beta,
            index=context.X.index, columns=self.targets)}

    def fit_statistics(self):
        n = len(self.residual)
        log_likelihood = (-.5 * n * np.log(2 * np.pi * 4)
                          - np.dot(self.residual, self.residual) / 8)
        return FitStatistics(
            float(log_likelihood), len(self.beta), n,
            "Gaussian variance=4 known; constants; k=regression coefficients",
        )

likelihood_candidates = [
    Candidate("intercept" if not linear else "line",
              lambda linear=linear: GaussianLine(linear),
              config={"linear": linear, "known_variance": 4})
    for linear in (False, True)
]
likelihood_selection = ModelSelection(
    likelihood_candidates, information_criteria=True,
    decision=MetricSelection("aic"),
).run(dataset, inner_plan, development_positions=development)
print(likelihood_selection.winner.report.studies[-1].result.tables["fit_statistics"])
print(information_criteria(FitStatistics(-10, 2, 20, "Example convention")))
```

The aggregate is explicitly **mean per-fit AIC/BIC**. Overlapping folds do not
become a single joint likelihood. Comparisons require matching fitted response
samples, sampling-unit counts and likelihood/parameter-count conventions.
Applicability of the supplied likelihood and effective degrees of freedom
remains the adapter author's responsibility.
See the [formulas and official references](model_selection_equations.md#fitted-likelihood-evidence).

## Failures and additional evidence

The default `on_error="raise"` stops on a candidate's fitting or numerical
callback exception. `"record"` keeps that failed trial visible and continues;
it cannot manufacture a winner if every candidate lacks usable evidence.
Declarative scope/preparation errors, final decision errors and storage errors
propagate. A model's invalid hyperparameter may be detected only by its fit
method and therefore follows the candidate error policy.

`evidence_reporters={"name": reporter}` optionally supplies additional
score-scoped numerical metric tables, including custom return calculations.
Such reporters use the shared metric schema and explicitly declared direction,
sample identity and calculation parameters. Their cost and assumptions are
explicit; selection does not automatically invent a betting strategy.

## Further reading

- [API and helper reference](model_selection_reference.md)
- [Equations](model_selection_equations.md)
- [Verification coverage](model_selection_documentation_checklist.md)
- [Notebook 13](../../notebooks/13_model_selection_quickstart.ipynb)
- [Saved notebook HTML](../../notebooks/outputs/model_selection_quickstart.html)
- [Verification record](model_selection_check.json)
- [Training controls](training_controls.md) and [post-training reports](post_training.md)

Fresh-kernel notebook execution and saved HTML interactions are checked
separately from live Jupyter frontend behavior, which remains unverified.
