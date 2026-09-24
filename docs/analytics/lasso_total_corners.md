# Lasso on total match corners

[Notebook 18](../../notebooks/18_lasso_total_corners_pipeline.ipynb) is a code-first,
end-to-end experiment. Open it with the project's `misc314_py314` kernel and use
**Run All**. The notebook discovers the repository root, imports existing public
APIs and shows the report in place. It does not require the UI server.

## Default experiment

| Stage | Configuration |
|---|---|
| Data | All available leagues in `data/xDiyo_data`, seasons `22_23`, `23_24`, `24_25` |
| Match inclusion | Awarded matches excluded; missing total-corner labels dropped at assembly |
| Raw inputs | `ALL / Match overview / cornerKicks`, plus pregame standings |
| History | Two perspectives per match, UTC kickoff; grouping by team and competition across seasons |
| Features | Own and conceded corners: lag 1, mean 5, mean 10, std 5, Z-score 5, EMA span 5; normalized standing |
| Label | `MatchTotal(corners)`: home corners plus away corners |
| Layout | One match per row, home and away feature columns |
| Outer split | One pooled chronological fold: 22/23–23/24 train, 24/25 test |
| Pre-analysis | Training-only distributions, Pearson/Spearman/Kendall correlations and feature timelines |
| Fitted feature selector | Disabled; optional top-k absolute Spearman, learned inside each fit |
| Preprocessing | Median imputation, retaining empty columns, followed by standardization; fitted on training rows only |
| Model | Lasso, alpha 0.1, 20,000 maximum iterations, tolerance 0.0001, cyclic coordinate selection |
| Search | `None`; optional five-value alpha grid scored by inner-fold MSE |
| Execution | CPU, one fold worker, one inner CPU thread |
| Training controls/checkpoints | `None`; native sklearn Lasso has no generic iterative observer or mid-fit checkpoint support |
| Post-analysis | MSE, MAE, R²; observed/predicted KDEs; residuals; timelines; filtered match results; surviving coefficients; saved-run leaderboard |
| Refit | Disabled; optional additional all-observed-row deployment fit after evaluation |
| Persistence | Numerical artifacts and HTML saved; fitted evaluation model explicitly saved and reloaded |
| Upcoming predictions | Disabled; optional round/status/as-of filters over published fixtures |
| Betting | Disabled: no fabricated odds, settlement quotes or decision policy |

All stage settings are visible in normal Python cells. The main configuration
cell is tagged `configuration`, making bounded execution copies straightforward.
Model-specific parameters that matter for Lasso are shown explicitly; irrelevant
classification, ranking and CUDA settings are not introduced into the model.

## Information timing and feature meaning

Historical operators exclude the match being predicted. Default
`CUTOFF_HOURS=None` uses earlier finished kickoffs as the retrospective availability
assumption. Set, for example, `CUTOFF_HOURS=48` for a two-day lead time. The same
offset determines feature cutoffs and split fitting boundaries.

`AVAILABLE_AT`, if provided, is aligned to **team-history rows** for feature and
rating construction. The split cell explicitly uses `available_at=None`. Real
split result-availability times must be supplied separately, joined by match
identity and aligned to the **assembled match dataset**; a history-aligned vector
must not be passed directly to a match-level splitter.

Rolling windows allow partial history with `min_periods=1`. Standard deviation
and Z-score use `ddof=1`; the most recent eligible value is included in the
Z-score's baseline window. Undefined standard deviations/Z-scores remain missing
until the fold-fitted imputer handles them. Missing standings default to zero.
Current observed corners are the label, not an unlagged predictor.

The model remains fixed throughout the held-out season. Later test fixtures can
use earlier completed test-season matches in their histories under the declared
information schedule. This is evolving feature information, not model refitting.

## Optional stages

### Ratings and warm-up

`USE_CORNER_RATINGS=True` adds an explicit saved Glicko corner-comparison stream
and team rating/RD features. It compares own versus opponent counts for a win,
draw or loss; it does not model the corner margin. Engine settings are visible.

`USE_FEATURE_WARMUP=True` wraps only each five-match rolling mean in a seeded EMA
with a one-round hard handoff. Other features retain their normal definitions.
`USE_RATING_TRANSITIONS=True` configures the optional Glicko transition adapter
when the rating stream is enabled. It uses standings-ranked cohorts, optional RD
inflation and configurable shrinkage. League movement requires the transition
helper's evidence; this notebook does not invent league tiers or promotion rules.

### Feature and model selection

`USE_TOP_K=True` attaches a train-scoped top-k selector to the candidate. It is
fitted again for inner candidate fits, outer evaluation and any deployment refit.
Descriptive correlation output never silently determines the model's inputs.

Set `SEARCH_GRID=ALPHA_GRID` to compare alpha values 0.01, 0.05, 0.1, 0.5 and 1.0.
The inner fold trains on 22/23 and scores 23/24, wholly inside the outer training
population. The winner is fitted freshly on the outer training rows before
24/25 evaluation. The final leaderboard receives the final evaluated run; search
trials remain selection detail.

The pooled split explicitly uses `calendar_by=()` and
`block_by=("source_season",)`. Provider season IDs differ across competitions and
would not represent one common season calendar. The splitter still removes any
candidate training fixtures beyond the earliest evaluation prediction cutoff.

### Refit, reuse and model persistence

`REUSE_RESULTS=True` restores an exactly matching completed result without
refitting. Completed grid trials also recover after interruption. There is no
claim of resuming an interrupted native Lasso coordinate-descent fit.

`FINAL_REFIT=True` fits an additional deployment model on all labelled rows after
evaluation. Evaluation predictions and the test report remain unchanged. This
deployment model has seen the former test labels and is intended for subsequent
fixtures, not a new score on that same test population.

The model is saved under the result's own `evaluation_model` or
`deployment_model` directory. Reruns load an existing model there. A restored
numerical result with no model artifact prints an actionable message; it does not
silently train. Set `REUSE_RESULTS=False` to request a fresh fit in that case.
Only load trusted model artifacts. The saved model includes fitted preprocessing
and expected feature order.

### Upcoming fixtures

`RUN_FUTURE_PREDICTION=True` activates the final cell. `PREDICTION_ROUNDS` accepts
an integer, a list or a league-name-to-rounds mapping; `None` keeps all published
unplayed fixtures. `PREDICTION_AS_OF` optionally limits scheduled kickoffs. Fixture
filtering happens after building features on the complete loaded history.

The requested historical seasons may contain no upcoming fixtures; the cell
handles that explicitly. To use live-season exports, extend the loaded data and
revise the evaluation split design deliberately rather than treating a four-season
dataset as the original three-season experiment.

## Reading the report

The match display uses team names and optional filters. A two-corner tolerance
controls row coloring only. Coefficients refer to standardized features; a zero
coefficient is suppressed in the surviving-coefficient study. Correlated inputs
can exchange weight, so coefficient survival is not a causal claim.

Lasso can predict negative or fractional totals. This baseline neither rounds nor
clips predictions. A predicted total is also not an over/under probability.
The notebook does not manufacture bet decisions or decimal odds from it.

The experiment leaderboard compares saved final runs in this named experiment,
including earlier sessions. Equal MSE/MAE weights act on direction-aware
percentile utilities, not raw values on incompatible scales. Comparison groups
retain the library's population and metric compatibility rules.

## Verification

Verified with the existing Python 3.14 environment, without installing packages:

- Notebook schema and every code cell compile successfully; the delivered copy
  has no executed outputs.
- The unchanged all-league preparation cells loaded **13 leagues / 39 league-season exports**:
  13,973 matches, 27,946 team-history rows, 13 feature expressions, 26 assembled
  columns and 13,971 labelled matches. Two missing targets were excluded.
  Outer train/test: **9,371 / 4,600**. Inner train/test: **4,767 / 4,604**.
- A fresh-kernel top-to-bottom copy using only `LEAGUES=["Premier_League"]`
  completed on all three real seasons: **1,140 matches**, train 760 / test 380.
  Reports, numerical artifacts and fitted-model save/load completed.
- A second fresh-kernel run reused that exact completed result. Reloaded
  evaluation-model predictions matched the retained test predictions.
- An optional-stage copy enabled the full five-alpha grid, top-12 Spearman
  selection, final all-1,140-row refit and future-fixture handling. It completed;
  all selected features were learned inside their fitting populations, the
  inner test rows were disjoint from the outer test, and the historical export's
  empty upcoming-fixture branch returned cleanly.
- Enabling the corner Glicko stream, rating transitions and seeded-EMA feature
  warm-up completed preparation on the same real population: 1,140 rows and
  30 inputs. This additional variant did not fit a model.
- A full synthetic execution used 23 labelled matches plus one published future
  fixture. The reloaded model predicted that unplayed fixture with its actual
  target still missing.

Bounded execution copies and artifacts are under
`.pytest_tmp/lasso_notebook18/` (`executed.ipynb`, `reuse.ipynb`, `optional.ipynb`,
`ratings_warmup_preparation.ipynb`, `future_synthetic.ipynb`), with a machine-readable
`verification.json` summary.
The full all-league **model fit** is intentionally left for the user's Run All.
Fresh-kernel execution verifies display generation; live Jupyter frontend iframe
behavior was not separately exercised. The environment emitted its existing
Windows ZMQ event-loop compatibility warning, without a notebook failure.

To rerun the delivered full experiment from the repository root:

```powershell
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' -m jupyter nbconvert --execute --to notebook --output 18_lasso_total_corners_executed.ipynb --ExecutePreprocessor.kernel_name=misc314_py314 --ExecutePreprocessor.timeout=1200 notebooks/18_lasso_total_corners_pipeline.ipynb
```

This writes a separate executed notebook and fits the default all-league model.
The checked-in notebook remains a clean configurable starting point.
