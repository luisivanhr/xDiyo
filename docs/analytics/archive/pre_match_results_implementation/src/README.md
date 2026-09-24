# Load and select football data for an experiment

Use `load_seasons` for an experiment spanning leagues and seasons, and
`inspect_season` to browse one publication's tables and columns.
Use `select_stats` to combine statistic bundles and exact measures after loading.

```python
from pathlib import Path
from xdiyo_analytics.data import inspect_season, load_seasons

project = Path("C:/Users/luisi/Documents/Programming/Python/xDiyo")
data_root = project / "data/xDiyo_data"
catalogue = inspect_season(data_root, "Premier_League_24_25")
catalogue[["rows", "column_count", "description"]]

experiment = load_seasons(
    data_root, ["22_23", "23_24", "24_25"], leagues=None,
    tables=["matches", "statistics", "pregame"],
    record_dir=project / "experiment/initial_population/selections",
)
experiment.matches.head()
experiment["statistics"].columns.tolist()
```

`leagues=None` uses all available leagues; pass an exact name such as
`"Premier_League"` or a list to narrow the selection. Tables stay separate, with
original values, nullable types and within-publication row order. Added
`source_league` and `source_season` columns identify each row. A table name alone
also works: `tables="statistics"`. Shots are independently selectable.

The first import on 16 September 2026 loaded **39 publications across 13 leagues**:
**13,976 matches**, **3,303,574 statistics rows** and **27,072 pregame rows**.
All 13 leagues have a publication for each selected season. These are collected
cohort counts, not a provider-wide fixture completeness claim.

## Choose statistics and standings

```python
from xdiyo_analytics.data import list_stat_bundles, select_stats

list_stat_bundles()
selected = select_stats(
    experiment,
    bundles=["attack_totals", "defense_first_half", "standings"],
    stats=[("ALL", "Match overview", "ballPossession")],
)
selected.statistics.head()
selected.pregame.head()
```

Selections form a union and preserve group identity, observations, nulls and row
order. Each category has one list shared by all periods: `*_all` keeps every
supplied period, while totals and halves filter `ALL`, `1ST` and `2ND`.
Totals are not sums of halves. `all_stats` includes uncategorized measures.
Defense means observed defensive metrics, not calculated opponent statistics.
Standings stay separate; matches pass through when loaded. Categories are
editable through `STAT_CATEGORIES` or a per-call `categories=` mapping.

## Build team histories

```python
from xdiyo_analytics.histories import build_team_history

history = build_team_history(selected)
history[["event_id", "side", "team_id", "opponent_id", "kickoff_at",
         "goals_for", "goals_against", "result"]].head()
```

The result has two observed rows per match with team/opponent statistics and
positions. Statistic columns preserve period/group/key identity; exact labels
are available in `history.attrs['stat_columns']`. UTC kickoff retains the full
time of day. Goals use current scores, and W/D/L requires a finished match with
both scores; penalty winners are not inferred. Missing observations retain rows.
Use the feature evaluator below to calculate historical predictors from these rows.

## Evaluate named features

```python
from xdiyo_analytics.features import (
    Stat, ForAgainst, Lag, RollingMean, IsHome, NormalizedStanding,
    evaluate_features, league_season_team_counts,
)

corners = Stat("ALL", "Match overview", "cornerKicks")
team_counts = league_season_team_counts(history)  # Full population before splits.
features = evaluate_features(history, {
    "home": IsHome(),
    "standing": NormalizedStanding(),
    "corners_lag1": Lag(ForAgainst(corners, "both")),
    "corners_mean5": RollingMean(corners, window=5),
}, team_counts=team_counts)
features.head()
```

Outputs keep the history index. Default history combines venues and crosses
seasons within team/competition. Windows count eligible matches; reductions use
finite values inside the window, while lags keep missing positions. Standings
normalize first place to 1 and last to 0, with missing/invalid values defaulting
to zero. The library also includes std, Z-score, H2H, nesting and optional EMA.

`cutoffs=None` and optional availability times need no extra inputs. Earlier
finished kickoffs are the default **retrospective availability proxy**; they do
not establish exact result-completion or publication times. For an earlier
boundary, pass `cutoffs=history["kickoff_at"] - pd.Timedelta(days=2)` after
importing pandas as `pd`. Current context is not reconstructed at that earlier time.

## Add result or statistic ratings

```python
from xdiyo_analytics.features import MatchResultGlicko, StatGlicko

rating_X = evaluate_features(history, {
    "wdl": MatchResultGlicko(),
    "corners": StatGlicko(corners),
})
```

Both sides' public rating, RD and volatility become ordinary training/reporting
columns. Corner ratings compare earned/conceded corners as win/draw/loss; they
are independent of result ratings and do not predict counts. The adapter reuses
the unchanged legacy Glicko engine, with tau=1.0 by default. Replay uses event
periods and equal-release batches, without automatic calendar idle inflation.
`build_ratings`, `RatingRun.save/load` and `Rating(name)` support reusable full
state with later model-column selection. Custom numeric states can use the same
snapshot interface; graph-model training remains future work.

- [Short notebook](../notebooks/01_loader_walkthrough.ipynb)
- [Loading and saved selections](../docs/analytics/season_loading.md)
- [Table descriptions and column discovery](../docs/analytics/season_inspection.md)
- [Statistic bundles and category definitions](../docs/analytics/stat_selection.md)
- [Team-history columns and semantics](../docs/analytics/team_history.md)
- [Feature quickstart notebook](../notebooks/02_feature_quickstart.ipynb)
- [Feature expressions and timing rules](../docs/analytics/features.md)
- [Exact cutoff formats and frozen rounds](../docs/analytics/feature_cutoffs.md)
- [Feature verification evidence](../docs/analytics/features_check.json)
- [Ratings quickstart notebook](../notebooks/03_ratings_quickstart.ipynb)
- [Ratings, saved runs and field scales](../docs/analytics/ratings.md)
- [Legacy-engine comparison](../docs/analytics/glicko_engine_comparison.md)
- [Rating verification evidence](../docs/analytics/ratings_check.json)
- [Recorded first-import evidence](../docs/analytics/multiseason_import_check.json)

`record_dir` saves one selection per publication outside the source directory.
Records pin versions; later discovery may include newly available publications.
Use `load_season(..., record_path=...)` for one explicitly named publication.
Full-file fingerprint checking is optional with `verify_hashes=True`.

## Fit one configuration on prepared folds

`xdiyo_analytics.training.TrainingRunner` fits a fresh adapter once per supplied
fold and predicts all held-out test rows. `fit_predict` provides the one-fold
form. `EstimatorAdapter` wraps an estimator or pipeline; custom adapters implement
the structural `ModelAdapter` contract. Put learned imputation/scaling inside
the fresh pipeline so its fitted state uses training rows only.

`FoldResult` and `TrainingResult` retain fitted models, original positions,
selected feature/target columns, metadata, true targets and named prediction
frames. Scoring subsets remain distinct from the full held-out predictions.
Optional stored feature selections require explicit naming and training-scope
containment; the caller owns their dataset provenance. Single/multiple targets,
label/probability outputs and numeric CPCV path reuse are documented.

All **914 analytics tests** pass, including **90 new synthetic training cases**.
Nine executable examples, the separate nine-cell/four-code-cell ninth notebook
and isolated wheel imports pass. The guide's 15 LaTeX expressions and the saved
notebook output were reviewed. All eight earlier notebooks remain unchanged.
This increment uses small synthetic datasets; real pilot fitting, search,
scheduled refitting remain later work. Post-training reporters are covered below. Existing
feature construction still owns histories, ratings and information cutoffs.

- [Practical training guide and equations](../docs/analytics/training.md)
- [Complete training API and helper reference](../docs/analytics/training_reference.md)
- [Coverage and limitations](../docs/analytics/training_documentation_checklist.md)
- [Ninth quickstart notebook](../notebooks/09_training_quickstart.ipynb)
- [Verification and preservation evidence](../docs/analytics/training_check.json)

## Control fitting, inspect models and refit

`TrainingRunner` can reserve explicit whole-match validation rows for compatible
adapters. `IterativeAdapter` and `PartialFitBackend` support early stopping,
plateau rate reductions, in-memory best-state restoration and fresh seeded
attempts. `LearningCurveReporter` and `CoefficientReporter` inspect retained
models. `refit_model` fits a fresh final configuration on declared development
rows. `ExperimentStore` records trial/final roles and an optional selected-trial
link; the default leaderboard displays final results.

The preserved controls revision passes **1,216 analytics tests** (138 new), seven
examples, notebook 11 and 26 offline browser checks. Its 49 LaTeX expressions
render correctly. Ten earlier notebooks remain unchanged. This is synthetic
verification of the recorded snapshot; the later MatchResult/viewer tree has a
separate pending check. Live Jupyter frontend behavior remains unverified.

- [Training controls guide](../docs/analytics/training_controls.md)
- [Complete API and helper reference](../docs/analytics/training_controls_reference.md)
- [Decision and coefficient equations](../docs/analytics/training_controls_equations.md)
- [Coverage checklist](../docs/analytics/training_controls_documentation_checklist.md)
- [Notebook 11](../notebooks/11_training_controls_quickstart.ipynb)
- [Recorded evidence](../docs/analytics/training_controls_check.json)

## Inspect predictions and compare saved runs

`PostTrainingAnalysis` composes named reports from retained `TrainingResult`
predictions. Declare per-fold, overall or timeline execution and test or score
rows. Metrics include configurable regression, classification, probability and
entropy calculations. Reports show performance, residuals, calibration, timelines
and observed/predicted KDE, ECDF or categorical-frequency overlays. Repeated
predictions require explicit pooling; overall metrics use pooled eligible rows.

`BetSpec` accounts for an existing `BetOption` with explicit decimal odds,
decisions and stakes. Settlement units must match the prediction layout.
`ExperimentStore` persists configured runs and optional predictions/reports;
`rank_runs` and `ExperimentLeaderboardReporter` compare saved metrics using
explicit weights, selectors and compatible evaluated populations. They perform
no model search or automatic strategy execution.

All **1,078 analytics tests** pass, including **164 new post-training cases**.
Twelve guide examples, the minimal tenth notebook, 33 offline browser checks and
isolated wheel use pass. All 54 LaTeX expressions render correctly. Nine earlier
notebooks, prior artifacts and the recorded data/selection files are unchanged.
All new fits/accounting are synthetic; live Jupyter frontend trust is unverified.

- [Practical post-training guide](../docs/analytics/post_training.md)
- [Complete API and helper reference](../docs/analytics/post_training_reference.md)
- [Calculation and comparison equations](../docs/analytics/post_training_equations.md)
- [Coverage and limits](../docs/analytics/post_training_documentation_checklist.md)
- [Tenth quickstart notebook](../notebooks/10_post_training_quickstart.ipynb)
- [Verification and preservation evidence](../docs/analytics/post_training_check.json)

## Pre-training analysis, reports and feature selection

`xdiyo_analytics.analysis.PreTrainingAnalysis` runs an explicit mapping of studies.
`xdiyo_analytics.reporting` supplies reusable result/artifact/viewer records,
distribution/KDE and association reporters, team/league timelines, a generic
`FeatureSelector` and `TopKCorrelationSelector`. Choose execution type and row
partition separately. Overall pooling uses unique observations; consensus combines
local selections. Named correlation reuse and selected-column transfer are explicit.

`AnalysisReport.to_html()` exports offline interactive HTML; `to_notebook()`,
`show()` and automatic rich display embed it in an isolated iframe without rerunning
analysis. Navigation, folds, plots and downloads were checked in Chrome, including
the executed notebook exported with nbconvert's Lab template. A live Jupyter
frontend was unavailable in the checked environments, so its trust behavior remains
unverified. Use a trusted HTML/JavaScript-capable frontend or the standalone export.

All 824 analytics tests pass, including 192 reporting cases. Ten guide examples,
both real-data layouts, the separate eighth notebook and isolated wheel imports
pass; no model fitting or pilot feature-set decision is included.

- [Reporting guide](../docs/analytics/reporting.md)
- [Complete API and helper reference](../docs/analytics/reporting_reference.md)
- [Coverage and limitations](../docs/analytics/reporting_documentation_checklist.md)
- [Eighth quickstart notebook](../notebooks/08_reporting_quickstart.ipynb)
- [Verification evidence](../docs/analytics/reporting_check.json)

## Splits and CV

`xdiyo_analytics.splits` provides `TemporalSplit`, `MatchKFold`, `GroupKFold`,
`CPCV`, `Fold`, `SplitPlan`, `create_split_plan` and `reconstruct_paths`.
All layouts preserve whole matches and original dataset positions. Temporal
training applies prediction/availability boundaries and complete-target checks;
score rows remain a subset of held-out test rows. CPCV supplies interval purging,
duration embargo and complete path membership. Retrospective membership needs
separate fold-aware feature/state preparation before a fitting experiment.

The corrected API passes **142 focused split/CV tests**. The preceding API passed
the full **632-test analytics suite**, including the prior 490 regressions, and
four-publication checks of 192,240 membership positions and 41,160 path-value cells.
Those algorithm checks retain their original source hashes; the splitting
algorithms are unchanged. Ten current guide examples, the refreshed seventh
notebook and rebuilt wheel imports pass. The final preservation audit confirms
33 stable source files, 164 unchanged earlier files and six unchanged earlier
notebooks, with both API revisions and their evidence preserved.
SplitPlan and create_split_plan own only membership, ordering and optional paths;
fitting policies belong to a future training-layer extension. No model fitting or
statistical report runs here.

- [Practical split/CV guide and equations](../docs/analytics/splits.md)
- [Complete API, metadata and internal-helper reference](../docs/analytics/splits_reference.md)
- [Coverage checklist](../docs/analytics/splits_documentation_checklist.md)
- [Separate minimal splits notebook](../notebooks/07_splits_quickstart.ipynb)
- [Verification evidence](../docs/analytics/splits_check.json)

## Dataset assembly, labels and next development stage

`assemble_dataset` joins identity-keyed features to one selected `LabelData`.
`evaluate_features(..., keyed=True)` attaches match/team/side keys; its default
keeps the original index. Match layout creates home-then-away feature blocks;
team-match layout preserves paired team rows. Both return aligned `X`, `y`,
metadata and exact match-group tuples through `ModelDataset`. Missing targets
remain by default; optional filtering removes whole matches. Settlement stays
in metadata. At the assembly checkpoint the analytics suite passed **490 tests**, including
**96 new assembly/keyed-feature cases**. The final source and preservation audit passed.

- [Dataset assembly practical guide and LaTeX equations](../docs/analytics/datasets.md)
- [Complete assembly API and schema reference](../docs/analytics/datasets_reference.md)
- [Assembly coverage checklist](../docs/analytics/datasets_documentation_checklist.md)
- [Separate assembly notebook](../notebooks/06_dataset_assembly_quickstart.ipynb)
- [Assembly verification evidence](../docs/analytics/datasets_check.json)

`create_labels` now builds realized targets through the separate label namespace:
`TeamValue`, `MatchTotal`, `Outcome`, `Above` and `BetOption`. Each named result
contains numeric `y`, identity metadata, a match/team-match unit and a perspective;
settled options also retain distinct win/loss/push/void/missing categories.
The label checkpoint passed **394 tests**, including 119 new label cases.

- [Labels practical guide and LaTeX equations](../docs/analytics/labels.md)
- [Complete labels API and schema reference](../docs/analytics/labels_reference.md)
- [Labels coverage checklist](../docs/analytics/labels_documentation_checklist.md)
- [Separate labels notebook](../notebooks/05_labels_quickstart.ipynb)
- [Labels verification evidence](../docs/analytics/labels_check.json)

The league/LOO and explicit warm-start/transition batch is implemented and verified:
all three handoffs are available, with unchanged unwrapped defaults. The full
analytics suite at that checkpoint passed **275 tests**, including 97 new cases. A bounded two-season
check, isolated wheel import and separate fresh-kernel notebook also passed.

- [League and warm-start practical guide](../docs/analytics/league_warmup.md)
- [Complete API and helper reference](../docs/analytics/league_warmup_reference.md)
- [Source-to-documentation coverage](../docs/analytics/league_warmup_documentation_checklist.md)
- [Separate minimal notebook](../notebooks/04_league_warmup_quickstart.ipynb)
- [Verification evidence](../docs/analytics/league_warmup_check.json)

Complete reusable orchestration before configuring the corner experiment. The
shared dataset contract now supplies aligned **X/y/metadata and declared prediction
layout**, match or team-match, with match/team/time/group identities.
The implemented labels provide quantities, perspective outcomes, strict thresholds
and explicit generic settlement. Split lines, parlays and bookmaker-specific
settlement remain outside the current label contract.

The fixed-configuration runner now connects assembled data and prepared splits
to training-only transforms, model adapters and held-out predictions. Remaining
layers include fold-aware persistent-state preparation, search/model selection
and scheduled refitting. Implemented pre-training reports consume the declared
layout and metadata; post-training reports add retained predictions and model
outputs. Saved-run comparisons consume persisted numerical records. Pilot recipe selection, tuning and
real-data fitting are deferred until these layers are ready.
See the [current implementation plan](../IMPLEMENTATION_PROGRESS.md) and
[working notes](../football_analytics_working_notes.md).
