# Deepthroat-gg research checkpoint

Research branch: `codex/deepthroat-gg-research`.
**Exact verified base:** `d38ae3192067f73df26f1c2f5baa8f1d148d215a`.
The branch is isolated from the uncommitted attention-reporter work in the
main checkout. It is not merged. No production data, frozen experiments,
collectors or GAT fits are changed by this implementation.

## Attribution and design decisions

Based on **DeepFoot-KNN**, Berrar, Lopes and Dubitzky (2024), *A data- and
knowledge-driven framework for developing machine learning models to predict
soccer match outcomes*, DOI [10.1007/s10994-024-06625-9](https://doi.org/10.1007/s10994-024-06625-9).
The supplied local article was inspected, including PDF pages 6–7, 10, 14–21
and 26–27. This is a documented research interpretation, not verified parity
with the authors' original code.

Paper-specified behavior: six inputs, own last-n-match form totals, normalized
rank `(N-R)/(N-1)`, n=9..100 selected by signed Pearson correlation, then separate
k=3..350 searches for score RMSE and RPS; at least six prior matches per team.
The article's prose mentions averaging ranks, whereas its worked table uses
the normalized form-table rank directly. The agreed implementation uses the
table interpretation: **no outer rolling average of rank**.

The user approved this explicit membership convention: rank the current
competition-season roster known at the prediction cutoff. Relegated teams
leave the table, but their competition-specific history is retained. Returning
teams recover that history across absences. New teams rank on their available
results (zero totals before any result); fixtures with fewer than six complete
prior results for either team are excluded from the case base. Membership is
an explicit table `(competition_id, season_id, team_id, known_at)`. `known_at`
must reflect evidence or a declared research assumption, never silently
inferred from a final roster. Exact points/GD/GF ties share the average ordinal
rank. These are our choices where the paper leaves details unclear.

Each team uses its own latest n eligible results, across seasons but within
its competition, pooling home and away appearances. Short histories of 6..n-1
use their actual count as the mean denominator. Incomplete paired goal scores
are excluded from both teams' windows. A result is released to both
perspectives only when both release times are known; the later time controls.
Native historical eligibility still requires a finished match, kickoff
strictly before the cutoff, result availability no later than the cutoff, and
exclusion of the target match. Omitted availability uses the native
retrospective kickoff proxy with reviewed completion bounds. That is an
assumption about availability, not demonstrated live information timing.

## Architecture and public entry points

`research.deepthroat_features.prepare_bundles` reuses native `MatchScore`,
`RollingMean`, historical eligibility, `MatchGoals`, `Outcome` and match-layout
`assemble_dataset`. It adds a separate cross-season form-rank producer;
`DerivedStandings` and its season/publication rules are unchanged. Rank queries
are shared across n values and bounded to the maximum requested history.

The recorded predictor order is exactly:

1. `home::mean_scored`
2. `home::mean_conceded`
3. `home::form_rank`
4. `away::mean_scored`
5. `away::mean_conceded`
6. `away::form_rank`

`research.deepthroat_gg.DeepthroatGGRecipe` has JSON and Python exports and
accepts a native team history and the explicit roster table. One configuration
and two native fitted models are created per competition. Training uses
`EstimatorAdapter`/`TrainingRunner` and native model persistence. The small
outcome adapter fixes the probability schema to home win/draw/away win
`[1, 0, -1]`, filling a class absent from fitting with zero probability.

This recipe is a separate research coordinator, rather than an ordinary UI
parameter grid: a grid optimizing n and k together would implement the wrong
selection procedure. The **generic** sklearn KNeighborsClassifier and
KNeighborsRegressor are available in the existing UI/catalog and normal
JSON/Python/notebook recipes. They support usual distance, weighting, search,
neighbor, leaf-size and CPU-thread parameters; preprocessing remains fold-local.
Training sample weights fail explicitly. Metric parameters use the native map
editor (scalar entries); advanced callable/vector-valued metric settings remain
available through the sklearn API. Precomputed distances need a separately
aligned distance-matrix workflow, not ordinary feature columns.

## Selection and evaluation

For each n, using development rows only:

```text
z = (home_mean_scored - home_mean_conceded)
    - (away_mean_scored - away_mean_conceded)
    + home_mean_scored - away_mean_scored
```

Select maximum **signed** Pearson correlation with observed home-minus-away
goals. Constant/insufficient candidates are recorded as rejected. Then fix n
and select k separately by:

```text
joint score RMSE = sqrt(mean(home_error**2 + away_error**2))
RPS = mean(sum((cumulative_probability - cumulative_outcome)**2, first 2 classes)/2)
```

Retrospective leave-one-match-out queries exclude the exact query index even
with duplicate vectors. k>N-1 is recorded as infeasible. All exact objective
ties are recorded, choosing the smallest candidate. One maximum-k sklearn
KD-tree query supplies cumulative sums/frequencies for all k. At equal
distances the installed sklearn KD-tree's maximum-k ordering defines the
prefixes; case order and dependency versions are retained. This tie convention
is not claimed to match FNN or rebuilding a separate tree for every omitted
match. Final fitted KNN models use sklearn's ordinary tie behavior (class-vote
ties follow its sorted class order); this is another explicit implementation
detail, not a paper-established choice.

Research assumptions: **Euclidean distance, uniform neighbor weights, no
additional scaling**. Final goal forecasts remain decimal neighbor means.
No integer rounding is applied to the metric or saved predictions. A future
paper-style rounded score display must declare its rounding rule separately.

The chronological holdout starts strictly after `development_end`; only
development labels released by that boundary can select n/k or fit the model.
Scored holdout labels must be released by `holdout_end`. An earlier holdout
result can update a later fixture's causal features, but never reselect or
refit the frozen development model. Retrospective selection scores and
chronological holdout scores are stored separately. No claim of forecasting
quality follows from the synthetic checkpoint.

## Working recipe

```python
import pandas as pd
from xdiyo_analytics.data import load_seasons
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.research.deepthroat_gg import DeepthroatGGRecipe, save_research

# Explicitly chosen development/holdout sources; this example is not auto-run.
data = load_seasons(data_root, seasons, leagues=[league], tables=('matches',))
history = build_team_history(data)
roster = pd.read_parquet(reviewed_roster_path)  # Includes explicit known_at.
recipe = DeepthroatGGRecipe(
    development_end='2023-06-30T23:59:59Z',
    holdout_end='2024-06-30T23:59:59Z',
    # available_at='released_at',  # Use when verified timestamps exist.
)
results = recipe.run(history, roster)
save_research(results, recipe, history, roster, new_output_directory)
```

`save_research` writes immutable-by-convention research inputs (ZSTD Parquet),
recipe JSON/Python, selected feature and metadata tables, full n/k search
tables, native goal/outcome model artifacts, selected n/k, all objective ties,
development/holdout positions, policies, class order, input/artifact hashes,
all library Python source hashes and Python/dependency versions. It requires a
new output folder and never overwrites an earlier study.

## Verification and research log

- Design checkpoint: read local PDF, verified base, isolated branch; existing
  attention work preserved. User then approved the current-season roster policy.
- Tables 3/4 independently verify City: 16 points, GF14, GA4; Liverpool: 13
  points, GF13, GA1. These establish their ordering, not league ranks 2 and 3
  without the rest of the roster.
- Independent replay checks compare own-window means/totals/ranks with
  incremental add/evict arithmetic, including delayed releases and postponement.
- Tests cover source isolation, returning/new teams, roster availability,
  incomplete pairs, unknown release, simultaneous/frozen round cutoffs,
  future perturbations, signed correlation, exact candidate ties, infeasible k,
  self-exclusion, missing outcome classes, multi-output regression and exact
  saved-prediction and recipe round trips. No new sklearn estimator is
  implemented: generic models remain sklearn's own estimators.
- Browser verification exercised both KNN selectors, distance field,
  optional metric-parameter map and thread control; the downloaded Python
  recipe retained `n_neighbors=8`, `weights='distance'`, `metric='euclidean'`.
  It caught and repaired an inherited generic field-name decoration that
  initially confused distance metrics with model evaluation metrics.
- Test XML and final hashed verification receipt are beside this document.
  Two existing `test_match_score` failures were reproduced on the exact base
  snapshot: full-frame equality predates added half-score columns; a pandas 3
  fixture assigns a nanosecond timestamp into a microsecond column. They are
  explicitly retained as inherited failures, not skipped or weakened here.

Final checks: **165 passed, 2 inherited failures** in the 167-test pandas 3
regression set; **91 passed** in the pandas 2 research/UI/standings set. The
20 new tests pass in both environments. Exact commands, versions, hashes and
the two base reproductions are recorded in `deepthroat-gg-verification.json`.

Changed-file groups:

| Files | Purpose |
| --- | --- |
| `src/xdiyo_analytics/research/__init__.py`, `deepthroat_features.py`, `deepthroat_gg.py` | Separate feature producer, two-stage selection, recipe and native models/artifacts |
| `src/xdiyo_analytics/ui/catalog.py`, `build_inventory.py`, `inventory.json` | Generic KNN catalog and UI controls |
| `tests/analytics/test_deepthroat_gg.py`, `test_ui_knn.py` | Numerical, temporal, model, export and persistence regressions |
| `examples/deepthroat_gg_synthetic.py` | Reproducible bounded benchmark and artifact generation |
| `docs/analytics/deepthroat-gg*` | Research log, test receipts, benchmark and browser evidence |

## Bounded run and next experiment proposal

`examples/deepthroat_gg_synthetic.py --output <new-folder>` exercises the full
92-by-348 sequential search on 144 synthetic fixtures, six teams and two
seasons. See `deepthroat-gg-benchmark.json` for measured time, sampled process
RSS, exact source hashes and selected configurations. Memory is sampled at
10 ms, so it is a measured lower bound on instantaneous peak RSS.

The final run took **3.53 seconds**: 3.19 s preparation, 0.16 s selection/fitting/
holdout scoring, 0.18 s persistence. Peak RSS was **183.6 MiB**, **13.2 MiB**
above its imported-runtime baseline. It selected n=30, goal k=72, outcome k=51
from 87 development fixtures and scored 38 held-out fixtures. These numbers
describe the synthetic fixture only.

The first **proposed, not launched** real study is one competition, at most
5,000 fixtures, an explicitly reviewed membership/availability table, one
chronological final-season holdout, n=9..100 then k=3..350, one CPU worker.
Review dates and included seasons before execution. Float64 six-feature bundles
alone cost about 22 MB at 5,000 fixtures; 350-neighbor indices and cumulative
two-goal/three-class arrays add roughly 84 MB before temporaries, pandas metadata,
native indexes and fitted models. Full process memory is larger. KD-tree query
time is data-dependent; the tiny synthetic time is not a reliable real-data
estimate. Start with a 1,000-fixture preparation/profile checkpoint, measure
time/RSS, then decide whether the 5,000-fixture study fits the agreed budget.

No real-data tuning campaign has been started. Membership evidence and exact
experiment scope remain the prerequisites for that next step; authors' code
parity and real predictive benefit remain unverified.
