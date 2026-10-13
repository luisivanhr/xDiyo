# Temporal team graph forecasts

The optional `xdiyo_analytics.graphs` package reconstructs a team graph at every
forecast issuance. It produces home/draw/away probabilities and reusable,
opponent-independent team embeddings. It has no learned team-ID vectors and no
persistent recurrent team state. The default dimensions and decay constants are
starting hyperparameters; the bounded pilot did not establish a predictive advantage.

An [optional heatmap CNN](heatmap_cnn.md) adds shared, jointly trained spatial edge
features. Its reconstruction branch is independently optional. Both are disabled
by default; the CNN also supports standalone classification and regression.

## Install and select a device

```powershell
python -m pip install -e ".[graphs,training,reporting]"
```

Install the appropriate CUDA-enabled PyTorch binary from the
[official selector](https://pytorch.org/get-started/locally/) when using an NVIDIA
GPU. The checked Windows environment uses Python 3.12, PyTorch 2.14.0+cu130,
CUDA runtime 13.0, an RTX 4070 SUPER (12 GB), and NVIDIA driver 610.88:

```powershell
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

The PyTorch binary supplies the runtime for this pure PyTorch implementation;
there are no custom CUDA extensions or PyG dependencies. This does not install
the separate CUDA Toolkit compiler (`nvcc`). Importing the base package or
`xdiyo_analytics.graphs` does not import PyTorch. CPU is the default;
`device="cuda:0"` requests a GPU explicitly and fails if unavailable.
`device="auto"` selects CUDA when available. `ExecutionPolicy` can configure
device and CPU threads through the existing runner. Thread limits are scoped
and restored. Seeds control initialization and fresh restarts; CUDA scatter
reductions may differ in final rounding and are tested with tolerances.

## Input contract

Use `build_team_history` on loaded matches and optionally selected statistics.
Every event needs two receiver-perspective rows with:

- exact positive integer `competition_id, season_id, event_id, team_id, opponent_id`;
- `side` equal to home or away, timezone-aware `kickoff_at`, and `status`;
- `goals_for, goals_against`, with missing values retained;
- `round`, and a stage column if round numbers repeat within a season.

Event identity is the competition/season/event triple. Team identity persists
across competitions and seasons. IDs never pass through a floating-point row
iterator. Duplicate perspectives, inconsistent paired outcomes/statistics,
self-matches, and fractional or negative scores are rejected. Repeated fixtures
with different event IDs remain separate match edges. Finished matches with
missing scores retain explicit score masks.

Supply authoritative `team_seasons` with `team_id, competition_id, season_id,
entry_at, movement`. `entry_at` is the effective season boundary, known at that
time. Use the library's movement records and explicit season metadata to prepare
it. Categories are retained, promoted, relegated, and unknown; `other_entry`
maps to unknown. Include intervening seasons even when match observations are
missing. Multiple domestic-league memberships for one team at the same boundary
are rejected. This first release models one current domestic-league context per
team; resolve overlapping cup/domestic memberships upstream.

Pass the complete fixture schedule to `TemporalEventStore(..., schedule=...)`
when history omits upcoming fixtures. The schedule has one row per event with
`home_id, away_id, kickoff_at`, competition/season/event keys, and round keys.
Without a separate schedule, home-perspective history supplies it. Public frame
properties return copies; inputs are never modified. A content fingerprint
covers history, membership metadata and schedule.

```python
import pandas as pd
from xdiyo_analytics.graphs import (
    TemporalGraphConfig, TemporalEventStore, TemporalGraphAdapter,
    TemporalGraphSerializer,
)
from xdiyo_analytics.training import TrainingControl, EarlyStopping

config = TemporalGraphConfig(timezone="Europe/Berlin", forecast_policy="by_round")
store = TemporalEventStore(history, team_seasons, schedule=fixtures,
                           provenance={"source": "reviewed local export"})
model = TemporalGraphAdapter(
    config, store, device="cpu",
    control=TrainingControl(max_steps=50, early_stopping=EarlyStopping(5), seed=17),
)
model.fit(fit_context)  # FitContext with match layout and one home/draw/away target
probabilities = model.predict(prediction_context)["predict_proba"]
```

The class order is always native home-perspective `1, 0, -1` (win, draw, loss).
Use `Outcome(perspective="home")`; text class labels are rejected. Probability
columns are the native `(target, class)` MultiIndex. Each returned frame has exactly
`context.X.index` and order. `predict` also returns the class label under the
fitted target's column name. Unselected X columns are ignored.

The native artifact format is `xdiyo.temporal-team-gat.v2`. The experimental
v1 text-label artifacts are not accepted by this format; create a fresh native
artifact after fitting. Existing experiment files are not rewritten.

## Forecast timing

An explicit IANA timezone is required.

**By Round** (`by_round`) issues one forecast at the beginning of each round.
The default issuance is the minimum kickoff in the complete supplied round
schedule. Grouping reuses the same minimum-anchor implementation as
`LeaguePopulation.round_info`. Keys include competition, season and round;
add a stage in `round_keys` for multi-stage competitions. Calendar weeks have no
role. A delayed fixture keeps its original round's issuance, even if played
months later. Selecting only that delayed fixture cannot move its anchor.

When a schedule revision would erase the original opening time, supply
`round_issue_at` on every schedule row. All copies within a round must agree.
The caller owns that known schedule anchor. An actual-kickoff retrospective
export cannot prove what an earlier schedule vintage contained.

**One Day Ahead** (`one_day_ahead`) defaults to kickoff minus 24 hours. An
explicit timezone-aware `issuance_at` can override this on a fixture.
Issuance cannot follow kickoff. By Round accepts an explicit issuance only
when it agrees with the frozen round anchor.

For issuance t, the exclusive history cutoff is local midnight at the beginning
of t's local date, converted to UTC. Finished matches before that boundary are
eligible under the operational guarantee that data through the previous
calendar day is available. Same-day and future matches cannot contribute.
Existing sparse reviewed `result_available_at` bounds carried by the library
are honored for interrupted/completed-later fixtures; no release-time collection
or new release infrastructure is required. Age is measured at issuance,
including the time between midnight and issuance.

The same policy applies to training examples, validation, fixture features, and
exports. Fitting and validation-selection labels must be known by the previous-day
boundary of every held-out issuance. A kickoff-only chronological split can
therefore be rejected if its issuance overlaps fitting. Inference never bypasses
this check; historical training representations are internal to fitting.

By Round freezes eligible graph history, age weights, priors, supports and rest
days at the round anchor. Fixture context can differ across matches, but must
have been known at that anchor.

## Match and fixture features

Default edge features are receiver goals for/against, receiver home/away role,
and original competition. Statistics are opt-in `build_team_history` columns:

```python
config = TemporalGraphConfig(
    timezone="Europe/Berlin",
    statistics=("team::ALL::Shots::shotsOnGoal::value",
                "opponent::ALL::Shots::shotsOnGoal::value"),
)
```

Both paired raw statistic columns must exist. Select actual match-level fields;
do not put rolling features on edges. xG is excluded. Numeric preprocessing uses
means/scales learned only from fitting examples and their eligible historical
inputs. Missing numerical values are filled with the fitting mean, yielding a
neutral standardized zero, and always receive a separate missingness mask.
Observed zero has mask zero. Competition maps are also fitted only on that
population; unknown categories have an explicit fallback.

### Explicit goal predictor selection

`include_goal_features=True` is the default and preserves the original ordered
numeric edge columns: `goals_for`, `goals_against`, then `statistics`. Set it to
`False` to use **only** the explicit statistics as numeric edge predictors:

```python
config = TemporalGraphConfig(
    timezone="Europe/Berlin",
    include_goal_features=False,
    statistics=tuple(
        f"{role}::{period}::Match overview::{stat}::value"
        for period in ("1ST", "2ND")
        for stat in ("goals", "shotsOnGoal")
        for role in ("team", "opponent")
    ),
)
```

Use the exact column names supplied by your prepared history; the example does
not create or reconcile period observations. Add any approved spatial covariance
channels explicitly to that same list. Each selected channel still requires its
paired receiver/opponent source column. Disabling goals with empty `statistics`
is rejected, even when a CNN is enabled. The setting must be a boolean.

This removes the two native goal predictor channels **and their masks**. It does
not modify canonical event-store scores, labels, completion status, available
history, edges, weights, side/competition inputs or the fixture head. Do not mask,
zero or rewrite canonical scores to suppress predictors. Optional causal fixture
inputs (including Bayesian inputs) keep their existing cutoff contract.

Missing period observations keep their native masks. Period-score reconciliation
belongs to data preparation: only independently verified regulation-time and
first-half goals may support a derived second-half goal count. Extra time,
penalties, suspicious zeros and unverified values must not be silently treated as
verified period goals. Other half statistics are not derived by subtraction.

In the builder, use **Model → Config → Include goal features**. Native JSON/Python
recipes and saved models retain the option. Old saved models without the field
load as `True`, with their existing model fingerprints unchanged. Saved ordered
feature lists remain validated; network width is inferred from the actual list
on either CPU or CUDA. Numeric edge width is twice the selected column count
(values plus masks), plus the home-side indicator and competition one-hot vector.

The fixture head receives competition, neutral venue, home/away rest days,
home/away discounted supports, and their numeric missingness masks. Rest is
elapsed time at issuance since the last eligible match; an unseen team has
missing rest. Neutral venue is 0/1/missing and must be supplied as known fixture
metadata.

Opt-in `fixture_features=("selected_name", ...)` reads only those X columns.
Build causal values with existing feature expressions and pass the graph's
exclusive history cutoff to `evaluate_features`. The context metadata must
carry `feature_cutoff_at` equal to that same cutoff for each row. This is an
auditable declaration, not an automatic proof that arbitrary caller values are
causal. No bank of rolling features is added automatically.

## Equations and synchronous layers

At t, team i starts from

```text
p_i(t) = p_global + u_current_league + v_current_season_entry
h_i^0 = p_i(t)
log w_ie(t) = -log(2) * age_days(e,t) / H + b_i(e,t) * log(rho)
S_i = sum_e w_ie
g_i = S_i / (S_i + kappa)
```

b counts actual receiving-team season boundaries after the match and at/before
issuance, using `entry_at`; there is no season-ID arithmetic. Every original
competition label is retained. Promotion/relegation changes current prior
context and discounts previous evidence without deleting history or changing
team identity. Retained entry is a zero reference. Unknown league uses the
global prior. League and entry adjustments initialize near zero and have
explicit quadratic regularization in addition to AdamW decay.

For each independent layer and head:

```text
s_ie = a^T LeakyReLU(Q h_i + K h_j + E x_ie)
alpha_ie = softmax_over_incoming_match_edges(s_ie + log w_ie)
m_i,head = sum_(j->i,e) alpha_ie * tanh(V h_j + U x_ie)
h_i^(l+1) = h_i^l + g_i W_l concat_heads(m_i,head)
```

All nodes read previous-layer states. Two layers mean two synchronous hops at
one snapshot, not two chronological match updates. Edge features enter values
as well as scores, so identical priors can produce different representations
after different outcomes. Repeated fixtures participate separately in softmax.
Empty neighborhoods have exactly zero correction.

Softmax combines weights in float64 log space, subtracts the receiver/head
maximum, then converts normalized weights to the model dtype. No age threshold
drops old edges. Extremely small absolute supports can naturally underflow to
zero, while relative attention remains well defined. Common aging cancels
from normalized attention; the separate support gate is what shrinks absolute
corrections. For fixed finite parameters and unchanged prior context, bounded
tanh messages give a correction bounded by g times a finite sum of output
matrix norms. Thus prolonged inactivity approaches the fixed prior. There is
no post-update normalization or dropout outside the gate.

The head concatenates `[z_home, z_away, z_home-z_away, z_home*z_away, c_fixture]`
with z equal to the last-layer state. Defaults: dimension 32, two layers, two
heads per layer, H=180 days, rho=0.5, kappa=5, and one 64-wide ReLU hidden layer
before three logits. All weights train jointly using weighted cross-entropy.

Fitting and validation observation weights are independently rescaled by an
exact power of two in float64 before conversion to the model's float32 dtype.
This keeps the largest stored weight between 0.5 and 1 and preserves relative
weights, including when the original positive weights are uniformly tiny or
their raw sum would overflow. No epsilon floor or clipping changes their ratios.
An excessively wide relative range that would erase a positive weight in
float32 is rejected explicitly. Zero weights remain zero. The denominator is
computed once per replay; optional reconstruction retains the issuance's share
of total fixture weight, independent of its historical edge count. Rescaled
objectives and gradients agree within float32 rounding. Existing artifacts are
unchanged; their inference does not recompute fitting weights.
One iteration is a full fitting-population gradient step; issuance groups share
one graph forward pass and backpropagate before the next group. Immutable
prepared graph tensors are reused, but embeddings are recomputed after every
optimizer update.

## TrainingRunner, selection and persistence

`TemporalGraphAdapter` implements `fit_controlled`, sample weights,
`configure_device`, and `configure_threads`. It uses the existing
`run_iterations` loop for early stopping, restarts, learning-rate scheduling
and best-checkpoint restoration. Validation uses inference mode without
preprocessing fitting or optimizer updates. Each restart owns new parameters,
optimizer and preprocessing.

```python
from xdiyo_analytics.training import TrainingRunner, ExecutionPolicy, save_model, load_model
from xdiyo_analytics.selection import Candidate

factory = lambda: TemporalGraphAdapter(config, store)
runner = TrainingRunner(
    factory, control=TrainingControl(max_steps=50),
    validation=validation_positions,
    execution=ExecutionPolicy(device="cuda:0", inner_threads=1),
)
result = runner.run(dataset, split_plan)
candidate = Candidate("temporal graph", factory, config={"graph": config.__dict__})
serializer = TemporalGraphSerializer(store)
save_model(result, "new-model-directory", fold_id=0, serializer=serializer)
loaded = load_model("new-model-directory", serializer=serializer)
```

ModelDataset needs a nonempty X selection for the existing runner, even when
the graph consumes no optional X features; a constant placeholder is sufficient.
Use a fresh factory per fold and candidate. Temporal checks reject inappropriate
nonchronological folds rather than treating them as causal evaluations.

The native serializer saves architecture/policy, weights, preprocessing/maps,
feature/class order, fitting/selection boundaries, representation version and
event-store fingerprint/provenance and training history. Native `save_model`
automatically embeds immutable history, membership and schedule tables using
ZSTD Parquet; `load_model` discovers this format automatically, including in the
UI. Artifacts can be large because full graph history is retained. Explicit
`TemporalGraphSerializer(store)` instead omits those tables and requires that
exact external store on load. Timestamp storage precision does not change the
fingerprint. Native calibrated adapters use the existing joblib wrapper format.
Tensor loading uses `weights_only=True`. Direct `save(adapter, directory)`
and `load(directory)` are available alongside generic save_model/load_model.

To add later observations, load against the original store and then call
`adapter.attach_event_store(extended_store)`. Existing events and memberships
must remain unchanged and frozen round anchors must be preserved. Revised
historical evidence requires a new fitting run. Loading is for prediction, not
resuming an optimizer mid-step. A new fit resets the backend.

## Exact-cutoff embeddings and downstream reuse

```python
queries = pd.DataFrame({
    "team_id": [team_a, unseen_team],
    "issuance_at": pd.to_datetime(["2025-08-01T12:00Z"] * 2),
})
export = model.encode_teams(queries)
```

Exports retain query keys/order, z coordinates, prior coordinates, support,
gate, context, exact issuance/cutoff, fitted boundary, model version, event-store
version, forecast policy and all potentially contributing match identities up
to the configured hop count. They include inactivity queries and contextual
priors for cold teams. No last-match snapshot is substituted for a later query.
This release provides direct exports rather than a RatingRun bridge; RatingRun
does not automatically age graph snapshots. There is no persistent embedding
cache. If caching externally, key on model version, event-store version,
forecast policy, exact issuance and requested team.

Use three chronological blocks for a separate estimator: fit/select the encoder,
freeze it and fit the downstream estimator, then evaluate. The maintained
[example](../../examples/temporal_team_gat.py) exports both downstream blocks
from the same selected encoder and enforces its temporal boundary. Never pool
raw coordinates from independently fitted encoders, even when dimensionality
matches.

## Bounded empirical example and limitations

See [the measured pilot](temporal_team_gat_pilot.md) for populations, six fixed
candidate fits, validation selection, test metrics and temporary artifact paths.

```powershell
python examples/temporal_team_gat.py --data-root PATH_TO_xDiyo_data --output NEW_EXTERNAL_DIRECTORY --device cuda:0 --max-steps 12
```

The example uses scores only and existing native 10-match rolling-goals features
with a logistic classifier as its baseline. It preserves all available history
within its declared 2010/11-2024/25 German two-division population. Full-history
compute and memory grow with edges and issuance groups; this implementation
does not claim archive-scale optimization. There is no neighbor sampling,
graph compression, hard age truncation, feedback probabilities or recurrent
memory. Saved publication hashes identify the actual data population.

The tests cover calendar cutoffs, paired/repeated edges, exact IDs, missingness,
numerical attention, two-hop influence, support decay, movement, cold teams,
joint gradients, validation isolation, weighted synthetic learning, native runner
folds, persistence, optional imports, CPU/CUDA behavior and provenance. Real
football superiority is not a software acceptance condition.

## Experiment builder and native reports

Select **Temporal team GAT** under Model & training. The setup button selects
native home-outcome labels and graph-owned masked scaling. Choose chronological
outer folds and chronological inner folds when tuning. Training and optional
validation labels must be known before the first evaluation cutoff; overlapping
rounds or an insufficient gap raise an error. The default model is a three-class
match-result classifier, not a corners model.

The nested configuration exposes every architecture, decay, statistic, fixture
input and optimizer setting. TrainingControl exposes steps, seeds, restarts,
early stopping and scheduling. Device and threads use the existing ExecutionPolicy.
Candidate validation, fitted class-weight reporters, probability calibration,
final refits, run reuse and model loading use the native pipeline.
Mid-fit checkpoint resume is not implemented by this adapter: CheckpointPolicy
uses its existing skip/raise behavior. Completed models remain recoverable.
Graph-only construction requires the graphs extra; plotting additionally requires
reporting. Catalog discovery remains available without importing Torch.

Memberships normally come from the selected dataset's team_seasons table. The
**first_fixture** policy explicitly uses each competition-season's first fixture
as its effective entry boundary. This is retrospective schedule context, not a
claim about publication time. Use **explicit** plus a CSV/Parquet file containing
entry_at for independently supplied boundaries. Include intervening memberships
when the selected history has gaps. A single current domestic-league context per
team is supported. Missing or ambiguous membership inputs fail clearly.

Graph preparation aligns optional assembled fixture predictors to the graph's
exclusive previous-day cutoffs. Generic cutoff overrides and top-level scaling
are rejected. Selected raw statistic columns must include both team and opponent
perspectives. Unselected assembled predictors are ignored. Architecture and
optimizer settings can be tuned through native grid paths such as
`model.params.config.params.learning_rate`. Membership or issuance-policy
changes require separate preparation/recipes, not a shared grid population.

Add these under **Post-training analysis**:

- **GATLossReporter**: stored training and validation cross-entropy per step,
  fold and restart attempt. Training loss is measured before that step's update;
  validation after it. These exclude the prior penalty and AdamW decay. No
  validation curve is invented when none was collected.
- **EmbeddingReporter**: **tSNE** by default, or **PCA**. Configure the projection
  in the builder. Names and available local badges identify teams; raw embeddings,
  projection coordinates, explained variance or tSNE diagnostics, historical
  support and prior/message gates remain downloadable. Small populations cap
  perplexity at n-1; identical representations have no invented projection.
  The rendered report has a fold dropdown when multiple folds are available,
  a square canvas with equal axis scale, and league buttons that hide points,
  names and badges together. Names are optional (off initially); hover still
  identifies each team. Use the team finder to zoom to a team, drag to pan,
  adjust badge size, or reset the view. Badge sizes stay constant while zooming.
  Missing or unreadable badges fall back to colored markers. These controls
  change presentation only; they do not recompute the projection or model.
  Reset always restores the original bounds, including after repeated searches
  and pan/zoom operations. Coincident teams keep their actual coordinates. Plotly's duplicate
  image-key behavior is avoided using imperceptible differences in badge width;
  no projection coordinates are jittered. Find team brings its badge forward.
  Native SVG downloads embed visible badges for offline
  viewing, detecting SVG/PNG content independently of filenames. Native catalogs
  embed local assets directly; custom relative image sources must be readable
  by the browser (serve such custom reports over HTTP if file access is blocked).
- **GraphAttentionWeightReporter**: use the rendered league and team dropdowns
  to inspect the strongest historical edges separately by layer/head, and
  download all weights. Concentration, entropy, effective edge count, support
  and gates give additional diagnostics. Attention is not causal attribution.
  Selected edges appear chronologically, newest on the left. Every team in
  the evaluation population is selectable, including teams without eligible
  edges, which show an explicit empty state. League grouping uses membership
  at the selected issuance; known league names come from evaluation metadata.
  Layer/head legends appear above the plot, separately from fixture labels.

Embedding and attention views use teams from the selected test/score population
at one common issuance per fitted fold (the latest selected issuance by default).
An explicit issuance must occur in that population. These reports reconstruct
cutoff-safe graph state using frozen weights; they never refit the model. Each
fold remains a separate coordinate system. Mean pooling is rejected. tSNE
neighborhoods are exploratory; coordinates and global distances are not rankings.
Badges use the same TeamCatalog and local assets as other native reporters.

A builder-importable example is provided by
[the recipe helper](../../examples/temporal_team_gat_recipe.py). It only writes a
recipe; it does not train or change any existing experiment.

### Bounded preparation and execution memory

Native GAT execution retains ordered issuance descriptors, copied fixture metadata,
selected fixture-head inputs and the immutable event store. It reconstructs one
raw snapshot and tensor group at a time for fitting, validation and prediction.
There is no graph/tensor cache, device offload pool or learned-embedding cache.
Existing builder recipes and native v2 model artifacts need no configuration change.

Preprocessing first scans fitting issuances one at a time to collect the same
snapshot league categories and aligned fixture numeric rows. Edge scaling still
fits on the **unique eligible fitting history**, not duplicated edges across
issuances. Category sorting, scaler arithmetic, missing masks, dtypes and fixture
row order are unchanged. The fixture numeric array is small and is discarded after
fitting preprocessing. Validation inputs are checked in a bounded pass before
optimization, without fitting their preprocessing or retaining their tensors.

Every training iteration zeros gradients once, consumes the original chronological
group order, uses the same full-population observation-weight denominator, adds
the prior penalty once and takes exactly one AdamW step. Training cross-entropy is
still measured before that update; validation is evaluated afterward under no-grad.
Early stopping, restart selection, scheduling and restored checkpoints retain their
native semantics. Reconstructing a graph does not change its cutoffs, contents,
round anchors, weights, node/edge order or model architecture.

Embedding and attention inspection also finish and release one issuance before
starting the next. Helper frames prevent the previous tensors or autograd graph
from lingering while a new graph is built. Errors and KeyboardInterrupt preserve
the traceback stack while clearing completed frames' temporary locals, so retaining
an exception does not pin a graph. Inference restores the caller's module mode even
on error. Results still accumulate their requested output rows: a very large
attention table itself can therefore require significant memory.

No source columns, inspection evidence or immutable-store fingerprint inputs are
projected away. Native save/load remains v2 and saves no runtime graph populations.
It is still a prediction artifact, not an optimizer-resume checkpoint. Private
`train_groups`/`validation_groups` are now restartable, indexable streams rather
than lists; explicitly calling `list(stream)` materializes all groups and defeats
the bound. Native callers never do that.

Live graph memory is bounded by the largest issuance graph and its autograd work,
plus fixed source/model/optimizer storage and fixture descriptors/output rows. It
no longer scales with the sum of repeated historical edges. This does **not**
guarantee any particular production fold fits: the largest individual graph,
source schema, output size and other processes still matter.

#### Exactness and lifetime checks

Tests compare the streamed path against frozen adapter and inspection sources
from `f8962af`, stored under `tests/analytics/reference`. They check:

- Preprocessing, category order, missing masks, group positions, tensor bytes,
  gradients immediately before every optimizer step, parameters and AdamW state
  immediately afterward, losses and restored best-state summaries.
- Both forecast policies; grouped and interleaved fixtures; delayed availability;
  movement-only league categories; cold-only training; missing/zero statistics;
  declared fixture-head inputs; weighted and zero-weight observations.
- Prediction, embeddings, attention, native embedded-store save/reload, frozen
  inference state/mode, repeated stream passes, caller-input mutation isolation,
  future-result mutation and a 100-day postponement with its original round anchor.
- Weak-reference checks during initialization, updates, validation, prediction,
  embeddings, attention and restarts. Previous snapshots and input tensors are
  gone before the next snapshot is built. RuntimeError and KeyboardInterrupt
  cleanup is checked while the exception traceback is deliberately retained.

The targeted suite passes **23 tests**. The wider graph, orchestration, UI,
persistence, calibration and reporting run passes **195 tests**, with **4 optional
LightGBM tests skipped** because that extra is absent.

#### Fresh-process synthetic resource measurements

`examples/benchmark_gat_streaming.py` compares the pinned eager adapter against
streaming using identical source data and configuration in separate processes.
The three populations share the same immutable store fingerprint, 2,000 historical
matches, 2,048 scheduled future fixtures and eight selected statistic channels.
There are 4,000 directed edges in each live graph. Seed 17, dimension 32, two layers,
two heads, half-life 180, CPU, one Torch thread; Python 3.12.14, pandas 3.0.6,
Torch 2.14.0+cu130. Each run initializes, performs one complete update including
validation, then predicts 24 issuances. No research experiment was run.

Decimal GB; training plus validation edge counts exclude another 96,000 prediction
edges. Validation contains 24 issuances in all three runs.

| Training issuances | Train + validation edges | Eager peak RSS | Streaming peak RSS | Eager resident tensor bytes | Streaming resident tensor bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 120 | 576,000 | 0.934 GB | 0.846 GB | 66,882,816 | 0 |
| 480 | 2,016,000 | 1.122 GB | 0.848 GB | 234,089,856 | 0 |
| 960 | 3,936,000 | 1.382 GB | 0.848 GB | 457,032,576 | 0 |

The 960-issuance run reduced peak RSS by **38.6%**. Streaming retains no prepared
graphs between operations; the largest transient prepared group was 464,464 bytes
in these inputs, excluding autograd activations. Prepared inputs, preprocessing,
updated model identity, optimizer state, losses and predictions have matching
hashes for every paired run.

At 960 issuances the per-phase sampled peaks (every 2 ms) and elapsed times were:

| Phase | Eager peak RSS | Streaming peak RSS | Eager time | Streaming time |
| --- | ---: | ---: | ---: | ---: |
| Preprocessing / initialization | 1.379 GB | 0.711 GB | 10.60 s | 8.79 s |
| Fitting, excluding validation | 1.357 GB | 0.848 GB | 11.00 s | 19.29 s |
| Validation | 1.357 GB | 0.848 GB | 0.12 s | 0.33 s |
| Prediction | 1.369 GB | 0.848 GB | 0.36 s | 0.34 s |

Process high-water RSS is recorded separately from the sampled phase peaks.
Rebuilding each graph each iteration is slower: the fitting phase here took about
**1.75 times** as long. These are single runs, not statistical speed estimates.
Those measurements predate the bounded source-row encoding cache described below;
they do not measure its speed or memory.

The full 41,732,050-edge production fold and 16,803,918-edge prediction population
remain **unmeasured**. The support bundle contains actual-cohort evidence for the
previous repair but no input Parquets for reproducing that experiment. These tests
establish bounded native graph lifetimes and synthetic scaling, not the complete
research fold's measured peak or runtime.

Evidence: [120 issuances](data/gat_streaming_120.json),
[480 issuances](data/gat_streaming_480.json),
[960 issuances](data/gat_streaming_960.json). The previous release-on-conversion
repair's initialization-only results remain in
[the 300-issuance profile](data/gat_preparation_memory.json) and
[the 1,806-issuance profile](data/gat_preparation_memory_1806.json); these describe
that older eager implementation, not the current streaming path.

```powershell
$env:PYTHONPATH = 'src'
python examples/benchmark_gat_streaming.py --issuances 120 --output streaming-120.json
python examples/benchmark_gat_streaming.py --issuances 480 --output streaming-480.json
python examples/benchmark_gat_streaming.py --issuances 960 --output streaming-960.json
```

The benchmark requires development-only `psutil`. Before allocating a population,
it estimates additional memory and refuses a run above 65% of available RAM.
This conservative benchmark guard is not a production memory-limit guarantee.

### Source-row preparation and loss-transfer optimization

The streaming path retains one CPU float32 source-edge encoding cache per fitted
preprocessor, shared by training, validation and prediction. Its combined value
and readiness arrays are capped at 128 MiB. Sources exceeding that cap use the
original per-snapshot encoding. No complete graphs, device copies or learned CNN
outputs are cached. Resident preparation memory therefore scales with source
rows plus the largest live graph, rather than the sum of overlapping issuances.

Rows are encoded lazily when eligible: future invalid observations cannot break
an earlier forecast. Encoding still uses float64 scaler arithmetic followed by
the original float32 cast. The cache identity includes the immutable store object
and content version, full graph configuration, ordered feature columns and fitted
preprocessing state. Store replacement, scaler changes and refitting invalidate
it. It is transient and absent from saved model artifacts and model fingerprints.

Store-level timestamp arrays and season-entry indices replace repeated boundary
scans. Each issuance still reconstructs its exact eligible row order, node
population, memberships, decay, support and rest days. Memberships use the full
issuance timestamp; two issuances sharing a midnight cutoff are not equivalent.
Exact team IDs are indexed once into compact source codes, avoiding repeated
Arrow scalar conversion for every edge. The original IDs are never cast to
floating point or restricted to signed 64-bit values. Latest eligible kickoff
times are reduced using the same receiver indices; rest days retain the original
scalar timestamp arithmetic, including its nanosecond rounding behavior.
Fixture column reads use temporary frames without attrs to avoid copying large
provenance dictionaries onto each Series. Returned metadata keeps its original
attrs and independent nested copies. These temporary frames are not cached.
Prepared graphs copy only the metadata and optional heatmap channels needed for
that issuance. Public snapshots retain their complete original edge columns.

Each group still performs its own backward call with the same global weight
denominator, followed by one prior backward and one AdamW update per pass.
Detached scalar losses are transferred together once per pass, then accumulated
as Python floats in their original order. Post-update training loss, validation,
selection, checkpoint metrics and reporting cadence are unchanged.

Optional diagnostic timing does not enter recipes or model identity:

```python
from xdiyo_analytics.graphs import GraphProfile

with GraphProfile(device="cuda:0") as profile:
    adapter.fit(context)
print(profile.timings)
```

The profile records snapshot/index construction, row encoding, tensor transfers,
forward/backward, loss transfer, optimizer, fingerprint and metric replay phases.
Native serializer saves are recorded as `checkpoint_export`; external checkpoint
observers can use `graphs.profiling.phase("checkpoint_export")`. CUDA boundaries
are synchronized, which introduces overhead. Phases can overlap through nesting,
so their totals must not be added together. Compare unprofiled wall time separately.

`examples/benchmark_gat_speed.py` benchmarks a completed scalar expanded-totals
fold against frozen `11ffa446` code. It reads saved preprocessing and weights,
checks model commitments, input hashes and model/store identity, then performs
disposable updates outside the study. It never resumes or rewrites the study.
Use separate processes for reference/native memory comparisons and pass Ayre's
support ZIP via `--evidence` to enforce its pinned hashes as well.

See the [measured speed audit](gat_speed_11ffa446.md) for timings, memory,
profiling overhead and exact-versus-nondeterministic CUDA comparison limits.
