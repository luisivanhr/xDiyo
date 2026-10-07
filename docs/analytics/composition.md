# Model composition and optional research staking

For explicitly assumed historical quote times, see
[research quote availability](research_quote_availability.md). Observed evidence
remains the default; the opt-in contract retains a separate timestamp and audit.

This addition uses the existing `ModelAdapter`, `TrainingRunner`, dictionary
recipe, catalog, ticket and report contracts. It lives on
`codex/model-composition-stake-policy` for review. It does not change existing
single-model recipes or introduce a second competing recipe model field.

## Runnable examples

From the repository root, in the existing Python environment:

```powershell
$env:PYTHONPATH='src;.'
python examples/composition_and_staking.py
```

The script uses synthetic data only. It demonstrates a regression mean, a
probability blend, regression and classification stacks, both residual modes,
whole-model save/load, unchanged fixed stakes, a complete-ticket AND gate,
single-model Kelly, asynchronous bankroll replay and a saved learned allocation
model. Capital, probabilities and fractions are illustrative. No experiments
against library data or financial executions take place.

## Model contracts

Import from `xdiyo_analytics.composition`:

| API | Purpose |
| --- | --- |
| `OutputSchema`, `OutputRef` | Declare target, class order, units, perspective, row keys, link, horizon, distribution and timing semantics; identify an output explicitly |
| `ModelNode` | Child model, output schemas, optional calibration and local class weighting, or a verified frozen artifact |
| `TransformNode`, `OutputFeatures` | Stateless named output concatenation, optional explicit original-feature passthrough and probability-to-logit conversion |
| `ReducerNode`, `Mean`, `HardVote`, `DistributionMixture` | Fixed typed aggregation |
| `PredictionGraphSpec`, `CompositeModelAdapter` | Validate and execute a static DAG through the native adapter interface |
| `ModelEnsemble` | Convenience constructor for parallel same-schema children plus a reducer |
| `ModelStack` | Convenience constructor for bases followed by one honest chronological meta layer |
| `Estimator` | Cloned sklearn child with its own preprocessors, prediction methods and optional target transformer |
| `TrainingPlan`, `TargetSpec` | Explicit fitting schedule, availability policy and target meaning |
| `ResidualModel` | Algorithmic correction or honest OOF error-meta learning |
| `UtilityTarget`, `LearnedTargetAdapter`, `SavedUtilityModel` | Explicit training-label rule, fitted utility learner and hash-pinned local artifact reference |

Adapters continue returning dictionaries of row-aligned DataFrames. Probability
columns have `(target, class)` identities in the declared class order. A shared
row count is insufficient: metadata identities, index/order and perspective must
match. Children fail explicitly on absent classes or outputs; there is no
undocumented class padding. Ensemble child outputs use names such as
`first/predict_proba`. General graphs can expose any declared output through
their public mapping.

`Mean(kind='probability')` averages event probabilities with finite nonnegative
weights. `HardVote` yields labels or explicitly typed vote fractions; its output
cannot masquerade as `predict_proba`. `DistributionMixture` currently supports
complete finite-support categorical PMFs only. Use existing distribution-to-event
adapters for NB/Poisson events; averaging their parameter vectors is not a mixture.
Every distribution edge, including a direct public output or restored schema,
requires `family='categorical_pmf'`, `parameters=('mass',)`, probability units,
identity link, distinct nonmissing support, matching column order, and finite
nonnegative mass summing to one (absolute tolerance `1e-10`). Other distribution
families require an explicit event adapter and are rejected here.

Child sklearn estimators and preprocessing are cloned for every fit, including
inside target-transform and device wrappers. A retained `CalibratedAdapter` is
rejected for refitting: declare its unfitted base with `ModelNode.calibration`
to rebuild calibration on each inner population, or explicitly freeze the
retained artifact with compatible provenance. Other retained native adapters
with learned attributes need an unfitted specification or a frozen artifact.
Custom fresh adapters remain responsible for resetting their own internal state.
Standalone graph transforms are
stateless `OutputFeatures`; put learned scalers, imputers and selectors inside
each `Estimator.preprocessors`. A learned combination of predictions is a meta
model, not a fixed `Mean` with secretly trained weights.

## Chronological training

Outer folds remain entirely under `TrainingRunner`. A typical inner plan is:

```python
TrainingPlan(
    mode='chronological', n_splits=3, min_train_groups=8,
    label_availability_column='available_at',
    feature_availability_column='features_available_at',
    embargo='0h', uncovered='drop', repeated='error',
    deployment='refit_bases', max_fits=1000,
)
```

An explicitly declared `availability_delay='3h'` can replace the label timestamp
column when appropriate for a research example. It is a proxy, not verification
of official result timing. `features_as_of_issue=True` similarly attests that
upstream feature construction was separately audited; it cannot make a raw
current-match statistic safe.

`issue_column` defaults to `time_column`, which defaults to `kickoff_at`.
Whole fixture partners remain together, including team-row datasets. `group_by`
can additionally keep league/season/round groups together. Equal issue-time
anchors stay in the same batch. `min_train_groups` counts distinct chronological
group anchors. Labels must be available at the inner fit boundary; embargo moves
that boundary earlier. A group with a late or missing outcome is excluded from
that inner fitting population. Original dataset row keys survive subsetting.

Each base, preprocessing pipeline, child weighting policy and calibrator fits
within its inner training boundary. `ModelNode.calibration` reuses the existing
temporal `ProbabilityCalibrator`; reserved calibration rows cannot include the
OOF rows being predicted. Set `ModelNode.weighting=ClassWeightPolicy(...)` for
local class balancing. Outer `features_from`, `weights_from`, or learned class
weighting are rejected for chronological meta learners because fitting those
once over the full outer training population would contaminate OOF construction.
Fixed, predetermined feature columns remain valid.

Initial rows without OOF history are counted and dropped, or rejected with
`uncovered='error'`. They are never filled with in-sample predictions. Repeated
OOF predictions and learned depth beyond bases plus one meta layer are rejected.
Meta labels must be available at outer `fit_at`. Bases are then refitted on the
eligible outer population and the OOF-trained meta model is frozen. This creates
an OOF-versus-full-refit input-distribution mismatch; the audit records it.
Prediction requests preceding the deployment fitting cutoff are rejected.
The same label-free timing validator checks original, outer-test and restored
predictions: issue cannot follow kickoff, and a declared feature-availability
timestamp must be present and no later than issue. Test labels and their release
times are not required for inference. Composition prediction contexts remove
the library's named settlement/outcome and training-label metadata. This does
not certify arbitrary custom feature semantics or an adapter's external data use.
Native retrospective `status` and `is_awarded` are also removed, since native
bet-label construction uses them for void/award outcomes. The original dataset
metadata remains available to reporting. Deliberate status-derived predictors
must use separately audited as-of features and the normal feature-availability
contract; changing a metadata field's name does not establish that it is safe.

Frozen children need `artifact_id`, `artifact_vintage`, `trained_through` and
matching `training_summary_['frozen_provenance']` retained with the fitted model.
Cutoffs and vintage are checked against each prediction issue time. This is a
provenance contract, not external attestation of a model's training history.

## Residual and utility targets

`ResidualModel` deliberately supports two distinct modes:

- `algorithmic_residual` with `TargetSpec('residual')`: fit a base and a correction
  to its in-sample residuals inside the outer training fold. This is an ordinary
  two-stage boosting-style procedure.
- `honest_error_meta` with `TargetSpec('oof_error')`: train the correction on
  chronological OOF errors, refit the base on eligible development rows, and
  freeze the correction.

Regression uses `y - prediction`. Binary classification requires
`loss='log_loss', link='logit'`, uses the pseudoresidual `I(y=class_1)-p`, fits a
regression correction, and reconstructs with
`expit(logit(clipped_base_p) + learning_rate * correction)`. It does not take the
logit of binary labels or add unconstrained corrections to probabilities.
Multiclass residual correction is explicitly unsupported.

`UtilityTarget.build(outcomes, cutoff=...)` constructs labels from a separate
outcome table only after official availability. `take_skip` marks net return per
unit above a declared threshold. `allocation` scales positive excess return and
caps it at a declared capital fraction. The latter is a concrete supervised
research objective, not an assertion of optimal portfolio sizing. Alternative
explicit objectives may supply their own finite targets.

`LearnedTargetAdapter` requires a selection identity, historical issued/OOF or
fixed-rule provenance, training cutoff and available utility labels. For learned
selection, use predictions issued without those outcomes. Train/evaluate these
models using separate outer folds just like any other learner. Threshold or
allocation tuning belongs inside training. `LearnedGate` consumes the declared
take/skip model; `LearnedAllocation` consumes its allocation counterpart.

The UI can reference a fitted utility model via `SavedUtilityModel(path,
manifest_sha256)`. It uses the existing trusted local artifact loader, validates
the manifest hash and payload hashes, and never fits or fetches a model.
Constructing the historical utility training dataset is an explicit preparation
step; the football builder does not infer utility targets from ordinary labels.

## Independent decisions

`DecisionLayer` accepts one or more **uncombined** probability streams and a
`DecisionContext`. `FrozenTable` copies immutable scalar/tuple values and excludes
known outcome-bearing fields. Do not disguise outcomes under different names:
arbitrary feature semantics remain the caller's responsibility. The new native
ticket path builds its decision table from an explicit allowlist and retains no
reference to the reporting context. Existing manual callbacks remain trusted,
unrestricted legacy code and have not become leakage-proof.

Candidates share economic keys, quote identity/time, decision time and decimal
odds. Model valuations must match those exact terms. Issue time, artifact vintage
and trained-through time are distinct; all must be valid at the original issue.
AND/OR gates operate on leg probabilities or complete-ticket binary EV. Strict
`>` is the default; `strict=False` deliberately chooses `>=`. Missing evidence
errors by default; explicit `missing='reject'` records an abstention. An AND gate
cannot pass a missing approval; OR requires at least one actual approval.

`AllCombinations(probability_columns={'a':'a::probability', ...},
ticket_gate=DecisionLayer(...), probability_mode='independent', payoff='binary')`
values the same completed tickets under every model and retains a per-model audit.
It also requires `model::issued_at`, `model::artifact_vintage` and
`model::trained_through` columns, alongside quote and decision timestamps.
Each supplied leg valuation is checked before taking any ticket-level maxima:
`trained_through < issued_at`, `artifact_vintage <= issued_at`, and all evidence
available at the decision. A later valid leg cannot certify another leg. Missing
probability rows or an absent model probability column follow the gate's declared
missing policy. Negative, infinite or out-of-range supplied values still error.
Quote timestamps cannot follow the candidate's own decision, even when the
surrounding decision context has a later timestamp.
`BetOutcomeReporter.probability_outputs` maps model names to retained upstream
outputs; `model_timing` supplies timestamps or explicit `{column: ...}` mappings.

Blending leg probabilities then multiplying is generally different from mixing
each model's joint ticket probabilities. Independence of legs is an explicit
assumption here; different model algorithms do not establish independence of
their errors. Binary EV gates reject nonbinary payoff declarations.

## Optional allocation

Existing `stake=1.0`, numeric/Series BetSpec stakes and template stakes retain
their original behavior when `stake_policy` is absent. They do not require an
initial bankroll and remain unconstrained research benchmarks.

The optional hook is available on `compose_bets`, `add_tickets`,
`BetOutcomeReporter` and `BetPerformanceReporter`. Supply `stake_policy`,
`stake_context` and optionally `risk_limits`. Singles use the same machinery;
`single_payoff='binary'` explicitly enables their binary Kelly valuation.

Selection and template expansion resolve the entire final batch first, including
`MultiBet(stake_mode='total')` nominal division. Policy amounts **replace** these
nominal amounts. Every template at the same decision instant shares one cash
snapshot. Static contexts accept only one timestamp; use replay for compounding.
Zero amounts remain selected-but-unfunded with zero cost/profit and separate
counts. Rejected candidates receive no funding. A downstream performance reporter
cannot accidentally compose or allocate a source's tickets for a second time.

Policies are `FixedStake`, `FixedFraction`, capped `FractionalKelly`, or
`LearnedAllocation`. Currency/units and bankroll basis must match the context.
Kelly uses `B * alpha * max(0, (p*O-1)/(O-1))`, then its fraction cap and independent
risk limits. Decimal odds must exceed one. Model probabilities need provenance
and issue timestamps, or an explicit manual attestation of decision-time
availability. Missing probabilities fail. Kelly rejects pushes, voids and partial
payout valuation; declaring binary does not convert such a market to binary.

`HistoricalRateSource` fits only eligible selected tickets for a named strategy,
with a frozen cutoff, optional strata/lookback and explicit shrinkage. Learned
selection requires issued or chronological OOF evidence. Applying its rates
requires `exchangeable_within_strata=True`: an explicit assumption that the fitted
rate applies to these offers, including their odds. A global draw frequency is
not automatically a ticket probability, and no default squares a historical rate.
Native recipes can supply `history=input.Table(...)` to fit this explicitly
provided history at the fixed cutoff on first use. Otherwise fit the source with
`.fit(history)` before allocation. It never consumes the current batch's outcomes.

`RiskLimits` independently applies cash, per-ticket and configured exposure caps,
including outstanding exposure. One conservative proportional factor satisfies
all caps; stakes are rounded down to a declared quantum. There is no order-driven
greedy dropping or redistribution of residual cash. Binary floating point near a
rounding boundary can leave an additional quantum unspent. Per-ticket Kelly on
overlapping tickets is an approximation, not joint portfolio Kelly.
Native league and round exposure identities support both competition/season IDs
and the existing `source_league`/`source_season` fallback. A configured exposure
cap requires complete, nonempty identities, including outstanding exposure.
Missing values or empty tuples raise instead of disabling a requested cap.
Empty and whitespace-only strings also raise, recursively inside selected and
outstanding exposure identities. Nonblank identifiers are preserved exactly;
allocation does not silently trim or otherwise remap them.

On the opt-in gate/allocation path, duplicate fixtures are compared using
decision evidence only. Conflicting retrospective outcome fields do not change
selection or allocation; contradictory settlement becomes missing for reporting.
The ordinary no-policy duplicate behavior is preserved.

## Closed bankroll replay

`BankrollLedger` maintains `wealth = initial + realized_net_pnl`,
`reserve = outstanding_cost_basis`, and `cash = wealth - reserve`.
Placement reserves cash without changing wealth. Settlement releases the original
stake and realizes `gross_return - original_stake - fee`. Several asynchronous
tickets can remain open. Fees cannot violate cash solvency.
Settlement preflights the proposed wealth, reserve and cash before mutating any
ticket or event. Overflow and invalid settlements leave the ledger unchanged.

`settlements_from_legs` supplies the conservative multiplicative-payoff default:
wait for the latest required leg result, even if an early loss determines the
return. Unknown leg outcomes leave a ticket pending. A void/refund leg's supplied
gross multiplier is one. Other payoff rules need an explicit settlement table.

`replay_bankroll` keeps selected tickets and outcomes separate, freezes the policy
for the run, uses UTC and processes officially available settlements before a
same-time decision batch. Postponed matches retain reserve until their actual
availability. It never releases money merely because a round advanced.
Its result includes allocations, event ledger, wealth, cash, reserve, return,
realized profit, log growth, drawdown and observed insolvency events.
`result.report()` returns native table artifacts. Observed insolvency events are
not a theoretical ruin probability. Existing cumulative-profit charts remain
outcome summaries; they are not relabeled as bankroll curves.

## Recipes, UI, persistence and migration

Use the existing `recipe['model']` field for an ensemble, stack, residual model or
explicit graph. Set top-level `preprocessors=[]`, `adapter=None`,
`target_transformer=None` and `candidate.calibration=None`; the builder offers a button to
clear these wrappers. Configure them on each child instead. This avoids fitting
outer preprocessing before constructing OOF inputs. Ordinary models keep the
existing controls. Numeric versus text class labels must retain their types.

The catalog supports nested components, output schemas, training plans, fixed
reducers, gates, allocation policies and serializer configuration in dictionary
recipes, JSON and Python/notebook exports. Native execution/preparation/reuse and
high-level artifact export are covered by regression tests. `CompositionReporter`
shows child outputs, inner row identities, fit count, OOF coverage and refit policy.
Bet reporters retain decision and allocation audits with actual-stake accounting.

`CompositeSerializer` saves all fitted children, preprocessing, calibrators,
schemas, graph, training plan and cutoffs through trusted joblib. Its inspectable
manifest records dependencies, registered configuration when portable, component
implementation fingerprints, outputs and audit. Unregistered Python extensions
are explicitly marked Python-only. The loader does not execute registration
instructions or dynamically import a component named by the manifest. Existing
artifact path and payload-hash validation remains in place. `ArtifactExport`
now forwards an explicit custom serializer both when saving and restoring.
Load only trusted joblib artifacts in a compatible environment: a manifest does
not make pickle a sandbox.

Composition introduces no cross-fold fitted-model cache. Existing experiment
identity includes the graph/training configuration and data/fold identities;
children are freshly fitted per population. Exact saved-run reuse and whole-graph
restore are tested with fitting disabled.

No existing recipe migration is required. To opt in, replace only the model or
add the allocation hook and its explicit contracts. To return to the old path,
use an ordinary model and omit `stake_policy`.

## Explicit limits

- CPU execution; one learned meta layer; no arbitrary callback workflow graph.
- Repeated OOF predictions error; warm-up drop/error; only `refit_bases` deployment.
- Binary logistic or ordinary regression residuals; no multiclass correction.
- Finite-support PMF mixtures; no averaging NB/Poisson parameters.
- Binary per-ticket Kelly; no joint portfolio optimizer, scenario learning,
  online policy updates or nonbinary Kelly payoff model.
- Standalone chronological replay is an explicit Python research API. Static
  allocation and fitted utility artifacts are available in the native builder;
  it does not silently run a bankroll replay or fabricate utility training data.
- No browser visual QA or matrix across all supported sklearn versions was run;
  native form handlers, catalog and recipe execution are tested locally.

See [implementation status](composition_status.md) for exact checkpoints and
validation results.
