# Bayesian football rating implementation boundary

This implementation follows the assessment of Ridall, Titman and Pettitt,
*Bayesian state-space models for the modelling and prediction of the results
of English Premier League football*, doi:10.1093/jrsssc/qlae075.

The agreed scope includes the bivariate Gamma-mixed Poisson score model,
converged mean-field inference (and the documented one-step approximation),
pooled or per-league offline parameter calibration, promotion/relegation
priors, selectable team summaries and optional fixture features, and complete
parameter/state persistence. Verification uses synthetic fixtures; production
research and any claims of predictive improvement remain separate.

## Existing contracts and the minimal adapter

`ratings.replay.build_ratings` supplies only a win/draw/loss scalar and an
opponent state to `PairwiseRatingEngine.update_period`. The bivariate score
likelihood needs both counts and a shared league home-advantage state. Feeding
it through this protocol would discard necessary observations.

The implementation uses the existing training runner, dataset assembly and
experiment orchestration. `RatingRun` already explicitly permits
other state producers. A model-specific Bayesian producer supplies that
contract; a subclass additionally owns league states, fixture predictions,
the parameter-training boundary, and complete checkpoints. Feature nodes
use the existing feature evaluator, and calibration/prediction adapters use
the existing `ModelAdapter` and native serializer interfaces. UI entries use
the existing constructor catalog and maintained discovery inventory.

The Bayesian producer is a numerical model adapter, not a separate experiment
pipeline. Output selection never changes its inference state. Opponent/venue
dependent quantities are fixture features, not team snapshots.

The existing labels supported statistic counts and result classes, but offered
no match-layout pair of native home/away goals. After reviewing this concrete
input incompatibility, the implementation adds `labels.MatchGoals` through the
existing `LabelExpr`/`create_labels` extension point. It produces one ordinary
`LabelData` containing both counts; dataset assembly and recipe target selection
remain unchanged. The native `current` score basis is explicit and does not
invent regulation-time scores. The existing model factory also dispatches the
native Bayesian adapter directly, as it already does for other native backends;
unsupported feature/target preprocessing is rejected.

## Choices to verify and document

- League team states and home advantage remain separate even when calibration
  pools parameters. Unobserved leagues do not silently inherit per-league fits.
- Mirrored log-scale entry priors are the default extension for promoted and
  relegated teams. An explicit fixed league-gap state bridge is optional.
- Time units, score basis, convergence, parameter cutoffs, result availability,
  season boundaries, and normalization must be serialized and documented.
- Forecasts precede updates; results released together are assimilated as one
  batch. Each fixture contributes one joint likelihood. Historical features
  must not use parameters calibrated after their prediction boundary.
- The paper's optional one-step approximation is distinguishable from
  converged VB. Shared Gamma match mixing is retained in predictive scores.

The existing run-stage `model_serializer` option is exposed in the builder,
and its prediction serializer selector includes the new native serializer.
The explicit serializer contract remains the same; model persistence does not
depend on a parallel store or an implicit fallback.

See the [model equations](bayesian_rating_model.md),
[usage guide](bayesian_rating_usage.md), and
[handoff with verification boundaries](bayesian_rating_handoff.md).
