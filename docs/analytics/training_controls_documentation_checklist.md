# Controls verification coverage

This checklist belongs to the frozen controls revision. The separate MatchResult
reporter batch changes shared viewer/export code and needs its own integration
verification. Recorded evidence is [training_controls_check.json](training_controls_check.json).

## Computational and boundary checks

- [x] Scripted independent patience, delta, monitor direction, exact-best, ties and restart decisions.
- [x] Nonfinite monitors, arbitrary exception propagation, protocol/freshness checks and copied contexts.
- [x] Learning-rate history, plateau order and below-floor behavior.
- [x] Whole-match validation, tied UTC batches, both layouts and supervised selector exclusion.
- [x] Independent regression/logistic SGD updates and fitting-only preprocessing/class discovery.
- [x] Lasso/ElasticNet closed-form coefficients, transformed names, multi-output and classifier classes.
- [x] Absent-feature and intercept-only denominator, all-zero and unavailable inspection cases.
- [x] Separate unequal histories, native extraction, model-only scope and no prediction recomputation.
- [x] Explicit fresh final refit, disjoint evaluation, different-dataset provenance and strict prediction outputs.
- [x] Final/trial roles, legacy defaults, explicit successful-trial linkage and serialized single-final contract.
- [x] Training JSON with predictions disabled and no leaderboard prediction reads.
- [x] Controlled-clock display handle, SVG coordinates, escaping, missing gaps and display point limits.
- [x] Full analytics suite: 1,216 passing cases, including 138 new cases, on the preserved source.

## Documentation and executable artifacts

- [x] Every public API, new option, record field and internal helper described in the reference.
- [x] Guide examples executed against the preserved source with recorded output hashes.
- [x] New minimal notebook 11 executed in a fresh kernel and exported; ten previous notebooks unchanged.
- [x] Browser navigation, fold selection, figures, table downloads, SVG and saved notebook iframe checked.
- [x] LaTeX rendered without errors and document tables/sections visually reviewed.
- [x] Local wheel built and imported without installation or source-checkout fallback.
- [x] Source snapshot, prior evidence/exports, prepared inputs and archived continuity files rehashed.
- [x] Continuity documents tied to the final evidence; separate reporter batch remains pending.

## Limits stated in the guide and reference

All fitting and numerical examples are synthetic. Fresh-kernel execution and
saved HTML are distinct from unverified live Jupyter frontend behavior. Snapshots
are in-memory only; there is no persisted resume, search engine or automatic
new-match retraining. Source repairs belong to the primary implementation task.
The original denominator failure and its revised source/evidence remain preserved.
