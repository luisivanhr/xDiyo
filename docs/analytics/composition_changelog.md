# Composition branch changelog

## 7 October 2026

### Follow-up to independent verification of `0d50956`

- Reject empty or whitespace-only strings recursively in configured selected and
  outstanding exposure identities, including source league/season fallbacks.
- Remove native retrospective `status` and `is_awarded` from composition
  prediction metadata while retaining the source dataset for reporters.
- Update the packaging smoke test to verify both existing command entry points
  and their help paths. Update the discovery harness with the existing odds
  configuration export and the real grid-fields module, preserving its assertions.
- Add native label-to-assembly-to-OOF/outer/restored inference tests and blank
  exposure regressions. See [the verification follow-up](composition_followup_0d50956.md).

### Repairs following the independent review of `25eb79a`

- Reset supported learned wrappers recursively; reject retained calibrators for
  refitting and retain the explicit fold-local calibration/frozen-artifact paths.
- Validate feature availability and issue/kickoff timing on outer and restored
  predictions without requiring test labels; remove named outcome metadata from
  composition inference.
- Reject incompatible residual schemas before fitting and validate reconstructed
  outputs. Validate complete categorical PMFs on every edge, including restored
  schemas; reject unsupported distribution families explicitly.
- Validate each model/leg's temporal provenance before aggregating tickets.
  Preserve missing valuations as declared AND/OR abstentions while rejecting
  invalid supplied probabilities. Bound quotes by the candidate's own decision.
- Preserve supported source league/season aliases in risk identities and reject
  missing exposure dimensions instead of silently bypassing a configured cap.
- Keep duplicate validation outcome-independent on the new opt-in path;
  contradictory retrospective settlement evidence becomes unresolved. Preserve
  the existing no-policy path. Preflight ledger settlement before mutation.

### Initial implementation

- Added typed native prediction graphs, ensembles and one-layer chronological
  stacks with local preparation/calibration, original-row audits and complete
  trusted fitted-model persistence.
- Added named algorithmic residual and honest OOF error-meta modes, binary logit
  reconstruction, explicit utility targets and fitted gate/allocation interfaces.
- Added independent AND/OR decisions over aligned probability streams and native
  complete-ticket gates on `AllCombinations`.
- Added optional final-batch staking, capped Kelly, historical strategy-rate
  sources and independent cash/exposure limits. Policy amounts replace resolved
  nominal stakes. Legacy no-policy outputs and defaults remain unchanged.
- Added closed research bankroll accounting, delayed settlement, asynchronous
  replay and native report tables.
- Added catalog/builder controls, child pipeline configuration, high-level custom
  serializer forwarding and composition diagnostics.
- Added offline synthetic examples and regression/adversarial tests. No dataset,
  frozen experiment, primary checkout or default-branch history was changed.

Migration: none for existing single-model or fixed-stake workflows. See
[the guide](composition.md) for opting into explicit child preparation, temporal
contracts, probability provenance and allocation contexts.
