# Composition branch changelog

## 7 October 2026

### Strict speed audit: native batching and explicit summary storage

- Batch nested quote validation within each invocation while retaining original
  per-leg checks and per-ticket timestamp/parser/product semantics.
- Retain a positional membership index, parse consumed model streams once, and
  index selection summaries without changing floating reduction order.
- Add native/UI/exported `AllCombinations(audit_level="summary")`, keeping full
  as default. Summary stores selected model values and a fingerprinted manifest
  with counts and explicit omissions; all computation and guards remain active.
- Prevent report working copies from propagating large attrs per row.
- Verification: 453 affected tests, 168 on each of pandas 2.2.3/3.0.6, exact e7
  comparisons and independent checks through 17,024 synthetic candidates.
  See [measurements, source pins and private-replay limits](strict_consensus_optimization.md).

### Strict speed audit: separate guard repairs

- Reject reserved native outcome names in namespace segments, including
  `a::won` and `a::settled_at`; ordinary `quote_status` metadata remains valid.
- Revalidate original per-model quote references at direct ticket finalization
  for explicit quote contracts, even after a previously valid composition.
- These intentionally reject inherited malformed paths; they are separate from
  behavior-preserving speed changes. Quote and performance regressions: 120 passed.

### Native consensus performance (parent `c31f215`)

- Publish complete audit metadata after internal settlement and summaries to
  avoid pandas copying it for every row/slice.
- Index ticket membership once and reuse all original ordered leg rows across
  model valuation and settlement, preserving validation and provenance.
- Add structural regressions and a bounded before/after synthetic profile with
  exact output/audit comparisons. See [verification and study replay limitation](consensus_performance_verification.md).

### Quote-contract review repairs (parent `b2a9927`)

- Initialize disclosure columns on the existing frame index before masked
  assignment, preserving valid empty/undersized research outputs on pandas 2.2
  and 3.x.
- Preflight all example candidate and model timestamps before grouping, retaining
  a complete audit even when OR rejects every fixture; support empty input.
- Reject configured reserved outcome names as probability sources before reading
  or relabelling ledger values, including mutations after template construction.
  Keep ordinary quote metadata allowed and concealed aliases outside the guarantee.
- Add review regressions and isolated pandas 2.2.3/3.0.6 checks. See
  [verification](quote_contract_review_repairs.md).

### Explicit research quote availability (separate extension branch)

- Add the opt-in `QuoteAvailability` contract, retaining observed and assumed
  timestamps separately with a stable identity, rationale and attestation.
- Validate complete matching per-model quote/provenance evidence on every
  eligible leg before native AllCombinations expansion; preserve strict observed
  defaults and missing-stream failure behavior.
- Retain per-leg evidence, research declarations and disclosures in native
  reports, Parquet storage, downstream performance reuse and UI recipe exports.
- Include an exact two-stage OR/AND consensus example and boundary tests.
  See [usage and review boundary](research_quote_availability.md).

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
