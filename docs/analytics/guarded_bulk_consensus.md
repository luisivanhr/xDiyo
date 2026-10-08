# Guarded bulk ticket execution

## Scope and revisions

Implemented on `codex/guarded-bulk-consensus`, based on
`57858007bdcd8c79e3f0479e4328137932860ec1` of
`codex/research-quote-availability`. Correctness repairs are commit `e5151cc`;
bulk integration is commit `ea622d2`. Main remains
`29994933902d213bccd13376bc91ab5af70a481b`; the parent composition branch remains
`f76edff8b2e42c7966a8e55c36e6076c8af32011`. No merge, training, dataset changes or
betting study was performed. The user separately authorized pushing this branch.

The consolidated handoff and all 217 manifest-covered support files were
verified byte-for-byte before use. Supplied evidence remains unchanged;
platform adaptations and new output are in disposable copies. The second
COLUMNAR_TICKET_PROTOTYPE_EVIDENCE archive also contains synthetic evidence,
not actual retained XGB/LGBM streams. Supplied pickles were not deserialized.

## Phase A: repaired reference

| Repair | Native change | Regression evidence |
|---|---|---|
| A1 | Require every nested leg's own keys before constructing any shared DataFrame. Explicit null observed times remain valid in research mode. | Every required key, full JSON and compact evidence, one deficient leg beside a complete ticket; unsupported outer observed time. |
| A2 | Resolve consumed rows by ticket ID and ticket bet. Reject conflicting explicit template labels; validate original model quote identities on those exact rows. | Forged unknown/blank/null template labels; duplicate/conflicting evidence; per-ticket cutoffs. |
| A3 | Call-local mixed-format date/time parsing and consumed-relation validation. Optional template column and unrelated orphan rows retain their old meaning. | Independent mixed ISO quote/model examples; timezone offsets, typed/naive values, missing/empty/malformed/numeric values; earlier-ticket cutoff attack. |
| A4 | Inventory uses mixed-format parsing independently from strict guards, including fractional seconds. | Correct December 2 maximum, null/all-null/empty inputs, unsupported values, unchanged manifest field names. |

Timezone-naive values retain the legacy UTC interpretation. Required temporal
fields reject numeric epochs, arbitrary objects and empty strings; absent or null
required model times reject. Research observed quote times may explicitly be
null/NaT, but a missing required key is never equivalent. Inventory coercion
counts null values as `missing` and unsupported non-null values as `unparseable`;
parseable count is row count minus these two existing counts. Inventory coercion
never authorizes decisions. No manifest schema/version change was introduced.

Intentional rejection changes also include invalid original evidence on rejected
or undersized fixture pools, unsafe quote extensions, integer price-product wrap
and aggregate accounting overflow. These are not baseline parity targets.

## Native architecture and dispatch

`compose_bets` runs native pool/preflight validation and the expansion cap before
calling a private, explicit capability predicate. The bulk path uses call-local
readonly arrays, positional ticket membership, ordered products and strict EV
comparisons. Settlement receives outcomes only after the decision kernel has
finished; loss precedes missing, and push/void default to removal. Unselected
outcomes are not inspected. Native table inference and quote JSON representation
are retained. Selected memberships are materialized late, with original sparse
candidate/member indices.

Both paths call the same final publication function for native metrics, selection
summaries, quote declarations and attrs. No new public result columns, backend
attrs, validation tokens, persistent caches, or replacement reporter are added.
Tests inspect `supports_bulk` and intercept `execute_bulk` for dispatcher
verification. The benchmark-only reference override is not a runtime API.

Bulk capabilities:

- Exactly one `tickets` AllCombinations template, native QuoteAvailability and
  native two-model DecisionLayer, complete original model probability streams.
- Full audit, fixed 1u stake, independent binary probabilities, AND EV strictly
  greater than zero, disabled min_ev prefilter, default identities and remove
  push/void rules, configurable positive integer leg count.
- Nonmissing event_id keys, or the complete ordered source_league/source_season/
  competition_id/season_id/event_id key. Caller indices may be arbitrary or
  duplicated. Native preflight still validates conflicts and deduplicates
  identical fixture evidence before call-local positions are assigned. uint64
  keys, including adjacent values above 2**53 and near 2**64, remain exact.
- Float64 or integral original odds; integral products must fit signed int64.
  No conversion of float32/object/numeric-string odds to qualify for bulk.

Other valid configurations use the repaired reference: summary mode, other gate
settings/custom hooks, multiple or differently named templates, unsupported or
mixed key types, alternative payoff/price representations, static allocations, bankroll
and zero funding. Invalid input is not caught and retried through another engine.
Original quote/model guards run before rejection/caps can conceal evidence.

The existing native fixture gate remains responsible for labelled PMFs and the
OR rule (original non_draw complement, inclusive boundary); it is unchanged.
The example adapter additionally accepts `ticket_legs` and `max_tickets`, with
backward-compatible defaults. The bulk dispatcher is automatic under existing
UI/recipe settings, with no new UI-only configuration or serialization class.
Raw artifact formats remain unchanged. Tests cover pickle, Parquet, recipe and
installed-wheel readers as well as HTML/CSV representation.

## Verification and limitations

The affected suite includes official model-composition, staking, quote, ticket,
summary, browser and isolated-wheel checks, plus new reference and bulk tests.
Exact comparisons include all frame values, dtypes, indices, order and recursive
attrs. The independent scalar oracle uses Python products and its own settlement
rules. Outcome mutation leaves selections, membership identities and decision
audits unchanged. k=1/2/3/4, undersized/empty pools, cap boundaries, signed/unsigned
prices, optional issue times, timestamp resolution and full/summary parity are
covered. No heavy QA ran concurrently with measured timing.

Supplied prototype checks: 204 scalar/adversarial cases, 14 full-output cases and
14 configurable-k/report cases passed against the installed dependencies/current
native implementation. The 204 cases exercise the supplied prototype; native
bulk safety has its own independent oracle and exact reference tests. Windows
harness adaptations only remove the unused Unix `resource` import and load the
example with runpy to avoid an unrelated installed `examples` package.

Actual retained draw/non_draw XGB/LGBM streams with their original identities,
times, opening quote references and settlements were unavailable. Local artifact
name inspection found older corners experiments, not this required paired input.
**Real-data replay acceptance remains pending.** Synthetic results below do not
establish historical tradability, ML training speed or full-season study speed.
Unknown observed quote timestamps remain unknown and carry the same explicit
research-assumed availability disclosure.

## Reproduction

Use a complete checkout with reporting/training/test extras. Explicitly set
PYTHONPATH to this checkout's src and root so another editable install is not used.
The checked Windows commands use PowerShell and Python 3.14.0 / pandas 2.3.3 for
broad regressions, and isolated Python 3.12.6 environments with pandas 2.2.3 and
3.0.6 for compatibility. Timing uses pandas 2.2.3.

```powershell
$env:PYTHONPATH="$PWD/src;$PWD"
$checks = rg --files tests/analytics | Where-Object {
  $_ -match 'test_(model_composition|composition_[^\\/]+|quote_[^\\/]+|guarded_bulk|consensus_performance|bet_tickets|ticket_ev|stake_bankroll|summary_audit|ui_composition_review)\.py$'
}
python -m pytest @checks -q -p no:cacheprovider --tb=short

$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'; $env:NUMEXPR_NUM_THREADS='1'
python examples/benchmark_guarded_tickets.py --source $PWD --output TEMP/reference --groups 18 --backend reference --audit-level full
python examples/benchmark_guarded_tickets.py --source $PWD --output TEMP/bulk --groups 18 --backend auto --audit-level full
```

Repeat with 72 groups and summary mode, in fresh serial processes. Discard one
warmup per backend/mode; randomize the three measured repeats using seed 81058.
Each group has eight fixtures and 28 doubles. Imports and synthetic input
generation are excluded; all per-call validation, conversion, gates, settlement,
table/audit materialization are included in native wall/CPU. Full native
`ticket_html` plus `AnalysisReport.to_html`, all evidence CSVs and local pickle
serialization are timed separately. Serialization includes the local pickle
write; CSV and HTML byte generation exclude disk writes. No report style changes
or browser-load speed claim are made.

## Matched synthetic benchmark results
Runtime commit: `ea622d2`; runtime source SHA-256: `115fad2692caa201dc0bf58e6de85789b12a4ec8760b86a429e5921aaab89080`.
Windows, AMD Ryzen 7 7800X3D (8 cores / 16 logical processors). Python 3.12.6, pandas 2.2.3; exact NumPy version is in every raw sample. One numerical thread, fixed hash seed 0. Other user applications were not controlled.
Three measured repeats per cell, after discarded warmups. Native seconds below include validation/conversion, composition/gates, settlement and full native table/audit materialization. These substeps were not separately timed. Peak RSS is process peak through full HTML generation.
| Candidates | Mode | Dispatch | Native median [range], s | CPU, s | HTML, s | CSV, s | Peak RSS, MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 504 | full | reference | 3.548 [3.505, 3.594] | 3.531 | 0.293 | 0.111 | 174.5 |
| 504 | full | auto | 0.523 [0.522, 0.529] | 0.531 | 0.300 | 0.122 | 170.6 |
| 504 | summary | reference | 3.600 [3.541, 3.735] | 3.562 | 0.196 | 0.055 | 123.7 |
| 504 | summary | auto | 3.747 [3.549, 3.889] | 3.688 | 0.207 | 0.060 | 122.7 |
| 2016 | full | reference | 13.592 [13.487, 15.633] | 13.516 | 1.082 | 0.441 | 402.4 |
| 2016 | full | auto | 2.001 [1.934, 2.037] | 2.000 | 1.108 | 0.451 | 396.3 |
| 2016 | summary | reference | 13.756 [13.641, 14.068] | 13.625 | 0.708 | 0.223 | 206.1 |
| 2016 | summary | auto | 13.780 [13.758, 13.905] | 13.703 | 0.683 | 0.217 | 205.8 |

Full native execution improves about **6.8x** at both sizes. Including the separately measured full HTML, CSV and pickle stages gives about **4.2x** at 2,016 candidates. HTML/CSV costs and report content are unchanged; no renderer speedup is claimed. Summary automatic dispatch uses reference, so its small run-to-run differences are measurement variation.
All 24 measured records have exact same-mode frames, recursive attrs, HTML SHA-256 and CSV SHA-256 against their matched reference. Two additional unchanged-5785800 valid-input probes match exact outputs; only source_revision/runtime_source_sha256 are excluded from summary manifest comparison. The manifest exclusion does not apply to same-source bulk/reference comparisons. No missing-timestamp semantics or omitted audit records are masked.
Full and summary are separate storage contracts and are not expected to have identical HTML/CSV to each other; computational parity is tested independently. Four discarded warmup samples and both baseline probes remain in the raw CSV. All 30 processes finished normally, with no timeout or excluded measured sample.
Raw measurements: [guarded_bulk_timings.csv](guarded_bulk_timings.csv). Full medians/ranges, sizes, exact command lines, dependency metadata and parity outcomes: [guarded_bulk_verification.json](guarded_bulk_verification.json). Benchmark generation does not read private model artifacts.

## Final verification outcomes

- Complete affected suite at runtime commit `ea622d2`: **604 passed**, seven
  dependency deprecation warnings, 59.26 seconds. Includes successful browser
  and isolated installed-wheel tests. No failures or skips.
- Focused compatibility matrix: **319 passed on pandas 2.2.3** (26.46 seconds,
  511 dependency warnings) and **319 passed on pandas 3.0.6** (40.12 seconds).
  An intermediate pandas 3 empty-string dtype mismatch was repaired; final
  matrix has no failures or skips.
- Supplied nested schema probe: **30/30 malformed cases rejected**. Supplied
  independent adversarial probe: **103 cases, zero findings**, including the
  manifest accuracy comparison. Original probe oracle logic was unchanged.
- Supplied prototype QA: **204/204 passed**; two exact native/prototype suites
  **14/14 each passed**. These complement, rather than substitute for, the new
  native independent scalar and dispatch tests.
- Matched timing parity: **24 same-source records exact** (frames, recursive
  attrs, HTML and CSV) and **two original-5785800 valid-input comparisons exact**
  except the explicitly excluded summary code-identity fields.
- Input immutability, outcome mutation, serialization, full/summary computation,
  fallback allocations and strict temporal checks passed. No real-data replay
  was run; it is blocked on the actual retained paired model/quote inputs.

The measured runtime was unchanged while benchmarking. Subsequent changes only
record these findings in documentation. This development branch is pushed under
the user's explicit instruction and remains unmerged; main is unchanged.

## Follow-up repairs to the 39aba81 audit

The following changes were verified on the same branch, with publishing
subsequently authorized by the user. The results above describe the initial revision; this section
supersedes its real-input limitation and event-only dispatch restriction.

1. **Original timestamps:** empty/whitespace strings reject for object, pandas
   StringDtype, Arrow strings and categorical columns, including before a pool
   is rejected or found undersized. Genuine null research observations remain
   null; required model timestamps still reject. Inventory coercion preserves
   the missing/unparseable distinction.
2. **Composite fixture identities:** both engines use native validated pools,
   native `_event` identities and stable candidate ordering. The bulk engine
   assigns dense positions independently of DataFrame labels, and serializes
   every supplied fixture key in quote evidence. It neither projects the study
   to event-only keys nor converts integer keys through floating point.
   Explicit model-prefixed native fixture keys, when supplied, must match the
   canonical keys in type and value. Older callers may omit those optional keys;
   their quote IDs alone do not independently authenticate a fixture join (see
   the direct-helper contract below).
3. **Native-precision PMFs:** the example validates finite bounded channels
   with per-column unit-roundoff budgets (`eps(dtype)/2 * abs(value)`, summed
   across channels), retaining the existing 1e-12 binary64 floor. Both channels
   remain unchanged. No normalization, complementary replacement or model
   averaging is performed; gate arithmetic and boundaries are unchanged.
4. **Finite accounting publication:** ordinary pandas reductions retain their
   existing rounding. A nonfinite reduction uses checked `math.fsum`; an
   unrepresentable total or ROI raises before publication. This also covers
   selected-stake summaries.

### Verification

- **921 passed** in the affected Windows suite on pandas 2.2.3, including the
  two previously blocked composition-browser tests, odds browser round trips,
  and isolated installed-wheel checks. No failures or skips.
- **694 passed** in the focused pandas 3.0.6 compatibility suite. This overlaps
  the affected suite; these numbers are not a combined unique-test count.
- The independent timestamp reproduction now rejects all 18 invalid cases.
  The native-precision reproduction accepts the three unchanged saved float32
  pairs, and all 4,767 retained PMFs from each model pass unchanged.
- A1–A4, native/prototype comparisons and independent differential probes are
  recorded in [the verification record](guarded_bulk_repair_verification.json).
  A temporary Windows probe copy moves the Unix-only `resource` import into
  its unused timing worker; correctness assertions are unchanged.

The first broad run exposed missing test-environment dependencies and joblib
1.6's removed vendored cloudpickle import. The successful run uses joblib 1.5.2,
setuptools and Playwright 1.55.0 with the existing Chromium installation. No
library dependency constraints or unrelated persistence behavior were changed.

### Unchanged five-key study replay

`examples/verify_guarded_repair_evidence.py` verifies the supplied archive hashes,
loads the original hash-pinned XGB/LGBM streams and opening quotes, and executes
only the recovered adapter's `keyframe`, `leg_gate`, `specification` and
`native_select` functions. Its environment mutation and full-study runner never
execute. Complete outputs are captured before the adapter removes report attrs.

| Bounded fixtures | Candidates | Selected | Automatic dispatch |
|---:|---:|---:|---|
| 30 | 109 | 73 | bulk, original five-key identity |
| 100 | 386 | 96 | bulk, original five-key identity |

Repaired reference and automatic execution agree exactly on frames, indices,
column order, dtypes, recursive attrs, IDs, quote JSON, native AnalysisReport
HTML, CSVs and pickle round trips. Original PMFs remain unchanged; original
scalar probability oracles agree. Input immutability and outcome-mutation
decision invariance also pass. No identity or metadata fields are excluded.

Run from the checkout with its `src` and root on `PYTHONPATH`:

```powershell
python examples/verify_guarded_repair_evidence.py `
  --evidence <extracted-audit-root> --output <verification.json>
```

This is a bounded correctness diagnostic, not a new repeated speed benchmark.
The separately labelled original event-only benchmark is not evidence for the
unchanged study caller. No models were refitted, predictions regenerated,
strategies searched or full study rerun. The original retrospective research
clock and unknown observed quote timestamps remain explicit; these checks do
not establish historical tradability. Numerical chunks remain capped at 4096;
complete audit storage still grows with candidates times leg count.

## Remaining boundaries from the 7714579 review

These repairs remain on `codex/guarded-bulk-consensus`, without a merge.

- Bulk ticket grouping columns now use the original typed Series scalars,
  matching reference inference for narrow signed/unsigned integers, float32,
  categorical and extension representations. Membership inference is unchanged;
  outputs are not blanket-cast to the input schema.
- Numeric categorical PMFs use the category dtype's existing precision budget.
  The example also supports a categorical non-draw channel by unboxing it only
  for gate arithmetic. Original input channels remain unchanged; no
  renormalization or wider tolerance is introduced. Unsupported representations
  raise a deliberate `ValueError`.
- `ticket_batch` and `finalize_tickets` verify consumed fixture membership
  against native `AllCombinations` ticket IDs, including all supplied fixture
  keys and league/season/stage/round/fold grouping. Duplicate fixtures, changed
  leg counts and inconsistent ticket grouping reject before publication.
  Under a quote-availability contract, available retained `quote_legs` must
  match the newly consumed ordered evidence; stale evidence cannot be silently
  replaced.

### Direct-helper compatibility contract

For native `AllCombinations`, the canonical ticket hash provides the original
fixture/group binding in both full and summary audit modes, even when callers
omit optional model-prefixed fixture keys. A valid re-finalization remains
supported, including outcome-only changes for retrospective settlement.
Existing ordered nested quote records, compact evidence and legacy
column-oriented JSON are checked when present. Equivalent timestamp spellings
are compared as instants using the existing strict parser; fixture identities
and exact quote values must agree. This is not a quote refresh operation.

Legacy manually assembled, non-`AllCombinations` tickets with neither a
canonical fixture hash nor original nested evidence have no prior full-key
binding to verify. A helper call in that state treats supplied membership as
the caller's declared fixture join. Quote IDs alone do not prove that join.
Use native `AllCombinations` for retained full-key checks; with a quote contract,
also retain the emitted nested evidence for subsequent checks. These are
consistency checks on supplied records, not
cryptographic authentication of an external source. No reusable approval token
or cross-call validation cache is introduced.

### Verification of these boundaries

- **1,074 passed**, no skips or failures, in the affected Windows pandas 2.2.3
  suite, including browser and installed-wheel checks.
- **847 passed**, no skips or failures, in the focused pandas 3.0.6 suite.
  Counts overlap; 153 new regression cases cover these repairs.
- The supplied independent composite matrix is **90/90 exact**, the grouping
  dtype matrix is **50/50 exact**, all six forged-membership/nested-evidence
  probes reject, and the outcome-mutation probe passes.
- Original malformed nested-evidence, timestamp, adversarial and accounting
  overflow probes retain their repairs. Numeric categorical draw compatibility
  is restored alongside float32, nullable/Arrow floats and object inputs.
- Repeating the hash-pinned original five-key adapter on 30 and 100 fixtures
  gives respectively 109/386 candidates and 73/96 selected tickets. Automatic
  execution actually dispatches to bulk. Reference/bulk frames, dtypes,
  indices, recursive attrs, IDs, quote JSON, native HTML, CSV and pickle output
  agree exactly; original PMFs and input tables are unchanged.

Detailed evidence and separate fresh-process timing results are recorded in
[the boundary verification record](guarded_bulk_boundary_verification.json).
No model refit, forecast regeneration, strategy search or full-study run was
performed. Unknown quote timestamps and the explicitly simulated research
clock are preserved. The 4096 numerical chunk limit and expansion cap remain;
full audit memory still grows with candidates times leg count.

Fresh serial timing workers used one numerical thread, three repetitions per
backend and alternating backend order. These are bounded retained-data
measurements, separate from correctness instrumentation:

| Fixtures | Reference native (s) | Bulk native (s) | Reference including reports/exports (s) | Bulk including reports/exports (s) |
|---:|---:|---:|---:|---:|
| 30 | 0.920 | 0.103 | 1.044 | 0.225 |
| 100 | 3.103 | 0.205 | 3.404 | 0.509 |
| 500 | 15.203 | 0.750 | 16.907 | 2.377 |

Values are medians; inclusive time includes native execution, HTML, CSV and
local serialization, excluding imports/input loading. At 500 fixtures the
bulk HTML/CSV stages alone took 1.021/0.588 seconds. Median process peak RSS
after reporting was 622.9 MiB for reference and 603.2 MiB for bulk; it includes
the interpreter, loaded data and report. All matched HTML/CSV hashes agree.
Report costs and full-audit memory remain substantial despite faster execution.
