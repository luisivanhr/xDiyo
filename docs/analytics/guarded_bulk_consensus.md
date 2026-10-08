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

## Compatibility repairs following the 166196c audit

The identity guard now preserves three additional boundaries:

1. **Nullable Float32 boxing (superseded).** Commit 2144ddc tried alternate
   binary32/widened spellings after a hash mismatch. The later audit proved
   that this could accept a different numerical value: the current value's
   representability cannot establish the original value. That fallback and
   its exponential spelling search have been removed. See the versioned
   identity-evidence repair below for the current contract.
2. **Generic fixture equality.** `Parlay` and `MultiBet` check duplicates using
   the same pandas multi-column equality as their native constructor.
   Integer `1` and text `"1"` remain distinct; boolean `True` and integer `1`
   in the same other key context remain duplicates. Canonical string hashes
   are reserved for `AllCombinations`. Reusing a leg across different system
   tickets remains valid.
3. **Partial grouping assertions.** Each supplied fold, league, season, stage
   and round field is checked independently against consumed membership.
   Correct partial or absent legacy metadata remains supported. Dropping
   `stage` no longer allows an inaccurate retained `round` to survive.

### Existing canonical type-equivalence boundary

Native `AllCombinations._identity` deliberately stringifies key components.
Its existing hash alone therefore cannot distinguish an integer key from the
same decimal text. Summary publication omits full nested quote JSON, so that
hash is not a stronger typed binding. Full retained quote JSON distinguishes
those key types. Public ID construction remains unchanged.

Optional model-prefixed fixture keys are checked by the model-evidence
validation in composition and `finalize_tickets`. The low-level `ticket_batch`
helper does not independently validate model streams; merely supplying those
columns to it does not strengthen its canonical-hash check. At 2144ddc no new typed-key
storage contract was introduced; the repair below adds conditional numeric
identity evidence without claiming source authentication.

### Verification and scope

- **1,167 passed** in the affected Windows pandas 2.2.3 suite, including
  browser and installed-wheel checks; **940 passed** in the overlapping pandas
  3.0.6 suite. Neither suite had failures or skips. The 93 new regression cases
  exercise both engines, both audit modes, helpers, allocation, serialization,
  original native IDs and negative mutations.
- All 20 independent nullable-Float32 grouping reproductions now compose;
  all eight retained-output helper checks pass. The four generic composition /
  allocation reproductions pass. All four partial-group contradictions reject.
- Prior composite **90/90**, dtype **50/50**, forged membership **6/6 rejected**,
  nested-schema **30/30 rejected**, timestamp **18/18 rejected**, adversarial
  **103 with no findings**, and finite-accounting probes retain their repairs.
  The additional categorical adapter checks pass **24/24** unchanged.
- The independent 219-case investigation now matches 207 raw expectations.
  The remaining nine cases illustrate the documented canonical type-alias
  boundary (including low-level helper calls); three float16 categorical
  constructions fail inside pandas before calling the library. These are
  recorded separately, not presented as new regression failures or passes.
- Original hash-pinned five-key XGB/LGBM replay on 30/100 fixtures still gives
  109/386 candidates and 73/96 selected tickets, with actual bulk dispatch.
  Reference and automatic frames/dtypes/indices/attrs, IDs, quote JSON, native
  HTML/CSV and pickle round trips agree exactly. All 4,767 PMFs per model remain
  unchanged, as do original unknown quote timestamps and the research clock.

[Verification and bounded timing evidence](guarded_bulk_identity_verification.json)
records source hashes, independent classifications, report/export overhead and
process peak RSS. Fresh serial single-thread timings are separate from the
correctness runs. The user authorized publishing this repair branch, not a
merge. No model fitting, regenerated forecasts, strategy search or full-study
run is involved. Full-audit storage remains O(candidates × legs), with the
existing expansion cap and 4096 numerical chunk bound.

## Numeric identity repair following the 2144ddc audit

`AllCombinations` captures original group and full fixture identities before
pandas boxes membership rows. If any identity value in a pool is floating
point, its tickets carry an `identity_evidence` JSON column (version 1):

- Ordered grouping/key column names and source dtype descriptors.
- Original scalar kinds and exact hexadecimal binary floating-point values,
  with integer, text and other scalar representations retained separately.
- Ordered membership cells, allowing the original public canonical hash to
  be reconstructed once and checked on every helper call.

Both native construction paths use the same capture. Float32 values remain
usable after ordinary boxing, allocation and pickle. Evidence stays in the
ticket table, standalone CSV and downloadable report CSV in full and summary
audit modes. Preserve it when saving or selecting ticket columns. Exact
numeric boxing is accepted; rounding and converting text keys to numbers are
not. The source dtype descriptors must parse, and encoded floating kinds must
round-trip to their exact recorded values. The current membership dtype may
legitimately differ after boxing; it is not used to guess the original value.

For example, binary64 `1.1` and binary32 `1.1` widened to binary64
(`1.100000023841858`) are different values. Mutating either into the other
rejects, even if their short canonical spelling happens to match. Retained
group assertions are checked independently; correct partial/absent group
columns remain supported. Full nested quote and model-evidence validation
remain in place. Generic `Parlay`/`MultiBet` retain pandas duplicate semantics.

**Legacy recovery:** floating identity membership without original evidence
fails explicitly with “rebuild from the original ledger.” A string hash alone
cannot determine whether an original token came from binary32 or binary64.
Re-compose with the original ledger, original dtypes, grouping, match columns
and policy, then retain the generated evidence. Do not synthesize provenance
from an already boxed legacy membership table. Nonfloating legacy tickets
retain the existing hash checks and optional grouping-column compatibility.

Public ticket IDs and round keys are unchanged. The ordinary integer/text
five-key workflow gains no column or attrs and retains its exact tables and
reports. New floating-identity evidence adds storage proportional to the
number of retained membership cells. Validation uses linear cell traversal
and one canonical hash, rather than an exponential product of spellings.

This evidence detects inconsistent retained inputs; it is not a cryptographic
signature or independent source authentication. Treat it as trusted retained
artifact metadata. Replacing evidence together with membership can yield a
mutually consistent record under an unchanged canonical ticket ID (and, for
summary fixture identities, without changing the other ticket fields).
The previously documented legacy integer/text
canonical aliases and the low-level `ticket_batch` model-stream limitation
remain separate from the repaired numeric inequality.

### Verification of the numeric repair

- **1,246 passed** on Windows with pandas 2.2.3, including browser and
  installed-wheel checks; **1,019 passed** in the overlapping pandas 3.0.6
  suite. No failures or skips. The 79 new cases cover value changes in both
  directions, adjacent values, text keys, persistence, allocation, grouping
  assertions, malformed evidence and bounded hash work at 4/6/8/32 keys.
- All **8** independent minimal numeric-rebinding cases and **88** directional
  mutations reject. The original 20 Float32 composition cases, eight retained
  helper calls, four generic allocation cases and four partial-group
  contradictions retain their expected behavior.
- Prior 90-case composite and 50-case dtype matrices pass; six forged
  membership cases, 30 malformed nested-evidence cases and 18 invalid
  timestamps reject. The 103 adversarial statuses are unchanged. Categorical
  adapter cases pass 24/24. The 219-case legacy investigation still has the
  separately documented nine canonical aliases and three pandas float16
  construction limitations; 207 raw expectations are met.
- Original cached five-key replay remains 109/386 candidates and 73/96
  selections at 30/100 fixtures, with actual bulk dispatch, exact reference
  tables/dtypes/attrs/IDs/quote JSON/pickle/report parity, unchanged original
  PMFs and outcome-independent decisions. Native HTML/CSV hashes also match
  the previously published repair's retained verification.

Fresh serial single-thread workers, three repetitions per backend, alternating
order after correctness QA finished, gave the following medians. Inclusive
times include native execution, HTML, CSV and serialization; imports/input
loading are excluded. These compare backends at this repair, not controlled
cross-commit performance.

| Fixtures | Reference native (s) | Bulk native (s) | Reference inclusive (s) | Bulk inclusive (s) |
|---:|---:|---:|---:|---:|
| 30 | 0.927 | 0.113 | 1.050 | 0.247 |
| 100 | 3.297 | 0.209 | 3.617 | 0.518 |

At 100 fixtures median peak process RSS after reporting was 219.0 MiB for
reference and 215.3 MiB for bulk, including interpreter and loaded inputs.
All matched HTML/CSV hashes agree. Detailed counts, source hashes, raw timing
samples, compatibility limits and the research-clock qualification are in
[the numeric identity verification record](guarded_bulk_numeric_identity_verification.json).
This work publishes only the development branch; no merge, model refit,
forecast regeneration or full research run is performed.

## Datetime boxing repair following the 837ac69 audit

Native construction can convert an object-column Python `datetime.datetime`
fixture key into a pandas `Timestamp`. Version 1 identity evidence now encodes
these two known scalar wrappers as a `datetime` cell. It retains the original
canonical time string, timezone implementation/name (or explicit naivety),
and fold. Both wrappers must produce the same complete encoding. The public
canonical hash still uses the original time string, so ticket IDs do not change.
Nanosecond precision is retained for original pandas timestamps.

Evidence is now emitted for pools containing either floating or these known
datetime identities. This also protects newly constructed datetime-only
tickets from text substitution in summary mode. Ordinary integer/text five-key
inputs remain unchanged. Object datetime fixture keys continue using guarded
reference fallback; this repair does not broaden bulk eligibility.

Changed times, naive/aware substitutions, changed timezone names or offsets,
and text keys reject. Caller strings are never parsed into datetime keys.
Only the retained datetime encoding is parsed during schema validation.
Arbitrary objects and datetime subclasses receive no new string-equality or
boxing exception. Older class-specific `other` cells keep their existing
strict same-class contract; they are not upgraded by guessing omitted timezone
provenance. Re-compose from the original ledger to produce the new encoding.

### Storage and supported scalar limits

Persistence of `identity_evidence` JSON does **not** make schema-free CSV
lossless. For example, an original Float32 grouping value may be written as
`1.1`; default `read_csv` loads a numerically different binary64 value. Even
`float_precision='round_trip'` cannot infer the original Float32 dtype from
that short text. Preserve the ticket grouping schema explicitly:

```python
tickets = pd.read_csv("tickets.csv", dtype={"round": "Float32"},
                      float_precision="round_trip")
members = pd.read_csv("members.csv", float_precision="round_trip")
```

This example applies to an otherwise ordinary fixture schema whose only
Float32 group assertion is `round`. Supply every relevant original grouping
dtype in your actual schema, and restore the existing timestamp columns from
their saved schema as well (for example, `pd.to_datetime` with the appropriate
timezone). In pandas 3, an inferred string column cannot receive native
timestamp values during re-finalization. Do not narrow membership values to repair a
failed guard: that could round genuinely changed data. Datetime fixture keys
also require explicit schema restoration, including their original timezone
representation, which CSV does not preserve. Prefer a dtype-preserving
retained artifact such as the existing pickle round trip when these keys are
present. Keep the original evidence rather than reconstructing it from boxed
memberships.

Floating identity support remains finite binary16/32/64. Extended-precision
`longdouble` (such as Linux float128) is explicitly rejected rather than rounded
to binary64. Legacy floating membership without original evidence still fails
with the explicit original-ledger recovery instruction. The independent
numeric-inequality guards and single-hash bound are unchanged.

### Datetime repair verification

- **1,279 passed** in the affected Windows pandas 2.2.3 suite, including browser
  and installed-wheel checks; **1,052 passed** in the overlapping pandas 3.0.6
  suite. No final failures or skips. All 33 new datetime/CSV cases also passed
  after the CSV test fixture explicitly restored its timestamp schema.
- All eight independent naive/UTC × full/summary × auto/reference
  reproductions compose and pass both helpers. These object-key cases record
  zero bulk calls, as expected for guarded fallback.
- The eight minimal numeric mutations and 88 directional mutations still
  reject. All 76 malformed/missing identity-evidence cases reject. Each valid
  or stale 4/6/8/32-key probe computes exactly one hash. Valid Float32, generic
  mixed-key and partial-group cases retain their established behavior.
- Prior 90-case composite and 50-case dtype comparisons remain exact; six
  forged memberships, 30 malformed nested records and 18 invalid timestamps
  reject. All 103 prior adversarial statuses are unchanged.
- Original 30/100-fixture replay retains 109/386 candidates and 73/96 selected
  tickets with actual bulk dispatch, exact reference tables/dtypes/attrs,
  quote evidence, native HTML/CSV and pickle. Published HTML/CSV hashes match
  the previous repair. Original inputs and all 4,767 PMFs per model are unchanged.

[Datetime verification record](guarded_bulk_datetime_verification.json) records
the source hashes, independent reproductions, suite results and storage limits.
No new performance claim, model refit, forecast regeneration, research study
or merge accompanies this compatibility repair.
