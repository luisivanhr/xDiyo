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
- Unique nonmissing event_id fixtures and a RangeIndex. uint64 fixture identities,
  including adjacent values above 2**53 and values above 2**63, remain exact.
- Float64 or integral original odds; integral products must fit signed int64.
  No conversion of float32/object/numeric-string odds to qualify for bulk.

Other valid configurations use the repaired reference: summary mode, other gate
settings/custom hooks, multiple or differently named templates, duplicate-row
indices, alternative payoff/price representations, static allocations, bankroll
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
