# Strict consensus audit follow-up

Date: 7–8 October 2026. Branch: `codex/research-quote-availability`, unmerged.
Reference: `e7ff491a11349275f42d972419caf78d8385f0e8`.

## Separate correctness changes

Commit `1e16f91` repairs the two inherited malformed paths identified in the
strict audit. Reserved native outcome names are checked in namespace segments
(`a::won`, `a::settled_at`), while ordinary `quote_status` remains allowed.
Direct finalization revalidates original model quote references under an
explicit quote contract; it no longer trusts a previous composition preflight.
These rejections are intentional behavior changes, separate from optimization.

## Full-mode optimization

- Nested ticket evidence is flattened once per validation invocation. Every
  original leg is checked, with its own ticket decision time and distinct
  fixture identity. No original stream is replaced with candidate evidence.
  Raw assumed timestamps must have a timezone before normalization. If string
  formats differ between tickets, parsing falls back to the original per-ticket
  parser, not permissive coercion. Per-ticket price products keep their original
  ordered pandas/NumPy reductions.
- Membership uses one positional index over the original frame. It preserves
  all conflicting rows, exact IDs, arbitrary/duplicate row labels and leg order.
  It does not retain thousands of small DataFrames. Model probabilities and
  timestamps are parsed per template; settlement uses just its required columns.
  Model timestamp strings that differ between tickets retain the original
  per-ticket parser fallback.
- Group-summary positions are indexed once. Stakes still use the original
  ordered `Series.sum`; no regrouped float reduction replaces it.
- No evidence cache persists across invocations, models, contracts or as-of
  times. Typed compact evidence is immutable data, never an approval token.

## Explicit audit storage

`AllCombinations(audit_level="full")` remains the default. Existing full-mode
outputs, ordering and complete attrs remain compatible on valid inputs.
`audit_level="summary"` is available in the native UI and recipe exports.
All templates in a slip must choose the same level. Parlay/MultiBet remain full;
use full for slips mixing those template types.

```python
from dataclasses import replace

compact_policy = replace(existing_all_combinations, audit_level="summary")
# Native builder: Compose tickets -> AllCombinations -> Audit level -> summary.

# The example adapter also configures compact fixture OR ballots explicitly:
result = consensus(legs, model_pmfs, quote_contract, audit_level="summary")
```

Summary mode runs the same guards and decisions. It keeps:

- Exact selected ticket/membership rows, original probabilities and EVs,
  stake/allocation and settlement/accounting results.
- Compact selected model ballots in `selected_model_values`, aggregated
  model/reason counts, group selection summaries and research declarations.
- `audit_manifest`: runtime source-content hash, available Git revision,
  dependency versions, ordered input and retained-model-stream fingerprints,
  snapshot/crosswalk pins, original time bounds, complete configured policies,
  candidate/selected/rejected counts, and explicit omitted-detail declarations.
  Model mappings are pinned separately per template; the research adapter also
  fingerprints both original PMF columns and their indices before reduction to
  a draw probability. Gate counts exclude templates that have no gate.

It avoids materializing verbose candidate dictionaries, repeated per-ballot
quote/timing payloads and repeated ticket JSON solely to discard them. Immutable
typed leg evidence is still validated normally before its verbose form is omitted.
The original member columns remain available. The example's fixture ballot
table retains compact numerical decisions, without repeated provenance copies.

`decision_policy_audit`, `ticket_candidates` and ticket `quote_legs` are absent
in summary mode by design. This does not mean zero candidates/rejections or a
complete empty audit. Native reports show the manifest, counts, selected model
values and explicit omission notes. Parquet/JSON preserve the manifest. `off`
is not supported. Full mode remains the detailed debugging/replay option.

Fingerprints identify supplied retained streams, not independently authenticated
model weights. Installed/snapshot sources without Git metadata report a null
revision and retain their exact runtime source hash and dependency versions;
callers should retain their source pin with the run. Unparseable timestamps on
unconsumed rows are counted in the manifest, without changing native eligibility
or weakening validation of consumed evidence. Observed quote times stay unknown
when unknown; research availability and retrospective simulation clocks do not
establish historical tradability or observed artifact issuance.

## Downstream full-audit consumption

Full-mode attrs intentionally remain public. Repeated pandas row/slice operations
can copy them. Preserve audits separately and make one working copy:

```python
audit_metadata = dict(tickets.attrs)
working_tickets = tickets.copy()
working_tickets.attrs = {}
# Analyze working_tickets; persist audit_metadata with the original output.
```

Do not discard the original metadata or describe the stripped working frame as
a complete audit. Benchmark report generation and serialization separately;
moving those costs outside the native clock does not remove them.
The native HTML renderer uses this pattern on both ticket and membership working
frames, preserving caller attrs while preventing per-row manifest copies.

## Verification and measurements

### Tests and exact comparisons

- **453 passed**, including the two native browser edit/export tests and the
  installed-wheel test (43.92s; seven dependency deprecation warnings).
- **168 passed on pandas 2.2.3** (12.22s; 166 dependency warnings) and **168
  passed on pandas 3.0.6** (15.81s). These cover full/summary computation,
  invalid evidence, missing/empty cases, allocation, conflicting metadata,
  reused IDs, per-ticket timestamp formats and per-template model fingerprints.
- The supplied independent harness has **13 exact supported output matches**
  against e7, including recursive attrs. Exactly the three inherited malformed
  cases in the separate guard commit now reject: edited direct model quote ID,
  `a::won`, and `a::settled_at`.
- The measured matrix adds **10 exact e7/full comparisons** and **12 exact
  full/summary computational comparisons**, including both large sizes.
  Summary selected model p/EV/votes, counts and omission declarations were
  compared separately; omitted attrs were not pretended equal.
- After final edge-case compatibility fixes, an additional fresh 1,008-candidate
  check in each mode confirmed final full/e7 exact recursive equality and final
  full/summary computational equality. Full native wall was 12.14s; summary
  was 12.63s. These single checks are not mixed into five-repeat medians.

No full-library certification is claimed. Test fits use small synthetic test
fixtures; no user experiment was fitted or resumed.

### Fresh-process timings

Windows, Python 3.12.6, pandas 2.2.3, NumPy 2.5.3; fixed hash seed and one
BLAS/OpenMP/NumExpr thread. Modest sizes are five randomized serial repetitions
per arm; each large size has one bounded sample per mode. Warm-ups and separate
tracemalloc samples are excluded. Timings vary with the host and are not SLAs.
All 40 raw records, including warm-ups and allocation probes, are in
[strict_consensus_timings.csv](strict_consensus_timings.csv).

| Candidates | Mode | Repeats | Native wall s | Native CPU s | Native peak MiB | HTML s | Serialization s | Serialized MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 84 | e7 full | 5 | 1.591 | 1.578 | 96.4 | 0.216 | 0.0035 | 0.268 |
| 84 | optimized full | 5 | 0.956 | 0.953 | 91.0 | 0.189 | 0.0039 | 0.268 |
| 84 | summary | 5 | 1.019 | 1.016 | 91.0 | 0.186 | 0.0035 | 0.121 |
| 1,008 | e7 full | 5 | 22.451 | 22.156 | 185.1 | 2.274 | 0.0103 | 3.069 |
| 1,008 | optimized full | 5 | 12.637 | 12.469 | 120.1 | 1.703 | 0.0099 | 3.069 |
| 1,008 | summary | 5 | 11.976 | 11.875 | 117.8 | 1.687 | 0.0063 | 1.242 |
| 5,040 | optimized full | 1 | 57.484 | 57.172 | 238.2 | 7.623 | 0.0379 | 15.316 |
| 5,040 | summary | 1 | 54.808 | 54.484 | 232.3 | 7.324 | 0.0182 | 6.135 |
| 17,024 | optimized full | 1 | 196.909 | 195.953 | 555.9 | 31.118 | 0.1429 | 51.789 |
| 17,024 | summary | 1 | 204.991 | 204.047 | 543.6 | 30.712 | 0.0698 | 20.682 |

The measured full-mode native median improves about **44% at 1,008 candidates**
and preserves the original serialized bytes. Summary removes 1,008 verbose
candidate records and 2,016 full model-audit dictionaries at that size, retaining
compact selected ballots, with about **60% smaller serialized output**. Summary
does **not** consistently improve native runtime or greatly reduce peak RAM;
at 17,024 it is slower. Native + HTML + serialization at that size is 228.17s
full versus 235.77s summary. This is a storage tradeoff, not a promised speedup.

Imports are measured separately (roughly 0.54–0.63s); input generation is
0.01–0.05s. Independent verification is also timed: at 17,024 it costs 4.59s
full versus 119.61s summary because the test oracle groups frames carrying the
manifest. That diagnostic cost is retained in the CSV and is not application
computation. Including import, generation and the oracle, the whole measured
worker stages at that size total about 233.4s full and 356.1s summary. The native
renderer strips only its temporary working-copy attrs; arbitrary downstream
consumers still need the documented working-copy pattern.

Separate 84-candidate tracemalloc probes report peak tracked allocations of
8.79 MB e7, 3.26 MB optimized full and 3.24 MB summary. Their instrumented wall
times (5.64s, 3.60s, 3.46s) are not unprofiled speed measurements. Process peak
includes imports and is not the same as tracked Python allocation peak; RSS
after reporting is recorded separately in the CSV.

The randomized matrix is pinned to runtime source hash
`61a1298bf4facb37e12bf266bfd4c15a896a5def03f0e5786274e06123884632`.
Final compatibility follow-ups preserve column-oriented legacy quote JSON,
per-ticket model time formats, unreferenced direct-helper members and mixed
gated/ungated summary counts. The final tested runtime hash is
`5a7514a546ffe3904d2132bc9796f8e7f8d13b334f7be0f6bc0f7bdc72bbd090`;
the additional final-source checks above validate it. Large timings belong to
the measured snapshot, not a second run of those final edge-case follow-ups.

An initial end-to-end matrix exposed summary-manifest copying in the renderer:
the 1,008-candidate summary report alone took about 33 seconds. The owned run was
stopped during the 5,040-candidate summary case, its incomplete sample was not
counted as a result, and its completed logs remain under `final-results`.
`REPORT_REPAIR_AMENDMENT.json` records this interruption. The repaired renderer
was measured in a separate fresh-process matrix under `repaired-results`;
the earlier matrix is superseded, not silently overwritten.

The supplied strict harness was run against isolated e7 and optimized sources.
Only its platform-specific memory counter was adapted to Windows
`PeakWorkingSetSize`; source/oracle checks were preserved. Missing native utils
in the first snapshot and a Windows counter signature were corrected before any
successful samples. Initial failed attempts are not timing observations.

The first matrix has five serial randomized pairs at 84, 112, 168, 504 and
1,008 candidates, plus seven edge-case pairs. All 33 complete output pairs
(including one discarded warm-up pair) match exactly, including recursive attrs.
Final full/summary measurements use `examples/benchmark_quote_storage.py` in
fresh processes with fixed thread limits and hash seed. Imports, construction,
native wall/CPU, independent oracle, HTML generation and serialization are
separately timed. Tracemalloc probes are separate from ordinary timing samples.
The large all-accepted oracle checks all unordered fixture pairs and original
probability/EV arithmetic; it is not an old/new parity measurement when e7 is
not run at that size. All outcomes in large samples are missing synthetic data.

Local plans, logs, trusted output pickles and measurements live under
`%TEMP%/xdiyo-strict-speed`. The private real slice and prior 24 saved-model
replays are excluded from the supplied packet and were not rerun. No refits,
dataset edits, full research resumption or merge are part of this patch.

## Further vectorization from db21b5f (8 October 2026)

The follow-up uses `db21b5f` as its control, rather than attributing the previous
e7 speedup to the new changes. The ticket membership index was already backed by
a hash map; introducing another hash table would not remove the dominant work.
The profile instead showed repeated wide-frame dictionary conversion, per-ticket
pandas model lookups/checks, and repeated native quote validation.

- Serialize only the columns consumed by economic identity, quote evidence and
  HTML presentation. Retained output tables still contain every original column.
  Dynamic `quote_*` fields remain included in quote evidence.
- Validate the native quote contract once over all consumed membership rows
  for each template inside `ticket_batch`. The validator still compares each
  original row with its own decision time. No duplicate rows are discarded, no
  model stream is replaced, and no approval persists across calls. Per-ticket
  timestamp parsing is retained as a fallback; custom contract subclasses keep
  their original per-ticket hook behavior.
- Validate original model probability ranges and provenance in vectorized
  batches with an explicit mapping to each ticket's own cutoff. Use integer
  array positions for subsequent lookup. Ordered per-ticket NumPy products are
  unchanged, with bitwise regressions for two, three, five and eight legs. No
  regrouped product, probability normalization, fast-math or new parallel
  floating reduction is used near the strict EV boundary.
- The benchmark oracle also now strips attrs only on a temporary working copy.
  Both control and optimized runs use this same revised worker. This removes
  the prior diagnostic-only manifest copying cost; it is not attributed to the
  library's native-call speedup. Output HTML hashes are checked separately.

The initial 504-candidate diagnostic profile was 14.06s native; after the first
two changes it was 8.87s. Profile overhead is substantial; the fresh-process
measurements below are the speed evidence. This follow-up retains full and
summary modes, recipe/API settings, all guards and output metadata contracts.

### Follow-up verification

- **464 affected regressions passed** (45.17s; seven dependency warnings),
  including browser and isolated-wheel checks. **179 passed on pandas 2.2.3**
  (12.64s; 194 dependency warnings) and **179 on pandas 3.0.6** (17.26s).
- **22 same-mode paired outputs and HTML hashes match exactly**: five repeats
  at 84 and 1,008 candidates in each mode, plus one paired 5,040-candidate run
  per mode. Full metadata matches exactly. For summary metadata, only the
  expected `runtime_source_sha256` change is excluded from equality.
- The supplied independent harness preserves all **13 supported outputs**
  exactly and all **32 acceptance/rejection outcomes**. There are no new
  intentional behavior changes in this follow-up.
- Both 17,024-candidate optimized runs pass the independent pair/probability/EV
  oracle. Their computational outputs match across modes. The full output also
  matches the retained large full output from the previous audit exactly.
  This is correctness evidence, not a fresh large baseline timing.
- New structural tests require one native quote check over all original rows
  during ticket-batch preparation, preserve each ticket's own cutoff and exact
  product bits, and prohibit serializing unrelated wide columns for quotes or
  HTML. Original tables, ordering and caller attrs are preserved.

### Follow-up timings

Same Windows/Python 3.12.6/pandas 2.2.3/NumPy 2.5.3 environment and fixed thread
limits as above. Five randomized serial fresh-process runs per modest arm;
one bounded run per large arm. A warm-up per arm and four separate allocation
probes are excluded from ordinary timing medians. There were no timeouts.
All **54 raw records**, including imports, input construction, independent
verification, native wall/CPU, HTML, serialization and memory, are retained in
[consensus_vectorized_timings.csv](consensus_vectorized_timings.csv).

| Candidates | Source / mode | Repeats | Native wall s | Native CPU s | HTML s | Native peak MiB | Serialized MB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 84 | db21 full | 5 | 0.928 | 0.938 | 0.185 | 91.3 | 0.268 |
| 84 | optimized full | 5 | 0.589 | 0.594 | 0.072 | 90.9 | 0.268 |
| 84 | db21 summary | 5 | 0.993 | 0.984 | 0.186 | 91.5 | 0.121 |
| 84 | optimized summary | 5 | 0.664 | 0.656 | 0.070 | 91.4 | 0.121 |
| 1,008 | db21 full | 5 | 10.773 | 10.750 | 1.553 | 120.4 | 3.069 |
| 1,008 | optimized full | 5 | 6.549 | 6.500 | 0.275 | 119.6 | 3.069 |
| 1,008 | db21 summary | 5 | 11.022 | 10.969 | 1.520 | 118.4 | 1.242 |
| 1,008 | optimized summary | 5 | 6.606 | 6.609 | 0.255 | 118.7 | 1.242 |
| 5,040 | db21 full | 1 | 54.322 | 54.109 | 8.844 | 238.4 | 15.316 |
| 5,040 | optimized full | 1 | 37.061 | 36.938 | 1.084 | 237.9 | 15.316 |
| 5,040 | db21 summary | 1 | 61.726 | 61.516 | 7.584 | 232.6 | 6.135 |
| 5,040 | optimized summary | 1 | 37.323 | 36.984 | 1.104 | 231.6 | 6.135 |
| 17,024 | optimized full | 1 | 128.330 | 127.922 | 4.155 | 554.4 | 51.789 |
| 17,024 | optimized summary | 1 | 124.285 | 124.000 | 3.765 | 547.3 | 20.682 |

At 1,008 candidates, native medians improve **39% full / 40% summary** relative
to db21; HTML improves **82% / 83%**. Serialized bytes are unchanged within each
mode. Memory is broadly unchanged. Native + HTML + serialization at 17,024 is
132.63s full and 128.11s summary; no fresh db21 control was timed at that size.
These synthetic measurements do not certify a universal speedup or authorize
resuming the private saved-model study.

Separate 84-candidate tracked-allocation peaks are 3.258 MB db21 full versus
3.233 MB optimized full, and 3.245 MB db21 summary versus 3.241 MB optimized
summary. Instrumented times are excluded from ordinary timing comparisons.

Source reference: `db21b5f`. Runtime snapshot hashes (raw file bytes, including
line endings) are `fb43a9167f8e3a66d2c7750c76eb189b78f6a62cc07ed161109fb6acedce90cd`
for the archived baseline and
`7bb1d47db0fd116f85cbd4d33a0ec79bfdc855ecd72d33b790f47fa496edce36` for the optimized
source. Git archive normalizes the baseline's line endings, so its runtime hash
differs from the earlier Windows working-copy hash without implying another
behavioral change. The final runtime source matches the measured snapshot.
Plans, profiles and trusted local output pickles are under
`%TEMP%/xdiyo-consensus-faster-db21`. Private cached-data/model replays remain
unavailable; no dataset, model, frozen experiment, main or parent branch changed.
