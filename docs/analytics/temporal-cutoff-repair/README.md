# Known dependency timing with missing forecast cutoffs

The review of `4266575f` identified a missing-boundary check: dependency availability
was compared with an optional forecast cutoff but not always with ticket decision
time. A missing cutoff could therefore hide a known later dependency and return
`unverified` instead of rejecting the candidate. The regression was reproduced
against the pre-repair source through `guard_candidates` before editing it.

## Repair

All recognized simulated dependency availability timestamps are checked directly
against decision time, as well as each applicable known forecast/training cutoff.
This covers row fields, prefixed model fields, bound forecast evidence, stages,
artifacts, training membership, stage inputs, nested dependency/input nodes and
model streams. A declared evidence cutoff is still checked when the row cutoff
is absent; the absent row evidence is not filled in or upgraded to verified.

The complete candidate information set is validated, including discarded rows.
Nested model evidence also respects the existing selection group's decision-time
bound. This uses the same forecast validator and keeps original bindings intact;
no timestamp is overwritten. The bound participates in the call-local validation
cache key, preventing reuse of a later-decision validation for an earlier ticket.

Availability equality remains allowed. Placement must remain strictly before
first kickoff. Missing/NaT cutoffs do not grant permission to use known future
information. Without a contradictory known ordering, incomplete provenance stays
`unverified`; eligible, explicitly declared retrospective replay remains
`conditional_research`. Physical `generated_at` and `created_at` are not simulated
availability and are not compared against historical placement as if they were.
Ambiguous legacy `trained_through` / `artifact_vintage` semantics are unchanged.

The certificate format remains v1. Existing certificates and forecasts are
revalidated on reload by the repaired validator. Previously accepted records with
a known contradiction now fail; reclassification alone cannot repair their inputs.
Unknown publication/model availability remains a limitation. Certificates still
provide integrity checking, not independent attestation of the supplied evidence.

No accounting, strategy grouping, SeasonCountdown, standings, experiment fits or
collector data were changed. Unrelated local attention-reporter work is excluded.

## Verification

`test_temporal_missing_cutoff.py` adds 145 cases through the public guard/composer
and reload path. They cover absent/null/NaT cutoff fields, each stage, nested
training evidence and model streams, discarded candidates, earlier group decisions,
equality/timezone equivalence, timely/unknown evidence, and remaining intermediate
cutoff enforcement. Existing retrospective-replay, scalar/bulk, stake, reserve,
quote and artifact tests run alongside them.

`verify.py` archives committed source and overlays only this scoped repair, keeping
unrelated dirty source out of the runtime. Receipts record exact commands, package
versions, input/runtime source hashes, log/XML hashes and source immutability.
The focused Python 3.14 / pandas 2.3.3 run passed **387 tests** with no skips.
The broader Python 3.12 / pandas 3.0.1 run passed **1,398 tests**, also with no
skips. Its three warnings concern sklearn's deprecated `n_jobs` argument and a
rank-deficient synthetic statsmodels fixture; none is a test failure. Both runs
record unchanged tested source. See `focused-py314-receipt.json` and
`broader-py312-receipt.json` for complete commands, hashes and results.

Run from the repository root:

```powershell
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' docs/analytics/temporal-cutoff-repair/verify.py focused
& 'C:/Users/luisi/Documents/Programming/Python/xDiyo/.venv/Scripts/python.exe' docs/analytics/temporal-cutoff-repair/verify.py broader
```

The scripts and receipts are stored without Git newline conversion. Numerical
source uses the repository's ordinary line endings; publication checks compare
tested and committed bytes after normalizing CRLF/LF only.
