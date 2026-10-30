# Mandatory betting temporal provenance

The shared betting layer now validates temporal evidence automatically. There is
no guard switch. This changes admission to verified comparisons, not the payoff
formula, strategy grouping, SeasonCountdown, or the independence assumption used
for multiplying probabilities. No experiment was rerun for this change.

## Timing contract

- Ticket `decision_at` must be **strictly earlier** than every leg's
  `first_kickoff_at`. For a resumed match this is its original start, not the
  resumption. Legacy `kickoff_at` supplies a contradiction check but cannot prove
  that an earlier first start did not occur.
- Each forecast's `forecast_as_of` must be at or before ticket decision time.
  An explicit `prediction_cutoff`, if present, must agree with it.
- Information dependencies must be available at or before the forecast cutoff.
  Fitted stages must use the train partition only, with training input and label
  availability at or before their training cutoff, and artifact availability at
  or before the forecast cutoff. Fitting availability cannot be later than the
  declared model availability. Availability equality is allowed; kickoff equality
  is not.
- New evidence timestamps require an explicit timezone. Equivalent offsets are
  compared in UTC. The existing legacy convention of interpreting naive values
  as UTC remains inspectable, but is classified unverified. NaT is missing
  evidence, not permission to bypass a known contradiction. Numeric epochs,
  empty strings and arbitrary objects are rejected.
- `generated_at` and stage `created_at` describe physical creation. They are
  never overwritten with simulated historical dates. Recomputing later from
  eligible historical inputs is legitimate **conditional retrospective research**
  when explicitly declared. It does not demonstrate that the physical model or
  forecast existed at that historical instant.
- `trained_through` and `artifact_vintage` retain the existing gated-stream
  comparisons. They are not automatically converted into model availability.

The complete offered pool is checked before selection, including `take=False`
rows and candidates subsequently rejected for price, probability, EV or gates.
Existing strategy groups define which rows can influence each other. Within
those groups, information cannot be later than the earliest declared decision.
This catches a discarded candidate with a perfectly valid *later* individual
forecast influencing an earlier ticket. The shared layer does not generate
common-cutoff forecasts or infer effective rounds; strategy adapters own that.

A months-long gap between legs does **not** prove leakage. In particular, the
draw-doubles audit's common nominal-round cutoff is not evidence of leakage
merely because some matches were postponed for months.

## Evidence and classifications

`evaluation.temporal_provenance` supplies producer and persistence helpers.
The ordinary `prepare_bets`, `evaluate_bets`, `compose_bets`, pool preparation,
bulk execution, `ticket_batch`, finalization, single allocation, immutable
allocation and bankroll entry points apply validation without caller opt-in.

Every classified result is one of:

| Classification | Meaning |
| --- | --- |
| `verified` | Required identity, content, pipeline, availability, quote and complete-pool evidence is present and internally consistent. |
| `conditional_research` | The otherwise complete evidence uses explicit bounded availability assumptions or a declared retrospective replay. |
| `unverified` | Required evidence is missing, inaccessible, ambiguous, or unsupported. Numerical accounting can remain inspectable. |

Contradictions always raise with offending identities/timestamps where applicable.
They are not converted into assumptions, dropped observations, revised cutoffs or
an unverified success result. A saved status label cannot certify anything:
classification is recalculated from the evidence. Betting metrics carry
`comparison_eligible`; native experiment ranking leaves conditional/unverified
betting metrics visible but unranked. `require_verified_comparison` applies the
same rule to independent consumers. Non-betting rankings are unchanged.

### Forecast producer contract

Call `bind_forecasts(frame, evidence_by_index, match_columns=...)` on a frame
with exact unique row identities. The mapping must match its index exactly;
there is no positional fallback. Include the complete composite fixture keys
when an event ID alone is not globally unique. The helper binds all decision
inputs, probabilities, fixture identities, schedule/group fields, and hashes.
Changing a retained timestamp, feature hash, probability or fixture identity
invalidates that binding. Regenerating a forecast must produce new evidence;
rebinding an already changed retained artifact is rejected.

Required forecast fields are `forecast_as_of`, `first_kickoff_at`,
`feature_hash`, `model_hash`, and `preprocessing_hash`, plus exact fixture keys
and the probabilities actually used. Ticket issue and quote fields are also
required for verified placement. Forecast evidence contains:

```python
{
    "generated_at": physical_creation_time,
    "forecast_as_of": historical_cutoff,
    "model_available_as_of": model_simulated_availability,
    "dependencies": [
        # At least the features, source, schedule and membership roles.
        {"id": identity, "role": role, "hash": content_hash,
         "reference": provenance_reference, "available_as_of": availability,
         "availability_kind": "observed", "content": content}
    ],
    "stages": [
        # Exactly preprocessing, ratings, calibration, tuning and fitting.
        {"kind": stage, "mode": "fitted", "artifact_hash": artifact_hash,
         "artifact": {"hash": artifact_hash, "path": absolute_artifact_path},
         "training_cutoff": train_cutoff, "available_as_of": ready_time,
         "created_at": physical_artifact_creation,
         "fit_rows_hash": exact_training_membership_hash,
         "fit_rows": {"hash": exact_training_membership_hash, "content": training_row_ids},
         "fit_partition": "train",
         "inputs": [{"role": "training_labels", "hash": input_hash,
                     "rows_hash": exact_training_membership_hash,
                     "available_as_of": latest_input_availability,
                     "content": input_content}]}
    ],
    "assumptions": []
}
```

Each unused stage must explicitly declare `mode="not_used"` and a reason;
fitting cannot be omitted. An identity preprocessing stage still has an explicit
artifact hash bound to the forecast. Supervised fitting/calibration/tuning must
include training-label dependencies, not just feature dates. The producer must
provide complete dependency inventories, exact training membership, and label
release/completion dates, including delayed administrative results.

Content is either JSON-compatible inline `content`, hashed with
`dependency_hash(content)`, or an absolute `path` with a SHA-256 digest of its
file bytes. A bare opaque hash/reference without accessible content cannot verify.
Changed content fails, and inaccessible content becomes unverified on reload.
File hashing and certificate parsing are reused only within one validation call;
there is no durable approval cache. Each subsequent call rechecks dependencies.

An assumed dependency needs a matching assumption ID with exactly bounded
`scope` equal to that dependency's ID, `rationale`, `reference` and
`available_as_of`. A retrospective replay requires an assumption with
`scope="retrospective_replay"` and the same mandatory declaration fields.
Missing evidence is not silently converted into an assumption.

For multiple forecast streams, `model_streams` maps each model name to its own
serialized `forecast_evidence`, constructed on the same fixture keys with that
stream's unprefixed fields. Stream-specific `model::...` fields override shared
fields during validation. Legacy stream timestamps alone remain unverified.

The working synthetic producer and all five stage declarations are exercised in
`tests/analytics/test_temporal_provenance.py::population` and
`test_verified_flat_and_exact_identity_persistence`. These are contract tests,
not verified historical production forecasts.

### Quotes, tickets and accounting

The existing `QuoteAvailability` contract still owns quote identity and observed
versus research-assumed publication. An opening/closing label is not a publication
timestamp. Unknown opening availability cannot verify a historical quote.

Version `xdiyo.temporal.v1` persists forecast evidence, complete candidate pools,
their policies, dependency hashes, exact ticket membership, quote values and
decision times in ordinary table columns. Frozen allocation tables keep that
evidence. Direct helpers validate retained membership and policy rather than
silently rebinding it. Optional legacy duplicated model identity assertions may
be absent, but that absence does not confer verification; conflicting assertions
still fail. Existing native identity checks remain in place.

Pool and comparison certificates use a lossless zlib/base64 envelope prefixed
`xdiyo.temporal.v1:z:` to keep repeated exports bounded in practice. They retain
the complete records, not a lossy summary. `inspect_certificate(value)` decodes
and revalidates them. Tables use the native round-trip string dtype. Repeated Arrow-backed pool
strings share their initial buffers rather than allocating a separate copy for
every row; ordinary assignment still works. File reloads follow pandas storage
behavior and may materialize these strings.

Allocation receipts bind the resolved amounts and allocation/risk policy. Ledger event hashes also bind placement/settlement
amounts, cash, reserves, wealth, and dependency receipts; reload checks these
bindings and conservation. Naive original settlement-availability times remain
unverified even when the legacy parser can normalize them to UTC.
Unknown fitted staking/gate provenance and a bare decision-availability boolean
do not establish a verified fitted policy. Legacy manual allocations remain
unverified. No outcome values enter the candidate certificate or allocation
inputs. Cash/reserve arithmetic is unchanged.

Legacy selection/stake callbacks retain `context.y.index` and column names for
alignment, but receive no observed label values or history. They receive copies
of prediction inputs and outcome-free metadata. Arbitrary callback provenance is
unverified; a callable's existence is not proof of its historical information set.
Serialized audit/certificate payloads are withheld from selector and staking
inputs: a full audit may contain later independent groups' forecasts.

`settlements_from_legs` still waits for **all** result-availability timestamps,
including a later leg after an already-known loss. Unknown results remain pending.
For complete evidence, pass full ticket membership with distinct `leg_id` values,
not just ticket/leg identifiers. Its receipt binds the complete membership,
release times and multipliers. Replay verifies it before funding. Direct ledger
settlement without complete evidence remains unverified. Settlement reserves are
released only at availability; simultaneous event ordering remains deterministic.

## Migration, reload and exports

- Existing recipes need no new setting. They run the guard automatically.
- Existing numerical payoff/stake outputs are preserved unless they contain a
  newly rejected temporal contradiction (including placement exactly at kickoff).
- Old artifacts gain unverified classification when consumed/recovered; no
  timestamps, training histories, first starts or quote publications are invented.
- Native recovery revalidates evidence and metric bindings. Portable
  `save_temporal_table`/`load_temporal_table` do the same for Zstandard Parquet.
  Ordinary report tables and exports retain JSON evidence columns and disclosure.
- Modifying retained dependencies, quotes, probabilities, membership or policy
  invalidates certificates. If source/model files have moved, restore their exact
  content at the declared path or regenerate an explicitly migrated evidence
  package; do not relabel the old artifact verified.
- Conditional research remains inspectable but is not silently mixed into native
  verified betting leaderboards. A false saved `temporal_status="verified"` or
  `comparison_eligible=True` cannot bypass this rule.

## Evidence limits

This is an integrity and temporal-consistency contract, not a signature scheme
or an independent attestation of a vendor's publication history. A malicious
producer that replaces all content, timestamps and hashes can forge a wholly new
story. The guard cannot discover omitted real-world sources or prove that a
producer's observed-availability declaration is truthful. Content references and
complete inventories must come from the actual producing pipeline.

Current legacy forecasts/models/quotes do not automatically acquire missing
historical provenance. Unknown schedule revisions, missing first-start evidence,
unknown model availability and incomplete fitted-policy lineage remain explicit
limitations. Physical recomputation today is not proof of historical production.
No collector, dataset, model training, model architecture or SeasonCountdown
behavior is changed by this implementation.

## Verification and changed files

`verify.py broader-1`, `verify.py broader-2`, `verify.py broader-3`,
`verify.py focused`, and `verify.py browser` reproduce the committed receipts
using the selected Python interpreter. The three broader partitions include every
test module in `verify.py broader`, without exclusions. They extract main's
tracked source and badge assets into an isolated temporary directory, overlay
only this patch's source, and run synthetic regression fixtures. Unrelated
attention-reporter edits in the working checkout are excluded. Each receipt
records the exact command, interpreter/package versions, source hashes before
and after execution, isolated runtime hashes, test counts, and log/XML hashes.
Use the accompanying `verification.md` for final results and limitations.

Changes by area:

- `evaluation/temporal_provenance.py`: one shared evidence contract, automatic
  validation, pool/ticket bindings, classifications and persistence helpers.
- Existing evaluation entry points (`betting`, `bet_decisions`, `tickets`,
  `all_combinations`, `bulk_tickets`, `ticket_allocation`, `decision_layer`,
  `stake_policy`, `bankroll`) call that contract and retain native quote,
  identity, numerical product, and reserve checks.
- Reporting contracts and betting reporters revalidate/export the evidence;
  experiment recovery and ranking prevent unsupported verified comparisons.
- Focused temporal regressions plus two small existing-test adjustments: a
  placement formerly made exactly at kickoff now uses a strictly earlier time;
  a performance fixture now supplies the same model names as its bound policy
  and accounts for the additional unique-quote checks. Neither adjustment
  weakens contradiction tests or changes payoff expectations.

Numerical accounting formulas, model mathematics and experiment selections are
unchanged. Newly rejected contradictory timestamps/policy substitutions are
correctness changes. Missing historical evidence is newly exposed uncertainty,
not a claim that the affected experiment leaked.
