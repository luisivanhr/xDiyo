# Explicit research quote availability

This extension is opt-in. Ordinary `DecisionContext` still requires an observed,
nonmissing `quote_at` no later than each candidate's `decision_at`. No vendor
schedule, download date or inferred pre-kickoff time becomes an observed quote
timestamp. Existing recipes and fixed numeric staking need no changes.

## Contract and retained evidence

```python
from xdiyo_analytics.evaluation import QuoteAvailability

availability = QuoteAvailability(
    mode="research_assumed",
    assumption_id="opening-round-minus-1h-v1",
    rationale="Unverified opening availability by league/season/stage/round first kickoff minus one hour.",
    reference="research-protocol/quote-assumption-v1",
)
```

The identity is a caller-supplied stable protocol version, not a claim verified
by the library. Retain the referenced attestation alongside the study. Change
the identity when the assumption changes. Rationale and reference are compared
as well as identity, so reusing an ID cannot conceal a contradictory definition.

Supply `assumed_available_at` separately for every fixture. Its timezone must be
explicit. The intended study rule is the earliest kickoff in the **full original
league/season/stage/round group minus one hour**, before model filtering, not the
earliest kickoff left after selection. Retain that schedule provenance separately.
The library validates supplied evidence; it does not independently verify this
assumption or infer it from a vendor schedule field with unknown timezone.

`availability.annotate(frame)` adds the static declaration columns below. It
neither creates nor fills either timestamp and rejects conflicting declarations.

| Field | Requirement |
| --- | --- |
| `quote_id`, `market`, `selection`, `odds` | Original per-fixture source quote and finite decimal price >1 |
| `quote_snapshot_hash`, `quote_crosswalk_hash` | Nonempty pinned source and mapping provenance |
| `quote_at` | Actual observed time, or missing when unknown |
| `decision_at` | Candidate decision time |
| `assumed_available_at` | Separate timezone-aware assumed availability, no later than decision |
| `quote_availability_mode` | `research_assumed` |
| `quote_assumption_id`, `quote_assumption_rationale`, `quote_assumption_reference` | Exact matching declaration |

An observed time later than assumed availability invalidates the assumption even
when both precede the decision. Mixed modes, blank pins, unknown/future model
evidence and contradictory quote references raise errors. Additional native
`quote_*` provenance and supplied `period`/`line` identities are also compared;
optional missing values match only other missing values.

## Native decisions and tickets

Pass the contract as `DecisionContext(..., quote_availability=availability)` or
`AllCombinations(..., quote_availability=availability)`. The latter requires an
explicit native ticket gate with `missing="error"`. Each model must supply its
own matching `model::field` reference for every quote identity/declaration field,
including its own `model::quote_at` (missing when unknown), and existing
`issued_at`, `trained_through`, `artifact_vintage` timing provenance.

Do not create model references by copying the selected quote merely to satisfy
the checks. They must describe the quote actually paired with that retained
model output. Both streams must agree on the same fixture's source, market,
selection, price, pins, assumption and decision; different fixtures keep their
different quotes and prices. Every eligible leg is checked before expansion,
including undersized groups. The library does not authenticate caller evidence.

Ticket `quote_legs` retains safe per-leg evidence and fixture membership as JSON.
Any unknown observed leg time keeps aggregate `quote_at` missing. Aggregate
maxima never substitute for validating individual legs. Native candidates are
constructed and gated before settlement; decision and allocation contexts do not
receive outcome fields. Arbitrary concealed callback data remains outside this
guarantee.

## Exact consensus recipe

See the runnable [synthetic example](../../examples/research_quote_consensus.py).
It calls existing native `DecisionLayer` twice, with no fitted decision callback:

1. Align both cached PMFs by explicit fixture keys and class labels; require both
   streams valid for **every** eligible fixture before selection.
2. For fixture eligibility only, declare `probability = 1 - original_P_non_draw`.
   Use OR, `metric="probability"`, `threshold=1 - .80`, `strict=False`,
   `missing="error"`. This implements either original non-draw probability <=.80.
   Literal `.20` is not bit-identical to `1-.80`. Do not substitute a separately
   stored draw probability for this complement.
3. Pass the selected fixture identities to `AllCombinations`, grouping by the
   full league/season/stage/round with `legs=2`, `stake=1` and independent products.
4. Use the **original unchanged draw probabilities** from each model for ticket
   valuation. The native ticket gate uses AND, EV, threshold 0 and strict `>`.
   Both original-model EVs must exceed zero with the same actual odds product.

This is not an intersection of separately selected ticket sets, an average PMF,
Kelly allocation or a learned ensemble probability. The two-class example uses
label-based access; it does not assume probability column order.

## UI, exports and reporting

AllCombinations exposes **Quote availability** in the existing builder. Leave it
disabled for the existing path; enable the typed component to select observed or
research-assumed mode. Research mode requires the three documentation fields and
prepared evidence columns. Selecting the form option alone does not manufacture
quote references or assumed times. JSON recipes, Python and notebook exports
retain the component and its explicit configuration.

Ticket, membership, metric, candidate and summary outputs retain assumption
labels. Decision audits retain labels and separate observed/assumed timestamps.
Ticket reporters export a quote-declarations table and display a research-only
notice, including when no tickets pass. Exported numbers under this mode describe
a retrospective simulation, not established historical tradability or live
execution. No source dataset is rewritten.

The declaration also survives Parquet round trips and saved-result reuse by
`BetPerformanceReporter`. Auxiliary report tables without per-row declaration
columns carry `quote_availability_declarations` as JSON, so an independently
downloaded table still identifies the applicable research assumption.

## Review boundary

The implementation and tests use synthetic evidence only. Independent review is
required before resuming the three-feature-set/four-season cached-model study.
No cached model refit, real-data consensus scorecard, dataset mutation, merge or
publication is part of this extension.
