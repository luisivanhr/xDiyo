# Whole-group combinations and binary draw decisions

`AllCombinations` creates every unordered combination of `legs` distinct eligible
matches in each league/season/stage/round and fold occurrence. It is a separate
policy: `Parlay` still creates disjoint batches, and `MultiBet` still combines
inside fixed-size batches. All three can be named templates inside `BetSlip`.

```python
from xdiyo_analytics.evaluation import AllCombinations, BetSlip

pairs = AllCombinations(legs=2)
comparison = BetSlip({
    "pairs": pairs,
    "triples": AllCombinations(legs=3),
    "quads": AllCombinations(legs=4),
})
```

Each template places additional independent stakes in the accounting sense;
this does **not** assume statistical independence of match outcomes. Comparing
sizes is configurable infrastructure, not an instruction to search sizes or bet.

## Eligibility, identity and stages

Only legs with `take=True` and finite decimal odds greater than one enter a pool.
Missing or invalid prices are excluded before counting. Native odds sources
retain their stricter quote validation and never substitute closing for missing
opening quotes. No settlement or profit is used for eligibility or ordering.

The grouping is `league_round`: fold, competition (or league), season, stage,
and round. The default stage column is `tournament_id`. Alternatively use
`stage_id` or `stage`. Missing or null stage identity fails clearly. Set
`stage_column=None` only to **explicitly declare single-stage seasons**. The
policy does not infer stages from dates or merge repeated round numbers.
Stage identifiers are retained from label creation through dataset metadata.

Reporters pass the dataset's full match identity to composition. Direct calls
should pass the same `match_columns`; row positions and external quote IDs are
never substituted. Large integer identifiers are kept exact.

Identical duplicate selections for one event collapse to one leg. Conflicting
options, prices, probabilities, grouping or settlement evidence raise an error;
the code does not choose the more profitable alternative. Row-position
bookkeeping and derived single-bet profit/payout are excluded from duplicate
comparison. One ticket cannot contain an event twice, but different tickets
intentionally share events. Ordering uses group identity, kickoff and exact
fixture identity. IDs hash template, group and unordered event membership;
row shuffling and outcome/profit changes do not change ticket membership or IDs.

## Counts, stake and expansion limit

For a group of \(n\) eligible distinct events and a chosen \(k\):

\[
N(n,k)=\begin{cases}\binom{n}{k},&n\geq k,\\0,&n<k.\end{cases}
\qquad S=s\sum_g N(n_g,k).
\]

`legs` must be a positive integer; booleans and fractional values are invalid.
Size one creates one single per eligible event. Empty and undersized groups
normally produce zero tickets. There is no fixed pool size or dropped tail.
The default stake \(s=1\) is **per ticket**, not per group. Original leg stakes
are replaced, so they are not charged again.

```python
from xdiyo_analytics.evaluation import preview_combinations, compose_bets

preview = preview_combinations(selected_leg_ledger, pairs,
                               match_columns=dataset.match_columns)
print(preview)
print(preview.attrs)  # total_tickets, total_stake, exceeded_limits
tickets, membership, metrics = compose_bets(
    selected_leg_ledger, pairs, match_columns=dataset.match_columns)
```

The preview includes group identity, eligible events, chosen size, ticket count,
expected stake, input events, unselected events, missing/invalid-price events,
missing-probability abstentions when decision evidence exists, and duplicate
rows removed. Exclusion counts can overlap. A manually supplied boolean mask
cannot explain why it excluded a row; probability exclusions then remain unknown
to composition, rather than inferred from outcomes.

`max_tickets=100000` limits the **sum across all groups per template in the report
scope**. Counts use exact integer arithmetic. Every AllCombinations template is
checked before any ticket is materialized, including other templates in its
BetSlip. Above the limit the error gives the count and limit; nothing is sampled
or truncated. Preview remains available above the limit. Reduce the eligible
population, change the size, or deliberately raise the limit. A BetSlip with
several templates adds their respective counts and stakes.

## Binary draw filtering

The predicted label is the exact native draw option:

```python
from xdiyo_analytics.labels import BetOption, Outcome
from xdiyo_analytics.evaluation import BinaryDrawThreshold, BetOffer

draw = BetOption(Outcome(perspective="home"), selection="draw")
baseline = BinaryDrawThreshold()  # None: every event with a valid quote
filtered = BinaryDrawThreshold(max_non_draw_probability=0.5)
offers = {"draw": BetOffer(draw, opening_draw_odds)}
```

The model target uses **1=draw and 0=non-draw**. The policy requires one draw offer
and a matching declared target definition. It consumes retained `predict_proba`
(or explicitly selected raw probabilities), mapping class 1 to win probability
and class 0 to loss probability. It never treats binary class 0 as the draw class
of a three-way outcome model. Calibration remains upstream.

For a supplied, frozen threshold \(t\), eligibility is

\[
I_i=\mathbf{1}\{P_i(\text{non-draw})\leq t\}
     \mathbf{1}\{\text{valid decimal quote}_i\}.
\]

Equality is retained. Missing probabilities abstain in the filtered strategy.
`None` is the unfiltered baseline, including validly quoted events whose
probabilities are missing. No EV ranking, minimum-odds threshold, tuning,
rebalancing or retrospective profitability filter is added.

Use native `OddsSeries(market="1x2", selection="draw", quote_type="opening", ...)`
with explicit seasons and reviewed fixture mapping. Paths and manifest pins
survive UI and recipe round-trips. Opening means earliest recorded prematch
quote; bookmaker identity and actual quote timestamps are unavailable. This
remains a price scenario, not evidence of simultaneous executable prices.

## Settlement and reports

Settlement uses existing native ticket rules. With all winning legs,

\[
\text{gross payout}=s\prod_{j=1}^{k}o_j,\qquad
\text{net profit}=\text{gross payout}-s.
\]

A losing leg loses the ticket. `on_push`/`on_void` preserve the existing choices:
`remove` uses a factor of one, `refund` returns the ticket stake unless a leg
loses, and `loss` treats it as losing. Missing outcomes leave payout and profit
missing unless an already losing leg or an explicit refund rule determines
settlement. Unresolved profit is never relabeled as settled zero.

Joint probabilities remain blank unless `probability_mode="independent"` is
explicitly selected. That displays the product with its independence label;
composition and observed settlement do not need this assumption.

`BetOutcomeReporter` and `BetPerformanceReporter` expose the ticket ledger,
membership, `combination_preview`, metrics and first/last leg times. A performance
reporter sourcing the outcome reporter reuses those tickets and their provenance
without staking legs again. Profit curves show cumulative **known** profit;
they are not bankroll simulations and unresolved tickets remain visible.

## Builder and portable example

In Post-training analysis, add a Bet Outcome Reporter, one draw offer with
Database odds → 1x2 → draw → opening, and choose **Binary Draw Threshold**.
Enable Maximum non-draw probability for filtering, or disable it for the baseline.
Under Compose tickets choose **All Combinations**, set legs, stage column,
per-ticket stake and maximum tickets. BetSlip supports several named templates.
Add a Bet Performance Reporter sourced from these decisions for the profit plot.

Whole-group count tables appear when reports have eligible prediction/quote
inputs. Data preparation alone cannot determine filtered ticket counts; the UI
states that explicitly. The standalone preview function accepts retained leg
ledgers without training or ticket expansion.

`examples/all_combinations_draw.py` exports a JSON recipe plus Python/notebook
versions containing baseline/filtered decisions and pairs/triples/quads. Paths
are relative; configure your reviewed mapping before use. Running the exporter
only writes configurations, never prepares, trains, predicts or places bets.

## Implementation verification

Run from the repository root in PowerShell, using the project Python environment:

```powershell
$env:PYTHONPATH='src;tests/analytics'
python -m pytest tests/analytics/test_all_combinations.py tests/analytics/test_ui_all_combinations.py tests/analytics/test_ui_odds_roundtrip.py tests/analytics/test_bet_tickets.py tests/analytics/test_bet_outcomes.py tests/analytics/test_post_training_betting.py tests/analytics/test_labels.py tests/analytics/test_ui_inventory.py::test_saved_inventory_has_widgets_help_and_packaged_file tests/analytics/test_ui_inventory.py::test_known_form_schemas_use_saved_inventory_without_signature_inspection tests/analytics/test_ui_inventory.py::test_schema_reads_are_isolated_from_persistent_defaults -k 'not ui_ticket_changes and not nb_pipeline_retains and not ui_classifier_retains' -q -p no:cacheprovider --tb=short
```

Verified on 2 October 2026: **186 passed, 3 deselected**. The three excluded
tests fit models; this verification used synthetic retained predictions,
ledgers, quotes and browser controls without fitting models or running the
research experiment. It covers native settlement, exact previews, binary class
semantics, stage isolation, report totals, recipe exports and preserved odds pins.
