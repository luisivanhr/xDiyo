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
\qquad S_{\mathrm{candidate}}=s\sum_g N(n_g,k).
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

Preview counts remain exact Python integers even beyond floating-point range.
Ordinary expected stakes and total stakes are floats; amounts beyond that range
are finite `decimal.Decimal` values, never infinity. Zero stakes remain zero
even for enormous counts. These are preview amounts only: the complete ticket
count still passes through the expansion limit before materialization.

## Binary draw filtering

The predicted label is the exact native draw option:

```python
from xdiyo_analytics.labels import BetOption, Outcome
from xdiyo_analytics.evaluation import BinaryDrawThreshold, BetOffer

draw = BetOption(Outcome(perspective="home"), selection="draw")
baseline = BinaryDrawThreshold()  # None: every event with a valid quote
filtered = BinaryDrawThreshold(max_non_draw_probability=0.80)
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

## Optional ticket expected-value filter

```python
legacy = AllCombinations(legs=2)  # min_ev=None: unchanged, unfiltered tickets
positive_ev = AllCombinations(
    legs=2, stake=1.0, probability_mode="independent", min_ev=0.0,
)
# Also accepts legs=1, 3, 4, or larger sizes within max_tickets.
```

This filter values **complete tickets**, after the existing single-leg decisions
and quote eligibility. A negative-EV leg can belong to a positive-EV ticket.
It never changes model inputs, predictions, calibration, or the single-leg policy.

For win probabilities \(p_i\), decimal quotes \(o_i\), and stake \(s\):

\[
P_T=\prod_{i\in T}p_i,\qquad O_T=\prod_{i\in T}o_i,\qquad
\operatorname{EV}_T=P_TO_T-1,
\]
\[
\operatorname{select}(T)=\mathbf{1}\{\operatorname{EV}_T>\texttt{min\_ev}\},
\qquad \mathbb E[\text{net profit}_T]=s\operatorname{EV}_T.
\]

`expected_profit` in the ticket ledger and candidate audit is **EV per unit
stake**, distinct from realized `profit` and `payout`. Changing stake never
changes selection. Equality rejects, including EV exactly zero at `min_ev=0`.
There is no rounding, epsilon, or inclusive toggle. This differs from inclusive
single-leg minimum-EV policies, whose semantics remain unchanged. Finite zero,
positive and negative thresholds are accepted; booleans, strings and nonfinite
thresholds fail. `None` disables the filter, including for old recipes.

**Assumptions:** enabling the filter requires an explicit `independent`
probability mode and win/loss probabilities. Multiplication assumes independence
across events, not independence between overlapping tickets. The formula does
not value prospective push/void refunds. Nonzero supplied `p_push` raises;
an absent `p_push` column is the caller's declaration of win/loss probabilities.
Actual outcomes never determine whether this assumption applies. Existing
realized push/void settlement rules still operate after selection.

Missing `p_win` columns raise a clear input error. Missing values reject each
affected candidate as `missing_probability`, without imputing probabilities.
All nonmissing eligible probabilities must be numeric, finite and in [0, 1];
zero and one are valid. Invalid evidence raises even if another leg is missing.
Only `take=True` legs with finite odds greater than one enter candidate pools;
nonnumeric quotes raise. No market-implied or closing-price fallback is added.

Combined odds and payout overflow still raise. Enabled filters also reject
unrepresentable aggregate selected stakes, payouts or profits. With the filter enabled, a
strictly positive probability product below the smallest normal float (including
underflow to zero) raises an unrepresentable-valuation error. An actual zero
probability remains valid. Nonfinite EV raises; comparison uses the computed
unrounded float. Extremely small differences that floating-point arithmetic
cannot represent are not assigned an artificial margin.

### Candidate counts, selected tickets and audits

Every template's exact candidate count is checked against `max_tickets` **before
any expansion**, even if the threshold would reject everything. The cheap
`preview_combinations` never evaluates EV: `ticket_count`, `expected_stake`, and
its `total_tickets`/`total_stake` attributes remain **candidate** quantities.
Selected quantities become known only after bounded valuation.

Only accepted tickets and their legs enter the placed ledger/membership. IDs
depend on template, group and event membership, so a changed threshold does not
rename surviving tickets. Rejections carry no actual stake, payout or profit.
The reporters export two additional ordinary tables, also available internally
as record lists in `tickets.attrs` for direct `compose_bets` callers:

- `ticket_candidates`: template/group/fold, stable ticket ID, JSON fixture
  membership, joint probability, odds, unit `expected_profit`, threshold,
  strict comparator, assumption, `take`, and rejection reason. Reasons are
  `missing_probability` or `ev_not_above_threshold`; accepted rows have no reason.
  With filtering disabled, candidate EV remains unavailable and all candidates pass.
- `ticket_selection_summary`: existing event/exclusion counts, candidate
  tickets/hypothetical stake, selected tickets/actual stake, rejected-by-EV and
  rejected-missing counts, threshold, comparator, assumption and enabled flag.

\[
N_{\mathrm{candidate}}=N_{\mathrm{selected}}+N_{\mathrm{rejected\ EV}}
 +N_{\mathrm{missing\ probability}},\qquad
S_{\mathrm{selected}}=sN_{\mathrm{selected}}.
\]

The two rejection categories are exclusive; existing leg exclusion counters
may overlap. All-rejected pools have zero bets, stake and known profit, with
undefined ROI. Unresolved selected tickets retain missing profit and stay out
of the settled-stake ROI denominator. `BetPerformanceReporter(source=...)`
copies tickets, audits, alternatives and quote provenance without recomposition.
Manual `BetSpec` inputs do not supply probabilities and therefore cannot use an
enabled ticket EV filter without an upstream probability-bearing decision source.
The public `compose_bets` return tuple remains `(tickets, membership, metrics)`.

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
**Minimum ticket EV (per unit stake)** starts disabled. Enabling it starts at
0.0; explicitly choose Independent probability mode. The builder shows a message
for incompatible settings and preserves the entered value. Disabling omits the
field (equivalent to `None`); reenabling starts at 0.0. Save/reopen, JSON and
Python/notebook exports preserve enabled thresholds and the other ticket/odds pins.
Add a Bet Performance Reporter sourced from these decisions for the profit plot.

Whole-group count tables appear when reports have eligible prediction/quote
inputs. Data preparation alone cannot determine filtered ticket counts; the UI
states that explicitly. The standalone preview function accepts retained leg
ledgers without training or ticket expansion.

`examples/all_combinations_draw.py` exports a JSON recipe plus Python/notebook
versions containing baseline/filtered decisions and pairs/triples/quads. Paths
are relative; configure your reviewed mapping before use. Running the exporter
only writes configurations, never prepares, trains, predicts or places bets.
`draw_recipe(min_ev=0.0)` enables the ticket filter for its configurable
pairs/triples/quads templates; `draw_recipe()` preserves unfiltered tickets.
The optional upstream non-draw cutoff defaults to the fixed 0.80 example;
0.90 is another explicit setting, not an automated threshold search.

## Implementation verification

Run from the repository root in PowerShell, using the project Python environment:

```powershell
$env:PYTHONPATH='.;src;tests/analytics'
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' -m xdiyo_analytics.ui.build_inventory
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' -m pytest tests/analytics/test_ticket_ev.py tests/analytics/test_all_combinations.py tests/analytics/test_ui_all_combinations.py tests/analytics/test_ui_odds_roundtrip.py tests/analytics/test_bet_tickets.py tests/analytics/test_bet_outcomes.py tests/analytics/test_post_training_betting.py tests/analytics/test_labels.py tests/analytics/test_odds_adapter.py tests/analytics/test_ui_inventory.py::test_saved_inventory_has_widgets_help_and_packaged_file tests/analytics/test_ui_inventory.py::test_known_form_schemas_use_saved_inventory_without_signature_inspection tests/analytics/test_ui_inventory.py::test_schema_reads_are_isolated_from_persistent_defaults -k 'not ui_ticket_changes and not nb_pipeline_retains and not ui_classifier_retains' -q -p no:cacheprovider --tb=short
```

Verified with ticket EV on 2 October 2026: **288 passed, 0 failed,
0 skipped, 3 deselected**. This includes **6 passing actual Chromium UI
tests** using the existing Playwright fixtures (Browser plugin unavailable).
The builder flow is Post-training analysis → nested AllCombinations → enable
minimum EV at zero → explicit Independent mode → edit/save/reopen/export →
disable/reenable. Checks cover legs 2/3/4, mixed default-off/enabled templates,
visible incompatibility messages, unchanged odds pins and no dispatched jobs.
Chromium used synthetic token-protected loopback servers at 1440 × 1000;
console/page errors were checked and desktop screenshots inspected. Other
browsers and mobile layouts were not tested for this change.
The three excluded
tests fit models; this verification used synthetic retained predictions,
ledgers, quotes and browser controls without fitting models or running the
research experiment. It covers native settlement, exact previews, binary class
semantics, stage isolation, report totals, recipe exports and preserved odds pins.

The review regressions cover two and three retained folds, different per-fold
binary class supports, unchanged inclusive thresholds, missing probabilities and
prices, and overall outcome-to-performance reporting with exact fold-isolated
ticket memberships. Extreme-count tests use 1,100 events and 550 legs, including
zero stakes and multiple groups, and forbid expansion to verify the guard runs
first. Aggregate stake overflow and mixed stage-column templates are covered too.
