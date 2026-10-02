# Parlays, bet slips and system bets

For every k-event combination across a whole variable-size stage-round, use
[AllCombinations](all_combinations.md). The existing batched templates below
retain their original behavior.

For every k-event combination across a whole variable-size stage-round, use
[AllCombinations](all_combinations.md). The existing batched templates below
retain their original behavior.

Both betting reporters accept optional `composition`. Its default is `None`,
which preserves the existing single-bet workflow. Composition uses retained
predictions and the selected-leg ledger. It never fits or calls a model.

## Three templates

| Template | Meaning | Example |
| --- | --- | --- |
| `Parlay` | One combined ticket per complete batch | `Parlay(size=3)` makes trebles |
| `MultiBet` | Every requested combination within each complete batch | `MultiBet(size=3, sizes=(2, 3))` makes three doubles and one treble |
| `BetSlip` | Named, separately staked templates | Singles plus doubles plus a system |

`Parlay(size=1)` creates singles. BetSlip is a container, not an additional
accumulator: it does not multiply its child tickets' odds or settle them together.
Reusing a leg across child templates places additional stakes.

## UI

1. In **Post-training analysis**, add **Bet Outcome Reporter** and configure its
   offered options and probability-selection policy as usual.
2. Enable **Compose tickets**, then select **Parlay**, **MultiBet** or **BetSlip**.
3. Select the grouping, leg/pool size, ordering, stake and settlement rules.
4. For MultiBet, add the desired **System sizes** and choose **Per ticket** or
   **Total** staking. For BetSlip, add named templates with the + control.
5. Add **Bet Performance Reporter**, selecting the outcome reporter in
   **Prepared decisions from**. Leave its own composition disabled to reuse the
   exact ticket ledger and metrics. It will not place the legs again as singles.

Alternatively, configure composition directly on the performance reporter using
manual `BetSpec` options or an uncomposed outcome-reporter source. Applying a
second composition to a source that already contains tickets gives an error.

Restart an existing builder server/kernel to load the new catalog. Saved recipes
without composition continue to use singles. Exported Python/notebook recipes
use the same registered constructors.

## Grouping and selection

| `grouping` | Eligible batch population |
| --- | --- |
| `league_round` (default) | Same league, season and round |
| `round` | Same round number and common `source_season`, across leagues |
| `day` | Same UTC calendar day, across leagues |

Round numbers across leagues do not imply simultaneous fixtures. `round` uses
the export's common season label; league-specific season IDs alone cannot align
different leagues. Missing grouping metadata gives an explanatory error.

Groups always stay inside one retained fold occurrence. Use a declared first,
last or mean pooling policy when reporting repeated evaluation predictions;
composition does not resolve repeated CV observations itself.

Within each group, order the selected legs by `kickoff` (earliest first),
`probability` (highest first) or `expected_profit` (highest first). The latter two
need the corresponding values from an outcome reporter. Ties use kickoff,
fixture identity and bet name. Retain at most one selected leg per fixture, then
form disjoint batches of `size` fixtures. An incomplete final batch is omitted;
there is no padding or automatic smaller ticket. Outcomes and settlements never
affect grouping, ranking or membership. This initial version does not generate
every combination across an entire round: MultiBet combines within each batch.

The outcome reporter first selects one option per prediction row using its
existing policy. Composition does not reconsider rejected alternative lines.

## Stakes and odds

Parlay `stake` is per ticket. MultiBet `stake_mode="per_ticket"` assigns that
amount to each generated combination; `"total"` divides it equally over that
batch. Original single-leg stakes remain in `leg_ledger` for inspection but are
not charged in the ticket ledger.

For a pool of size \(n\) and system sizes \(K\), the number of tickets is

\[
N=\sum_{k\in K}\binom{n}{k}.
\]

A total batch stake \(S\) allocates \(S/N\) to each ticket. By default,
`max_tickets=1000` limits tickets per template per reporter scope. Exceeding it
raises an error rather than silently truncating the requested system.

Quoted accumulator odds are the product of the leg decimal odds:

\[
O=\prod_{i=1}^{k}o_i.
\]

Missing odds remain missing; composition invents no quote. The outcome reporter's
existing explicit `default_odds` may supply a fixed-odds scenario before
composition. This release implements product pricing, not bookmaker-specific
boosts or bespoke combined quotes.

## Settlement rules

Both `on_push` and `on_void` independently accept:

- `remove` (default): remove the leg from the payout product, equivalent to odds 1.
- `refund`: refund the entire ticket unless another leg loses.
- `loss`: treat this leg as losing.

Precedence is explicit: any losing leg settles the ticket as a loss; otherwise a
whole-ticket refund rule settles it as refunded; otherwise an unresolved leg
leaves it unresolved. Remaining winning legs determine the payout. If every leg
was removed, return the stake, using `void` when all were void and `push`
otherwise. Missing results are never silently treated as voids.

For ticket stake \(s\) and surviving winning legs \(A\):

\[
\mathrm{payout}=s\prod_{i\in A}o_i,
\qquad \mathrm{net\ profit}=\mathrm{payout}-s.
\]

A loss pays zero. A refund returns \(s\). Unknown winning-leg odds leave the
payout unresolved. A known loss or refund can be accounted without all quotes.
The performance reporter still requires quoted odds for the selected tickets.

Ticket ROI uses settled ticket stakes, including refunded tickets:

\[
\mathrm{ROI}=\frac{\sum_{t\in\mathrm{settled}}\mathrm{profit}_t}
{\sum_{t\in\mathrm{settled}}s_t}.
\]

Profit/ROI are marked partial while tickets remain unresolved. No tickets means
zero profit and undefined ROI. A BetSlip retains metrics per named template;
its ticket ledger can also be summed for a whole-slip total.

## Probability and time interpretation

`probability_mode="none"` leaves joint probability blank. Explicitly selecting
`"independent"` displays

\[
P(\text{all legs win})=\prod_{i=1}^{k}p_i.
\]

That is an independence assumption, not a dependence model, calibrated joint
forecast, probability of positive net profit after pushes, or ticket-level EV.
Only one selection per fixture is used; same-game parlays remain unsupported.

All retained predictions for a prospective ticket must have been available
before its first fixture started. Grouping a historical round does **not** prove
this timing condition. Use the feature pipeline's common prediction cutoffs when
constructing those predictions. These reporters summarize outcomes; their
kickoff-ordered profit curve is not a settlement-time bankroll simulation.

## Python and retained artifacts

```python
from xdiyo_analytics.evaluation import BetSlip, Parlay, MultiBet

composition = BetSlip({
    "singles": Parlay(size=1, grouping="league_round", stake=1),
    "doubles": Parlay(size=2, grouping="round", stake=1),
    "day_system": MultiBet(
        size=3, sizes=(2, 3), grouping="day",
        stake=4, stake_mode="total",
    ),
})
# Pass composition=composition to either betting reporter.
```

For independent numerical use, `compose_bets(ledger, composition,
match_columns=(...))` returns `(tickets, ticket_legs, metrics)`. Pass the full
fixture identity columns from the prepared dataset when IDs are not globally
unique. Leg rows need `fold_id`, `row_position`, `bet`, `take`, `odds`,
`settlement`, `kickoff_at`, fixture identities and the selected grouping columns.

Composed reporters retain `leg_ledger`, `tickets`, `ticket_legs`, `ledger` (the
ticket accounting ledger), and `bet_metrics`. The performance reporter also
exposes those metrics as `metrics` for experiment summaries. Expand a displayed
ticket to inspect its teams, badges when supplied, selected options, probabilities,
observed results and settlements. Ticket headings show stake, available odds,
optional independence probability and net profit. Green indicates a winning
ticket, red a losing ticket; refunds and unresolved tickets are neutral.

Saved analysis snapshots preserve these tables and can be refreshed from an
existing run without refitting when only the composition settings change.

Deferred extensions: explicit hand-picked cross-match legs, same-game dependence,
bookmaker-specific pricing, split-quarter lines, dynamic staking and bankroll
cash-flow simulation. These are not implied by the template names.
