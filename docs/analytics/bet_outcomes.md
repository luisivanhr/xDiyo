# Probability-driven bet outcomes

`BetOutcomeReporter` consumes retained predictions, chooses an offered option,
then settles it using the observed outcome. It never fits a model or calibrator.
`CalibrationReporter` remains a class-level reliability study. Optional
`ProbabilityCalibrator` belongs to training and supplies calibrated probabilities
to all downstream consumers.

## Configure it in the builder

1. Open **Post-training analysis**, add a reporter and select **Bet Outcome Reporter**.
2. Name it, for example **Bets**. Choose **Overall** or **Per fold** and the test/score
   rows to analyze. **Last** pooling keeps one prediction per observation if folds overlap.
3. Under **Offers**, add a **Bet Offer** for each available option. Its **Bet Option**
   uses the same label source as the model target. New options copy the current
   target definition; select **Under**, **Over**, or another supported selection.
   For example: Under 4.5, Under 7.5 and Over 10.5 for total corners.
4. Choose a **Policy** and set its thresholds. Each policy exposes its own controls.
5. Leave **Output = auto** to use the upstream class probabilities, calibrated when
   enabled, or the Negative Binomial distribution. Use `predict_proba_raw` explicitly
   only when you want the raw probabilities retained by a calibrated model.
6. Optionally enter decimal odds per offer and a fixed stake (default 1). Alternatively,
   enable **Default odds** for an explicitly assumed fixed-odds scenario. Disabled
   means no fallback quote; odds/profit columns disappear when unavailable.
7. To inspect profit, add **Bet Performance Reporter** afterward. Enable
   **Prepared decisions from** and choose **Bets**. Use the same analysis scope,
   rows and pooling. It reuses the exact ledger, without selecting or settling again.

Classifier recipes automatically retain `predict_proba` when the model supports
it. Negative Binomial recipes retain `count_distribution` with mean and dispersion,
including when the point prediction uses the mode. Existing saved artifacts that
lack a distribution cannot reconstruct it from point predictions alone.

Save your recipe, restart the notebook kernel and relaunch the builder to load new
backend classes. A page refresh alone cannot update an already-running Python backend.

## Probability definitions

For an exact-count classifier with probabilities \(p_k=P(Y=k\mid X)\):

\[
P(\text{Under }7.5)=\sum_{k\leq 7}p_k,
\qquad P(\text{Over }7.5)=\sum_{k\geq 8}p_k.
\]

There is no midpoint interpolation. At an integer line \(L\), the equality mass
\(P(Y=L)\) is a push when `on_equal="push"`, or a loss when `on_equal="loss"`.
The helper also supports `Outcome` win/draw/loss and `Above` yes/no labels.
Count probabilities require exact nonnegative integer classes; interval classes
and overflow bins need an explicit probability adapter and are not silently split.

For NB2, the retained conditional mean \(\mu\) and dispersion \(\alpha\) define

\[
\operatorname{Var}(Y\mid X)=\mu+\alpha\mu^2,
\qquad r=\alpha^{-1},\qquad p=(1+\alpha\mu)^{-1}.
\]

CDF/survival-function calculations include the entire count tail. The predicted
mode is never substituted for \(\mu\). The helper supports the Poisson limit at
\(\alpha=0\). This does not imply the NB probabilities have been calibrated.

When **mean pooling** combines repeated NB predictions, option probabilities are
computed from each original fold distribution and then averaged. For \(J\)
retained predictions of the same match and an event \(A\),

\[
P_{\mathrm{pooled}}(A)=\frac{1}{J}\sum_{j=1}^{J}P_j(A).
\]

The reporter does not substitute averaged NB parameters into a single CDF.
Original fold distributions must remain available for this mixture calculation.

Missing probability cells remain unavailable. When concatenation introduces
absent-class columns, the reporter recovers the originating fold's declared class
support; it does not fill arbitrary missing probabilities with zero. Mean pooling
across incompatible class supports may therefore yield unavailable decisions.

## Selection policies

### TightestLine

`min_probability=0.9`, `max_probability_loss=0.03`, `min_ev=None` by default.
For each side, let \(p_*\) be the greatest win probability among its offered lines.
An eligible option must satisfy

\[
p_{\text{win}}\geq p_{\min},\qquad
p_{\text{win}}\geq p_*-\delta.
\]

Among eligible Under lines, choose the lowest; among eligible Over lines, the
highest. If both sides supply a candidate, choose the one with higher win
probability. Remaining ties follow offer insertion order. The reference \(p_*\)
stays fixed, so the permitted probability sacrifice cannot accumulate across lines.

Example: Under 4.5 has probability 0.91 and Under 7.5 has probability 0.93.
With a minimum of 0.90 and maximum loss of 0.03, choose Under 4.5.
With a maximum loss of 0.01, choose Under 7.5.

This rule only compares offered O/U lines. An optional `min_ev` also requires odds
and filters on expected net profit. It does not automatically maximize profit.

### HighestExpectedProfit

`min_probability=0.0`, `min_ev=0.0` by default. Select the eligible option maximizing
expected net profit per stake unit:

\[
\mathrm{EV}=p_{\text{win}}(o-1)-p_{\text{loss}},
\]

where \(o\) is the decimal quote. Push returns the stake and contributes zero.
Missing odds make an offer ineligible. Probability and EV thresholds both apply;
ties follow offer insertion order. No qualifying offer means **No bet**.

Selection uses predictions and offered quotes only. Observed outcomes are used
afterward for settlement. A reporter represents one target/market and selects at
most one offer per row: one per match for match layout, or one per team for team
layout. It does not optimize a combined portfolio across reporters or team rows.

## Display and shared accounting

The fixture table shows **Home · Away · Selected bet · Probability · Actual result**,
then **Odds · Net profit** when available. It uses existing local badges and team
names, green/red outcome shading, and league/season/team/round filters. Team-layout
results appear under separate home/away label headings. Expand a selected option
to see alternatives, probabilities, expected profit and rejection reasons.

The result retains `bets`, `alternatives`, `ledger` and `bet_metrics` tables. Settlement
uses the existing `BetOption` rules, including explicit push/void/missing outcomes.
The performance reporter consumes the ledger and metrics from an earlier matching
reporter. Odds are required for profit analysis of placed bets; absent quotes do
not become assumed returns. `default_odds=1.1` is allowed when explicitly supplied
and is labelled as an assumption. Profit timing and ROI conventions remain those
of `BetPerformanceReporter` (settled stakes include push/void refunds).
Only the performance reporter publishes the shared accounting as leaderboard
metrics, so using both reporters does not duplicate the same profit metric.

Post-training changes reuse matching saved training results and fitted models;
they recompute the analysis. Changes to calibration or model-selection evidence
remain training changes. This upgrade does not migrate old cache identities or
invent missing probability outputs in old runs.

## Python example

```python
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.evaluation import BetOffer, TightestLine
from xdiyo_analytics.features import Stat
from xdiyo_analytics.labels import MatchTotal, BetOption
from xdiyo_analytics.reporting import BetOutcomeReporter, BetPerformanceReporter

corners = MatchTotal(Stat("ALL", "Match overview", "cornerKicks"))
analysis = PostTrainingAnalysis({
    "Bets": BetOutcomeReporter(
        type="overall", pooling="last", history=history,
        offers={
            "Under 4.5": BetOffer(BetOption(corners, "under", line=4.5), odds=2.5),
            "Under 7.5": BetOffer(BetOption(corners, "under", line=7.5), odds=1.5),
        },
        policy=TightestLine(min_probability=0.90, max_probability_loss=0.03),
        catalog=team_catalog,
    ),
    "Profit": BetPerformanceReporter(
        type="overall", partition="score", pooling="last", source="Bets",
    ),
})
report = analysis.run(training_result)
report.show()
```

For custom numerical workflows, `prepare_bets(context, offers, policy, ...)`
returns named `BetSpec` objects and an alternatives table. Call `evaluate_bets`
afterward for settlement. Existing manual `BetPerformanceReporter(bets=...)`
configuration remains supported. Point-prediction confidence policies, joint
parlays and dynamic stakes remain future extensions.
