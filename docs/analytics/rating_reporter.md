# RatingReporter

`RatingReporter` is a post-training study of rating states over the fixtures in
the selected evaluation folds. Add it in **Post-training analysis → Add → Rating
Reporter** in the experiment builder. It also works in exported Python recipes,
notebooks, and offline HTML reports.

## What it displays

| Rating producer | Team plots | Shared plot | Uncertainty |
|---|---|---|---|
| Match-result Glicko, statistic Glicko, named Glicko runs | Rating | None | Normal approximation using rating deviation (RD) |
| BayesianRating, BayesianFixture, named Bayesian runs, fitted BayesianScoreAdapter | Attack and defensive vulnerability | League home-advantage multiplier | Equal-tail Gamma posterior intervals |
| Custom numeric RatingRun | Each state field | None | None unless supported explicitly |

Defensive vulnerability is the model's native defense quantity: **lower means
stronger defense**. Attack and home advantage are positive multipliers. The home
advantage is shared by teams **within a league**. Pooled Bayesian parameter
training does not turn it into a single state shared across disconnected leagues.
Glicko volatility (`sigma`) is not used as the rating's error band.

## Interactive controls

The rendered report contains league, rating-source/stream and team selectors.
The rating-source selector is hidden when only one source/stream is available.
Choose **All teams** to compare the league, or one team to inspect it. A team has
the same line color across panels. Badge/name buttons below the plots hide or show
that team's lines and intervals together; **Show all lines** restores visibility.
Missing badges fall back to names. The uncertainty checkbox toggles the bands.
The shared home-advantage panel remains visible when selecting a team.

Plots are constructed for the current view, rather than caching a figure for
every team. Switching controls uses retained numeric values and never fits a model,
calls prediction, replays outcomes or updates a rating. The data is embedded in
offline HTML; its size grows with retained evaluation queries. Team assets are
embedded once per team. The viewer works in the existing notebook iframe.

## Time, folds and information boundaries

- `partition="test"` (default) uses all test fixtures in the selected folds.
  `partition="score"` uses their scored subset, such as rounds after a warm-up gap.
- The horizontal coordinate is **kickoff time**. Each ordinate is the state
  available at that fixture's **prediction cutoff**, which can be earlier.
  These are pre-prediction ratings, not post-match ratings.
- No training fixtures are included. Curves contain only the evaluation fixtures,
  even when test windows are disjoint. A line joins sampled evaluation points;
  it does not assert that missing days or excluded fixtures were observed.
- `type="overall"` and `type="timeline"` include the selected folds together;
  `type="per_fold"` adds the existing fold selector.
- `pooling="occurrences"` retains fold occurrences. Each fold has a separate
  curve, with dash styles distinguishing folds; independently trained ratings
  are not averaged. `first` and `last` are also supported. `mean` is rejected.
- Fitted Bayesian models expose their exact query-only rating run, including
  season-entry states. Its diagnostic copy is saved by `BayesianScoreSerializer`;
  the fitted checkpoint remains unchanged. Model curves are scoped to their own
  fold. Their forecast queries use the adapter's kickoff timing.

## Interval equations

Let the requested interval probability be \(c\), with \(0<c<1\), and let
\(z=\Phi^{-1}((1+c)/2)\), where \(\Phi\) is the standard normal CDF.
For a Glicko rating \(r\) and rating deviation \(RD\), the displayed interval is

\[
\left[r-z\,RD,\quad r+z\,RD\right].
\]

For a Bayesian Gamma state with mean \(m\) and standard deviation \(s>0\),
the equivalent shape \(a\) and scale \(\theta\) are

\[
a=\left(\frac{m}{s}\right)^2,
\qquad \theta=\frac{s^2}{m}.
\]

The displayed equal-tail credible interval is

\[
\left[
F^{-1}_{\operatorname{Gamma}(a,\theta)}\left(\frac{1-c}{2}\right),
F^{-1}_{\operatorname{Gamma}(a,\theta)}\left(\frac{1+c}{2}\right)
\right].
\]

These are intervals for latent ratings under the rating model, **not predictive
intervals for goals or corners**. The default is \(c=0.95\). Unavailable uncertainty
produces no band; zero standard deviation produces a degenerate interval.

## Python and retained resources

```python
post = PostTrainingAnalysis({
    "Ratings": RatingReporter(type="overall", partition="test", interval=0.95),
})
result = experiment.run(prepared, model=candidate, post_analysis=post)
```

The UI preparation retains saved/named runs and generated inline rating features
automatically. Shared producers used by BayesianRating and BayesianFixture are
retained once. This also preserves uncertainty fields when the input features
select only rating means. Post-report configuration changes reuse stored models
and preparation resources through the existing completed-run recovery path.

For a custom preparation function, retain generated producers explicitly:

```python
rating_runs = {}
features = evaluate_features(
    history, feature_definitions, cutoffs=cutoffs, rating_runs=rating_runs,
)
# Put these resources in PreparedExperiment.outputs, or pass them directly:
report = post.run(training, resources={
    "history": history,
    "rating_cutoffs": cutoffs,
    "ratings": rating_runs,
    "team_catalog": team_catalog,
})
```

The optional `ratings` reporter argument selects named retained sources in Python.
By default all sources are discovered and selected in the rendered viewer.
Older experiment bundles may not retain inline or loaded rating runs; those
cannot be reconstructed from means alone. Supply the original rating resources
or prepare them again. The reporter explains when no retained ratings are found.
It never silently rebuilds them. Download **rating_states** for the plotted values,
intervals, exact kickoff/cutoff timestamps and source/fold/team identities.
