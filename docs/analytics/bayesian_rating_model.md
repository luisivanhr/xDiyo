# Bayesian score ratings: model and implementation

This optional rating complements Glicko2 with separate attack and defensive
vulnerability distributions. It implements the Gamma-mixed Poisson model in
[Ridall, Titman and Pettitt (2025)](https://doi.org/10.1093/jrsssc/qlae075),
with explicitly documented extensions for several leagues and relegation.
See [usage and UI configuration](bayesian_rating_usage.md), the
[adapter design](bayesian_rating_design.md), and the [handoff](bayesian_rating_handoff.md).

Install the optional numerical dependency with `pip install -e ".[ratings]"`.
No research fit or evidence of improved football predictions accompanies this
implementation. Defaults are starting values, not calibrated xDiyo ratings.

## Observation model

For home team \(i\), away team \(j\), and competition-specific home advantage
\(\gamma\), let attack be \(\alpha\) and defensive vulnerability be \(\beta\).
Smaller \(\beta\) means stronger defence. Conditional intensities are

\[
\lambda_H=\alpha_i\beta_j\gamma,\qquad
\lambda_A=\alpha_j\beta_i.
\]

The bivariate observation model is

\[
\epsilon\sim\operatorname{Gamma}(\kappa,\kappa),\qquad
X\mid\epsilon\sim\operatorname{Poisson}(\epsilon\lambda_H),\qquad
Y\mid\epsilon\sim\operatorname{Poisson}(\epsilon\lambda_A),
\]

with conditional independence of \(X,Y\). Every Gamma uses **shape/rate**.
Integrating the shared multiplier gives

\[
p(x,y\mid\lambda_H,\lambda_A,\kappa)=
\frac{\Gamma(\kappa+x+y)}{\Gamma(\kappa)x!y!}
\frac{\kappa^\kappa\lambda_H^x\lambda_A^y}
     {(\kappa+\lambda_H+\lambda_A)^{\kappa+x+y}}.
\]

Thus \(E[X]=\lambda_H\),
\(\operatorname{Var}(X)=\lambda_H+\lambda_H^2/\kappa\), and
\(\operatorname{Cov}(X,Y)=\lambda_H\lambda_A/\kappa\), conditional on the
team intensities. This is a bivariate negative-binomial construction, with
positive dependence and marginal overdispersion tied to the same parameter.
It is different from an additive shared-count bivariate Poisson.
`BayesianConfig(bivariate=False)` sets the multiplier to one and supplies the
independent-Poisson control.

Only paired goal counts enter the likelihood. Correct team, competition,
season, home/away, kickoff, availability and movement metadata are also
necessary for chronological replay. Scores alone cannot establish league
comparability or promotion direction.

## Filtering and uncertainty

The filter keeps independent variational Gamma factors for each team's attack
and defence and for each league's home advantage. If a factor has parameters
\((a,b)\), its mean is \(a/b\) and variance is \(a/b^2\).
A discount \(w\in(0,1]\) changes it to \((wa,wb)\): the mean stays fixed,
and the variance increases by \(1/w\).

For one fixture, write bars for current factor expectations and superscript
minus for the fixed discounted prior. Coordinate ascent updates include

\[
q(\epsilon)=\operatorname{Gamma}\left(
\kappa+x+y,
\kappa+\bar\alpha_i\bar\beta_j\bar\gamma+
        \bar\alpha_j\bar\beta_i\right),
\]

\[
q(\alpha_i)=\operatorname{Gamma}
(a_{\alpha_i}^-+x,\ b_{\alpha_i}^-+\bar\beta_j\bar\gamma\bar\epsilon),
\quad
q(\beta_j)=\operatorname{Gamma}
(a_{\beta_j}^-+x,\ b_{\beta_j}^-+\bar\alpha_i\bar\gamma\bar\epsilon),
\]

\[
q(\alpha_j)=\operatorname{Gamma}
(a_{\alpha_j}^-+y,\ b_{\alpha_j}^-+\bar\beta_i\bar\epsilon),
\quad
q(\beta_i)=\operatorname{Gamma}
(a_{\beta_i}^-+y,\ b_{\beta_i}^-+\bar\alpha_j\bar\epsilon),
\]

\[
q(\gamma)=\operatorname{Gamma}
(a_\gamma^-+x,\ b_\gamma^-+\bar\alpha_i\bar\beta_j\bar\epsilon).
\]

For several results released together, each shape and rate increment is summed
over the relevant fixtures. There is one home-advantage factor for that entire
league batch. Iterations always reuse the original prior; a goal is counted
once, rather than once per iteration. `method="vb"` iterates epsilon, attack,
defence and home blocks to relative convergence and raises on nonconvergence.
`method="one_step"` computes epsilon first and then all state updates using
the frozen prior means, following the paper's one-step approximation.
Its completion flag does not certify convergence of VB.

`clock="team_match"` discounts a team's factors once per assimilated appearance;
`clock="days"` uses elapsed kickoff days divided by `time_unit_days` (default 7).
The paper's team clock is a round. The appearance clock coincides with that
clock in a regular schedule but differs with postponements or missing fixtures.
Home advantage is discounted once per assimilated match. Retained teams and
home advantage also have separate offseason discounts. Snapshot lookup adds
no further inactivity inflation; it reports the stored distribution.

Proper initial Gamma priors anchor the implementation's scale. The paper
initially fixes Arsenal's vulnerability to one and drops that constraint after
the first season (Section 3.3.3); this implementation uses proper priors from
the start. In the likelihood,
rescaling all attacks by \(c\) and all vulnerabilities by \(1/c\) leaves
intensities unchanged. Individual attack/defence coordinates and the derived
strength index therefore depend on this convention. They are not automatically
an absolute cross-country ranking.

## Several leagues and movement between them

`mode="pooled"` learns one complete set of shared hyperparameters by aggregating
the eligible league objectives. Team and home-advantage states remain separate
by competition. This is complete parameter pooling, not hierarchical partial
pooling. `mode="per_league"` fits a separate bundle for each competition and
rejects unknown competitions at prediction. Scores in disconnected leagues
cannot identify their relative quality merely by being pooled.

The default `transition="mirrored"` uses destination-league entry priors. With
baseline means \(m_\alpha,m_\beta\) and nonnegative shifts \(d_\alpha,d_\beta\),

| Entry | Attack prior mean | Vulnerability prior mean |
| --- | --- | --- |
| Promoted | \(m_\alpha e^{-d_\alpha}\) | \(m_\beta e^{d_\beta}\) |
| Relegated | \(m_\alpha e^{d_\alpha}\) | \(m_\beta e^{-d_\beta}\) |

Both directions share the two shifts and the two entry shape parameters.
This avoids introducing another poorly estimated relegation-specific set.
It is a symmetry assumption to test, not a finding of the paper. The previous
league's history remains stored, while the destination begins from its mirrored
prior. Use the bridge when individual previous-season state should carry over.

`transition="bridge"` accepts fixed `bridge_gaps=((source, destination, gap),)`.
Positive gap means a stronger source competition. If \(\Delta\) is the gap,
the source posterior is transformed by

\[
\alpha_{\rm dest}
=\alpha_{\rm source}\frac{m_{\alpha,\rm dest}}{m_{\alpha,\rm source}}e^\Delta,
\qquad
\beta_{\rm dest}
=\beta_{\rm source}\frac{m_{\beta,\rm dest}}{m_{\beta,\rm source}}e^{-\Delta}.
\]

Scaling a Gamma variable by \(c\) divides its rate by \(c\), preserving shape.
`bridge_discount` then inflates uncertainty. Reverse lookup negates the gap.
Source posteriors are frozen before applying all transfers at one boundary,
so simultaneous promotion/relegation does not depend on input order.
A missing predecessor or missing gap falls back to the mirrored prior.
The gap is an explicit externally supplied parameter, not an estimated league
quality. Learning gaps jointly, hierarchical pooling and asymmetric transition
effects are future extensions. This implementation restricts bridge-enabled
calibration to pooled mode because independent calibration replays each league
in isolation. Externally fixed gaps could support a future implementation that
combines independently fitted parameters and their source-state replays.

Movement tables use existing `team_seasons` and `season_starts` contracts.
A new appearance alone does not imply promotion or relegation. Unknown entries
receive ordinary priors. An incremental new-season replay for an already known
team requires explicit movement and both predecessor fields; explicit nulls
acknowledge a reset. This avoids inferring retention from an incomplete slice.

## Prediction, columns and persistence

Prediction uses posterior means to form \(\lambda_H,\lambda_A\) and integrates
the Gamma match effect. It does **not** integrate the full posterior uncertainty
of attack, defence, home advantage or fitted hyperparameters. Under VB their
reported standard deviations are approximate marginal uncertainties.

For score probabilities, total goals satisfy
\(N=X+Y\sim\operatorname{NB}(\kappa,\kappa/(\kappa+\lambda_H+\lambda_A))\)
and \(X\mid N\sim\operatorname{Binomial}(N,\lambda_H/(\lambda_H+\lambda_A))\).
The implementation sums total-goal strata until the omitted mass is at most
`tail_tolerance`; it raises if `max_total_goals` cannot meet that bound.
Outcome probabilities are not silently renormalized. Both-score and over-2.5
probabilities use closed forms. The numerical helper also returns omitted mass.
Joint score log probabilities use a rising-factorial identity to avoid
cancellation at large finite dispersion shapes. Total-goal probabilities use
a complementary incomplete-beta tail when the usual negative-binomial
parameterization loses precision. This preserves the finite-shape model without
switching to a Poisson approximation. Unsupported nonfinite special-function
results raise explicitly.

`BayesianRating.fields` chooses team columns, preserving order. Defaults are
`attack_mean` and `defence_vulnerability_mean`. Optional columns are
`attack_sd`, `defence_vulnerability_sd`, `log_attack`, `log_defence_strength`,
`strength_index`, and the four attack/defence `shape`/`rate` values.
The log columns are
\(E[\log\alpha]=\psi(a_\alpha)-\log b_\alpha\) and
\(-E[\log\beta]=\log b_\beta-\psi(a_\beta)\);
`strength_index` is their sum, not a separate fitted state.
The generic `Rating(name, fields=None)` and direct `run.features(fields=None)`
retain the existing contract of exporting all numeric team state fields.

`BayesianFixture.fields` defaults to `expected_home_goals` and
`expected_away_goals`. Optional fields are `expected_total_goals`,
`expected_goal_difference`, `strength_difference`, `home_advantage_mean`,
`home_advantage_sd`, `p_home_win`, `p_draw`, `p_away_win`, `p_both_score`, and
`p_over_2_5`. These retain canonical home/away orientation on both history
perspectives. Selecting fields never changes the inference state or the
checkpoint; unrequested probabilities need not be computed for feature lookup.

A `BayesianModel` JSON contains frozen parameters, model conventions, fitting
cutoff and provenance. A `BayesianRatingRun` additionally saves all shape/rate
states, league states, filtered snapshots, update frontier and processed event
identities. Native artifacts are checksummed and refuse overwrites. A generic
`RatingRun.load` cannot load them and discard the fitted-parameter time barrier.

The normal training adapter forecasts from its frozen fitted checkpoint. It
applies supplied season-entry declarations through an outcome-free scheduled
replay, preserving that checkpoint. Native serialization retains those movement
declarations and season-start anchors. It never reads held-out goal targets or
optional source-history outcomes during prediction. Actual newly observed
results require an explicit rating update.

## Temporal and data boundaries

- Each completed fixture needs both reciprocal history rows with consistent
  scores, kickoff and availability. Awarded matches are excluded. Missing scores
  are skipped; invalid finite/integer constraints raise errors.
- `provider_current` uses the provider's current goal counts. It does not infer
  regulation scores from extra-time/penalty outcomes. `regulation` requires a
  caller-prepared history with `history.attrs['score_basis']='regulation'`.
- Supply actual result availability where available. The default kickoff proxy
  is a compatibility convention, not a claim that results were known at kickoff.
  A contributing kickoff must be strictly before the prediction cutoff; result
  availability may equal it. Simultaneous fixtures cannot see one another.
- An earlier match released exactly at a season boundary is assimilated before
  a transfer. State recursions follow actual kickoff order; available matches
  with equal kickoff in a competition form one joint update. Distinct kickoffs
  remain separate even when their scores arrive in the same release batch.
- A late result triggers chronological re-filtering of the saved observation
  and entry journal. It refreshes teams, shared home advantage and downstream
  retained-season or promotion/relegation bridges. Revised states are published
  when the result becomes available. Earlier snapshots and stored forecasts
  retain the information available at their original prediction times. This
  is re-filtering of the configured approximation, not Bayesian smoothing.
- Inferred season anchors may move earlier when an older fixture first arrives.
  Explicit season-start declarations remain authoritative. A forecast-only
  update cannot add a new entry before the durable frontier without a new
  result release; rebuild full history for that membership correction.
- Future scheduled entry anchors may produce query snapshots but cannot move
  the resumable observation frontier. An update accepts only new releases
  strictly later than that frontier. Revisions and partial equal-time append
  batches require rebuilding from complete history.
- Saved runs now retain the observation/entry journal. Older artifacts remain
  readable for forecasts, but must be rebuilt from full history before updating.
  Ordinary ordered observations retain the incremental path. A late append
  replays retained history; a full-history call containing delayed releases
  may replay prefixes at every release boundary, with quadratic worst-case
  work. This also increases calibration cost on such datasets.
- Feature queries before `training_cutoff` are forbidden even if the state
  lookup itself is chronological. Hyperparameters fitted on future results would
  leak. Use separate calibration windows or refit inside each temporal fold.
  Stored prequential fitting diagnostics do not constitute held-out evaluation.
- In the normal training adapter, a declared fold `training_boundary` constrains
  every selected observation: kickoff must be earlier, and result availability
  must be at or before it. If only `fit_at` is declared, it supplies that limit.
  The adapter rejects an inconsistent exact fit population instead of dropping
  rows or advancing the fitting time. The saved model's `training_cutoff` is
  `fit_at` when supplied; the calibration report retains the separate data cutoff
  and actual last observation times. Without either fold timestamp, standalone
  fits and final refits retain the observation-derived cutoff.
- Prediction checks the declared fold `fit_at` and data boundary as well as
  target kickoff. The existing `PredictionContext` retains the earliest fold
  prediction time, not the original per-row cutoff vector. An `explicit` cutoff
  marker is not a timestamp or column name. Exact row-specific rating cutoffs
  remain available through the rating feature/replay APIs; the adapter does not
  guess missing timing information.

## Calibration and reasonable history

Calibration minimizes chronological mean outcome log loss (default) or joint
score log loss, with a weak penalty around the supplied initial parameters.
The bounded optimizer is deterministic L-BFGS-B. Nine dynamics, dispersion and
entry parameters are candidates by default; baseline means and initial shapes
are supplied as fixed settings. Entry fields with no actual mirrored-prior
application, and dispersion in univariate mode, are reported as inactive and
held fixed. Successful bridges bypass those entry parameters. Reports include
actual mirrored/bridge mechanism counts; historical mirrored uses remain
counted if later information enables a bridge. Active optimization coordinates
are not evidence of statistical identification.
Inspect success, counts and parameter
provenance before exporting a fitted bundle. Optimization success does not
establish identifiability or predictive calibration.

The default gates are **three observed seasons per league for pooling** and
**five per independently fitted league**. These are practical starting
heuristics, not minima demonstrated by the paper. A useful pooled pilot would
have roughly four or five comparable leagues with three consecutive complete
seasons each. One or two seasons can initialize team states when hyperparameters
are already trained, but offer little evidence for offseason or entry effects.
Reserve at least one, preferably two, later seasons for rolling evaluation.

The code counts observed season labels, not complete schedules. Completeness,
consecutiveness and the number of actual promotion/relegation examples require
caller review; repeated fixtures within a season do not replace independent
season-boundary evidence. Compare histories of 2/3/5/8 seasons on the same future
windows, checking log score, outcome calibration, draw probabilities and the
first rounds after team movement. Compare with Glicko2 and the independent
Poisson control. Old data can add transition evidence while introducing changing
scoring regimes; more seasons are not automatically better.

The paper's long EPL study does not verify a short-history, several-league or
lower-league deployment. Additional limitations include unmodelled player and
manager changes, positive-only shared score dependence, mean-field uncertainty,
score-basis consistency, unequal league sample sizes in pooling, and the cost of
replaying the filter many times during calibration. Synthetic tests establish
implementation behavior, not production accuracy, betting value or throughput.
