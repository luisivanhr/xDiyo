# Training-control decisions and coefficient diagnostics

The [guide](training_controls.md) shows usage; the [reference](training_controls_reference.md)
defines every option. Equations below describe the frozen implementation, not a
claim of convergence or predictive quality. A backend supplies the scalar metric
and model update; the control loop makes monitoring decisions from those values.

## Fitting and validation populations

Let \(D_f\) be the declared development positions for fold \(f\), \(V_f\) its
validation subset, and \(F_f\) the rows passed to fitting:

\[
V_f\subset D_f,\qquad F_f=D_f\setminus V_f,\qquad F_f\ne\varnothing.
\]

Both subsets consist of whole matches. For `ValidationTail`, let \(B_f\) be the
number of distinct UTC kickoff times in development and \(q\in(0,1)\) the fraction.
Reserve the latest \(b_f\) complete timestamp batches:

\[
b_f=\lceil q B_f\rceil,\qquad 1\le b_f<B_f.
\]

Preprocessing and supervised feature selection use \(F_f\). Validation is
transformed with that fitted preprocessing. Outer test/score rows do not enter
these fitting contexts. These membership checks do not repair upstream feature
leakage or determine whether an observation was available at prediction time.

## Exact best state versus patience

Let \(m_t\) be the monitored scalar after step \(t\). Use the oriented value
\(z_t=s m_t\), with \(s=1\) for minimization and \(s=-1\) for maximization.
The exact best step is the first minimum among finite observed values:

\[
t^*=\min\operatorname*{arg\,min}_{1\le t\le T,\ z_t\ \mathrm{finite}}z_t.
\]

Strict improvement of this best value updates the snapshot. Separately, let
\(p\) be the last significant-progress reference, initially infinity, and
\(\delta\) be `min_delta` when early stopping is configured, otherwise zero.
A step resets both stale counters only when

\[
z_t<p-\delta.
\]

On such a step set \(p=z_t\). Otherwise increment the counters. Stop when the
early-stopping count reaches its patience. For example, minimizing
\((10,9.8,9.7,9.6)\) with delta 0.5 and patience 3 stops at step 4; exact best
restoration still retains step 4. Equality at the delta boundary does not reset
patience. This distinction prevents interpreting the patience reference as the
best checkpoint value.

## Plateau rate schedule

When the scheduler counter reaches its patience and early stopping has not already
ended the attempt, update the rate for the next step:

\[
\eta_{t+1}=\max\{\min(\eta_t,\eta_{\min}),\ \gamma\eta_t\},
\qquad 0<\gamma<1.
\]

This clamps normal reductions at the floor while leaving an already sub-floor rate
unchanged. The history records \(\eta_t\), the rate used for the completed step,
before that possible change. Reduction resets the scheduler counter only.

## Fresh attempts and retained state

For `restarts=r`, run \(r+1\) independent fresh attempts with

\[
\mathrm{seed}_a=\mathrm{seed}_0+a,\qquad a=0,\ldots,r.
\]

With restoration enabled, each usable attempt contributes its exact best monitored
state. Without restoration, it contributes its final state and final metric only
if that metric is finite. Select the first attempt with the best oriented retained
value. A nonfinite monitor ends an attempt; earlier finite values can salvage it
only with restoration. All performed history rows remain recorded. An arbitrary
exception is raised immediately rather than treated as another trial.

## Loss and illustrative SGD update

`PartialFitBackend` computes loss using the existing metric layer after each
update. For a single complete regression target and MSE:

\[
L_F(\theta)=\frac{1}{|F|}\sum_{i\in F}(y_i-\widehat y_i(\theta))^2,
\qquad
L_V(\theta)=\frac{1}{|V|}\sum_{i\in V}(y_i-\widehat y_i(\theta))^2.
\]

For the verification fixture's unregularized constant-rate `SGDRegressor`, each
ordered sample with squared-error loss updates a coefficient vector and intercept:

\[
e_i=y_i-(w^\top x_i+b),\qquad
w\leftarrow w+\eta e_i x_i,\qquad b\leftarrow b+\eta e_i.
\]

One backend step performs one `partial_fit` call over the fitting population.
The native estimator owns sample ordering, schedules and penalty behavior.
The independent classification fixture checks the analogous binary logistic
update using \(p_i=(1+\exp[-(w^\top x_i+b)])^{-1}\) and error \(y_i-p_i\).
The external monitor is an evaluation metric; it need not equal a penalized
optimization objective. Native histories retain their estimator-defined meaning.

## Lasso and ElasticNet coefficients

For fitted input matrix \(X\), Lasso's usual squared-loss objective is

\[
\frac{1}{2n}\|y-Xw-b\mathbf1\|_2^2+\alpha\|w\|_1.
\]

ElasticNet with L1 ratio \(\rho\) adds its quadratic term:

\[
\frac{1}{2n}\|y-Xw-b\mathbf1\|_2^2
+\alpha\rho\|w\|_1+\frac{\alpha(1-\rho)}2\|w\|_2^2.
\]

For centered orthogonal fixture columns with \(X^\top X/n=I\) and unpenalized
coefficients \(c_j\), the independent expected coefficients are

\[
w_j^{\mathrm{Lasso}}=\operatorname{sign}(c_j)(|c_j|-\alpha)_+,
\qquad
w_j^{\mathrm{EN}}=\frac{\operatorname{sign}(c_j)(|c_j|-\alpha\rho)_+}
{1+\alpha(1-\rho)}.
\]

These closed forms apply to that fixture, not arbitrary data. The reporter reads
actual fitted coefficients without re-solving either problem. If a feature is
standardized as \(x'_j=(x_j-\mu_j)/\sigma_j\), the displayed coefficient multiplies
\(x'_j\); it is not automatically converted back to original units.

## Feature survival denominator

For target/class output \(o\), let \(I_o\) contain every inspected fold exposing
that output. Let coefficient \(w_{f,o,j}\) be zero for a feature absent from an
inspected fold. At tolerance \(\tau\ge0\),

\[
S_{f,o,j}=\mathbf1\{|w_{f,o,j}|>\tau\},\qquad
\mathrm{frequency}_{o,j}=\frac{\sum_{f\in I_o}S_{f,o,j}}{|I_o|}.
\]

Intercept-only outputs count in \(I_o\); unsupported outputs that were not
inspected do not. Intercepts themselves never become features. If one fold has
coefficient 2 for feature a and another exposes only the same target's intercept,
feature a's survival frequency is \(1/2\), not 1. Frequency is descriptive stability
across fitted models, not feature importance or a confidence probability.

## Display and final evaluation

Curve traces keep each fold/attempt/metric separately; there is no implicit mean
across unequal step counts. Display downsampling changes plotted point count only,
and skipped nonfinite observations continue to break line segments.

For same-dataset final evaluation, with declared development \(D\), test \(T\)
and score \(S\):

\[
T\cap D=\varnothing,\qquad S\subseteq T.
\]

An explicit final refit may use all development rows and a preselected step budget.
It must not infer that budget from final test outcomes. Group/trial/final artifact
roles supply provenance; they do not enforce research independence or perform
selection. Leaderboard numerical scaling remains in the
[post-training equations](post_training_equations.md).
