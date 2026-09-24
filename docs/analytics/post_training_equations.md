# Post-training calculations and comparison rules

These equations describe the implemented numerical outputs. They do not assign
statistical significance, choose a strategy or establish out-of-sample performance.
The [guide](post_training.md) shows executable usage and the
[reference](post_training_reference.md) records parameters and output schemas.

## Populations and complete cases

Each prediction occurrence has identity \((f,i)\): fitted fold and original dataset
row position. A study uses either all held-out rows or the fold's scoring subset.
Let \(O\) be the chosen occurrence population and \(E\subseteq O\) its eligible
complete cases for one metric. Coverage is

\[
n=|E|,\qquad n_{\mathrm{total}}=|O|,\qquad
n_{\mathrm{missing}}=|O|-|E|.
\]

Numeric metrics require finite observed/predicted pairs. Label metrics require
nonmissing labels. Probability metrics require a nonmissing observed class and
every probability finite. Binary entropy requires only the complete probability
vector. Empty eligible sets produce missing values with `no_valid_observations`.
The sample fingerprint hashes the evaluated index, supplied metadata values and
dtypes, and the observed target; uncertainty metrics substitute a constant for
that target. Predictions themselves are excluded from this fingerprint.

Repeated occurrences require a declared rule. Occurrence pooling keeps every
row; first/last use selected-fold order. For mean pooling, with occurrences
\(O_i\) of row \(i\), each numeric output cell is

\[
\bar p_i=\frac{1}{|O_i|}\sum_{o\in O_i}p_o.
\]

Any missing contributing cell makes that pooled cell missing. Targets and metadata
must agree across repeated rows. Mean-pooled rows have fold marker `-1`. Per-fold
metrics and metrics on the pooled population answer different questions.

## Regression errors

On the eligible rows, define residual \(e_i=y_i-\widehat y_i\). The built-in errors are

\[
\operatorname{MSE}=\frac1n\sum_i e_i^2,\qquad
\operatorname{MAE}=\frac1n\sum_i |e_i|,\qquad
\operatorname{RMSE}=\sqrt{\operatorname{MSE}}.
\]

With \(\bar y=n^{-1}\sum_i y_i\), the coefficient of determination is

\[
R^2=1-\frac{\sum_i e_i^2}{\sum_i(y_i-\bar y)^2}.
\]

This layer reports missing `r2` for fewer than two eligible rows or a constant
target, before invoking the backend. Otherwise it can be negative. MSE/MAE/RMSE
are minimized; R2 is maximized. The residual reporter preserves paired rows and
uses ordinary histogram counts, with the final bin including its right edge.

## Label classification

Let \(C_{ab}\) count true class \(a\) predicted as \(b\), and let \(s=\sum_{ab}C_{ab}\).
For a class \(k\), its one-vs-rest counts give

\[
P_k=\frac{\mathrm{TP}_k}{\mathrm{TP}_k+\mathrm{FP}_k},\qquad
R_k=\frac{\mathrm{TP}_k}{\mathrm{TP}_k+\mathrm{FN}_k},\qquad
F_{1,k}=\frac{2\mathrm{TP}_k}{2\mathrm{TP}_k+\mathrm{FP}_k+\mathrm{FN}_k}.
\]

Precision/recall/F1 default to the arithmetic mean over the label set used by the
backend, with zero-division value zero. Explicit binary averaging selects a
positive class. Micro and weighted averaging, selected labels and other supported
parameters are forwarded to scikit-learn; macro averaging does not weight classes
by support. Accuracy and default balanced accuracy are

\[
\operatorname{accuracy}=\frac{\sum_k C_{kk}}s,\qquad
\operatorname{balanced\ accuracy}=\frac1{|C_y|}\sum_{k\in C_y}R_k,
\]

where \(C_y\) contains classes present in the observed labels. For multiclass MCC,
let \(c=\sum_k C_{kk}\), \(t_k=\sum_b C_{kb}\) and \(p_k=\sum_a C_{ak}\). Then

\[
\operatorname{MCC}=\frac{cs-\sum_kp_kt_k}
 {\sqrt{\left(s^2-\sum_kp_k^2\right)\left(s^2-\sum_kt_k^2\right)}}.
\]

The backend returns zero for a degenerate MCC denominator. Classification defaults
and supported averaging parameters follow the
[scikit-learn metric API](https://scikit-learn.org/stable/modules/model_evaluation.html).

## Probability losses and uncertainty

Class labels identify columns; their physical order never identifies the observed
class implicitly. Complete probability vectors must contain at least two classes,
values in the unit interval and a row sum within absolute tolerance `1e-6` of one.
They are not renormalized. For cross-entropy, define
\(c_\epsilon(p)=\min(1,\max(\epsilon,p))\), with default \(\epsilon=10^{-15}\):

\[
L_{\mathrm{CE}}=-\frac1n\sum_i\log c_\epsilon(p_{i,y_i}).
\]

`log_loss` and `cross_entropy` use this same calculation. The binary form requires
exactly two class columns. If \(z_i\) indicates one of the classes,

\[
L_{\mathrm{BCE}}=-\frac1n\sum_i
\left[z_i\log c_\epsilon(p_{i,+})+(1-z_i)\log c_\epsilon(p_{i,-})\right].
\]

The logarithm is natural, so these losses use nats. Observed classes absent from
the probability vocabulary raise. The mathematical loss is consistent with the
[log-loss definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html);
this library retains its own explicit epsilon and labeled-column contract.

Binary predictive entropy uses the two probabilities without consulting observed
labels, with the convention \(0\log0=0\):

\[
H=-\frac1n\sum_i\sum_{k\in\{-,+\}}p_{ik}\log p_{ik}.
\]

Entropy has no universal maximizing/minimizing direction. A confident wrong
forecast may have low entropy. Binary Brier score uses an explicitly identified
positive class, default label `1`:

\[
\operatorname{Brier}=\frac1n\sum_i(p_{i,+}-z_i)^2.
\]

For binary AUC, with positive/negative eligible sets \(I_+\) and \(I_-\),

\[
\operatorname{AUC}=\frac{1}{|I_+||I_-|}
\sum_{i\in I_+}\sum_{j\in I_-}
\left[\mathbf1(p_{i,+}>p_{j,+})+\tfrac12\mathbf1(p_{i,+}=p_{j,+})\right].
\]

A single observed class reports a missing AUC with a reason. Multiclass defaults
to one-vs-rest macro AUC; declared class order is preserved by integer encoding.
Other valid `multi_class`/`average` choices use the backend's definitions.

## Calibration and distribution diagnostics

For a class and nonempty probability bin \(B\), calibration plots

\[
\left(\frac1{|B|}\sum_{i\in B}p_{ik},\quad
      \frac1{|B|}\sum_{i\in B}\mathbf1(y_i=k)\right).
\]

Uniform bins partition zero to one; exact interior boundaries go to the bin on
their right and probability one belongs to the last bin. Quantile bins use unique
empirical quantile edges, so ties can reduce the number of bins. Empty bins are
omitted. No calibration transformation is learned.

Both distribution curves use the same paired eligible observations. For a
univariate sample \(x_1,\ldots,x_n\), the Gaussian KDE is

\[
\widehat f(x)=\frac{1}{nh\sqrt{2\pi}}
\sum_{i=1}^n\exp\!\left[-\frac12\left(\frac{x-x_i}{h}\right)^2\right].
\]

The bandwidth is a covariance factor times the sample standard deviation
(`ddof=1`). Scott uses \(n^{-1/5}\); Silverman uses \((3n/4)^{-1/5}\); a supplied
positive scalar is the factor itself. Each KDE integrates to one over the real
line; the displayed grid is finite and is not independently renormalized.
Constants/single observations use vertical markers. These conventions follow
[SciPy gaussian_kde](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html).

The alternatives use empirical cumulative and categorical relative frequencies:

\[
\widehat F(x)=\frac1n\sum_i\mathbf1(x_i\le x),\qquad
\widehat p(k)=\frac1n\sum_i\mathbf1(x_i=k).
\]

The ECDF table retains one sorted entry per observation, including ties.
Frequency plots use the union of observed/predicted labels and zero for absent
categories within either distribution. No absent class-probability values are
invented by these descriptive frequency counts.

## Bet accounting

For a selected, fully settled option with stake \(s_i\) and decimal odds \(o_i\),
the payout and net profit are

\[
q_i=\begin{cases}
s_io_i,&\text{win},\\
0,&\text{loss},\\
s_i,&\text{push or void},
\end{cases}
\qquad g_i=q_i-s_i.
\]

Quotes must exceed one and stakes must be finite and nonnegative for selected
bets. Missing quotes/stakes/settlements leave the selected option unresolved.
Nonselected rows have zero actual stake, payout and profit. Let \(S\) be the set
of selected bets with known settlement and complete quote/stake inputs:

\[
G=\sum_{i\in S}g_i,\qquad
V=\sum_{i\in S}s_i,\qquad
\operatorname{ROI}=G/V\quad(V>0).
\]

Push and void stakes are included in this denominator. Zero settled stake gives
undefined ROI; no selected bets gives profit zero. Profit and ROI are `partial`
while any selected bet remains unresolved. Counts and settled totals remain
explicit. After ordering each named option by its configured time column,

\[
G_j^{\mathrm{known}}=\sum_{i\le j,\ i\in S}g_i.
\]

Default kickoff ordering summarizes known outcomes; it does not reconstruct the
time cash actually became available. Both layouts use exact option identities;
team-level identities include the team. Repeated offers require per-fold study
or explicit unique-row pooling.

## Weighted experiment comparison

Let nonnegative requested weights be \(w_k\), with positive total. Zero-weight
metrics are ignored. Normalize once, and retain the same weights for every run:

\[
\widetilde w_k=\frac{w_k}{\sum_jw_j},\qquad
S_r=\sum_k\widetilde w_k u_{rk}.
\]

Only runs with finite `ok` values and matching definitions/sample/scope are ranked
together. For percentile scaling in a comparison group of size \(m>1\), rank the
raw value for maximization or its negative for minimization in ascending order,
using average ranks for ties:

\[
u_{rk}=\frac{\operatorname{rank}_{\mathrm{average}}(v_{rk}^{\mathrm{oriented}})-1}{m-1}.
\]

A singleton receives utility `0.5`; an all-tied metric also yields `0.5`. For
declared finite increasing fixed bounds \((a_k,b_k)\),

\[
z_{rk}=\operatorname{clip}\!\left(\frac{v_{rk}-a_k}{b_k-a_k},0,1\right),\qquad
u_{rk}=\begin{cases}z_{rk},&\text{maximize},\\1-z_{rk},&\text{minimize}.\end{cases}
\]

The exported contribution is \(\widetilde w_k u_{rk}\). Final rank is one plus the
number of comparable candidates with a strictly larger total score; ties share
the minimum occupied rank. Missing required metrics do not cause weights to be
redistributed. Failed runs remain visible. Scores from distinct comparison groups
cannot establish an ordering between those groups.
