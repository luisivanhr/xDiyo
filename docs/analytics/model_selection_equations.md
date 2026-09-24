# Model-selection equations

These equations describe the implemented calculations and their scope. All
references to a winner refer to a configuration chosen from development data.
The independent evaluation remains separate.

## Populations and fresh fitting

Let \(D\) be the declared development row positions, \(T_f\) an inner
training set, \(E_f\) its test set and \(S_f\) its score subset. Then

\[
T_f\subseteq D,\qquad S_f\subseteq E_f\subseteq D,
\qquad T_f\cap E_f=\varnothing.
\]

Each population preserves complete matches, including both observations in a
team-match layout. If monitoring validation is \(V_f\subset T_f\), actual
fitting and consumed feature selection use \(F_f=T_f\setminus V_f\).
Each candidate/fold creates fresh fitted objects. A feature rule and scaler
are learned from \(F_f\), then applied to \(E_f\).

For a selected configuration's outer fold, its training rows \(T_o\) lie in
\(D\), while \(E_o\cap D=\varnothing\). The selected rule and model are
fitted freshly on the allowed outer fitting population.
These set conditions do not establish chronological causality; the splitter
and feature availability rules establish that separately.

## Prediction metrics and repeated rows

Without repeated score rows, pooled MSE is

\[
\operatorname{MSE}_c=
\frac{\sum_f\sum_{i\in S_f}(y_i-\widehat y_{c,f,i})^2}
     {\sum_f |S_f|}.
\]

This calculation gives each scored observation equal weight. It differs from
an unweighted mean of fold MSEs when fold sizes differ. The shared metric
implementation retains counts, missing status and sample identity; incomplete
required evidence is not ranked.

With repeated score rows, choose the policy explicitly:

| Policy | Population and prediction |
|---|---|
| `occurrences` | Retain each fold/row pair; repeated appearances each contribute. |
| `first` / `last` | Retain one prediction per original row, following retained fold order. |
| `mean` | Average numeric predictions per original row, then calculate the metric once on unique rows. |

For mean pooling over \(A_i\), the folds predicting row \(i\),
\(\overline y_{c,i}=|A_i|^{-1}\sum_{f\in A_i}\widehat y_{c,f,i}\).
A missing output cell in any contributing occurrence remains missing.
Outcomes/metadata for repeated row identities must agree. Mean pooling is
not categorical voting.

## Weighted utility

Positive weights \(w_j\) are normalized as
\(\widetilde w_j=w_j/\sum_{k:w_k>0}w_k\). For candidate \(c\),

\[
U_c=\sum_{j:w_j>0}\widetilde w_j u_{cj}.
\]

With percentile scaling among \(m\) comparable eligible candidates, order raw
values from worst to best and assign average ranks \(r_{cj}\). Then
\(u_{cj}=(r_{cj}-1)/(m-1)\) for \(m>1\); a single candidate receives
\(u_{cj}=1/2\). All tied values also receive utility \(1/2\).
Minimizing metrics reverse the ordering. Exact total-utility ties follow the
original candidate order. Percentile utility can change when candidates change.

With fixed reference bounds \(a_j<b_j\), maximizing utility is
\(u_{cj}=\operatorname{clip}((x_{cj}-a_j)/(b_j-a_j),0,1)\).
For a minimizing metric, use one minus that quantity. Fixed bounds do not change
with the candidate set.

Every required metric must be finite and complete on comparable observations.
Missing required metrics leave the candidate unranked. There is no
candidate-specific redistribution of weights. Separate sample/definition
groups cannot compete in one built-in selection decision.

## Parsimony

Let \(b\) be the best scalar metric value, with its declared direction, and
\(\delta\ge0\) the absolute practical tolerance. The near-best set is

\[
\mathcal N=\{c:\ c\text{ is ranked and }|x_c-b|\le\delta\}.
\]

For a supplied complexity measure \(q_{c,f}\), the aggregate is
\(\overline q_c=K^{-1}\sum_{f=1}^{K}q_{c,f}\).
A missing measure in any fit makes this aggregate missing. Choose the finite
smallest complexity in \(\mathcal N\), then the better metric, then candidate
order. This is a declared tolerance rule; it does not estimate a standard error
or perform a significance test.

## Fitted likelihood evidence

For fitted unpenalized data log likelihood \(\ell\), justified parameter
count/effective degrees of freedom \(k\), and sampling-unit count \(n\),

\[
\operatorname{AIC}=-2\ell+2k,\qquad
\operatorname{BIC}=-2\ell+k\log n.
\]

These are the conventions exposed by the official
[statsmodels AIC helper](https://www.statsmodels.org/stable/generated/statsmodels.tools.eval_measures.aic.html)
and [BIC helper](https://www.statsmodels.org/stable/generated/statsmodels.tools.eval_measures.bic.html).
The adapter supplies likelihood constants, response units, parameter convention
and sampling units explicitly. A schema check cannot establish that a likelihood
or effective degrees-of-freedom formula is suitable for the fitted estimator.

For the guide's Gaussian regression with known variance \(\sigma^2=4\),
ordinary least squares supplies

\[
\ell=-\frac n2\log(2\pi\sigma^2)
      -\frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i-\widehat y_i)^2.
\]

Only the fitted regression coefficients contribute to \(k\) in that example;
variance is fixed, not estimated. This convention is not inferred for Ridge,
Lasso or ElasticNet from their penalties or nonzero coefficients.

For \(K\) inner fits, the recorded selection criteria are explicitly

\[
\overline{\operatorname{AIC}}_c
 =\frac1K\sum_{f=1}^{K}\operatorname{AIC}_{c,f},
\qquad
\overline{\operatorname{BIC}}_c
 =\frac1K\sum_{f=1}^{K}\operatorname{BIC}_{c,f}.
\]

Each individual fit remains in the evidence table. This arithmetic mean
is not a single full-population AIC/BIC and does not treat overlapping fits as
independent likelihood factors. Comparability binds fitted response identities,
convention IDs and sampling-unit counts.

## Nested evaluation

For each outer fold \(o\), inner selection uses only \(T_o\), returns a
configuration \(c_o\), and a fresh fit predicts \(E_o\).
Final post-training analysis receives only those outer predictions. There is
no universal configuration selected by comparing the outer outcomes.

The [guide](model_selection.md) executes these workflows and the
[reference](model_selection_reference.md) lists every option and helper.
