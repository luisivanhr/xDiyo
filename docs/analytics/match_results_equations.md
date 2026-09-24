# Match-result comparison and display equations

These equations describe the reporter's decisions over retained predictions.
They define display agreement, not a training loss or aggregate performance
metric. See the [guide](match_results.md) and [reference](match_results_reference.md).

## 1. Populations and fixture identity

Let \(i\) be an original row position, \(f\) a selected fold and \(t\) a
selected target. The retained test or score occurrences form

\[
\mathcal O=\{(f,i):i\in I_f\},
\qquad I_f=\begin{cases}
I_f^{\mathrm{score}},&\text{score partition},\\
I_f^{\mathrm{test}},&\text{test partition}.
\end{cases}
\]

The shared orchestrator applies the declared pooling policy to obtain
\(\mathcal P\). Occurrence pooling retains \(\mathcal O\); first/last pooling
retains one occurrence per row position in selected fold order. Mean pooling
averages numeric prediction cells and uses the synthetic fold ID \(-1\).
A missing contributing cell stays missing. This reporter does not average
per-fold agreement rates or choose a pooling policy for repeated rows.

If \(m(i)\) is the full declared match identity, the fixture occurrence is
\(q=(f,m(i))\). A match layout has at most one row for \(q\); a team layout
has at most one row for each of \((q,\mathrm{home})\) and
\((q,\mathrm{away})\). Both sides must agree on fixture context.

## 2. Numeric agreement

For available real observed and predicted values, the signed error is

\[
e_{fit}=\widehat y_{fit}-y_{fit}.
\]

With finite nonnegative tolerance \(\tau\), the color decision is

\[
g_{fit}=\begin{cases}
\mathrm{good},&|e_{fit}|\leq\tau,\\
\mathrm{bad},&|e_{fit}|>\tau.
\end{cases}
\]

The default \(\tau=0\) requires exact agreement after the implementation's
numeric conversion. The boundary is inclusive: result 10, prediction 9.75 and
tolerance 0.25 give error -0.25 and a green group. Missing/nonfinite inputs, or
push/void/missing settlements, are neutral instead of entering this comparison.
The export retains `within tolerance`/`outside tolerance` as internal status;
the fixture table communicates it with color and the numeric Error column.

## 3. Categorical agreement

For available categorical observations and decisions,

\[
g_{fit}=\begin{cases}
\mathrm{good},&\widehat y_{fit}=y_{fit},\\
\mathrm{bad},&\widehat y_{fit}\ne y_{fit}.
\end{cases}
\]

Class labels may be strings or numbers. Numeric class codes need explicit
categorical comparison when auto mode cannot infer a categorical definition.
There is no signed error for categorical comparison. The reporter keeps each
selected target separate; it does not combine correctness across targets.

## 4. Probability validity and optional decisions

For an ordered set of at least two distinct stored classes
\(\mathcal C=(c_1,\ldots,c_K)\), a numeric vector is available exactly when

\[
\left(\forall c\in\mathcal C:\ p_{fit,c}\text{ is finite},\quad
0\leq p_{fit,c}\leq1\right)
\quad\text{and}\quad
\left|\sum_{c\in\mathcal C}p_{fit,c}-1\right|\leq10^{-6}.
\]

There is no relative tolerance or renormalization. Class-schema errors fail;
invalid vectors remain unavailable. Without a decision rule, the vector is shown
but \(\widehat y_{fit}\) is absent, so the group is neutral. When argmax is
explicitly selected, ties resolve to the first stored maximizing class:

\[
j^*=\min\{j:p_{fit,c_j}=\max_{1\leq k\leq K}p_{fit,c_k}\},
\qquad \widehat y_{fit}=c_{j^*}.
\]

For a declared binary pair \(\{c_+,c_-\}\) and threshold
\(\theta\in[0,1]\), the alternative decision is

\[
\widehat y_{fit}=\begin{cases}
c_+,&p_{fit,c_+}\geq\theta,\\
c_-,&p_{fit,c_+}<\theta.
\end{cases}
\]

The schema must contain exactly these two classes, including `positive_class`,
even when no vector is available. Invalid vectors never receive a decision.
The default auto comparison treats the chosen class categorically. A companion
vector beside a point prediction has no effect on that point's comparison.

## 5. Paired row markers

Let \(S_q\) be the available side observations for a fixture and target, with
their independently computed colors. The row marker is

\[
G_q=\begin{cases}
\mathrm{bad},&\text{some side is bad},\\
\mathrm{good},&\text{all required sides are present and good},\\
\mathrm{neutral},&\text{otherwise}.
\end{cases}
\]

A match row requires its one observation; a paired row requires home and away.
An available correct side and a missing side therefore give a neutral row
marker, while the available side retains its own green cells. This is a display
rule, not a metric or an imputed result for the absent side.

## 6. Filters, counts and downloads

Let \(L,S,T,R\) be the initial or current league, season, team and round choices,
and let \(\ast\) mean All. Visible observations are

\[
\begin{aligned}
\mathcal V=\{o\in\mathcal P:\;&(L=\ast\ \lor\ \ell(o)=L)\\
&\land(S=\ast\ \lor\ s(o)=S)\\
&\land(T=\ast\ \lor\ T\in\{h(o),a(o)\})\\
&\land(R=\ast\ \lor\ r(o)=R)\}.
\end{aligned}
\]

Team membership uses exact canonical IDs. Filtering retains the complete stored
population \(\mathcal P\). The displayed observation count is
\(|\mathcal V|\), while the fixture count is the number of distinct fixture
identities in \(\mathcal V\). Pagination operates on fixtures. The filtered CSV
exports all observations in \(\mathcal V\), across all pages; the Data CSV exports
\(\mathcal P\). Sorting changes display order, not CSV observation order or scope.
