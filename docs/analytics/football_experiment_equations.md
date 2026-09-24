# Recovery and scope invariants

## Scope remains explicit

For an outer fold, training positions \(T\) and test positions \(E\) are disjoint,
and scored positions \(S\) are a subset of the test positions:

\[
T\cap E=\varnothing,\qquad S\subseteq E.
\]

Holdout selection uses only the declared development population \(D\). Its inner
folds use original dataset positions. Nested selection first maps an outer
training subset to a local dataset, then maps inner folds back to original
positions for retained evidence. The selected model is fitted freshly on the
outer training rows. Deployment refit has a separately declared population.

## Recovery identity

The execution key is a deterministic digest of the recorded input description:

\[
k=H(\text{data, order, definitions, splits, configuration,
factory code, analyses, policies, source, runtime}).
\]

This is an identity for reuse, not a statistical guarantee or proof that hidden
external state is unchanged. A matching completed result is loaded. Otherwise
completed search trials with matching keys may be loaded while missing work is
executed. Explicit external revisions belong in configuration or cache_key().
Input mutation during an execution and concurrent writers are outside the
contract. Prepared auxiliary outputs alone do not enter this key.

## Equivalent resumed fitting

For the guide's illustrative momentum SGD, with row gradient \(g_t\), weights
\(w_t\), momentum \(v_t\), and fixed rate \(\eta\):

\[
v_{t+1}=0.4v_t+g_t,\qquad w_{t+1}=w_t-\eta v_{t+1}.
\]

A checkpoint must retain weights, momentum, fitted normalization, random state,
the completed cursor and history. Restoring weights alone changes later updates
when momentum or shuffled observation order differs. Independent tests compare
an interrupted/resumed path with an uninterrupted reference, including exact
predictions, weights and loss history for this deterministic synthetic adapter.
Those checks do not claim reproducible floating-point paths for every device or
framework. See the [guide](football_experiment.md) for the complete native example.
