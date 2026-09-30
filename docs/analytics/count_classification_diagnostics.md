# Exact-count classification diagnostics

`CountClassificationReporter(type="overall", partition="score", pooling="last",
tolerance=2, epsilon=1e-12, group_by=("source_league",))` consumes the upstream
`predict` and `predict_proba` outputs. It does not train or calibrate a model.
Observed labels, predicted classes and class identities are nonnegative integer
counts. `targets=None` uses all retained targets. `probability_output` can select
retained raw probabilities when a calibrator also saved `predict_proba_raw`.

The summary includes MAE, MSE, RMSE, exact accuracy, macro F1, tolerance rate,
unseen-count frequency, probability availability and log-loss diagnostics.
For tolerance \(d\), its reported rate is

\[
\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\{|y_i-\widehat y_i|\le d\}.
\]

For each match, probability-weighted count is

\[
\widehat\mu_i=\sum_{c\in\mathcal C_i}c\,p_i(c).
\]

Here \(\mathcal C_i\) is the class support of the model that produced that row.
When folds have different supports, the reporter uses each source fold's actual
probability columns. An absent class has probability zero; a present column with
missing probability makes the probability diagnostics for that row unavailable.
It never fills genuine missing probability values with zero.

On rows with complete valid probabilities, clipped log loss is

\[
-\frac{1}{N_{\mathrm{valid}}}\sum_i
\log\max\{p_i(y_i),\varepsilon\}.
\]

Supported-only log loss applies the same floor to rows whose observed count
belongs to their model's support. The exact-infinite flag records whether any
eligible observed class received zero probability, including unseen classes.
The floor is explicit and does not turn this into an unqualified finite exact
log loss. Mean pooling retains its existing missing-probability policy.

The majority baseline uses the mode of the actual fitting labels after validation
and calibration reservations. Ties follow pandas' first sorted mode. Different
folds retain different baselines. Older runs without that metadata, and mean-pooled
predictions with no single source model, report the baseline as unavailable.
Grouping produces the same summaries per selected metadata group, such as league.

`FeatureImportanceReporter(type="overall", partition="model", top_k=30,
importance_type="gain", include_zeros=True)` inspects retained XGBoost boosters.
The table includes zero-gain features when requested; only the plot is limited
to top-k per model. Transformed names are preferred when known, otherwise retained
input names are used only when their mapping is established. Unknown mappings
keep native identifiers. This reporter does not select predictors or interpret
gain as a causal effect.

`PredictionTimelineReporter(..., max_points=300)` plots the first chronological
300 observations of each series while keeping its full table. All three reporters
are selectable in the experiment builder and their tables participate in
`ArtifactExport`. See [the UI settings guide](feature_inventory/current_notebook_ui_settings.md).
