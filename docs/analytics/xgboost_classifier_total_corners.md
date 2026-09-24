# XGBoost classification of total corners

[Notebook 20](../../notebooks/20_xgboost_classifier_total_corners.ipynb) treats each observed integer total as a class: 7, 8 and 12 corners are distinct labels. Its prediction is the most probable count. It also retains the probability of each count class.

## Features and populations

The notebook imports the **same preparation function** as notebook 19 and preserves its feature settings: windows 3/5/10/20, lags 1/2/3, EMA spans 5/10, all three periods and ratings enabled. The user's expanded selection is all available leagues for `20_21`, `21_22`, `22_23`, `23_24`, `24_25`; notebook 19's own three-season selection is unchanged. This includes the existing team, opponent, league, leave-one-out, head-to-head, rating, rest, calendar and match-combination features. Optional fitted feature selection is disabled by default. The generated manifest records the actual available columns; exports lacking a statistic cannot produce that statistic's features.

The latest loaded season, ordered by its earliest kickoff, is held out; all preceding loaded seasons form development data. With the five selected seasons, `20_21` through `23_24` form outer training and `24_25` is held out. Statistic discovery uses **all four development seasons**, so a statistic introduced in the third or fourth season is eligible while a holdout-only field is not. The default expanding inner split produces three folds, evaluating `21_22`, `22_23` and `23_24` after fitting their preceding seasons. A custom `INNER_SPLITTER` is respected; only its folds contained wholly within outer training are retained. Both splitters pool leagues by `source_season`, rather than provider-specific season IDs. Historical feature construction and cutoff behavior are unchanged. Changing `SEASONS` updates the outer span automatically.

## LOO, H2H and historical warm starts

The original feature bank already contained LOO corner means over 3/5 completed
league rounds and H2H for/against corner means over 3/5 previous encounters. These
columns remain unchanged. The notebook now exposes their switches, windows and
reducers and adds spread/Z-score variants rather than duplicating the old means.

| Visible setting | Current choice and meaning |
|---|---|
| `INCLUDE_LOO`, `LOO_WINDOWS`, `LOO_REDUCERS` | Enabled; 3/5/10 completed rounds; mean, std and Z-score |
| `INCLUDE_H2H`, `H2H_WINDOWS`, `H2H_REDUCERS` | Enabled; 3/5/10 previous pair encounters; mean, std and Z-score, separately for/against |
| `WARM_FEATURE_POLICY` | `SeededEMA(alpha=0.5, handoff=Hard(rounds=1), league_weight=0.5)` |
| `WARM_STAT_KEYS` | `None`: append warmed variants for all discovered rolling statistics; a tuple such as `("cornerKicks",)` limits them |
| `WARM_RATING_POLICY` | `GlickoTransition(phi_scale=1.1, movement_phi_scale=1.0, shrinkage=0.5, top=3, bottom=3, rank_by="standings", aggregate="mean")` |
| `TEAM_SEASONS`, `SEASON_STARTS` | Optional predecessor/movement records and entry times; `None` uses supported inference |

LOO first selects the completed-round window, then removes the focal team's own
contributions without refilling the window. Its standard deviation uses individual
team observations with `ddof=1`, not round averages. Its Z-score compares the
team's latest eligible corner count with that LOO population. H2H uses ordered
team/opponent history; its default Z-score reference is the latest eligible
encounter, included in the encounter window. Insufficient spread/history stays
missing. The population additions concern **full-match corners**; other existing
statistic features are retained.

Warm starts append `warm::` columns to existing mean/std/Z-score features,
including team, league, LOO and H2H variants. All ordinary columns remain in the
same model input for direct inspection. Lag, ordinary EMA and standings features
are not wrapped. For a prior-seeded mean and population variance, each new
eligible observation updates

\[
m_t=(1-\alpha)m_{t-1}+\alpha x_t,
\qquad
v_t=(1-\alpha)\left[v_{t-1}+\alpha(x_t-m_{t-1})^2\right].
\]

The initial mean uses eligible previous-team history; known movers can blend it
with the previous destination-league mean. Initial variance uses the matching
previous-league population. H2H priors retain the opponent restriction and LOO
priors retain the exclusion. The default hard handoff uses the seeded moments
until one complete league round, then restores the original rolling calculation
and its sample-variance convention. Before any eligible prior exists, the ordinary
calculation is used; no synthetic first-match observations are invented. Available
`LinearFade`/`ObservationCount` policies can replace `Hard`; their consistent
moment blending is described in the [warm-start guide](league_warmup.md).

Both result and corner Glicko streams get **separate warmed variants**, using each
stream's own prior state. Season entry preserves location and inflates uncertainty
by the configured factor. With explicit supported movement context, shrinkage can
use the configured top/bottom destination cohort. Standings ranking uses available
pre-transition pregame standings, not a reconstructed final table. `None` context
infers retained membership where supported; it does **not** guess promotion or
relegation. Supply the documented [team-season records](league_warmup_reference.md#context-inputs)
or results from its movement adapter to enable those transfers.

These policies operate on historical features/rating states, **not on XGBoost
optimizer continuation**. To compare a separate unwarmed run, set both warm
policies to `None`; the configuration/cache identity changes. Policies, population
choices and supplied transition context are recorded in preparation configuration;
the feature manifest adds `variant` and `family` columns. The current model grid
does not search warm-up policies. Notebook 19 retains its previous default feature
definitions because helper defaults preserve the original populations and disable
new warm variants.

## Optional correlation feature selection

Two settings in the configuration cell control the existing selector:

```python
FEATURE_SELECTION_PROPORTION = None  # e.g. 0.8 enables selection
FEATURE_SELECTION_CORRELATION = "spearman"  # or "pearson", "kendall"
```

`None` retains 100% of assembled inputs, including nonrankable columns. When
enabled, `TopKCorrelationSelector` ranks absolute feature/target correlation and
requests

\[
k=\lceil p\,d\rceil,
\]

where \(d\) is the number of assembled input columns before removing undefined
associations. For example, 0.8 requests 9 of 11 columns. Fewer than \(k\) can be
retained when fewer columns have finite correlations; constants or entirely
missing columns commonly have undefined scores. `1.0` requests every **rankable**
column. This notebook converts it into an integer count because the common
selector's literal `k=1` means one feature. `None` and `1.0` therefore differ.

Selection belongs to each candidate's `pre_analysis` with `type="per_fold"`,
`partition="train"`, consumed through `features_from`. Every inner training fold
fits its own ranking; the selected candidate fits a new ranking on the complete
outer training population. Neither inner validation outcomes nor outer holdout
outcomes enter that ranking. The separate exploratory Spearman report is not
used for fitted selection. Selector options participate in candidate/cache identity.

The notebook displays the winner's actual retained count/fraction; exact names
are exported to `classifier_summary/selected_features.csv`. Its summary JSON also
records the requested proportion/method and actual count/fraction. Candidate fold
rankings/names remain in their fitted reports. `TOP_GAIN_FEATURES` only limits the
importance plot, while XGBoost `colsample_bytree` independently samples available
columns per tree **after** any fitted selection.

## Grid and class balancing

The user's current grid has **54 candidates** (preserved when adding features):

| Parameter | Choices |
|---|---|
| Maximum tree depth | 2, 3 |
| Minimum child weight | 20, 30, 60 |
| Trees | 100, 150, 200 |
| Class-balance power | 0, 0.05, 0.1 |

The remaining model settings are visible in the configuration cell. Candidate selection minimizes inner-fold **MAE of the predicted count**. It then fits the selected configuration on all development seasons and evaluates the untouched last season. With the current five seasons this means 54 candidates × 3 inner folds, followed by the selected outer fit. Native early stopping is disabled: the tree count is selected through chronological CV. This also avoids requiring every inner validation count to have occurred in the fitting sample.

For fitting sample size \(N\), number of observed classes \(K\), class frequency \(n_c\) and configurable power \(p\), each observation receives

\[
u_i=\left(\frac{N}{K n_{y_i}}\right)^p,
\qquad
w_i=\frac{u_i}{N^{-1}\sum_{j=1}^{N}u_j}.
\]

Thus weights have mean one. Power zero is unweighted; 0.5 balances mildly; 1 gives equal total weight to each observed class. Frequencies are computed **inside each fit**, from its optimization rows only. They are not computed from validation or test rows. If another caller supplies a validation tail, those rows are excluded from the frequencies too; native validation labels must then be supported by the fitting classes.

The existing adapter encodes sparse counts into XGBoost's contiguous class IDs and maps both predictions and probability columns back to the original counts. For example, fitting labels 3, 7 and 12 become internal IDs 0, 1 and 2; the report still shows 3, 7 and 12.

## Results and probability interpretation

The report contains MAE, MSE, exact accuracy, macro F1, count-frequency plots, a leaderboard and **MatchResultReporter with numeric tolerance 2**. A match is within tolerance when

\[
|y_i-\hat y_i|\leq 2.
\]

Numeric comparison is intentional: exact-class comparison would ignore this tolerance. The additional tables show the within-two rate, per-league diagnostics, fitting class frequencies/weights, gain importance and an empirical majority-count baseline. Probability-weighted expected counts are retained as a separate diagnostic; they do not replace the classifier's modal prediction.

Held-out counts absent from the fitting class universe remain in every point-prediction metric and are flagged explicitly. Their assigned probability is zero. Accordingly, exact log loss is infinite if any observed class has probability zero. `log_loss_clipped_all` includes every row and uses the declared floor \(\epsilon=10^{-12}\):

\[
L_\epsilon=-\frac{1}{n}\sum_{i=1}^{n}\log\max\{p(y_i\mid x_i),\epsilon\}.
\]

`log_loss_supported_only` is a supplementary, explicitly restricted diagnostic. The notebook also reports the unsupported count rate; it never silently removes those outcomes from the main evaluation or expands the fitted class universe using test labels. Log loss is not the selection criterion.

Balancing changes the training objective and can worsen common-count predictions. Weighted probabilities are **not automatically calibrated** to actual match frequencies. This notebook makes no betting-probability or profit claim and supplies no invented odds.

## Execution and saved outputs

CPU is the default, with one candidate/fold job and four native threads. Set `DEVICE="cuda:0"` only when you want to use a supported GPU. The configuration cell exposes grid choices and thread limits.

Completed candidate artifacts use the pipeline's recovery mechanism. `REUSE=True` reuses compatible completed results; `False` requests fresh fitting. Saved outputs include candidate comparisons, all count probabilities, predictions, metrics, per-league results, the feature manifest, fitting class weights, a fitted model and the navigable report. The final notebook cell reloads the model and checks prediction equality without refitting. No separate deployment refit is enabled by default.

Open notebook 20 and select **Run All**. From the repository root, the equivalent command below writes a separate executed copy:

```powershell
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' -c "import nbformat; from nbclient import NotebookClient; p='notebooks/20_xgboost_classifier_total_corners.ipynb'; n=nbformat.read(p,4); NotebookClient(n,timeout=7200,kernel_name='misc314_py314').execute(); nbformat.write(n,'notebooks/20_xgboost_classifier_total_corners.executed.ipynb')"
```

## Verification and execution boundary

Seven focused tests passed in **2.62 seconds**. They exercise actual XGBoost fitting/search, independently calculated weights, validation-tail isolation, sparse class encoding, unseen test counts, probability alignment, the numeric tolerance report, persistence and exact completed-result reuse. They also compare notebook 20's feature configuration with notebook 19 and validate/compile the notebook.

A separate copy completed **every notebook cell in 5.95 seconds** on the existing small synthetic export: 23 matches, 117 available inputs, 16 training rows, 7 held-out rows and a two-candidate CPU grid. It retained the complete feature configuration; the fixture contains fewer available statistics. Report rendering and reloaded-model prediction parity passed. The executed proof is at `.pytest_tmp/classifier_notebook_check/20_xgboost_classifier_total_corners.executed.ipynb`.

**The default full-data 24-candidate grid has not been run during verification.** Running the notebook launches that grid. Synthetic results demonstrate functionality, not predictive performance on real football matches. Saved user experiments and running production jobs were not modified.

After the user expanded the season selection, **15 affected helper tests passed in 6.04 seconds**, including five-season discovery, one latest-season outer holdout, three expanding inner folds, and preservation of a custom sliding inner splitter. Notebook configuration, code cells and existing outputs were preserved; only the outdated season explanation changed. Restart an existing notebook kernel before Run All so it imports the updated shared helper. No expanded real-data grid was executed.

The subsequent LOO/H2H/warm-start extension passed **50 affected tests in 9.03
seconds**. New hand-calculated cases cover individual-observation LOO variance/Z,
missing first encounters, prior-free first-season behavior, one-round handoff,
explicit cross-league priors, rating uncertainty inflation and unchanged predictions
after perturbing current/future outcomes. These tests exposed and fixed a narrow
unsigned-ID comparison overflow in H2H prior filtering; identifiers above
\(2^{63}\) now retain exact comparisons.

A fresh disposable notebook copy completed all cells in **9.35 seconds** with
23 synthetic matches, **105 expressions / 293 inputs**, two candidate fits,
16 outer training rows and 7 held-out rows. All newly enabled policies remained
active, and reports/model reload passed. Its output is
`.pytest_tmp/classifier_warmup_notebook_check/20_xgboost_classifier_total_corners.executed.ipynb`.
Only fixture seasons/data and compute limits were overridden in that copy.
The user's five-season selection, edited 54-candidate grid and existing notebook
outputs were preserved. The expanded real-data grid remains unrun by this work.

The optional selection controls passed **13 focused tests in 5.40 seconds**.
Actual XGBoost candidate/winner fits verified independent training populations,
Pearson/Spearman/Kendall ranking, upward rounding, nonrankable-column handling,
the disabled path and recorded selected names. A separate exact notebook copy
with `0.8` Spearman selection completed in **6.86 seconds**, retaining **235 of
293 inputs** on the synthetic fixture, including reports and saved-model reload.
Its executed notebook is under `.pytest_tmp/classifier_selection_notebook_check/`.
The delivered setting remains `None`; no full real-data search was executed.
