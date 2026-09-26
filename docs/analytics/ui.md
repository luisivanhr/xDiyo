# Football experiment builder

The local experiment builder configures the same `FootballExperiment` pipeline used in Python. Its forms produce a portable recipe; the server compiles that recipe into the existing loaders, feature expressions, labels, splits, reporters, adapters, and selection rules. It does not implement a separate training pipeline.

## Open the builder

From a Python session or notebook in the project environment:

```python
from xdiyo_analytics.ui import launch_ui

builder = launch_ui(
    workspace=r"C:\Users\luisi\Documents\Programming\Python\xDiyo",
    open_browser=True,
)
```

For an embedded notebook view:

```python
builder.show(height=850)
```

[Notebook 17](../../notebooks/17_experiment_builder.ipynb) is a minimal launcher with one import/launch cell and commented `show`/`close` calls. It is saved unexecuted so opening it does not start a server.

Keep the Python process or notebook kernel running while using the UI. When finished:

```python
builder.close()
```

For feature checkboxes in a notebook-launched builder, open **Pre-training analysis → Discover feature columns** (or press **Prepare**) first. It builds histories/features/folds without fitting a model and keeps the discovery page open. Once complete, enable the reporter's **Features** setting and select the actual assembled columns; leaving it disabled includes all features. A completed experiment run also publishes these choices. Refresh discovery after changing data, feature definitions, the target, or layout.

After updating frontend files, refresh the builder page. A notebook kernel restart is not needed for the feature-discovery controls. Backend Python changes, such as publishing previews from completed run jobs, require reloading the relevant package modules and relaunching the builder. Save your recipe first, close the old builder, and use the updated package; closing and relaunching alone does not reload modules already imported in the same kernel. The notebook and saved experiment artifacts do not need to be recreated.

The installed command is `xdiyo-ui --workspace <project-folder>`. Alternatively, run `python -m xdiyo_analytics.ui.server --workspace <project-folder>`. `--no-browser` prints the local URL without opening a browser; `--port` selects a port instead of allocating one automatically.

Use the complete URL returned by `launch_ui`, including its fragment. The fragment contains a temporary access token used by API requests. The service binds to `127.0.0.1`; it is a local tool, not a hosted multi-user service. Do not share that URL. The static interface is included in the Python package and uses the report viewer's stylesheet.

The environment needs the analytics reporting/training dependencies. LightGBM and XGBoost appear only when already installed. Opening the builder does not install models, CUDA, or other dependencies.

## Forms and defaults

Built-in forms use a persistent, packaged inventory of parameter-specific controls, defaults, and help. Required and common settings appear directly; **Additional settings** exposes less common arguments. Optional fields with a `None` default have an enable checkbox; disabling one omits that override.

**None is an explicit value**, not a universal instruction to use the default. Its meaning belongs to the selected parameter: for example, an absent cutoff retains the documented retrospective timing assumption, while a missing search configuration disables model selection. For a parameter whose default is not `None`, omitting it and setting it to `None` can behave differently.

Controls use the parameter's domain: statistic and model selectors, numbers, checkboxes, lists, and named entries. Components can nest: for example, `RollingMean` → `ForAgainst` → `Stat`. The form does not ask the user to choose a generic Python value type. Advanced imported recipes retain the registered reference/callback/factory format; custom Python behavior needs host registration rather than arbitrary code entered into a form.

The inventory stores readable descriptions and parameter help. A form exposing a parameter does not make every possible value valid: the existing component checks still apply. Training controls, GPU selection, and checkpoint recovery depend on adapter capabilities. See [inventory operation and maintenance](ui_inventory.md) for the regeneration command and the distinction between normal form loading and explicit developer schema generation.

## Workflow

| Stage | What to configure |
|---|---|
| Data | Data folder, discovered leagues/seasons, tables, awarded-match policy, optional publication records/hash verification, statistic bundles and extra statistics, history fields. |
| Target & layout | Named `TeamValue`, `MatchTotal`, `Outcome`, `Above`, or `BetOption` labels; the selected training target; match or team-match layout; assembly column selection and missing-target handling. |
| Features & ratings | Named expressions, for/against, lags and rolling windows, EMA, H2H, league populations and LOO, warm-start policies, named rating streams, historical timing/grouping, and optional auxiliary inputs. |
| Evaluation | Temporal, grouped, match K-fold, CPCV, or explicit split plans; splitter parameters, timing/group inputs, and an optional subset of folds. |
| Pre-training analysis | Exploratory studies and separately configured fitted feature selection. |
| Model & training | Estimator, input preprocessing order, optional target scaling, adapter, prediction methods, candidate columns, validation, stopping/scheduling/restart controls, observer and fit-statistic extensions. |
| Model selection | Optional candidate grid, decision rule and evidence metrics, inner splits, holdout or nested evaluation, and candidate recipe overrides. |
| Execution & refit | Fold workers, device preference, thread/GPU limits, checkpoint policy, optional final deployment refit, reuse and run settings. |
| Post-training analysis | Any selected combination of performance, distribution, residual, calibration, timeline, betting, coefficient, learning-curve, match-result, and leaderboard studies. |
| Run & results | Experiment name, artifact folder, metadata, preparation previews, execution status, retained reports, and fitted-model saving. |
| Future predictions | A trusted saved model, upcoming statuses, optional as-of filter, and selected rounds. |

The **Scale the target** option defaults to disabled. Choose Standard, Min–Max, Max Absolute, Robust, Quantile (uniform/normal), or Power (Yeo–Johnson/Box–Cox) transformation. For supported regression models, it fits only on fitting labels and automatically restores predictions to original units, including after saving/loading. Input preprocessing offers these families independently. Keep Lasso/Elastic Net's intercept enabled ordinarily; centered inputs alone do not center the target. Transforming targets changes the regularization scale, so retune alpha when enabling it. Positive-label objectives such as Poisson/Gamma/Tweedie are incompatible with negative transformed labels. Box–Cox requires positive labels, and quantile inversion is limited to the fitting target range. Quantile transformation is distinct from quantile regression. See [scaling and prediction units](ui_scaling_verification.md) for equations, coefficient interpretation and supported adapters.

For count targets such as total corners, the model selector also includes
`training.NegativeBinomialRegressor`. It uses an NB2 distribution with a log link,
fixed configurable dispersion, optional L1/L2/Elastic Net penalties, and a mean/mode prediction choice. The mode is zero when dispersion is at least one (including the default). Keep target
scaling disabled for this model; input preprocessing remains available. Its
coefficient report uses log-mean units. See [negative binomial regression](negative_binomial.md)
for equations, assumptions and parameters.

### 1. Discover and inspect the data

Choose the ordinary prepared-export folder, then discover leagues and seasons. Discovery reads published season manifest filenames; league names containing underscores are preserved. Season inspection shows matches, statistics, pregame, and shots with row counts and short descriptions.

Statistic, team, and round choices follow the selected leagues and seasons. The statistic picker resolves its internal group and offers available periods. A statistic is still identified by its period, group, key, and value field; a friendly key alone does not establish that every selected season contains it. The usual loader handles availability when preparing the recipe.

Awarded matches remain excluded unless explicitly enabled. Current-season publications may contain both completed and upcoming fixtures. Preparation uses history from the selected publications; future prediction later selects the requested fixture rows without discarding the earlier history needed by features.

### 2. Compose features and labels

Use **+ Add** to create a named expression. For a five-match corners average, choose `RollingMean`, set `window=5`, and set its source to `Stat(period="ALL", group="Match overview", key="cornerKicks")`. Wrap that stat in `ForAgainst` to select own, opponent, or both contributions. `Stat` is a source expression; current-match raw observations are not accepted as stand-alone predictors.

League-wide features use `League` as their population and a historical reduction such as `RollingMean` or `RollingStd`. `LeaveOneOut` excludes contributions after choosing the window. `WarmStart` opts in per expression, with `SeededEMA` and an explicit `Hard`, `LinearFade`, or `ObservationCount` handoff. These retain the definitions documented in the feature library.

Named rating streams are built using the parameters of `build_ratings`; a `Rating` feature refers to a stream by its name. `MatchResultGlicko` and `StatGlicko` can also build their streams within feature evaluation. Imported recipes can use `input.RatingRun` to load an existing stream for the feature evaluator. Optional movement tables and season boundaries use the existing transition preparation functions.

**Hours before kickoff** supplies a convenient per-row cutoff. Explicit cutoff/availability vectors and auxiliary tables remain available through the parameter forms. Feature construction handles historical eligibility. Labels deliberately describe the observed outcome, and upcoming labels remain missing.

Choose the label under **Target & layout** and select the model's row layout. One row per match is appropriate for a match total; team-match layout keeps both team perspectives. The builder preserves the library's explicit assembly rules.

### 3. Prepare before fitting

**Prepare** loads data, builds histories/ratings, evaluates features and labels, assembles the dataset, and creates folds. It displays feature, label, and metadata previews plus fold sizes. No model is fitted by preparation.

Reporter feature choices use those exact assembled column names, including home/away or period expansions, rather than only the names entered under **Named features**. Discovery results are applied only if the submitted preparation settings still match the current recipe. Editing reporter selections keeps the discovered names; changing preparation inputs clears them until discovery is refreshed. Navigating away from the results panel while preparation is running does not prevent the names from arriving.

**Run experiment** uses a snapshot of the recipe submitted for that job. Editing the forms afterward does not modify that running job. UI jobs are queued in a single server worker; the execution policy can parallelize folds within a job.

If the current server has a completed **Prepare** job with exactly the same resolved recipe, a run reuses that prepared object. Otherwise it prepares new inputs. Any recipe change, including model or report settings, prevents this exact-recipe preparation reuse. To pick up changed files on disk while keeping the recipe unchanged, press **Prepare** again before running; the previously inspected snapshot is not silently replaced. This in-memory preparation reuse is separate from the experiment store's completed-result reuse.

The prepared outputs also retain matches, keyed features, labels, rating runs, and an automatically built team display catalog, alongside the team history. These are available to registered components through the appropriate prepared-data references.

### 4. Select features, train, and optionally search

Under **Evaluation**, click **Discover folds** to populate **Choose folds** from the complete outer split plan. Discovery stays on that page and works even if the enabled selection is currently empty. All Prepare actions submit an unfiltered copy (`fold_ids=None`) for discovery while preserving the live recipe's existing selection. A later Run uses that selection. For a single holdout search, select exactly one discovered outer fold; leave the setting disabled to evaluate all available folds.

Distribution reporters include an optional **Labels** checklist for observed
label histograms/KDE plots alongside input features. Disabled Features selects
all inputs; disabled Labels omits label plots. When either checklist is enabled,
select at least one item. Run names an empty reporter field and opens its analysis
page before any work starts. Prepare and Discover feature columns still work
with empty checklists, so you can load the actual columns and then choose them.
See [label distribution behavior](reporting.md#include-observed-labels-in-distribution-studies).

For a temporal split across leagues, **Unit = seasons** now groups shared source
seasons into pooled folds. **Unit = league_seasons** retains the previous separate
league-season behavior. To train on four seasons and test the next starting at
round 11, set **Train size = 4**, **Test size = 1**, **Unit = seasons**,
**Gap = 10**, and **Gap unit = rounds**. The first ten observed rounds are excluded
independently in each league; they are not added to training, and the test still
ends with its nominal season. Leaving Gap unit unset uses the window unit and
skips whole blocks instead. See the [split guide](splits.md#use-a-round-gap-inside-a-season-test-window)
for postponed fixtures, timing, and explicit calendar overrides.

Outer fold choices are kept separately from post-training report fold choices. A run containing only a selected subset does not replace the complete list of outer choices with its locally numbered fitted folds. Changing preparation inputs clears the lists until discovery runs again. Configuring a model search does not itself fit models or prevent discovering the available outer plan.

Report choices use the prospective fitted IDs in the selected order. For example, selecting outer folds `[2, 0]` offers report **Fold 0 (outer 2)** and **Fold 1 (outer 0)**. Changing the outer selection or its order clears an explicit report subset so local ID 0 cannot silently switch to a different outer fold. Choose the report subset again if needed; leaving it disabled reports all evaluated folds. Editing report settings or repeating discovery for the same selection preserves the subset.

Put descriptive studies under **Exploratory reporters**. Put the selector and any correlation study it reuses under **Fitted feature selection**, then set the candidate's **Features from** to the selector's name. This route reruns fitted selection inside eligible training populations. A descriptive overall/all ranking is not automatically a valid train-only selection.

Opened reports use an embedded frame sized to the viewport, with room for the
builder toolbar. **Expand report** opens the same retained HTML in a larger
dialog; **Close expanded report** returns to the embedded view. Expansion does
not train or fetch new experiment results. It creates a separate report view,
so interactive choices made inside one frame are not synchronized with the other.

The final report displays both the fitted correlation study and its selector
under `fitted/` names. Their rows come from the actual fitting population,
excluding reserved validation and test rows. Search reports show the evaluated
winner's studies; nested evaluation offers each outer fold's winner. Inner
search trials are not repeated as final report panels. New saved runs retain
these studies for reopening; older artifacts that never stored them cannot
recover them merely by refreshing the page.

Preprocessors are ordered and fitted within each training fold. The default adapter wraps the configured estimator. Installed LightGBM/XGBoost models use the UI's `BoostingAdapter`, which handles native fitting options, transformed validation inputs, and whole-match ranking groups. Its rankers require team-match layout. XGBoost classification encodes the fitting population's label values for its native API, then restores the original class values in predictions and probability columns; validation cannot introduce an unseen fitting class. Custom adapters can receive the configured estimator through a prepared-data reference; see [catalog extensions](ui_extensions.md).

Boosting stopping settings use the estimator's native parameters or the adapter's `fit_kwargs`, including registered native callbacks where required. The common `TrainingControl` loop requires `IterativeAdapter` or another adapter implementing those controls; supplying it to `BoostingAdapter` raises a clear error. The `pre`, `post`, and `fitted` analysis settings also expose their report titles separately from reporter-specific parameters.

For a simple grid, select a model parameter such as `alpha` and add values such as `0.01`, `0.1`, and `1.0`. In an imported recipe, a bare parameter name addresses `model.params`; a full path can address another candidate setting, such as `preprocessors.0.params.strategy`. Candidate grids do not rebuild source data or feature preparation; unsupported preparation paths raise instead of silently ignoring a requested change. Inner folds evaluate candidates. A holdout search requires one selected outer fold; nested search constructs an inner plan separately within each outer training population.

Model-selection reports display human-readable parameters. Internal trial IDs and hashes remain in machine-readable selection evidence and saved artifacts for recovery, but are omitted from the displayed comparison. Distinct comparison populations receive readable population labels.

One fold is one execution job. CPU is the initial device default. An explicit CUDA request requires a capable adapter; it is not silently applied to a CPU-only estimator. Native checkpoints require resumable-fit support. Final deployment refitting is optional and separate from retained evaluation predictions. Refit controls offer `training` or `all`; imported recipes can also provide a row-position list. After nested selection, provide an explicit candidate recipe override, for example a model component with the deployment parameters; outer-fold winners need not agree, so the builder does not invent a universal winner.

### 5. Reports, reuse, and saved models

The report viewer opens inside the UI and retains its existing study navigation, collapse controls, and fold selection. Changing report filters does not refit a model. The default `MatchResultReporter` uses the automatically prepared `team_catalog`, built from source match names; absent names retain the reporter's ID-based fallback. Override that catalog to supply other display names or optional local badges. `input.TeamCatalog` reads an existing catalog JSON file, resolving badge paths relative to that file.

The experiment name and output folder define the experiment store. Earlier final runs in that same store contribute to the leaderboard; internal search trials remain separate. Generated run names include scalar model settings to help distinguish configurations; custom run names remain available. Saved-run listing can reopen a report without preparing data or fitting a model.

Reuse is enabled by default. The engine identifies a completed compatible run from its data and configuration; completed search trials can also be recovered. Recovered experiment results are data-only and do not secretly deserialize an executable fitted model. To retain a model for prediction, save a live fitted fold model or the optional final refitted model. A multi-fold run requires choosing a fold when saving one fold model.

Model loading uses the library's explicit saved-model format. Load only trusted artifacts; the default joblib format can execute Python when read.

### 6. Predict selected upcoming rounds

In **Future predictions**, choose the saved-model folder and optional round filter. `None` includes all eligible upcoming fixtures. A round list applies across selected leagues; a league-to-round mapping selects league-specific rounds. A status filter identifies upcoming fixtures, rather than assuming that every empty statistic represents an upcoming match.

The prediction recipe prepares the full history, retains missing future targets, aligns the selected fixtures, and calls the loaded model. It does not retrain. Feature names, layout, and definitions must be compatible with the saved model.

## Save, reopen, and export

**Save recipe** writes readable JSON under `<workspace>/experiments/_recipes/`. Saving the same recipe filename replaces it. **Open recipe** reads those files; **Import recipe** reads a selected JSON file. Duplicate an experiment by changing its name before saving/running.

**Export Python** and **Export notebook** generate inspectable code using `prepare_recipe` and `run_recipe`. Exporting does not run the code. The notebook separates preparation, a small feature preview, and execution. Paths resolved by the local builder are exported as absolute paths; update them when sharing with a colleague on another machine.

The [bundled example recipe](../../examples/bundles/README.md) is usable without
the UI. Run this from the repository root; it prepares the published data and
fits the configured fixed Lasso experiment, writing new results under the ignored
`experiments/example_bundles/` directory:

```python
from xdiyo_analytics.ui import read_recipe, prepare_recipe, run_recipe

recipe = read_recipe("examples/bundles/corners_lasso.json")
prepared = prepare_recipe(recipe)
result = run_recipe(recipe, prepared=prepared)
result.show()
```

## Current boundaries

- Built-in graphical controls come from the saved inventory. New implementations need catalog registration and appropriate presentation descriptors; regenerate and review the inventory when changing built-in controls.
- The recipe search path currently enumerates a finite parameter grid and candidate overrides. An adaptive optimizer such as Optuna requires its own integration; typing its name does not create one.
- Optional model packages and physical CUDA availability are properties of the host environment. Their presence in a recipe does not install or validate a GPU runtime. LightGBM `auto` conservatively chooses CPU; XGBoost `auto` requires both a CUDA-capable build and the optional Torch device probe before choosing CUDA. An explicit XGBoost CUDA request that falls back to CPU is reported as an error. Actual GPU training was not part of the recorded UI verification.
- Report presentation and synthetic recipe/API tests are separate checks. CPU synthetic tests do not establish real-data model quality, CUDA execution, or native checkpoint recovery for every third-party adapter.
- Closing a browser tab does not cancel a running fit. Closing the server stops queued tasks, but a fit already running in its worker may finish; the current UI does not expose forceful training cancellation.

See [UI verification](ui_verification.md) for the recorded verification scope and current evidence.
