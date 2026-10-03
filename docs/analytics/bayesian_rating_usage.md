# Bayesian ratings in Python and the local builder

The Bayesian model uses the existing rating snapshots, feature evaluation,
label, training-adapter, recipe and persistence interfaces. Its numerical
conventions and limitations are described in [the mathematical model](bayesian_rating_model.md)
and [the implementation design](bayesian_rating_design.md).
The examples below are configuration examples. They do not run production
research or establish that Bayesian features improve a downstream model.

## Fit and export a parameter bundle

`history` is paired team history built by `build_team_history`. The optional
`available_at` argument accepts a caller-supplied, aligned result-availability
series. Without it the existing earlier-finished-kickoff proxy is used.
Kickoff determines the order of rating updates; availability determines when
the resulting state can inform a forecast. A newly received old result triggers
automatic replay of saved observations in kickoff order, including downstream
season transfers. Forecasts made before its arrival keep their original values.
The new result must have an availability time strictly after the checkpoint's
observation frontier. Corrections to existing scores and partial equal-release
batches require a rebuild from complete history. Saved runs retain the replay
journal; artifacts saved before this addition need rebuilding before updating.

```python
from xdiyo_analytics.ratings import BayesianConfig, train_bayesian

trained = train_bayesian(
    history,
    mode="pooled",                  # or "per_league"
    cutoff="2024-07-01T00:00:00Z",  # explicit information boundary
    config=BayesianConfig(bivariate=True, method="vb", transition="mirrored"),
    available_at=availability,
)
trained.model.save("artifacts/goal-parameters.json")
```

Pooled training shares fixed parameters while keeping each league's team and
home-advantage states separate. Independent training fits separate bundles
and rejects unseen leagues. Its transition rule must be `mirrored`; an
explicit cross-league bridge is a pooled-mode option. The default minimum
season-label counts are three pooled and five independent. The caller must
still audit season completeness and consecutive coverage. These are pilot
eligibility defaults, not statistical guarantees.

The parameter file retains its training cutoff, conventions and calibration
provenance. It is distinct from a checkpoint containing accumulated team and
home-advantage states. Saving uses a new path and does not overwrite an artifact.

## Select features and retain historical state

To retain all available warm-up history, build a complete run first, then
query only rows whose prediction cutoffs are at or after parameter training:

```python
from xdiyo_analytics.features import BayesianFixture, Rating, evaluate_features
from xdiyo_analytics.ratings import build_bayesian_ratings

run = build_bayesian_ratings(
    history, model=trained.model, available_at=availability,
    team_seasons=movement_table,
)
target_history = history.loc[history.kickoff_at >= trained.model.training_cutoff]
features = evaluate_features(
    target_history,
    {
        "strength": Rating("goals", side="both", fields=(
            "attack_mean", "defence_vulnerability_mean", "attack_sd",
        )),
        "fixture": BayesianFixture(name="goals", fields=(
            "expected_home_goals", "expected_away_goals", "p_draw",
        )),
    },
    ratings={"goals": run},
    keyed=True,
)
run.save("artifacts/goal-state")
```

The saved run retains complete Gamma shape/rate states, league state, parameter
bundle and checkpoints, regardless of selected feature fields. A static saved
run does not assimilate results simply because new rows are passed to
`evaluate_features`. Call `run.update(...)` for strictly later releases, or
rebuild from complete history when earlier results are revised, then save a
new artifact. For backtests, the run may contain later snapshots, but queries
only use state eligible at each prediction cutoff.

For an independent cold start, use `BayesianRating(model=trained.model, ...)`
and `BayesianFixture(model=trained.model, ...)` directly. They share one filter
per model within feature evaluation. Every requested row must obey the trained
parameter cutoff; selecting a later split does not make earlier feature rows
safe. A parameter bundle alone starts from its priors and does not restore
the team states accumulated during calibration.

Larger defensive vulnerability means worse defence. `log_defence_strength`
reverses this direction. Team features are focal-team/opponent summaries;
fixture fields always mean the actual home/away teams. Feature evaluation marks
these columns with explicit fixture scope. Match-layout assembly now emits one
`fixture::<name>` column for each selected fixture output, after the home and away
team blocks. For example, `BayesianFixture` named `fixture` with multiple fields
produces `fixture::fixture::p_draw`; a single-field feature named `draw` produces
`fixture::draw`. The former `home::fixture::p_draw` and `away::fixture::p_draw`
copies are no longer produced. Update explicit selectors or derived-column
references in older recipes accordingly; existing saved artifacts are not
rewritten, and fitted models expecting the old duplicated schema need their old
input schema or retraining with the new schema.

Both copies must have the same nonmissing prediction cutoff and exactly equal
values before assembly collapses them. Two missing values agree; one missing
value or unequal values/times raises an error. `BayesianRating` and generic
`Rating` retain both teams' columns even if their values happen to be equal.
Team-match layout retains the original two rows and feature names unchanged.
Field order follows the requested
tuple. Empty, repeated or unknown Bayesian fields are rejected.

## Recipes and builder controls

The **Features & ratings** page lists **Bayesian Rating** and **Bayesian Fixture**
in the ordinary feature selector. Their **Fields** controls select outputs.
The optional **Model** chooses a loaded `input.BayesianModel` or explicit
`ratings.BayesianModel`. Feature preparation never trains those parameters.
The existing **Named rating states** form continues to configure Glicko runs.

To reuse a saved run without editing JSON, open **Saved rating runs**, click
**Add**, give the source a name, select **Bayesian Rating Run**, and enter its
artifact folder. Select that name in a **Rating** feature for team summaries,
or enable **Saved Bayesian run** in a **Bayesian Fixture** feature. The team
field choices follow the selected source; fixture name choices include only
saved Bayesian runs. When changing sources, incompatible selected fields remain
visible with a warning so you can remove them or choose a compatible source.
Saved and generated names must be distinct; duplicate names fail validation. Saving,
reopening and exporting the recipe retain this mapping. **Inspect preparation**
loads the artifact and reads eligible snapshots without training or updating it.

A recipe can reuse a complete saved Bayesian run through the existing named
rating input:

```python
from xdiyo_analytics.ui.recipe import node

recipe["feature_options"]["ratings"] = {
    "goals": node("input.BayesianRatingRun", path="artifacts/goal-state"),
}
recipe["features"] = {
    "strength": node("features.Rating", name="goals",
                     fields=["attack_mean", "defence_vulnerability_mean"]),
    "fixture": node("features.BayesianFixture", name="goals",
                    fields=["expected_home_goals", "p_draw"]),
}
```

Use a target history whose cutoffs obey the bundle's training boundary. The
saved-run input and all selected fields survive recipe JSON, Python and
notebook exports. Relative `input.*` paths resolve against the builder's
workspace. `input.RatingRun` is the generic snapshot loader;
`input.BayesianRatingRun` preserves the Bayesian time boundary and full state.

For parameter training as a normal model in the existing pipeline:

```python
recipe["labels"] = {"score": node("labels.MatchGoals", score_field="current")}
recipe["target"] = "score"
recipe["assembly"]["layout"] = "match"
recipe["features"] = {"venue": node("features.IsHome")}
recipe["model"] = node(
    "ratings.BayesianScoreAdapter",
    home_goal_target="score::home_goals",
    away_goal_target="score::away_goals",
    mode="pooled",
    team_seasons=node("input.Table", path="data/known-season-entries.parquet"),
)
recipe["preprocessors"] = []
recipe["target_transformer"] = None
recipe["adapter"] = None
recipe["candidate"] = {}  # no iterative TrainingControl or ValidationTail
recipe["run"]["model_serializer"] = node("training.BayesianScoreSerializer")
```

`MatchGoals` adds two observed target columns through `LabelData` and the
existing assembly contract. Scores use the provider's native current basis,
which is not universally regulation time. Paired score perspectives must
agree; finished observed counts must be nonnegative integers. Unfinished
fixtures retain missing targets.

The native adapter uses the fitting partition's scores and match identities;
its ordinary feature matrix is unused. It does not update with held-out
targets when predicting. Feature/target preprocessing and a second adapter
wrapper are rejected. Inspect the calibration report, then export the fitted
adapter's `model_` for another pipeline. To load its fitted prediction
artifact, supply `BayesianScoreSerializer` again, including in the recipe's
`prediction.serializer` setting. The builder exposes the explicit model
serializer under run options and lists it in the prediction page's existing
custom serializer selector.

When predicting a known team in a new season from a checkpoint, provide its
season-entry evidence: `competition_id`, `season_id`, `team_id`, `movement`,
`previous_competition_id` and `previous_season_id`. Both predecessor fields
must be explicit, including for a retained team. Explicitly null both to
request a fresh prior when predecessor evidence is unavailable. This prevents
a gap or league change from silently being treated as ordinary retention.
These are known membership facts; do not insert future match results.

The adapter supports numeric expected-goal outputs and a separate
home/draw/away probability output. Outcome probabilities are not per-target
goal-count class probabilities; generic per-target classification reports
should not interpret them as such.
