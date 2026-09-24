# Choose prediction rounds and save a fitted model

Load completed and future matches from the usual `data/xDiyo_data` season
publications. Choose the published fixtures you want to predict, compute their
features using the full loaded history, and apply a previously fitted model.
The folder name has no special daily meaning; there is no inferred “next round.”

For your existing export, the usual entry point is:

```text
batch = load_prediction_fixtures("data/xDiyo_data", "26_27",
    rounds={"Premier_League": 5, "La_Liga": [6, 7]})
```

That call is shown without executing it here. The runnable examples below build
tiny synthetic publications in a new demo folder. They do not change collector
exports or train a live football model. [Notebook 15](../../notebooks/15_prediction_persistence_quickstart.ipynb)
is the shorter workflow. Its [synthetic helper](../../notebooks/helpers/prediction_publications.py)
only writes the example publications.

## 1. Load prediction fixtures while keeping completed history

```python
from pathlib import Path
import numpy as np
import pandas as pd
from notebooks.helpers.prediction_publications import create_demo_publications
from xdiyo_analytics.data import load_prediction_fixtures
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import Stat, IsHome, RollingMean, evaluate_features
from xdiyo_analytics.labels import MatchTotal, create_labels
from xdiyo_analytics.datasets import assemble_dataset
from xdiyo_analytics.training import EstimatorAdapter, refit_model, save_model, load_model
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

root = Path.cwd()
output = root / "experiment/prediction_persistence_demo/guide"
data_root = create_demo_publications(output / "synthetic_exports")
batch = load_prediction_fixtures(data_root, "26_27", tables=["matches", "statistics"],
    rounds={"Alpha": [5, 6], "Beta": [6, 7]}, verify_hashes=True)
assert len(batch.data.matches) == 14
assert len(batch.completed_matches) == 4
assert len(batch.fixtures) == 4
print(batch.fixtures[["source_league", "event_id", "round", "status"]])
```

By default, all three analytics loaders exclude explicitly awarded matches
(`is_awarded=True`) and their event-linked rows in every requested table.
Missing/unknown flags stay included. `include_awarded=True` explicitly retains
those rows and the original selected-tables-only read behavior. Default exclusion
may read matches internally even when only statistics or shots were requested;
it does not add matches to the returned requested-table collection.

`.data` retains the entire loaded analytical population: completed, ongoing and
future rows. `.fixtures` is the status/round/time-selected prediction population.
`.completed_matches` is an inspection view of status='finished'; it does not
choose training rows or remove other history. Awarded metadata is also retained
through histories and label metadata when the rows are explicitly included.

## 2. Compute features first, then align the prediction rows

```python
history = build_team_history(batch.data)
corners = Stat("ALL", "Match overview", "cornerKicks")
features = evaluate_features(history, {
    "home": IsHome(), "recent": RollingMean(corners, window=2),
}, keyed=True)
labels = create_labels(history, {"corners": MatchTotal(corners)})
dataset = assemble_dataset(features, labels["corners"], layout="match", drop_missing_targets=False)
prediction_data = batch.align(dataset, rounds={"Alpha": 5, "Beta": 6})
assert len(dataset.X) == 14 and len(prediction_data.X) == 2
assert prediction_data.y.isna().all().all()
assert prediction_data.metadata.source_league.tolist() == ["Beta", "Alpha"]
```

This preserves early current-season matches for history and training. Alignment
only selects already assembled rows by exact source/league/season/event and team
identity; it does not construct features or impute missing values. Match layout
retains one row per fixture. Team-match layout requires and orders home then away
for each fixture. All X/y/metadata frames are reindexed together. Missing targets
and feature values remain missing. Do not drop unknown targets during assembly.

`rounds` may be an integer, a list of integers, or a league-name mapping. A mapping
selects only its named leagues. `None` keeps the current fixture selection; an
empty list/map or unpublished round produces empty rows with the same schema.
The load/select/align filters intersect: alignment cannot reintroduce fixtures
excluded when loading. All choices preserve source history and dataset inputs.

Selection uses explicit statuses, default `("notstarted",)`, regardless of missing
scores/statistics. There is no wall-clock filter by default. Optional `as_of`
keeps kickoffs at or after a supplied UTC instant (naive times are treated as UTC).
Missing kickoffs sort last without as_of and do not satisfy an explicit cutoff.
Catch-up fixtures and differing league round numbers remain available as published.

## 3. Fit once, save the fitted pipeline, and predict after loading

```python
training_rows = np.flatnonzero(
    dataset.metadata.status.eq("finished").to_numpy()
    & dataset.y.notna().all(axis=1).to_numpy()
)
def model_factory():
    return EstimatorAdapter(make_pipeline(
        SimpleImputer(keep_empty_features=True), StandardScaler(), Ridge(alpha=.2),
    ))
fitted = refit_model(dataset, model_factory, train_positions=training_rows)
expected = fitted.predict(prediction_data)
model_path = fitted.save(output / "ridge_model")
restored = load_model(model_path)
actual = restored.predict(prediction_data)
pd.testing.assert_frame_equal(actual["predict"], expected["predict"])
predictions = prediction_data.metadata[["source_league", "event_id", "round", "home_id", "away_id"]].copy()
predictions["predicted_corners"] = actual["predict"].iloc[:, 0]
predictions.to_csv(output / "predictions.csv", index=False)
print(predictions)
```

The imputer, scaler and estimator are saved together after fitting. Loading does
not access the training source or fit anything. Feature order, target names,
layout and class-probability columns are preserved. Missing prediction labels are
valid. The numbers here are only a smoke demonstration on four synthetic training
matches, with no performance claim.

`save_model(fitted, path)` is equivalent to `fitted.save(path)`. A TrainingResult
can also be saved; multiple folds require an explicit `fold_id` and no fold is
automatically chosen from evaluation scores. Paths must be new directories.
Reusing an existing model directory raises instead of overwriting it. To rerun
this demo, choose a fresh output folder; to reuse a model, call load_model.

The default JoblibSerializer supports EstimatorAdapter/sklearn pipelines.
Joblib/pickle loading can execute Python, so load artifacts from a trusted producer
in a compatible Python/library environment. Hashes check file integrity, not the
producer's authenticity. The API adds no mandatory trust flag or approval step.

## 4. Supply a native serializer for a custom adapter

Custom frameworks explicitly provide `format_id`, `save(adapter, directory)` and
`load(directory)`. The returned adapter must implement predict(context). Here a
small least-squares adapter saves its fitted scaling, weights and names as NPZ/JSON.

```python
import json

class NativeLinear:
    def fit(self, context):
        self.columns, self.targets = tuple(context.X), tuple(context.y)
        self.mean = context.X.mean().to_numpy()
        self.scale = context.X.std(ddof=0).to_numpy()
        self.scale[self.scale == 0] = 1
        design = np.c_[(context.X.to_numpy() - self.mean) / self.scale, np.ones(len(context.X))]
        self.weights = np.linalg.lstsq(design, context.y.to_numpy(), rcond=None)[0]

    def predict(self, context):
        design = np.c_[(context.X.to_numpy() - self.mean) / self.scale, np.ones(len(context.X))]
        return {"predict": pd.DataFrame(design @ self.weights, index=context.X.index, columns=self.targets)}

class NativeSerializer:
    format_id = "example.native-linear.v1"
    def save(self, adapter, directory):
        np.savez(directory / "state.npz", mean=adapter.mean, scale=adapter.scale, weights=adapter.weights)
        (directory / "names.json").write_text(json.dumps({"columns": adapter.columns, "targets": adapter.targets}))
    def load(self, directory):
        adapter = NativeLinear()
        with np.load(directory / "state.npz", allow_pickle=False) as state:
            adapter.mean, adapter.scale, adapter.weights = state["mean"], state["scale"], state["weights"]
        names = json.loads((directory / "names.json").read_text())
        adapter.columns, adapter.targets = tuple(names["columns"]), tuple(names["targets"])
        return adapter

# This small custom example chooses complete known-context columns explicitly.
native = refit_model(dataset, NativeLinear, train_positions=training_rows,
                     feature_columns=["home::home", "away::home"])
serializer = NativeSerializer()
native_path = save_model(native, output / "native_model", serializer=serializer)
native_loaded = load_model(native_path, serializer=serializer)
pd.testing.assert_frame_equal(native_loaded.predict(prediction_data)["predict"], native.predict(prediction_data)["predict"])
```

A prediction model artifact is separate from a numerical experiment result and
from an interrupted-training checkpoint. FootballExperiment.load/reuse still
returns model=None and never loads joblib implicitly. Its `CheckpointPolicy`
requires an adapter that retains optimizer/RNG/cursor/history state; saving a
fitted prediction model does not establish mid-fit resumability.

Each save serializes into a new pending directory, records data-only FittedModel
metadata and file hashes, then publishes the completed directory. Serializer
failure leaves the destination unpublished. Loading rejects missing/corrupt files,
escaping paths, wrong schema and mismatched serializer format. Custom native
state and environment compatibility remain the serializer's responsibility.

See the [complete API reference](prediction_persistence_reference.md),
[coverage checklist](prediction_persistence_documentation_checklist.md), and
[verification record](prediction_persistence_check.json). All source publications,
fourteen prior notebooks and previous verification evidence remain preserved.
