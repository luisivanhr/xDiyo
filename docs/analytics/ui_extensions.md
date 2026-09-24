# Extending the experiment builder catalog

The catalog translates a named component plus its parameter mapping into a Python object. Registration happens in trusted Python code before the local server starts. A recipe cannot import an arbitrary module or evaluate a Python string.

```python
from xdiyo_analytics.ui import catalog_for_ui, launch_ui

catalog = catalog_for_ui()
catalog.register(
    "my_models.MyAdapter",
    MyAdapter,
    category="training",
    title="My football model",
    description="Adapter for the model used by our group.",
    fields={"width": {"kind": "number"}},
)
builder = launch_ui(catalog=catalog, workspace=".")
```

`MyAdapter` must already be imported or defined by the host. Its constructor signature defines the form fields; optional arguments become defaults. Field hints can provide choices, suggestions, a preferred value kind, or clearer titles. Reporter/selector implementations retain their existing shared contracts. The relevant categories include `feature`, `label`, `rating`, `warmup`, `split`, `pre_reporter`, `post_reporter`, `model`, `preprocessor`, `training`, `selection`, `evaluation`, `display`, and `input`.

## Recipe values

The following are data structures, not executable strings:

```python
{
    "component": "features.RollingMean",
    "params": {
        "window": 5,
        "source": {
            "component": "features.Stat",
            "params": {"period": "ALL", "group": "Match overview", "key": "cornerKicks"},
        },
    },
}
```

The catalog recursively constructs nested lists, mappings, and components. Tuple parameters used as immutable AST cache keys are restored from JSON lists. Dataclass defaults, including nested warm-up policies, are encoded recursively. Special values such as infinity and NumPy numeric types have explicit `constant` representations rather than invalid JSON tokens.

`{"ref": "estimator"}` refers to the estimator currently being assembled. Other contexts expose prepared objects such as `history`, `matches`, `dataset`, `labels`, or `features` where available. References are stage-specific: a form listing a reference does not make that object available before it is built. Unknown/unavailable references raise a clear error.

`input.Table(path=..., column=..., index=...)` reads a CSV or Parquet auxiliary table, optionally selecting an index and column. It does not load pickle files. The caller owns the auxiliary data's alignment with the relevant history/dataset. `input.TeamCatalog` reads a local display catalog.

For reusable rating data, `input.RatingRun` loads the library's saved snapshot format. The registered transition context and team-season helper constructors accept their own existing public arguments. Saved-model prediction also accepts a registered custom serializer; `training.JoblibSerializer` is available explicitly when that format is appropriate.

## Factories and callables

Some adapter contracts expect a factory or callback, rather than an already constructed object. The recipe has explicit representations:

```python
# Construct a new backend when the training loop asks for one.
{"factory": {
    "component": "training.PartialFitBackend",
    "params": {"estimator": {"ref": "estimator"}},
}}

# Pass an already registered callable without invoking it here.
{"callable": "my_helpers.device_inventory"}
```

Register callbacks under a named catalog key before using them. Native device setup, checkpoint writers/readers, custom likelihood calculations, and exotic model adapters belong in those Python implementations. The browser supplies their configuration; it does not invent framework-specific behavior.

Factories used for training restarts must create fresh mutable model and preprocessing state. The built-in deferred factory copies its context, clones a referenced sklearn estimator, and applies the supplied restart seed to its `random_state` parameters. A seed-aware backend may reference the factory's `seed` context. An estimator pipeline lacking `partial_fit` does not become incremental merely by wrapping it; place compatible train-only preprocessing in the backend's preprocessor argument when using `PartialFitBackend`.

## Recovery and reproducibility

The catalog supplies a recovery identity for registered constructors. The experiment engine also tracks recipe, data, split, model, and code identity. Custom objects that capture external state need an appropriate `cache_key()` or explicit configuration describing that state, following the existing experiment recovery contract.

Saved recipes store names and data. A collaborator must have the corresponding registered implementations and compatible packages before opening and executing an extended recipe. Registering a component does not automatically grant it GPU support, a native checkpoint implementation, a persistence serializer, or every reporter scope; those remain explicit capabilities of the component.
