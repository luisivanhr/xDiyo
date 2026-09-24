# Prediction fixtures and fitted model persistence reference

## Imports

Use `xdiyo_analytics.data` for `PredictionFixtures`, `load_prediction_fixtures`,
`select_prediction_fixtures`, `load_season`, `load_season_table` and `load_seasons`.
Use `xdiyo_analytics.training` for `save_model`, `load_model`, `JoblibSerializer`
and `FittedModel.save`. No additional package was installed for verification;
the default joblib format uses the existing sklearn environment.

## Awarded-match loading

All three loaders accept `include_awarded=False`. A boolean is required.
Only rows explicitly equal to True in matches.is_awarded are excluded, together
with their matching event IDs in every requested table. Unknown/missing flags
and child rows with unknown event IDs remain retained. Tables without event_id
are unchanged. Filtering remains within each league-season publication.

Default exclusion may read and validate matches internally for a nonempty
requested event-linked table. It keeps only requested tables in the result.
include_awarded=True restores the selected-table-only read path. Source exports
remain read-only. Existing table validation and verify_hashes behavior are retained.
The source record includes include_awarded, excluded_awarded_event_ids and
awarded_flag_available; combined loads retain per-source records. _event_membership
uses exact integer membership, preserving signed/unsigned IDs beyond `2**53` and
`2**63` without float casts. is_awarded is carried through histories/label metadata.

## Fixture loading and selection

| Parameter | Meaning |
| --- | --- |
| `data_root` | Ordinary season-publication directory, including data/xDiyo_data; daily folder names have identical semantics |
| `seasons=None` | Discover published season labels; explicit label/list narrows them |
| `leagues=None` | All published matching leagues, or exact league-name selection |
| `tables=("matches", "statistics", "pregame")` | Requested tables; matches is always included for fixture selection |
| `statuses=("notstarted",)` | One explicit status or sequence; never inferred from missing scores/statistics |
| `as_of=None` | No time cutoff by default; explicit timestamp means kickoff >= cutoff, UTC; naive interpreted UTC |
| `rounds=None` | Keep all already selected fixtures, or integer/sequence/league-name mapping |
| `include_awarded=False` | Apply the ordinary loader exclusion described above |
| `record_dir=None` | Optional per-publication version selection records, outside source root |
| `verify_hashes=False` | Optional full-file hashes; mandatory structural/size/count checks still apply |

`select_prediction_fixtures(data, ...)` takes a SeasonData with matches and applies
the same status/time/round selection without source access. Required fields are
status, event_id, home_id, away_id and kickoff_utc. Missing/unreadable times sort
last; an explicit as_of excludes them. Stable kickoff order preserves source order
for ties. It retains data itself and copies selected fixture rows.

Round numbers are nonnegative integers; strings, booleans, fractional values and
negative numbers are rejected. NumPy integer values are accepted. A mapping
uses exact source_league labels and includes only named leagues. Missing round
values are retained with None and do not match explicit integers. Requested rounds
must already be published: missing rounds produce empty rows, without fabricated
fixtures. Empty sequence/map selects none. A round column is required only when
rounds is requested; mapping additionally requires source_league. _select_rounds
and its numbers helper perform this filter without changing history.

## PredictionFixtures and alignment

`.data` retains the full loaded analytical source, including early current-season
completed matches, ongoing rows and future fixtures. `.fixtures` is the selected
matches table; `.completed_matches` returns a copy of finished rows for inspection.
Neither property defines a model training set or rebuilds features.

`align(dataset, rounds=None)` narrows the current fixture selection and returns
a ModelDataset in fixture order, with a fresh shared index for X/y/metadata and
copied definitions. It requires matching nonmissing exact match keys including
event_id. Published selected keys must be unique. Each match-layout fixture
requires exactly one prepared row and matching home_id/away_id. Team-match layout
requires exactly two rows, one home and one away, with matching team/opponent IDs;
the returned order is home then away. Missing, duplicated or mismatched prepared
rows raise. Extra unselected dataset rows remain untouched.

Compute history/features/labels on .data first and assemble with
drop_missing_targets=False. Alignment retains missing y/features and empty result
schema; it does not fit, impute or recalculate anything. A second round filter
intersects the first. For different issuance cutoffs, use the existing feature
cutoff API explicitly; this helper does not freeze availability or infer next rounds.

## Saving and loading a fitted model

`save_model(model, path, fold_id=None, serializer=None)` accepts a FittedModel or
TrainingResult. A multi-fold TrainingResult requires explicit fold_id; one fold
may be omitted. Missing fold IDs raise KeyError, and fold_id cannot be used with
a FittedModel. `_fitted_model` preserves the chosen model's feature/target order,
train/fit/validation positions, layout, identity/match columns, target perspective
and definitions. Numerical-recovery model=None is not saveable. A checkpoint
proxy is unwrapped to its underlying prediction adapter before serialization.

The default `JoblibSerializer(format_id="xdiyo.joblib.v1")` supports an
EstimatorAdapter, including its complete fitted sklearn pipeline. It preserves
class labels and probability output schema. Custom adapter classes must remain
importable; a compatible Python/library environment is required. Joblib/pickle
can execute Python: load artifacts from a trusted producer. Hashes detect changed
bytes but do not authenticate who produced them. There is no mandatory trust flag.

For another framework, pass an object with nonempty format_id and callable
save(adapter, directory)/load(directory). Save writes model state into the supplied
directory; load returns an adapter with predict(context). Preserve all learned
preprocessing and prediction state. Format mismatch is rejected; custom formats
require an explicitly supplied serializer on load. The guide includes an executable
data-only NPZ/JSON example.

`path` must be a new directory. A pending sibling receives model files, schema-1
metadata.json and model.json with format/Python version/file hashes, then _publish
makes the directory visible. Existing artifacts are never overwritten. A failed
writer leaves the destination unpublished and may leave a pending directory for
inspection. Model-state files must be present and symlinks are rejected on save.
There is no scheduled cleanup or concurrent-writer coordination.

`load_model(path, serializer=None)` validates schema, format, required files,
contained paths and hashes before restoring metadata and adapter. It returns a
FittedModel; use `.predict(dataset, positions=...)` for retained outputs. It does
not train or load original source data. `.save(path, serializer=None)` is a
convenience method on a live FittedModel.

## Three distinct artifact purposes

| Artifact | Restored content | Entry point |
| --- | --- | --- |
| Completed numerical experiment | Inputs, predictions, reports and history; model=None | FootballExperiment.load/reuse |
| Fitted prediction model | Executable fitted adapter, preprocessing and prediction contract | save_model/load_model |
| Native training checkpoint | Adapter-defined optimizer/RNG/cursor/history continuation state | CheckpointPolicy + fit_resumable |

Saving a prediction artifact does not make an ordinary sklearn fit resumable.
Experiment recovery never loads executable model artifacts implicitly.

## Exact verified signatures

```text
PredictionFixtures(data: xdiyo_analytics.data.loading.SeasonData, fixtures: pandas.core.frame.DataFrame) -> None
```

```text
load_prediction_fixtures(data_root, seasons=None, *, leagues=None, tables=('matches', 'statistics', 'pregame'), statuses=('notstarted',), as_of=None, rounds=None, include_awarded=False, record_dir=None, verify_hashes=False)
```

```text
select_prediction_fixtures(data, *, statuses=('notstarted',), as_of=None, rounds=None)
```

```text
load_season(data_root, season_stem, *, tables=('matches',), record_path=None, verify_hashes=False, include_awarded=False) -> xdiyo_analytics.data.loading.SeasonData
```

```text
load_season_table(data_root, season_stem, *, table='matches', record_path=None, verify_hashes=False, include_awarded=False)
```

```text
load_seasons(data_root, seasons, *, leagues=None, tables=('matches',), record_dir=None, verify_hashes=False, include_awarded=False) -> xdiyo_analytics.data.loading.SeasonData
```

```text
save_model(model, path, *, fold_id=None, serializer=None)
```

```text
load_model(path, *, serializer=None)
```

```text
JoblibSerializer(format_id: str = 'xdiyo.joblib.v1') -> None
```

```text
PredictionFixtures.align(self, dataset, *, rounds=None)
```

```text
FittedModel.save(self, path, *, serializer=None)
```

```text
JoblibSerializer.save(self, adapter, directory)
```

```text
JoblibSerializer.load(self, directory)
```

## Evidence and limits

The [guide](prediction_persistence.md), [notebook 15](../../notebooks/15_prediction_persistence_quickstart.ipynb),
[coverage](prediction_persistence_documentation_checklist.md), and
[verification record](prediction_persistence_check.json) describe the synthetic
checks. There is no collector, raw export or backfill change. Source freshness,
framework compatibility, custom serializer completeness and real forecasting
quality are not established by the synthetic persistence checks.
