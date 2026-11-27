# Daily standings, folds and prepared recipe reuse

Set `recipe["league_timezones"] = "native"` to use the native competition calendars for daily standings and temporal folds. In the builder this field is under **Historical timing and grouping**. A recipe can instead provide a JSON object keyed by competition ID, such as `{"11621": "America/Mexico_City", "17": "Europe/London"}`. The native calendars are project conventions. For example MLS uses New York time and the A-League uses Sydney time. Validation resolves the preset to literal IANA zone names for portable recipe exports. An unknown competition needs an explicit timezone.

The League calendar uses the start of the issuance day. A finished result on that day enters the next day, with a resumed match using its completion date. An earlier explicit issuance and stricter known result bounds still apply. Temporal folds fit once at the earliest test issuance and use the training match’s league calendar for its label. Daily standings ranks use all eligible teams and retain the existing ranking arithmetic. A saved local-calendar table stores its calendar configuration. Frozen recipes without this field and existing UTC standings artifacts retain their former behavior. The setting does not change rolling averages or the GAT graph clock.

For direct table construction pass `timing="kickoff_day", league_timezones="native"` to `build_standings`. For a direct temporal split pass `league_timezones` in its fold options.

## Reusing preparation

Prepare once, then call `run_recipe(recipe, prepared=prepared)` for each estimator or reporter configuration. A change to inputs, feature order or definitions, labels, cohort, calendar, folds, or graph input construction requires fresh preparation. Fitting-only changes can reuse the same matrix.

## Refreshing a publication selection

Preview the update from the workspace holding the recipe and its relative paths:

```powershell
python examples/refresh_recipe_data.py recipes/daily.json --output recipes/daily-v2.json --dry-run
```

Repeat without `--dry-run` to publish the new recipe and pin records. A broad native query can admit new season publications. A missing record for an existing member still fails. When the recipe uses statistics backfill, preview and publication verify the selected manifest and its partition hashes. They preserve the chosen backfill release. Feature construction still joins its observations by native fixture identity.
