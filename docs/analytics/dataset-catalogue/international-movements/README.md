# International season-entry flags

The current local international publication is enriched with the same native
promotion/relegation semantics as the main dataset. These describe entry into
the season, not relegation at the end of that season.

- 9,676 fixtures, 39 competition-seasons, 582 observed team-seasons.
- 80 promoted, 14 relegated, 488 retained; zero unknown flags.
- Scope: the already admitted Premier League clubs, including their admitted
  Championship fixtures. This does not expand the quarantined international
  identities or assert complete league membership for partial imports.
- All previous match values, statistics and odds compare exactly after reload.
  All four tables use ZSTD.

The new immutable version is
`69694ab5ad6ff0a5622031e708df1a16b92db42b101defd48347d92cb1c21940`.
The previous `893ff59b...` publication remains unchanged for pinned recipes.
Select the new version explicitly and regenerate the selection/feature artifacts
when opting in. There is no automatic change to existing experiments.

## Evidence and derivation

2010 onward uses hash-verified native `team_seasons` records joined by exact
competition/season/team identities. Full native evidence is restricted to the
imported population only after movement derivation. Earlier Premier League
seasons use the league's complete [promotion history](https://www.premierleague.com/en/news/412265)
for 2000/01–2009/10, encoded as explicit boundary decisions passed to the existing
`derive_movements`. Non-promoted top-flight participants are retained, not
relegated from an imagined higher division. Missing teams in partial imports
are never treated as proof of promotion.

`publication.json` records boundary IDs, native manifest/table hashes, the
reproducible script hash and output hashes. Source data and derived Parquets stay
in the private local catalogue; redistribution rights remain unestablished.

## Native use and UI

In **Data → Named datasets**, select the new international version and include
`goals,movements` in required capabilities. Preview and accept that selection.
Add the existing **TeamMovement** feature, choosing promoted or relegated; JSON
and Python exports use the same API. With match layout this produces one flag
per side for each selected movement type.

```python
from xdiyo_analytics.data.catalogue import select_data, resolve_selection, selected_history
from xdiyo_analytics.features import TeamMovement, evaluate_features

selection = select_data(
    root, snapshots={"international_dataset": version},
    requirements=["goals", "movements"],
)
data = resolve_selection(root, selection)
features = evaluate_features(selected_history(data), {
    "was_promoted": TeamMovement("promoted"),
    "was_relegated": TeamMovement("relegated"),
})
```

Matches expose `home_got_promoted`, `home_got_demoted`, `home_season_entry` and
the corresponding away fields. `team_seasons` stores the native evidence and
predecessor identities. Native history exposes team/opponent equivalents.

The `reviewed_season_entry_v1` policy and all table hashes enter publication
identity. Reload checks evidence and materialized fields; cross-publication
contradictions are rejected. Old publications without this table still load.
Unknown evidence stays nullable and is excluded when movements are required.
This enrichment does not enable unsupported standings or membership features.

Retrospective reviewed flags are not archived historical announcement times.
The existing conditional retrospective research profile applies; strict as-of
selections requiring movements reject unsupported movement availability. No
historical timestamps are invented.

## Reproduction and verification

Set `PYTHONPATH=src` and run:

```text
python examples/enrich_international_movements.py --catalogue <private-catalogue> --version 893ff59ba396080e6c03303b07cc95fc08ac9e83937271754a9dd29fba15f697 --native-root <native-data-root> --receipt docs/analytics/dataset-catalogue/international-movements/publication.json
python docs/analytics/dataset-catalogue/international-movements/verify.py <private-catalogue>
python -m pytest tests/analytics/test_named_datasets.py tests/analytics/test_named_dataset_repairs.py tests/analytics/test_movement_enrichment.py tests/analytics/test_movement_review.py tests/analytics/test_named_movements.py tests/analytics/test_explicit_warmup.py tests/analytics/test_ui_explicit_warmup.py tests/analytics/test_native_dated_aliases.py tests/analytics/test_named_dataset_browser.py -q
```

The new regressions cover immutable/repeat publication, ZSTD, materialized-field
consistency, missing/contradictory evidence, exact IDs, partial membership,
strict availability, selected subsets, conflicting unions, tampering, native
feature evaluation and JSON/Python recipe preparation. Existing import, wheel,
movement and UI tests are included. No model fitting or collector work is run.

The full native feature check covers 19,352 team-perspective rows and verifies
every flag against the loaded data. A reusable accepted selection is saved as
`international-movements-selection.json` in the private catalogue.

Run the two environments sequentially: wheel tests share the checkout's
setuptools build directory. An initial concurrent run collided in that directory;
its generated build output was cleared and final checks were rerun serially.
Windows sandbox restrictions also require the existing hard-link/wheel tests to
run with normal local filesystem permissions.

Final receipts: **158 passed** under Python 3.14/pandas 2.3.3;
**154 passed, 2 skipped** under Python 3.12/pandas 3.0.1. The skips are missing
Playwright in the latter environment; browser checks passed in Python 3.14.
`verification.json` binds the final source and test receipt hashes.
