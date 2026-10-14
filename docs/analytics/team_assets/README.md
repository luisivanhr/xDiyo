# Team display catalog and offline SVG badges

Updated 2026-10-12 against every published league-season in `data/xDiyo_data`.

## Coverage and identity

- **483 distinct team IDs across 289 published league-seasons**, all with local SVG badges.
- Includes historical teams, reserves, Japan and Mexico. The previous collection covered only 289 teams in 39 recent European exports.
- IDs are the exact `home_id` / `away_id` values in published matches. `aliases`, `leagues` and `seasons` describe their dataset appearances; they are not model inputs.
- This is a display catalog, not a historically versioned crest timeline. Renamed clubs may display a historical or current crest. The Llagostera crest is explicitly mapped to the stable club ID subsequently named Costa Brava / Badalona Futur. Reserve sides using the parent club's crest have an explicit `shared_badge_team_id` and identity source.

`catalog.json` maps decimal team IDs to names, relative `badge_path`, actual file format, SHA-256, source URLs and identity/provenance notes. Reports embed these assets locally and require no network requests. Every current catalog entry uses SVG; six original SVG artworks contain embedded raster tiles. They remain self-contained. The two previous PNG catalog selections have been replaced by SVG crests.

## Sources and transformations

Most badges come from [Jose Arroyave's football-logos collection](https://github.com/JoseArroyave/football-logos), pinned to `1b8e57c3ea3658001cafba18c4a6db730b420ad6` or `a80039b889ba3308aa83e52cd20d0967be4b833a`. Source SVGs credit football-logos.cc. The repository's declared MIT text is retained in `SOURCE_LICENSE.txt`; it does not establish ownership of each club's artwork or trademark.

Additional explicitly identified crests come from Wikimedia, Logotyp, Seeklogo, Freebie Supply and Logoneate. Each entry records its source URL/page and available license/attribution. Artwork and trademark rights remain with their owners. Matching uses country-scoped normalized names and reviewed aliases, not automatic fuzzy assignment. `reviewed_aliases.json` records the selected repository paths; the catalog records their exact revisions.

Lobos BUAP, Lorca FC and Llagostera were supplied as AI/CDR vector originals and converted using Inkscape 1.4.2 (`--export-type=svg --export-plain-svg --export-text-to-path`). The export page is fitted to the drawing with `--export-area-drawing`. Troyes and Lokeren also have excessive source canvas margins trimmed with that option. These originals are preserved under `originals/`, with conversion details and hashes in the catalog. Standalone SVG document-type declarations from some sources are removed before native validation; source and resulting hashes are recorded separately. Git attributes preserve exact asset bytes across Windows checkouts.

## Repeating the checks

Run from the repository root in the analytics Python environment:

```shell
python docs/analytics/team_assets/inventory_teams.py
python docs/analytics/team_assets/check_assets.py
python -m pytest tests/analytics/test_team_asset_inventory.py tests/analytics/test_match_results_catalog.py tests/analytics/test_graph_reporter_browser.py
```

The inventory reads **all** publication manifests and preserves existing badge/provenance fields. It never downloads or guesses badges. A new unmatched team causes the catalog coverage regression to fail until a reviewed asset is added.

Artifacts:

- `inventory.json`: contributing publications and exact team IDs.
- `source_tree.json`, `source_tree_20261012.json`, `fallback_source_tree.json`: pinned source inventories.
- `collection.json`: scope, counts and unresolved identities.
- `integrity.json`, `renderer_compatibility.json`: exact hashes, XML/content validation and native renderer acceptance.
- `browser_validation.json`: all 483 SVGs decode and paint nonempty pixels in offline Chromium.
- `sample.html`: self-contained badge sample.

The embedding browser regression covers coincident team coordinates, league visibility, repeated find/reset/pan/zoom, content-based PNG/SVG detection and offline SVG export. Badges with identical positions are retained without changing their embedding coordinates; Find team raises the selected badge above coincident ones. Unavailable or unreadable custom assets fall back to colored points and names.

No match tables, model weights or experiments are modified by this asset update.

## Integrated badge coverage

The main-branch coverage expansion (`bfe249c`) is preserved alongside the SVG
selections: all 483 identities remain covered, all added raster assets remain
available, and 199 alternative badge records retain their exact source metadata
under `alternative_assets` in the catalog. Reports continue to select the
reviewed SVG for every current team. The missing-badge collection helper from
main is retained for future acquisitions.

## Portable report packaging

`TeamCatalog.builtin()` loads this complete catalog both from a checkout and
from an installed wheel. The wheel includes the catalog, selected SVGs and
source/license documentation. The UI uses this same loader; it does not depend
on an external `docs` directory after installation.

For an independent report bundle:

```python
from xdiyo_analytics.reporting import TeamCatalog

catalog_path = TeamCatalog.builtin().export_bundle("report/team_assets")
catalog = TeamCatalog.from_json(catalog_path)
```

The destination must be new. By default **every** catalogued team is included,
including historical teams, Japan and Mexico. An explicit `team_ids` iterable
can select a subset. Missing entries/files, invalid images and incorrect recorded
SHA-256 values fail before the output directory is created. The exported catalog
uses relative paths and content-detected MIME types, so it remains usable after
moving the report. It preserves provenance; alternative asset records describe
unselected source assets and are not copied. Selected image bytes are unchanged.
Include this source/license documentation when redistributing the collection.

For a report generated with an old recovered subset, replace its generation
catalog with the complete bundle and regenerate presentation from retained
projection tables. Do not refit models or recompute t-SNE/PCA. Native embedding
traces now retain exact team IDs in Plotly `ids`, independently of labels and
coordinates, to support subsequent presentation repairs. Already generated HTML
is not automatically rewritten when the library is upgraded.
