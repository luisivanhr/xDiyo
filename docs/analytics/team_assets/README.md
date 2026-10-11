# Team display catalog and offline badges

Initially collected on 2026-09-18 for three experiment seasons; expanded on **2026-10-12** to all current published seasons.

## Coverage

- **483 teams**, enumerated from the published `matches.parquet` tables of all **289 exports / 18 competition labels** in `data/xDiyo_data`, including Japanese and Mexican competitions and historical seasons.
- **483 local badges:** 284 SVG, 170 PNG and 29 WebP. There are **no missing badges** in this inventory.
- All 286 previously acquired badge files and their source metadata are preserved. The expansion adds 197 badges, including Chamois Niortais (1665), Villarreal B U23 (24338), and SPAL (113755).
- Team IDs are the exact `home_id` / `away_id` values used by the analytics histories. Names come from saved match metadata. If names changed between exports, `name` is the latest encountered name and `aliases` retains every encountered variant.

`catalog.json` is a JSON object keyed by the decimal team ID string. Every entry has `name`; acquired badges also have `badge_path`, `format`, `source_url`, `identity_match`, `sha256`, and a source-license note. Repository sources have a pinned `source_commit`; provider images have a retrieval timestamp and dimensions. Paths are relative to `catalog.json`. The additional `leagues`, `seasons`, and `aliases` fields are descriptive provenance, not model features.

## Sources and identity matching

Most assets come from [Jose Arroyave's football-logos collection](https://github.com/JoseArroyave/football-logos), pinned to commit `1b8e57c3ea3658001cafba18c4a6db730b420ad6`. SVG files themselves credit `football-logos.cc`. The repository's declared MIT text is preserved in `SOURCE_LICENSE.txt`; this does not establish ownership of each club's trademark or artwork.

FC Köln's and Troyes' PNGs come from [leoratzlaff/football-badges](https://github.com/leoratzlaff/football-badges), pinned to commit `ba8ec269d88498a3842567887677198af43e3dd8`. Troyes uses the collection's 2022/23 season asset because the first source's 7 MB SVG wrapper exceeds the report's embedding limit; the original SVG is retained and its provenance is recorded in `previous_asset`. A separate image-specific license was not identified. Club artwork and trademark rights remain with their owners.

FC Martigues and Jahn Regensburg use Wikimedia Commons files whose descriptions explicitly identify their clubs: [Martigues 2020 crest](https://commons.wikimedia.org/wiki/File:Logo_FC_Martigues_-_2020.svg) and [Jahn Regensburg 2014 crest](https://commons.wikimedia.org/wiki/File:Jahn_Regensburg_logo2014.svg). Both description pages identify the artwork as PD-textlogo, with trademark restrictions; author attribution, source page, download URL and SHA256 are recorded in the catalog.

The 197 new assets come from Sofascore's public team image endpoint, `https://img.sofascore.com/api/v1/team/{team_id}/image`, using the exact provider IDs already present in published match data. This also resolves reserve-team identities directly, including [Villarreal B (24338)](https://www.sofascore.com/football/team/villarreal-b/24338). Each catalog record preserves the actual image URL, retrieval timestamp, dimensions and SHA256. No separate image license was identified; club artwork and trademark rights remain with their owners. UCAM Murcia's single-frame GIF is losslessly encoded as RGBA PNG because the renderer does not accept GIF; its original source hash and conversion are recorded. Other new files are stored unchanged.

The original collection matched exact normalized names within the competition's country and explicit aliases in `reviewed_aliases.json`; the expansion matches exact provider IDs. No fuzzy assignments are used. This is a display catalog of acquired club crests, **not a historically versioned crest catalog**: a current crest may differ from one used during a particular match season, especially for renamed or relocated clubs. Provider identities and source names are preserved without attempting to resolve historical club succession.

## Collection and inspection artifacts

- `inventory.json`: every contributing publication, league, and match count.
- `source_tree.json` and `fallback_source_tree.json`: pinned source-tree inventories.
- `collection.json`: acquired count, unresolved identities, and retrieval errors.
- `integrity.json`: file/hash checks, SVG XML inspection, PNG decoding and format counts.
- `renderer_compatibility.json`: acceptance checks using the current report renderer's actual helper source.
- `sample.html`: self-contained sample of 32 acquired team badges, including 16 from the expansion. All 197 new badges were visually inspected in labeled contact sheets; all 483 assets pass the renderer helper, hash and format checks. The HTML page itself has not been browser-tested.
- `check_assets.py`: repeat hash/integrity/renderer checks and recreate the sample.
- `inventory_teams.py`: refreshes the inventory from every current local publication; requires the existing analytics Python/pyarrow environment. Preserves existing badges, their source metadata and previously recorded aliases.
- `collect_missing_badges.py`: downloads only missing assets by exact provider team ID; requires Pillow for decoding and the documented static GIF conversion. Retains all acquired assets on repeated runs. Run this after refreshing the inventory, then run `check_assets.py`.
- `collect_badges.py`: original collection helper for the 13 European leagues, retained as provenance. SVG files with script, foreign-object or external `href`/`src` references are rejected; the report renderer also checks embedded raster signatures. Four selected SVGs contain raster tiles as part of the original artwork and are preserved unchanged. Use `collect_missing_badges.py` for the expanded inventory.

No collection or network request is needed when displaying the report. Unavailable badges should fall back to the real team name. No model, training data or notebook was modified by this collection.
