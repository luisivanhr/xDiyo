# Team display catalog and offline badges

Collected on 2026-09-18 for the first experiment's **2022/23, 2023/24 and 2024/25** seasons.

## Coverage

- **289 teams**, enumerated from the current published `matches.parquet` tables of all **39 exports / 13 leagues** in `data/xDiyo_data`.
- **286 local badges:** 284 SVG and two PNG fallbacks (1. FC Köln and Troyes).
- Three entries deliberately remain name-only: Chamois Niortais (1665), Villarreal B U23 (24338), and SPAL (113755). No badge was acquired and confirmed for these identities in this bounded collection pass.
- Team IDs are the exact `home_id` / `away_id` values used by the analytics histories. Names come from saved match metadata. If names changed between exports, `name` is the latest encountered name and `aliases` retains every encountered variant.

`catalog.json` is a JSON object keyed by the decimal team ID string. Every entry has `name`; acquired badges also have `badge_path`, `format`, `source_url`, `source_commit`, `identity_match`, `sha256`, and a source-license note. Paths are relative to `catalog.json`. The additional `leagues`, `seasons`, and `aliases` fields are descriptive provenance, not model features.

## Sources and identity matching

Most assets come from [Jose Arroyave's football-logos collection](https://github.com/JoseArroyave/football-logos), pinned to commit `1b8e57c3ea3658001cafba18c4a6db730b420ad6`. SVG files themselves credit `football-logos.cc`. The repository's declared MIT text is preserved in `SOURCE_LICENSE.txt`; this does not establish ownership of each club's trademark or artwork.

FC Köln's and Troyes' PNGs come from [leoratzlaff/football-badges](https://github.com/leoratzlaff/football-badges), pinned to commit `ba8ec269d88498a3842567887677198af43e3dd8`. Troyes uses the collection's 2022/23 season asset because the first source's 7 MB SVG wrapper exceeds the report's embedding limit; the original SVG is retained and its provenance is recorded in `previous_asset`. A separate image-specific license was not identified. Club artwork and trademark rights remain with their owners.

FC Martigues and Jahn Regensburg use Wikimedia Commons files whose descriptions explicitly identify their clubs: [Martigues 2020 crest](https://commons.wikimedia.org/wiki/File:Logo_FC_Martigues_-_2020.svg) and [Jahn Regensburg 2014 crest](https://commons.wikimedia.org/wiki/File:Jahn_Regensburg_logo2014.svg). Both description pages identify the artwork as PD-textlogo, with trademark restrictions; author attribution, source page, download URL and SHA256 are recorded in the catalog.

Identity matching uses exact normalized names within the competition's country, followed by the explicit club-name aliases recorded in `reviewed_aliases.json`. Fuzzy suggestions were not used to automatically assign assets. Villarreal's first-team asset was not silently assigned to Villarreal B. This is a display catalog of acquired club crests, **not a historically versioned crest catalog**: a current crest may differ from one used during a particular match season.

## Collection and inspection artifacts

- `inventory.json`: every contributing publication, league, and match count.
- `source_tree.json` and `fallback_source_tree.json`: pinned source-tree inventories.
- `collection.json`: acquired count, unresolved identities, and retrieval errors.
- `integrity.json`: file/hash checks, SVG XML inspection, PNG decoding and format counts.
- `renderer_compatibility.json`: acceptance checks using the current report renderer's actual helper source.
- `sample.html`: self-contained sample of 16 acquired team badges. Opening this local HTML through the computer-use browser was blocked by its URL security policy; no alternate browser/server workaround was attempted and **full visual verification is pending**. The two PNG assets were viewed directly and show the named Troyes and FC Köln crests.
- `check_assets.py`: repeat hash/integrity/renderer checks and recreate the sample.
- `inventory_teams.py`: rebuilds the name inventory from local match data; requires the existing analytics Python/pyarrow environment. **It resets badge metadata**, so run the collector afterward.
- `collect_badges.py`: downloads the explicitly mapped assets using Python's standard library. SVG files with script, foreign-object or external `href`/`src` references are rejected; the report renderer also checks embedded raster signatures. Four selected SVGs contain raster tiles as part of the original artwork and are preserved unchanged. Already-hashed SVG files are reused locally, and previously acquired alternatives are retained. The source license file is retained separately. Rebuilding the name inventory resets alternative metadata too; retain the catalog to preserve those separately sourced choices.

No collection or network request is needed when displaying the report. Unavailable badges should fall back to the real team name. No model, training data or notebook was modified by this collection.
