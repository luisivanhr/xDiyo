# Supplied international team mapping

The mapping supplied on 2026-11-08 contains 1,204 unique team identities across
38 competition countries, all with exact positive canonical team IDs. There
are no duplicate source keys, duplicate target IDs or missing target IDs.
All source keys and names were checked against the original workbook. The
413 supplied identities also covered by native fixture corroboration agree
with that evidence. The remaining identities retain the user's supplied
web-profile/external-mapping provenance; they are not described as independently
web-reverified by the local agent.

## Integration

`load_reviewed_team_directory()` loads the content-bound, packaged directory.
`apply_reviewed_team_directory(registry, directory)` returns a new registry.
Existing canonical IDs and names must agree. Aliases are country-scoped to
already-reviewed club leagues and bounded by actual workbook appearance dates.
They are not extrapolated to earlier seasons. Conflicting aliases or identities
fail before a result is returned. The original registry is never mutated.

All 1,204 identities are retained, including teams whose competitions have not
yet been admitted. Applying the directory does not create competitions, seasons,
fixtures, membership or squad eligibility. After additional competitions are
reviewed, reapplying the directory adds aliases for those countries. Japan and
Mexico retain an explicit compatibility mapping for legacy native region fields.
The import example applies the original fixture-corroborated decisions first,
then this supplied directory. Existing dated distinctions remain intact.

The published representation contains canonical IDs, names, country, bounded
aliases, evidence URLs and input hashes. Source workbook join IDs are transient
during preparation and are not added to the directory, registry or match schema.
Country denotes the reviewed competition context, not inferred geography; for
example, clubs from New Zealand can participate in the Australian competition.

Identity and artwork are independent. A supplied identity can be registered
without an available badge; a conflicting existing catalogue name is rejected.
In this delivery, every supplied identity does have a validated local image.

## Badge coverage

- 417 supplied IDs already had local badges; their catalogue entries are unchanged.
- 687 genuine SVGs were added from the pinned existing artwork collection.
- 100 remaining images were acquired by exact reviewed team ID: 89 PNG and 11 WebP.
- Total supplied coverage: 1,204 of 1,204, comprising 1,104 SVGs and 100 raster images.
- The complete catalogue now has 1,271 entries. All 484 preexisting entries and
  selected asset bytes remain unchanged, including the native league SVG inventory.

All newly acquired images retain source URLs, hashes and format metadata. SVG
sources were verified against pinned Git blob hashes and native renderer checks.
Three require documented nonvisual sanitation (empty Adobe metadata or JPEG MIME
spelling); source and installed hashes are separate. Raster crests remain in their
actual formats; no raster file is renamed to SVG or presented as vector artwork.
Current crests are display assets, not historical crest-availability evidence.

The 1,204-team offline bundle was reloaded and compared against the builtin
catalogue for exact names and image content. The 100 raster crests were also
inspected in a contact sheet; none is empty or a repeated placeholder. This is
image/packaging verification, not a browser interaction test.

## Reproduction

Prepare the same directory from the original inputs:

```powershell
$env:PYTHONPATH='src'
python examples/prepare_supplied_team_mapping.py --mapping <supplied-csv> --workbook <original-xlsx> --output <new-directory-json>
```

Preparation validates the supplied source keys, original names and competition
countries, retains exact integer identities and derives alias date bounds from
the workbook. The complete original workbook was reread and produced a directory
identical to the shipped JSON. The CSV and workbook hashes are stored in it.

```python
from xdiyo_analytics.data.catalogue import (
    Registry, load_reviewed_team_directory, apply_reviewed_team_directory,
)
from xdiyo_analytics.reporting import TeamCatalog

directory = load_reviewed_team_directory()
updated = apply_reviewed_team_directory(Registry.load("registry.json"), directory)
updated.save("reviewed/registry.json")
TeamCatalog.builtin().export_bundle(
    "reviewed/badges", team_ids=[row["team_id"] for row in directory["teams"]],
)
```

Use a new bundle destination. No original dataset publication or saved model is
rewritten. New publications must explicitly use the new registry and its actual
content hash; there is no silent saved-artifact ID migration.

## Verification and limits

See `supplied_team_mapping_verification.json` and the adjacent test XML receipts
for final source hashes, commands, runtime versions and results. Tests cover
country/date scope, legacy native region names, exact large integers, collisions,
duplicate IDs, malformed URLs, tampering, idempotence, reload, input preservation,
original workbook identity validation, wheel installation and portable badge MIME.

Coverage here is **the supplied 1,204 mappings**. The workbook contains additional
identities, and the historical text source has additional names and older seasons.
This delivery does not claim that all of those are reconciled, or that the existing
42-team named publication has automatically expanded. Full vector coverage also
remains incomplete: the 100 raster selections need genuine SVG replacements if
an SVG-only international collection is required.
