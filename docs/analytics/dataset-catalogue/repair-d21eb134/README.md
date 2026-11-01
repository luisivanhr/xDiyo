# Named dataset audit repairs

Base reviewed: `d21eb1344cd28dcd761b97bcf6d369bef137c64b`.

## Repairs

- Explicit wheel packaging includes `xdiyo_analytics.data.catalogue`. Native imports no longer depend on the checkout-only scraper competition registry. An isolated installed-wheel test exercises legacy and catalogue imports outside the source checkout.
- Selection checks effective-dated squad classifications across every selected snapshot, including publications with no retained matches. Conflicting decisions fail before selection; a reserve correction cannot be undone by an older alternate publication. Decisions outside the fixture dates do not cause false conflicts.
- Native entity numbers are candidates, not proof of identity. Occupied team, competition and season IDs require compatible facts or a reviewed scoped alias; dependent fixture/statistic keys use the resolved canonical IDs. Suspicious existing-ID or nearby/reversed fixture occurrences require a reviewed occurrence or distinct-fixture decision.
- Dataset previews capture their root and generation. Input edits, replacement requests and cancellation invalidate old responses. Acceptance binds the root used for the accepted preview. Desktop/mobile and delayed-response Chromium tests cover the actual native panel.
- Recipe migration is an explicit plan/apply workflow. Selection schema 2 freezes each fold's ordered training/evaluation/scoring memberships; normal temporal CV may reuse a previous test fixture in a later training fold, without crossing roles within a fold. Schema 1 selections remain readable and are revalidated under the stronger checks.

## Migration

```python
from xdiyo_analytics.data.catalogue import (
    plan_recipe_dataset_migration, apply_recipe_dataset_migration,
)

plan = plan_recipe_dataset_migration(
    "old-recipe.json",
    catalogue_root="/path/to/catalogue",
    snapshots={"comprehensive_dataset": "<exact publication hash>"},
    # mapping={"<legacy fixture ID>": <canonical fixture ID>},
)
print(plan["status"], plan["blockers"], plan["diff"])
# Only after inspecting a ready preview:
result = apply_recipe_dataset_migration(plan, output_path="new-recipe.json")
```

`xdiyo-migrate-recipe old.json another.json --catalogue-root <root> --snapshot comprehensive_dataset=<hash>` prints a per-file JSON dry run. Directories expand their immediate `*.json` files. Add `--apply --output-dir <new-directory>` to write new recipes. `--overwrite` explicitly backs up an existing output before replacement. Output filenames include a source-path hash, avoiding collisions between equal basenames from different folders. Batch exit code 2 means at least one recipe was blocked; every input retains its own result.

By default competition/season scope, target/history membership, fold hyperparameters and paired fixture rows remain fixed. The planner prepares features but never fits models. It checks retained scores and required statistics, feature/label parity for exact migrations, eligible history sequences and target timing. Large IDs stay integers or decimal strings, never JavaScript numbers. Original recipes remain byte-for-byte unchanged unless explicit backed-up overwrite is requested.

`mode="extend_history"` freezes targets and folds while explicitly extending earlier history in the same competitions. `mode="union"` explicitly constructs the chosen scope's union and freezes the existing splitter's resulting folds. Population differences and membership fingerprints are included in the preview; applying any changed population requires `approve_population_changes=True` (CLI `--approve-population-changes`). Expanding competitions/seasons requires explicit Python scope arguments; it is never inferred from migration consent. Changed source observations are blocked even for an expansion.

Unknown top-level legacy fields remain opaque and hash-bound in a versioned migration envelope, and are retained alongside all model/feature settings. Normal recipes still reject unknown sections. Changed population/split definitions require a new preview. Apply recomputes the plan, rejecting stale originals, selections or dependencies. A matching already migrated recipe is a no-op; a different snapshot/policy requires the original recipe and a fresh preview. Root relocation is a separate explicit operation, not a snapshot upgrade.

## Boundaries

Unsupported graph/spatial/rating/membership dependencies, path-based split topology, legacy cutoff/history overrides, ambiguous or missing identities, coverage losses and unavailable training labels block migration with reasons. A fixture-ID mapping does not authorize changing participants, competition, season or date; those changes require canonical review first. Missing required fixture IDs are reported, not silently dropped. There is no automatic migration of saved models or experiments.

The retrospective profile remains conditional research: nominal reported scores, unarchived corrections and unknown publication times are not converted into verified historical evidence. Existing private catalogue and recipe files are checked read-only; no raw data, derived Parquets or model runs are published with this repair. Initial reviewed coverage is unchanged. Review expansion of unresolved leagues/teams separately.

Fresh test/environment/source-hash receipts are in this directory. Earlier verification receipts are preserved.
