# International team identities and badges

This work audits the complete historical results file and the complete workbook,
not merely the earlier 42-club English publication. The inputs contain 218,916
historical fixtures and 146,000 workbook fixtures. They include competitions and
clubs outside the native dataset; the workbook also includes international cups
and national teams.

## Implemented identity bridge

`load_team_identity_decisions()` reads the shipped, content-bound review file.
`apply_team_identity_decisions(registry, decisions)` returns a **new** registry.
It does not edit its input, rewrite a publication, remap a saved model, classify
squads, establish competition membership or make unknown results available.
The import example installs these reviewed aliases before resolving source rows.
Competition/season admission and existing source-quality checks still apply.

The shipped file has 5,233 dated alias decisions for 471 team IDs. These include
469 fixture-corroborated native identities, an additional independently reviewed
Lommel SK mapping, and Bradford City (Sofascore ID 22). Bradford's SVG comes from
the existing pinned artwork collection. The other IDs reuse existing SVGs.
Every other existing catalogue record and selected artwork is preserved.

The automatic discovery pass requires multiple ordered native fixtures, on the
same comparison dates and in the same competition, against at least two distinct
opponents. Exact country-scoped native names and already-reviewed aliases seed
the pass. Scores and odds are never matching keys. Discovery is **candidate
evidence**, not a general fuzzy matcher or permission to publish new fixtures.
The retained decision file is explicit and separately validated on application.

Dates are inclusive. A name can map to different teams in disjoint periods.
For example, the historical `Murcia` name resolves to Real Murcia (4487) in the
reviewed 2011–2014 appearances and UCAM Murcia (55785) in 2016–2017. It must not
be a global name merge. The earlier/later Querétaro identities likewise remain
separate. A comparison calendar date does not establish the workbook timezone,
result availability or historical badge validity.

Application checks the decision digest, exact positive integer IDs, existing
canonical identity, catalogue name, SVG readability and artwork hash. Any
overlapping alias referring to a different ID fails the entire application.
The registry revision and its content change; old source/recipe hashes are not
copied into the new registry. Old published snapshots retain their original
registry and semantics. Regenerate a new publication explicitly when its
population is ready; this badge task has not silently expanded one.

```python
from xdiyo_analytics.data.catalogue import (
    Registry, load_team_identity_decisions, apply_team_identity_decisions,
)
from xdiyo_analytics.reporting import TeamCatalog

old = Registry.load("catalogue/registry.json")
decisions = load_team_identity_decisions()
updated = apply_team_identity_decisions(old, decisions)
updated.save("reviewed/registry.json")
ids = sorted({row["team_id"] for row in decisions["records"]})
TeamCatalog.builtin().export_bundle("reviewed/team_assets", team_ids=ids)
```

The destination bundle must be new. All selected assets are copied with relative
paths and verified hashes; the bundle works offline after relocation. The wheel
ships both the identity decisions and the complete badge catalogue.

## Reproduction and remaining scope

Run `examples/audit_international_team_identities.py` with `--historical`,
`--workbook`, `--native-root`, and `--output`. It reads every input fixture and
every published native match table, validates native table hashes, and writes
the private reconciliation CSV/JSON plus input/script/catalogue hash receipts.
Source workbook join IDs are used transiently and are not written to the audit.
Private source rows are not included in this repository.

The full discovery pass finds 5,227 supported source-season identity records
covering 469 native IDs. It retains 17,507 unresolved source-season records;
these are **not** 17,507 distinct clubs. Many belong to earlier seasons, other
countries, cups, national teams, or aliases requiring additional review.
The native catalogue's 483 IDs are not the universe of the international sources.

Full-source badge coverage remains unfinished. Unknown clubs are not silently
assigned a similarly named club's ID or crest. In particular, historical clubs,
successor clubs, reserve teams and national teams need separate identities.
Direct Sofascore search and public bulk register downloads returned HTTP 403
during this pass. Local browser use was disabled in Work settings. These access
limitations affected bulk acquisition; individual public pages remain searchable.
The remaining external identity and badge research is not complete. Current
XML/MIME/hash and relocation checks are not a substitute for a fresh browser
rendering check.

The research register investigated as a discovery source is not imported into
the canonical registry: its available frozen CSV had only 25 populated Sofascore
team identifiers, and the newer bulk tables could not be downloaded. No guessed
provider IDs, invented crests or successful-completion labels were introduced.

## Verification receipts

The final targeted and related suite passed 83 tests on Python 3.14.0 with
pandas 2.3.3, and 83 on Python 3.12.14 with pandas 3.0.1. This includes installed
wheel loading, decision and registry persistence, portable badge relocation,
alias collision rejection, exact large IDs, and fixture-supported discovery.
The audit script digest matches the full-input audit receipt. All 483 preexisting
catalogue entries were compared with the base commit and preserved.

See `team_identity_verification.json`, `team_identity_tests.xml` and
`team_identity_tests_py312.xml` for commands, runtimes, hashes and test results.
The private `remaining-identity-review.csv` groups unresolved source names across
seasons. Its groups are review items, not a deduplicated census of real clubs.
These checks certify the implemented bridge and asset packaging; they do not
certify complete coverage of the international sources.
