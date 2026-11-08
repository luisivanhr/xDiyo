# Combined fixture rounds and supplied team identities

This integration combines the exact audited round commit
`f7d94c666418a502ff2d495ab2dd841a0e5c8e49` and supplied identity/badge commit
`8a20df9933a34aeb7ca7f4b3c43761fd5bf19fea` on
`codex/v59-rounds-team-identities`. Their common base is
`ec19875ad13de83187a42762e9094e016c778042`. Both changes are retained as merge
parents. There were no textual conflicts. The round-release payloads, round
integration code, canonical catalogue implementation, team directory, badge
catalogue and selected badge assets are unchanged from their respective parents.

The existing clean isolated checkout was reused. No new worktree was created;
the other identity checkout and both original branch tips were left unchanged.
Main was not merged or updated.

## Data compatibility

The V59 release contains 61,488 workbook-keyed fixtures. All 122,976 home/away
appearances resolve uniquely using the supplied directory's exact country/name
aliases and inclusive effective-date bounds. There are 814 distinct canonical
teams in that population, all with valid local badges. No fuzzy match, score
match, or inferred cross-date fixture remapping was introduced.
The 61,488 country/league/season/date/canonical-pair keys remain unique, with no
home/away identity collapsed into a self-match.

Workbook `match_id`, `home_team_id` and `away_team_id` remain in their original
source namespace. The supplied canonical team IDs are used for identity
resolution and badge display; they do not replace the source fields against
which the round join is validated. The combined test also applies the supplied
directory to a synthetic registry, confirms that neither the input registry nor
the four Australian source fixtures are mutated, and checks that no fixture,
season or squad eligibility is created by that application.

The seven selected round/stage fields still use the V59 whitelist. The four
Australian choices remain provider fallbacks, with original-draw verification
false and all competing source claims retained. The 368,928 field-ledger rows,
794 historical flags, three explicit clears, 60 MLS series ordinals, 13,168 null
season rounds and 745 named stages without numeric rounds remain intact. Audit
and source metadata remain outside model feature columns.

The supplied directory retains all 1,204 teams. Their badge coverage remains
1,104 SVG, 89 PNG and 11 WebP images. The complete catalogue has 1,271 entries.
All 484 entries and selected artwork hashes present before the supplied-team
commit are preserved. Current crests are display assets, not evidence of
historical availability; raster files remain correctly labelled raster files.

## Verification

The combined suite covers both parent suites plus two interoperability tests:

```powershell
$env:PYTHONPATH = 'src'
$env:TEMP = '<checkout>/scratch/combined-pip'
$env:TMP = $env:TEMP
$env:PIP_NO_INDEX = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
python -m pytest -q `
  tests/analytics/test_fixture_rounds.py `
  tests/analytics/test_round_team_compatibility.py `
  tests/analytics/test_reviewed_team_directory.py `
  tests/analytics/test_team_identity_reconciliation.py `
  tests/analytics/test_badge_bundles.py `
  tests/analytics/test_native_dated_aliases.py `
  tests/analytics/test_named_datasets.py `
  tests/analytics/test_named_dataset_repairs.py `
  tests/analytics/test_team_asset_inventory.py::test_inventory_includes_old_seasons_and_preserves_badge_provenance `
  tests/analytics/test_loading.py `
  tests/analytics/test_awarded_loading.py `
  tests/analytics/test_supplemental_fixtures.py `
  tests/analytics/test_named_movements.py
```

The final combined suite passed **180 tests in each environment**, with no
failures, errors or skips. The executed XML receipts and `verification.json`
record the results.
The runtimes are Python 3.12.14 / pandas 3.0.1 / PyArrow 25.0.1 and
Python 3.14.0 / pandas 2.3.3 / PyArrow 23.0.0. The first Python 3.14 run passed
174 tests and failed six because that interpreter lacked optional reporting,
workbook, scientific and wheel-build dependencies. Those dependencies were
installed into a checkout-local environment, leaving the source code and global
Python installation unchanged, before rerunning the full suite.
`verify.py` independently checks both preserved source trees, the complete
round-release hashes, all dated round/team matches, all supplied badge hashes,
and exact preservation of the prior 484 catalogue entries. It does not publish
data, prepare features or train a model.

## Limits

This combines the annotation and identity capabilities without silently
expanding an existing named publication, rewriting a registry snapshot, or
remapping a saved model. V59's workbook IDs are not canonical fixture/event IDs;
a future canonical round integration still needs an explicit fixture crosswalk.
The supplied list does not claim to reconcile every identity in all historical
sources. The team branch's retained provenance and 413 native-corroborated
overlaps are unchanged; the additional aliases were not newly web-reverified.
No new browser rendering audit was performed.

Forecast timing, phase grouping and feature-time readiness remain separate
work. No experiment was prepared or run and no forecast policy was changed.
