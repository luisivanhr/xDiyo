# Explicit named datasets

The opt-in catalogue adds immutable `comprehensive_dataset` and `international_dataset` publications, coherent union selection, and extended history with fixed training/evaluation fixtures. Existing schema-2 loaders, recipes, datasets and experiments keep their previous behavior. No model experiment is run by the importer or verification example.

## What is reviewed and what remains quarantined

The initial squad-classification batch covers senior Premier League participants, with aliases anchored to existing canonical club identities. Fixtures between those reviewed clubs in other already-mapped competitions can also be admitted; this is not a claim that every international league has been reconciled. Four independently reviewed reserve identities remain in the registry and are excluded. Other unresolved competition/season/team identities and squad classifications stay ineligible. Unknown teams are never guessed from a suffix, score or similar spelling.

The entire input population is validated, including all five workbook sheets, before publication. Aggregate dispositions distinguish competition review, season review, identity review, squad review, reserve exclusion and accepted observations. Expanding coverage requires canonical review decisions and a new immutable version. The original broad-statistics dataset remains fully accessible through its legacy loader; the new filtered publication is a separate product.

The data remain **local research publications**. Redistribution rights have not been established. This commit contains implementation, synthetic tests and aggregate receipts, not either original file or derived match/statistics/odds Parquets.

## Import locally

From the repository root, with `src` on `PYTHONPATH`:

```powershell
python examples/import_named_datasets.py `
  --legacy-root data/xDiyo_data `
  --results-txt '<unchanged historical v2 TXT>' `
  --workbook '<unchanged results workbook>' `
  --output '<separate private catalogue directory>'
```

The command preserves the durable `registry.json` on repeat imports. Canonical IDs are allocated monotonically after reserving existing native IDs; temporary workbook join IDs never enter publication. Exact, scoped and effective-date aliases resolve entities. Reviewed date corrections use `Registry.map_occurrence`; nearby same/reversed pairs are quarantined until reviewed. An unrelated new fixture can be allocated without requiring a comprehensive counterpart.

Each named dataset has a version directory with ZSTD Parquets, canonical registry and manifest. Data and metadata are staged before one atomic directory rename. Pending directories are not listed; an interrupted run cannot publish a partial version. Repeating the same inputs and decisions reproduces the content version. Creation time is physical import time and is excluded from content identity. It is never historical availability evidence.

The manifest and table schemas permit canonical fields only. They retain whole-input hashes, dataset identity, policy, validity and coverage, but no supplier IDs, raw row/sheet pointers or field attribution. Reconstructing an attribution audit requires returning to the unchanged originals. Canonical club names and reusable aliases are identity facts, not row attribution.

## Tables and semantics

Matches retain canonical competition/season/team/fixture IDs, edition/stage, reported date/local civil datetime, precision, timezone knowledge, status, score basis and nullable goals. Missing half scores remain missing. Unknown UTC kickoff and publication timestamps remain null.

The workbook adapter reuses the native statistic keys and units for corners, yellow/red cards, expected goals, possession, total/on-target/off-target shots, fouls and goalkeeper saves. Full time, first half and second half use `ALL`, `1ST` and `2ND`. Possession is 0–100; counts are nonnegative integers. Blank cells stay missing; true zeros remain observations; malformed optional values become missing with `validity=invalid`. Shots off target are never derived by subtraction. Workbook FT/half goal consistency is checked; reported FT is mapped to the regulation-period contract, while historical TXT goals remain `reported_unknown` and native current goals remain `reported_current`.

Unsupported detailed events, lineups, player heatmaps and match incidents are not fabricated. Native statistics without a reviewed compatible period definition are retained for inspection but cannot satisfy a `regulation_period` statistic requirement. Whole observed maps, ratings and membership are not inferred from fixture presence.

All 40 supported workbook odds fields retain opening/closing labels, market, side, line and price validity. No quote publication time or bookmaker is invented. These observations do **not** satisfy executable-odds requirements and do not unlock return/backtesting comparisons.

## Conservative temporal profiles

`retrospective_reported_goals` is an explicitly conditional research view. For unknown timezone/date-only records it uses the full UTC−12 to UTC+14 civil-date interval: the prediction bound is UTC midnight minus 14 hours; the result/statistic bound is midnight plus 60 hours (end of the possible civil day plus an assumed 24-hour publication delay). Known actual kickoff is retained; a later known release takes precedence. A known release before a known kickoff is rejected in either profile.

These are **logical bounds under assumptions**, not invented actual kickoffs or verified publication times. Raw uncertain UTC timestamps stay null. The team-history view records its time kind and complete frozen selection, and treats reported results as observed results only within the declared retrospective profile. Same-day uncertain fixtures cannot inform each other. Future observations are excluded; this does not establish that historical corrections were already known at issuance. Unusually delayed publication may violate the assumed bound.

`strict_as_of` requires aware UTC-compatible actual kickoff and result availability, finished status and regulation goals. Required statistics also need compatible definitions and known availability. Unknown timezone, publication, status or score semantics are exclusions; known contradictions raise. Strict eligibility is an input-availability classification, not a certificate for an entire trained strategy.

Standings, membership-dependent features, ratings, spatial/graph models and executable odds are gated until their additional dependencies have an approved contract. No relegation, membership, official standings, complete schedule or rating state is inferred from the filtered tables. Import completeness never implies real-season completeness.

## Selection and native recipe use

```python
from xdiyo_analytics.data import list_datasets, inspect_coverage, select_data, resolve_selection

selection = select_data(
    private_root,
    snapshots={"comprehensive_dataset": exact_version_a,
               "international_dataset": exact_version_b},
    mode="extend_history",
    requirements=["goals"],
    train_ids=reviewed_train_ids,
    evaluation_ids=reviewed_test_ids,
)
data = resolve_selection(private_root, selection)
```

`single_dataset` takes one exact snapshot. `union` expands the eligible population deliberately. `extend_history` requires explicit fixed target IDs and admits earlier eligible observations without changing them. Training, evaluation and history identities are stored separately. Train/test overlap, rounded/fractional IDs and ineligible fixed targets fail. Whole physical fixtures stay paired.

Eligibility and material-conflict checks precede dataset preference. The default prefers comprehensive only among coherent eligible candidates. An incomplete preferred record cannot hide a usable alternate; a disputed score/status/identity cannot be resolved by preference. The importer chooses one coherent observation and associated statistics/odds, never cell-wise enrichment. V1-only historical records are not admitted; the aggregate revision audit identifies the revision gap without concatenating versions.

`examples/named_dataset_recipe.py::make_recipe` is a working native two-channel rolling goals-for/goals-against recipe factory. Supply exact versions and target IDs, then call `prepare_recipe(recipe)` to prepare without training. JSON and Python exports preserve `data_selection`; preparation propagates it into feature audits and dataset/model definitions. Required statistics must use explicit period and numeric value. Unsupported feature dependencies fail closed. Existing team-plus-competition history grouping is retained.

Frozen train/evaluation selections create the corresponding native fold and validate training-label availability at the earliest evaluation cutoff. Other splitters also receive label-availability checks. The recipe cannot override the catalogue's fixed timing/history with legacy cutoff/history options. Named historical selections are not live future-fixture inputs.

The selection fingerprint covers snapshot versions, record choice, membership, requirement closure, reserve policy, temporal assumptions, scope and exclusions. Resolution recomputes and compares it, and reload verifies every payload hash. A correction requires a new selection and feature artifacts; it cannot mutate an old experiment. Existing models are not automatically rebound.

## Builder and migration

In Data → Named datasets, enter a local catalogue folder, list versions, choose a mode/profile/capabilities and optional fixed targets, and preview coverage and exclusions. Only an accepted preview becomes the recipe selection. The selector keeps its membership JSON opaque in the browser to avoid rounding integer IDs. Native downstream team/statistic choices refresh from that selection. Legacy source controls are hidden while it is active; Use legacy loader restores them without rewriting those settings.

Existing recipes need no migration. To opt in, save a copy with `data_selection={"root": catalogue_root, "selection": frozen_selection}` and remove incompatible legacy history overrides. Inspect exclusions and the held-out population before training. Keep the old recipe/artifacts as the rollback. Adding another file or publication never expands an accepted selection.

## Verification

See `verification.json`, `input-verification.json`, `import-repeat.json` and the test receipts in this directory. `verify_local.py` repeats the aggregate v1/v2 audit and real fixed-target history demonstration; it prepares features but fits no model. Browser QA covers desktop and mobile preview/accept/restore, correct selected-dataset discovery, console errors and horizontal overflow. Browser plugin was unavailable, so the checks used local Playwright Chromium against the native authenticated server.

Initial squad review references: [Premier League competition definition](https://www.premierleague.com/en/premier-league-explained), [Barcelona reserve team](https://www.fcbarcelona.com/en/football/barca-b/news/1858999/the-four-barca-b-captains), [Real Sociedad filial](https://ticketing.realsociedad.eus/), [Castilla](https://www.realmadrid.com/en-US/news/football/academy/reports/castilla-bilbao-athletic-13-02-2026), and [Bilbao Athletic](https://www.athletic-club.eus/en/news/2024/08/08/bilbao-athletic-players-registered-in-laliga/). Classification decisions are independent of a name containing B, II, Jong or Youth.


## Local rollout results

The reviewed initial publication contains 10,854 comprehensive fixtures and 9,676 international fixtures. The goals-profile union admits 14,235 distinct canonical fixtures; this is the reconciled **initial subset**, not the size of the full unreconciled archives. The comprehensive publication excludes 464 confirmed-reserve fixtures and 81,665 fixtures requiring squad review. Publication and task eligibility are separate: an awarded result can remain inspectable while being excluded from the goals research selection.

The native fixed-target demonstration keeps four training and four evaluation fixtures. There are 21 baseline history fixtures available by the final target bound, versus 3,407 under the extended view (3,386 additions). Thirty rolling-feature cells change; target IDs and fold indices are identical. The uncut baseline history population is larger because it also contains later fixtures, which the causal feature evaluator cannot use for these targets.

Private catalogue location on the implementation machine: `C:/Users/luisi/Documents/Programming/Python/xDiyo/scratch/named-datasets-20261102`. It is covered by the existing `/scratch/` Git exclusion. Enter that folder in the builder running this main revision; other checked-out research branches are not changed automatically.
