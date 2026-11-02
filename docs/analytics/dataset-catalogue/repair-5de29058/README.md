# Fixture-dated native team identities

Reviewed base: `5de290583e975de044e362655a9f1272cb2a1593`.

The new regression reproduced the audit on the base: a September fixture using a reviewed alias assigned to canonical team 22 from August onward was incorrectly imported as team 1 because the adapter resolved the alias on January 1.

## Repair

Native team binding now occurs for each fixture side using its actual UTC kickoff date. A cache includes native ID, spelling and fixture date, so the same incoming ID can resolve to different canonical clubs within one season. Match participants, displayed canonical names and fixture identities all use that per-fixture resolution. Statistic identities are checked against the original fixture side before remapping through the identical binding. A contradictory statistic team/side fails with fixture identity and review reason.

Existing reviewed team aliases are used without adding or widening their intervals. A newly observed native spelling is recorded for its fixture date only. Subsequent source imports cannot turn time-limited review evidence into an unbounded alias. Missing, invalid or nonfinite kickoff timestamps fail explicitly instead of skipping the fixture or substituting a season date. Overlapping active alias decisions retain the registry's fail-closed ambiguity check.

Competition/season binding, sporting results, selection policy, feature mathematics and recipe migration remain unchanged. No published registry is rewritten, and no catalogue is automatically reimported. Fresh imports with previously unreviewed spellings outside recorded dates may need additional reviewed alias coverage; this repair does not infer that coverage. This synthetic audit does not establish that the private union was affected.

## Verification

`test_native_dated_aliases.py` covers the exact September counterexample, both sides and fixture orders across July 31/August 1, one source ID reused within one season, corresponding statistic values/identities, saved-publication reload, repeated-import idempotence, preserved alias intervals, bounded newly recorded aliases, missing/invalid dates, ambiguous aliases and contradictory statistic identities.

The broader suites repeat the catalogue, migration, loading, native history, feature composition/assembly, venue and UI recipe checks in both local Python environments. The installed-wheel check runs under Python 3.14 only; it is deselected under Python 3.12 to avoid concurrent package-build mutation. See `verification.json` and the log/XML receipts for exact counts, commands and source hashes. No browser code changed, so the prior browser receipts are not presented as a new browser run.

No production experiment was fitted or rebound. Private inputs/catalogues and the active collector were not modified. Broader tests include small synthetic estimator fits.
