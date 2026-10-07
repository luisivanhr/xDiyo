# Research quote extension verification

Date: 7 October 2026.
Parent: `f76edff8b2e42c7966a8e55c36e6076c8af32011`.
Branch: `codex/research-quote-availability`.

## Scope

Native quote contract and propagation only, plus a synthetic two-stage consensus
example. The existing `codex/model-composition-stake-policy` branch remains at
the reviewed parent. Main remains at `29994933902d213bccd13376bc91ab5af70a481b`.
No datasets, frozen experiments, cached models or permission settings changed.
No real-data study, model refit, scorecard, push or merge was performed.

## Acceptance evidence

- Strict observed defaults reject missing/future times and assumed-mode input.
  Explicit observed contracts also validate matching source pins and identities.
- Research mode requires a separate timezone-aware timestamp and a complete
  declaration. Contradictory observed timestamps cannot be overridden.
- Both model streams reference the same original price/market/selection/source
  pins and availability evidence for each individual fixture. Different fixtures
  retain different quotes and prices. Preflight includes undersized pools.
- Boundary matrix covers non-draw .80 and both representable neighbours for
  each model, missing streams, strict zero EV rejection and the declared
  complement `1 - original_P_non_draw`. Original draw probabilities remain intact.
- Model/fixture/class-column permutations, conflicting duplicates, nested
  membership, concealed late evidence, outcome mutation and nullable metadata
  checks pass. Missing optional values agree only with matching missing values.
- Browser editing, JSON catalogue round trips, Python/notebook exports, full
  Parquet table/audit restoration and downstream saved performance reuse retain
  the research disclosure. Reuse is tested with recomposition forbidden.
- Five synthetic legacy controls were run using both the parent source from
  `git archive f76edff src` and the extension. Parlay, MultiBet, AllCombinations,
  minimum-EV and observed-gate outputs match exactly via `assert_frame_equal`
  (`check_exact=True`), including dtypes, ordering, complete metric tables and
  DataFrame attrs. Parent evidence is temporary local verification data only.

## Test results

Final full analytics suite: **4582 passed, 5 skipped, 55 warnings in 315.82s**.
The skips are the optional sklearn Array API checks (`SCIPY_ARRAY_API` unset).
The 55 warnings are existing dependency deprecations, convergence and numerical
test warnings. Installed-wheel, saved-model restoration and Chromium UI checks
passed. No tests failed.

Focused quote and ticket EV tests: **173 passed in 6.75s**. Node tests:
**23 passed**. The synthetic example runs without loading data or fitting models.

Environment: Python 3.14.0, sklearn 1.9.0, pandas 2.3.3, NumPy 2.4.6, joblib 1.5.2.
No runtime/dependency updates. Full-suite command:

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest tests/analytics -q -p no:cacheprovider --tb=short
```

Log: `%TEMP%/xdiyo-quote-full.log`. The quote-focused test file has 72 cases;
the additional Chromium test edits and exports the new native UI component.

## Independent review still required

These are implementation checks, not independent review or evidence that the
opening quotes were historically obtainable. The cached-model study must remain
stopped until independent review approves this contract and the study explicitly
enables its documented assumption. No model/quote provenance is authenticated
by this API; callers remain responsible for truthful retained evidence.
