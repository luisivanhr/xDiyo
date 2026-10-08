# Historical repair re-audit — 8207e29

2026-10-08 · Read-only verification of the five fixture repairs and completion-aware availability

## Verdict

**The five-fixture repair hold can be cleared for the specifically audited native pooled-Bayesian workflow**, using the explicit kickoff +3h policy clamped to reviewed completion bounds. Its eligible early-history population is **23,655 matches**, not the old five-case quarantine's 23,653. No research calibration, feature-bank rebuild or betting run was started.

This is not blanket approval of every library API or an all-13-league earlier betting rerun. Three bounded API issues remain below; none reproduced on the canonical repaired data, explicit+3h Bayesian calibration or fixed raw, uncalibrated XGB route. Exact original 40/44/46 feature eligibility and native opening-odds joins remain a separate preparation gate.

## Commit, source and data verification

- Main pinned to **8207e29c49bfa20e09c75be16105d8847b0eef3a**, a direct child of **4e4b951e9bb512ed16fc6c5edb34f135f68cbbad**
- 42 changed paths, including five native match tables, their publication/native manifests and nested season files, the evidence register and availability propagation code
- All 186 inspected non-cache source/runtime files and every changed text file match the pinned Git tree
- All 65 native match binaries pass SHA-256, Git blob identity, bytes, row-count and ZSTD checks; the 60 unaffected match binaries are byte-identical to the prior audit
- Across all 23,658 raw matches, every pre-existing column except the intended effective award flag is unchanged. That includes goals, halves, status/status codes, kickoff, round, fixture identity and movement flags. The prior 310 half-score exceptions and 65 unusual numeric statuses are preserved; no final scores were reconstructed from faulty halves
- Charleroi–Cercle1378121 remains unchanged, correctly retaining the played 23 March 2011 fixture
- The large nested season binaries were unavailable through the connector (empty base64 response), so their byte-level publication roundtrip was **not independently reverified**. Their new pins and publisher evidence are retained; this does not block the verified native-table calibration path

## Five fixture outcomes

| Event | Verified reviewed representation | Default treatment |
|---|---|---|
|2893007 Cagliari–Roma | Provider 0–3 preserved; played score null; administrative 0–3; `not_played`; effective award true, original provider flag false | Excluded |
|1399769 St Pauli–Schalke | Provider/played-at-abandonment 0–2; administrative 0–2; `abandoned`; effective award true | Excluded |
|3944327 Nantes–Bastia | Played/provider 2–0 preserved; later-forfeit classification; administrative score left unknown; effective award true | Excluded |
|1861771 Padova–Torino | Played 1–0; `resumed`; award false; original 3 December kickoff preserved; release bound **14 December 2011 23:00 UTC** | Included only after bound |
|1795067 Granada–Mallorca | Played 2–2; `resumed`; award false; original 20 November kickoff preserved; release bound **7 December 2011 23:00 UTC** | Included only after bound |

Both bounds mean the midnight following the documented completion day in Rome/Madrid. They are conservative historical release bounds, not fabricated exact whistles. The effective policy is **max(configured release, reviewed bound)**. Explicit later releases stay later; explicitly unknown releases stay unknown. Unreviewed rows retain the existing proxy assumption; sparse null review metadata does not certify an ordinary completed match.

Default loader exclusion removes exactly 2893007, 1399769 and 3944327. Both resumed fixtures survive load→history→features→labels→assembled match metadata, without their completion fields becoming predictors. Actual-fixture cutoff tests reject each resumed result one nanosecond before release, admit it exactly at/after release under native inclusive semantics, and reject the awards even when loaded explicitly for inspection.

## Temporal and regression evidence

- **50 targeted repository tests passed**; one real nested-publication test was first blocked by missing binary input and then explicitly deselected. This is not a full-suite or publisher-total claim
- **800 independent history eligibility comparisons** and **144 temporal/CPCV folds** passed across layouts, shuffled/custom indexes, missing availability and earlier cutoffs
- Delayed-result mutation did not change intervening lag/rolling/EMA/weighted/league features; later features respond after release
- Guarded SVC margin splitting and small end-to-end synthetic SVC tests preserve preprocessing, fit and calibration isolation. These tiny regression fits are not a research model study
- Bayesian checks passed 130/130 knowledge-set queries, 14/14 calibration-objective checks and 14/14 incremental-release checks; both actual-fixture windows passed. Overall 396/400 checks passed, with four failures confined to the Glicko edge case below. All 60 explicit+3h tie checks passed. Details are recorded in `bayes_completion_order.md` and its structured results

Bayesian replay retains immutable knowledge-time versions. After a late completion it may rebuild the **current** posterior in kickoff order using only results released by that boundary; it does not mutate earlier forecasts. This distinction matters: it is not a single kickoff cursor that blocks all later completed games, nor does it expose the late score before release.

## Bounded remaining API findings

1. **Glicko default kickoff proxy, exact release tie:** a batch containing an earlier result and a fixture kicking off exactly at the same release/cutoff is stamped with that current kickoff. Snapshot lookup discards the whole batch, omitting the valid earlier result. Independent reproduction shows a large rating difference. Uniform explicit+3h removes the triggering kickoff=release condition; the tested explicit+3h release ties and Bayesian path pass
2. **Legacy predict_proba calibration:** its inner ValidationTail split does not honor completion-release bounds, although the outer temporal split does. The guarded `decision_function`/SVC path does honor them. The requested three raw, uncalibrated XGB models do not use this legacy probability-calibration route
3. **Manually asymmetric paired metadata:** match-label creation can discard a release bound supplied only on the away row. Canonical `build_team_history` copies the identical reviewed match bound to both sides and passes; direct paired split helpers conservatively honor either side

These are explicit scope limits and reproducible maintenance findings, not a reason to relabel the five repaired fixtures again. Scripts and expected/actual results accompany the report. Custom adapters or hand-constructed histories/folds are not certified by this audit.

## Frozen population and next-step contract

- Keep original 13 leagues, 2010/11–2014/15 selected regular-stage history; no new leagues, playoffs or invented matches
- 65 partitions  / 23,658 raw  / 23,655 default non-awarded matches; no additional numeric-status or half-sum exclusion in the primary population. The prior 23,653 five-case quarantine remains a separately labeled sensitivity if desired
- Coefficient cutoff remains **2015-07-24 17:30 UTC**. Strictly prefilter whole matches with kickoff<cutoff and effective availability<cutoff. Native admission remains inclusive at the bound; the stricter study gate must be explicit
- Latest effective release is **2015-06-07 21:00 UTC**, safely before the earliest 2015/16 decision. Freeze fitted parameters before producing any 2015/16-or-later features
- Use original round-first-kickoff minus 1h decisions, explicit release policy and the full stage/round identity; retain original decisions for postponed fixtures
- The original pre-fit configuration was recovered from the hash-verified original launcher and pinned source: pooled outcome_log_loss, nine original fit fields, regularization 0.01, days clock/7, mirrored transitions; L-BFGS-B maxiter 100 and ftol 1e-8. Exact initial parameters, bounds and provenance are frozen in `FROZEN_CALIBRATION_SPEC.json`. Do not use coefficients fitted on 2015–2019. Later-fitted coefficients would leak future information as optimizer initial values and as the regularization center
- Copy old recipes before native `refresh_recipe_data`; preserve original selections, cached features, predictions and results. The helper is unchanged and rejects existing output paths. Use new code/data/recipe/cache identities and `verify_hashes=True`; never relabel old results with new hashes
- Parent coordinates one controlled calibration run after the specification is fixed. No model grid, new home-advantage feature or changed XGB hyperparameters

The earlier source-coverage findings remain unchanged: Scotland's early history is pre-split only; Championship 2014/15 and Belgium 2014/15 omit declared abandoned fixtures. Earlier exact common-universe feature preparation and quote matching are unfinished. A possible first test is 2016/17 after one training season; the primary proposal remains train 2015/16–2016/17, test 2017/18, with causally justified staggered spatial entry. Five leagues lack early opening draw prices; price presence elsewhere is not proof of contemporaneous executability.

Detailed evidence: updated 65-season CSV, native table hashes, five repaired rows, actual pipeline checks, independent temporal/replay oracles, scoped API reproductions and calibration specification accompany this report.
