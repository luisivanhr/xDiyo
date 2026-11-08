# Workbook-only actual-playing-date GAT study

This is a new controlled comparison, not an exact reproduction of earlier native-source results. It supersedes the 7 November preparation's native/workbook mixture and by-round timing. The original-13 mapping was committed and pushed before this work. Do not merge main as part of this study.

## Fixed experiment

Three datasets: original 13 leagues; those plus all 38 listed first divisions; those plus all nine listed second divisions. Lists overlap the original 13, so their unions contain 13, 43 and 47 leagues. Every qualifying workbook fixture and team is retained, including unresolved round annotations and supported missing statistics. No native fixture outcomes or statistics enter any stage. Workbook match IDs remain the source identity; derived event IDs are 3,000,000,000 plus workbook match ID. Team IDs use the reviewed dated identity directory. Original-13 competition IDs remain canonical; other competition IDs are 1,000,000,000 plus workbook league ID.

Families B/N, C/E1 and B/CD_DW use the retained architecture, goals, total shots, shots off target and corners for both teams. CD_DW uses window 10 throughout. Seeds 17, 42, 73; 27 development selections and 108 fresh fold refits. No fitted warm starts. Expanded league vocabularies mean equal seeds do not imply bit-identical initial tensors across datasets. Preserve initial state digests and semantic coordinate order.

History begins 1 July 2019 for context; fitting begins 1 July 2020. Development splits on 11 February 2022 and ends 1 July 2022. Four expanding test intervals start on 1 July 2022, 2023, 2024 and 2025 and each last one calendar year. Use identical boundaries for all stages. Development selects raw equal-fixture H/D/A CE, at most 200 updates, patience 15, min_delta 0, restoring the exact best update. Each stage/family/seed selects once. Fresh outer refits use that fixed update count without validation or test selection. Score outer tests only after every fit is complete.

## Time and availability

`actual_playing_date_v1` issues forecasts at Europe/Berlin midnight starting the fixture's actual workbook playing date. All graph inputs, observed gaps, memberships, age weighting and phase context use that cutoff. Original round/stage labels are separately retained and cannot anchor issuance. Results and statistics use the actual kickoff as the user-accepted retrospective availability proxy, respecting any later explicit release bounds; histories and fitting/validation boundaries are strictly before the forecast cutoff. Same-day matches are excluded regardless of kickoff order.

The workbook does not declare a timezone. Europe/Berlin interpretation of its civil clock is the explicit study convention, not verified UTC precision. At repeated autumn hours use the later feasible UTC instant; retain both candidates in `ambiguous_civil_clocks.parquet`. This affects 29 selected fixtures, zero original-13 fixtures, and changes neither date nor cutoff. Nonexistent spring-clock values fail. Source disagreements with native actual dates remain unadjudicated: workbook-only here consistently uses the workbook date, not a silent native correction.

## Features and limitations

Old by-round/one-day-ahead models and V1 research profiles retain their behavior. New factories are separately versioned `GATActualDate*V1`. Train-only fitting transformations, per-coordinate missing masks and strict statistic releases apply. No odds, outcomes of target fixtures, audit decisions, evidence, source IDs or review metadata are model features. Past goals remain legitimate historical edge inputs; target outcomes are labels only.

No historical schedule-publication log exists in the workbook. **Countdown is unavailable and masked in every row**; it is not reconstructed from the final season-end date. Consequently B/CD_DW has its declared countdown coordinate and missing mask, but its active contextual information is window-10 form rank. Do not report this as a fully observed countdown experiment.

The new rank policy is explicitly distinct from the prior complete-roster native standings policy. For each regular-stage competition-season and actual playing-date cutoff, form ranks use only participants already observed in strictly earlier available regular-stage results. Points 3/1/0, goal difference, goals for, goals against and competition ties follow existing arithmetic; each team's last 1/5/10/20 own matches determine the corresponding rank. Normalize as `(N-rank)/(N-1)` using the observed participant count. New teams, insufficient populations, unresolved stages and reset/knockout stages remain missing. No final roster, future result or invented carry/reset rule is used. These are observed-member form ranks, not official table positions. This change and the missing countdown limit comparison to older research baselines.

Membership becomes effective at the first observed fixture's playing-date cutoff, when the current fixture's participating teams must be known. Promotion/relegation category is unknown. Original-13 phase values and all raw fields are identical across dataset stages; only graph history and learned league dimensions expand. Round provenance remains outside the model table's selected features, including all four explicit Australian provider fallbacks and the 367 unresolved original-13 fixtures in the context window.

## Execution and evaluation

Local RTX 4070 SUPER, Python 3.14 / PyTorch 2.9.0+cu130; deterministic cuBLAS `:4096:8`, one CPU thread, highest matmul precision, optional speedups off. No RAM guard. The existing 4 GiB input cache budget is a cache policy, not a process memory guard. Resume only the same job/control/source seal. Smoke tests perform one update on 32 fitting and 32 validation fixtures for each of nine stage/family combinations; complete population checks independently cover every fold before sealing.

Report raw natural-log CE, normalized RPS (half the sum of two cumulative H/D/A squared errors), and C/E1 home/away goal RMSE on all test fixtures. Keep per-seed/fold/league results and distinguish pooled fixture-weighted from equal-fold means. Score seed losses independently; do not ensemble probabilities. Compare dataset-size effects on exactly paired common original-13 fixtures. Paired calendar-block intervals average seed losses first; four-week primary, two/eight-week sensitivity; exploratory comparisons conditional on fitted seeds.

Financial simulations use original-13 fixtures only, workbook-labelled opening draw odds, probability of draw at least .20, strict positive expected value, flat one-unit stakes. Missing/invalid prices are explicit exclusions. Quote capture times and bookmaker are unverified: these are historical simulations, not executable-price claims. Opening odds are kept in a separate analysis-only file and sealed. Continuous cumulative profit, exposure, ROI, win rate, maximum drawdown and daily unit-PnL Sharpe-like metric follow existing conventions. Include paired original-13 comparisons across stages.

## Completion

Freeze source, protocol, data and preflight hashes before launch. Preserve all checkpoints, selection decisions, loss histories, prediction exports and analysis receipts. After compute, the runner calculates final analysis automatically. Native tickets, embeddings/attention, report rendering and browser validation remain final reporting work; use the current Ayre reporting standard and accepted FourGAT presentation. Upload final verified results to Ayre Drive with existing sharing unchanged. No interim metric claims, no fabricated diagnostics, no destructive cleanup of unrelated worktrees.
