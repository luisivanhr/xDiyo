# Football analytics working notes

## Purpose and sources

This is the new working reference for the continuing football analytics design conversation. It consolidates the original analytical ideas, the handwritten refinements, and the recent discussion so that ordinary planning can proceed without rereading the PDF. The analytical content remains relevant; the architecture is open to reconsideration.

**Confirmed end-product goal:** build a **reusable football analytics library** that eventually makes experiments easy to set up and run. The total-match-corners experiment is its first pilot for exercising and refining reusable components. The library's scope includes multiple targets, model families, and training libraries. Ease of use and reproducibility are goals. (D)

The [First pipeline idea](C:/Users/luisi/Documents/Programming/Python/xDiyo/first_pipeline_idea.md) records the Lasso corner-count experiment as the first test of the reusable library. The user now wants the actual system scaffold and its initial feature library built while real-data preparation continues. Feature availability is separate from experiment selection: prepare lags, means, deviations and composition so different subsets can be compared. This reference retains the broader design context. (D)

This document preserves the user's requested separation between the football system's general idea and the scraping-refactor plan appended to the older `planning.md`. The standalone general-idea Markdown was deliberately created to make that separation. Creating these notes does not update the originals, select a final architecture, or authorize implementation, scraping changes, or data collection.

| Source | Reference | How it is used |
| --- | --- | --- |
| **G: General idea** | [Football analytics system general idea.md](<C:/Users/luisi/Documents/Programming/Python/xDiyo/Football analytics system general idea.md>) | Primary written account of artifacts, templates, operators, dependencies, targets, and expression composition. Its workflow section is unfinished. |
| **H: Handwritten planning** | [Football analytics system planning.pdf](<C:/Users/luisi/Dropbox/Sent files/Football analytics system planning.pdf>) | Fourteen handwritten pages, visually read using existing rendered pages. References such as “H, p. 6” refer to PDF page order. |
| **P: Pipelines scaffold** | [README](C:/Users/luisi/Documents/Programming/Python/Architecture/pipelines/README.md), [architecture](C:/Users/luisi/Documents/Programming/Python/Architecture/pipelines/docs/architecture.md), [feature construction](C:/Users/luisi/Documents/Programming/Python/Architecture/pipelines/docs/feature_construction.md) | A separate source of architectural inspiration and documented implementation boundaries. |
| **PF: Scaffold feature library** | [Feature library README](C:/Users/luisi/Documents/Programming/Python/Architecture/pipelines/features/README.md), [composition](C:/Users/luisi/Documents/Programming/Python/Architecture/pipelines/features/composition.py), [time series](C:/Users/luisi/Documents/Programming/Python/Architecture/pipelines/features/time_series.py), [extraction](C:/Users/luisi/Documents/Programming/Python/Architecture/pipelines/features/extraction.py), [base contracts](C:/Users/luisi/Documents/Programming/Python/Architecture/pipelines/features/base.py) | Source inspection of existing composition and temporal-feature boundaries; no execution or extension was performed. |
| **LF: Learned-feature extension sources** | [FeatureBuilder contract](C:/Users/luisi/Documents/Programming/Python/Architecture/pipelines/contracts.py:118), [fold runner](C:/Users/luisi/Documents/Programming/Python/Architecture/pipelines/runner.py:161), [feature-construction guidance](C:/Users/luisi/Documents/Programming/Python/Architecture/pipelines/docs/feature_construction.md:89) | Read-only checks of the fitting/application boundary and supervised-feature extension point, alongside PF; no learned-feature workflow was implemented or run. |
| **L: Loader planning sources** | [Canonical export reader](C:/Users/luisi/Documents/Programming/Python/xDiyo/scraping/sofascore/export.py:107), [compatibility loading helper](C:/Users/luisi/Documents/Programming/Python/xDiyo/utils/data_loading.py:12), [prepared-export storage](C:/Users/luisi/Documents/Programming/Python/xDiyo/scraping/sofascore/README.md:47) | Narrow source checks for the proposed analytics loader; no sample dataset was loaded or new loader implemented. |
| **D: Current discussion** | **Pipeline Brainstorming**, local task `01a09156-205d-7310-8376-52359d8df65c` | User-confirmed goals and assistant proposals conveyed in the originating task's handoff and subsequent clarifications. |
| **R: Existing reshaping helper** | [utils/reshaping.py](C:/Users/luisi/Documents/Programming/Python/xDiyo/utils/reshaping.py) | Source inspection of `to_team_match_long` only; no execution or validation of its behavior. |
| **C: Existing CV prototype** | [train/cross_validation](C:/Users/luisi/Documents/Programming/Python/xDiyo/train/cross_validation) | The full extensionless source file defining `SeasonRoundTimeSeriesSplit` was inspected, not executed or tested. Its behavior differs from the requested round-based CV. |
| **GR: Existing rating core** | [utils/glicko_rating.py](C:/Users/luisi/Documents/Programming/Python/xDiyo/utils/glicko_rating.py) | The complete source was inspected; no execution or validation of the rating algorithm was performed. |
| **U: Utility inventory** | [Saved-project utils directory](C:/Users/luisi/Documents/Programming/Python/xDiyo/utils) | Earlier source inventory, with a targeted current refresh of the season-entry helpers and their movement-builder dependency. Other entries retain the earlier review scope. |

The originals remain the sources of truth for their content. Current user clarifications establish the present direction. Historical instructions and examples are recorded as design material, and discrepancies remain visible below.

Status terms used throughout:

- **Confirmed direction:** explicitly stated or confirmed by the user in D.
- **Original design:** an idea or convention recorded in G or H; its presence does not make it a current implementation contract.
- **Working proposal:** an assistant recommendation from D, still open to revision.
- **Documented capability:** a claim in P, or behavior visible in the inspected sources PF, LF, L, R, C, GR, and U; no new runtime verification is implied.
- **Open:** a decision or interpretation that the discussion has not settled.

Next, discuss league-wide rolling histories, focal-team leave-one-out (LOO), and optional season/movement warm starts before the remaining orchestration layers. The section below records the latest priority and pending choices. The verified population, features and ratings remain available components; pilot recipe selection, model tuning and fitting are deferred. The goals, artifacts, timing/CV decisions, and later template reference remain the broader context.

## Latest design priority — league histories and optional season transitions

**Confirmed discussion order:** review league population/window/LOO semantics, then optional season-transition policies, before returning to the shared dataset contract and reusable orchestration. The rules confirmed below refine the design; these extensions are **not implemented**. The existing **no-warm-up, available-history default remains unchanged** (`warmup=None`). This update authorizes discussion only, with no code, tests, notebooks or pilot configuration requested. (D)

- **League population and timing — confirmed round rule:** retain the last completed-round baseline while the next round is incomplete; freeze one shared reference snapshot for all predictions in the target round. Kickoff mode may evolve as eligible results arrive. Observation unit (one team's contribution versus one match total), window length and the definition of completion for postponed/staged rounds remain explicit choices. Current eligibility helpers require `team_id`; league histories need explicit support, not just different grouping columns.
- **LOO — confirmed order, preferred wrapper proposal:** **select the eligible history/window → exclude focal-team contributions → aggregate, with no refill from older observations**. The user favors `League` and `LeaveOneOut(..., exclude="team_contributions")`; whole-fixture exclusion remains a distinct optional policy. Reuse league/team sufficient summaries where valid, recomputing retained moments and count after exclusion; never subtract or average standard deviations. A frozen round population still yields team-specific LOO baselines. A league-relative team Z-score needs a known historical team numerator in matching units. LOO excludes feature-population observations; it differs from LOCO feature-importance refitting.
- **Pooled moments — confirmed:** a three-round std pools individual eligible observations around their pooled mean, not the three round means. Unequal round sizes give an observation-weighted mean. Recompute moments and denominator after LOO, with `ddof` explicit; neither equal weighting of round means nor an unchanged pre-exclusion denominator represents this choice.
- **Season transitions — proposed shared helpers:** detect season entry/movement, obtain eligible pre-transition priors and apply each feature/rating policy exactly once. Feature construction owns this policy, independently of CV. Reuse the existing ID-based movement core with new-export inputs; a newly observed team is not automatically promoted. Continuing-team uncertainty inflation could preserve rating `mu`; movement could blend `mu` toward a destination prior and optionally inflate `phi`, leaving `sigma` unchanged unless explicitly configured. These adjustments and their parameters remain optional and pending.
- **Priors and state transfer — refined design:** destination top-m/bottom-n cohorts support configurable ranking by standings (preferred default) or rating. Freeze their snapshots before the transition and exclude the moving team; cohort sizes and mean/median remain configurable choices. Current ratings are competition-scoped, so retrieving previous-league state and calibrating/transferring league rating scales remain explicit operations; membership evidence alone does not settle them.
- **Mean warm starts and handoff — optional design:** seed a mean EMA from previous-season and/or destination-league means, with explicit blend, decay and time unit. Handoff is selectable after **1, 2 or 3 completed rounds**, then uses the ordinary configured rolling window with available samples, no padding or automatic full-window requirement. **One round when enabled is a tentative recommendation, not a confirmed default:** it mainly supplies the opening-round prior, whereas at least two rounds let later predictions use the updated EMA. Continuing an EMA indefinitely is a separate policy. Global `warmup=None` remains the default; James-Stein and more elaborate borrowing remain deferred.
- **Variance warm starts — confirmed prior source, details pending:** when enabled, use the previous season's league-wide variability, or the destination league's preceding-season variability for a mover, matching statistic, period and observation unit. Seed a centered second moment/variance with appropriate prior weight; update mean and spread consistently with the same exponential weights. Do not combine a warmed mean with an unrelated rolling std or silently turn missing prior variance into zero. Weighting/sample correction, missing-prior handling and possible handoff discontinuity still need explicit policies.

The confirmed window/LOO order, completed-round handling and pooled-moment definition supersede their earlier pending-choice status. Wrapper APIs, remaining observation/transition settings and warm-up recommendations still need review. The earlier orchestration-first instruction is superseded only as the immediate next step: reusable orchestration still precedes configuring the corner experiment. Historical proposals below remain reference material. (D)

## Current loading and experiment population — 16 September 2026

The user selected **all available leagues for 2022/23, 2023/24 and 2024/25** from
`data/xDiyo_data`. The first actual import used `load_seasons` with
`seasons=['22_23', '23_24', '24_25']`, `leagues=None` and separate `matches`,
`statistics` and `pregame` tables. It loaded **39 publications across 13 leagues**:
**13,976 matches**, **3,303,574 statistics rows** and **27,072 pregame rows**.
All 13 leagues have a publication in each requested season; there are no missing
league-season combinations in this imported snapshot. These are collected
cohorts, not a claim that all provider fixtures were independently discovered.

`load_seasons` adds `source_league` and `source_season` without changing source
observations. Access tables through `.matches`, `.statistics`, `.pregame`,
`.tables` or brackets. Shots remain independently selectable; no joins, features
or model fitting were performed in this import. `load_season` remains available
for one named publication and `inspect_season` lists tables, descriptions and
columns. All 117 imported table partitions matched their source Parquet values,
types, row order and nulls. Those 55 loading cases remain in the current 109-test
suite, alongside 28 statistic-selection and 26 team-history cases.

The **39 saved selection records** are in
`experiment/initial_population/selections`. They pin publication versions, not
discovery membership; later matching publications can expand a new call. The
[import evidence](docs/analytics/multiseason_import_check.json) records the exact
population, versions, per-table rows and fingerprints. The
[twelve-cell notebook](notebooks/01_loader_walkthrough.ipynb) shows practical usage
and match counts by league/season; [the guide](docs/analytics/season_loading.md)
explains selections and table access.

### General statistic selection — implemented and verified

`select_stats` now combines attack/defense/all bundles with exact
`(period, group_name, key)` statistics and optional standings. Every category has
one period-independent list: `*_all` retains every supplied period, and totals
or halves filter that shared selection using provider `ALL`, `1ST` or `2ND`.
Totals do not sum halves. `all_stats` includes uncategorized and new measures.
The initial attack/defense lists are editable through `STAT_CATEGORIES` or
replaceable/extendable per call; `list_stat_bundles` lists the available names.

Defense selects observed defensive metrics; it does not construct opponent
statistics. Standings are the original match-specific pregame positions in a
separate frame. Matches pass through when loaded. Group identities, values,
nulls, order and types remain intact; missing observations are not fabricated.
This selection step keeps tables separate. The history builder below reshapes
the selected observations; feature construction and fitting remain future work.

The representative Premier League 2024/25 union of `attack_totals`,
`defense_first_half`, `standings` and exact `ALL / Match overview / ballPossession`
contained **22,762 statistic rows**, **740 standings rows** and **380 matches**.
All 12 category/period combinations matched independently selected source rows.
The same notebook example across the unchanged experiment population selected
**746,880 statistic rows**, retaining all **13,976 matches** and **27,072 pregame
positions**. The [selection guide](docs/analytics/stat_selection.md) records the
default definitions and customization semantics; [verification evidence](docs/analytics/stat_selection_check.json)
records the stable source state, tests and fresh-kernel notebook execution.

### Observed team histories — implemented and verified

`from xdiyo_analytics.histories import build_team_history` supplies the next
step: `history = build_team_history(selected)`. The frame contains two rows per
match, reversing team/opponent IDs, names, current goals, statistics and pregame
positions. Source league/season plus event ID identify each match. Statistic
columns retain period/group/key/field identity with escaped components, and
`history.attrs['stat_columns']` records their original labels. Missing
observations remain missing and do not remove match rows.

`kickoff_at` is timezone-aware UTC converted from actual `kickoff_utc` Unix
seconds, retaining time of day. Missing/unrepresentable times become NaT and
sort last. W/D/L compares current provider scores only for finished matches with
both scores present. Penalty scores do not break a current-score tie. Unknown
or unfinished statuses retain their rows with missing result.

The real selected history contains **27,952 rows × 83 columns** over the unchanged
39-publication population. All context/result rows were independently checked;
the representative Premier League 2024/25 history also matched **47,120 statistic
cells** and **1,520 position cells** against direct source Parquet. The final
**109-test suite** and fresh 12-cell notebook passed after the primary fixed an
out-of-range timestamp overflow. See the [history guide](docs/analytics/team_history.md)
and [verification evidence](docs/analytics/team_history_check.json).

These rows contain observed outcomes. The separate feature evaluator below
supplies historical eligibility and cutoffs. Scheduled kickoff alone remains
insufficient to establish exact completion/publication time. Model fitting remains
deferred; the verified rating integration is described below.

### Feature expressions — implemented and verified

`xdiyo_analytics.features` provides `Stat`, `ForAgainst`, `H2H`, `IsHome`,
`NormalizedStanding`, `Lag`, `RollingMean`, `RollingStd`, `RollingZScore` and
optional `EMA`, evaluated through a named mapping with `evaluate_features`.
`Stat(None, ...)` expands supplied periods separately. For/against selection
uses existing observed columns; H2H follows ordered team/opponent identity
through venue reversals. Direct observed-statistic roots are rejected.

**Current choices supersede earlier open proposals below:** default history
groups by team and competition, combining venues and crossing seasons. Windows
count eligible matches including missing-valued matches; reductions use finite
values inside that window, while lags retain missing positions. `min_periods=1`;
std/Z-score default to `ddof=1` and require `n > ddof`. Z-score includes its latest
eligible observation in the baseline, and remains missing for zero spread or a
missing latest value. Nested children are evaluated at their own historical
cutoffs. EMA initializes at the first observed value with alpha `2/(span+1)` and
skips missing values without decay; it is optional, with no selected pilot prior,
warm-up or seasonal blending.

Standings use `1 - (position - 1)/(team_count - 1)`, with distinct team counts
computed from the full league-season population **before filtering or folds**.
Missing/invalid positions or denominators default to **zero**. This is specific
to standings; undefined lag/rolling results stay missing. Current home context
is allowed.

`cutoffs=None` uses target kickoff; result-availability times are fully optional.
Past candidates must be finished and strictly earlier than cutoff, with explicit
availability no later than cutoff when supplied. Target identity and boundary
ties are excluded. The default is an earlier-finished-kickoff **retrospective
proxy**, not evidence of exact completion/publication time. Earlier cutoffs do
not reconstruct historical versions of supplied match context such as standings.
See [exact cutoff formats](docs/analytics/feature_cutoffs.md).

**Verification:** 141 tests passed, including 32 new feature cases. Independent
arithmetic matched **760 Premier League 2024/25 rows × 22 feature columns** at
default and two-day-earlier cutoffs, with the named-column form matching the
aligned Series. Full-population team count was 20; direct Parquet lookup matched
4,560 corner cells. These real cutoffs happen to yield identical values;
synthetic cases exercise actual eligibility changes. All 12 package sources,
195 source fingerprints and 39 saved records remained stable. The new separate
[8-cell feature notebook](notebooks/02_feature_quickstart.ipynb) executed in a
fresh kernel; the 12-cell loading notebook remains unchanged. See the
[feature guide](docs/analytics/features.md) and
[evidence](docs/analytics/features_check.json).

### Latest ratings implementation — 16 September 2026

Result and statistic-comparison rating streams are now implemented and verified.
`ratings.Glicko2` is a thin adapter around unchanged `utils.glicko_rating`, the
single retained numerical engine. It inherits tau=1.0, with initial public
rating/RD/volatility 1500/350/.06 and epsilon1e-6. Public `rating`/`rd` and internal
`mu`/`phi` have distinct snapshot names; the legacy object's mu/phi are public-scale
names. The primary's [comparison](docs/analytics/glicko_engine_comparison.md)
records exact legacy parity after consolidation.

`MatchResultGlicko`, `StatGlicko(Stat(...))` and named saved/custom `Rating` nodes
return both perspectives into ordinary training/reporting `X`. Statistic values
are compared as greater/equal/less to 1/.5/0, optionally reversed; missing pairs
skip updates and periods use separate state. This is comparative strength, not
a statistic-count forecast. All state fields survive build/save/load even when
the model selects fewer fields.

Replay uses event periods, batching equal release times against shared preceding
states. Availability remains optional with a retrospective kickoff proxy.
Lookup requires recorded time at/before cutoff and every contributing kickoff
strictly before cutoff. Default scope is competition across seasons; scope=()
crosses competitions and an explicit season scope resets state. Explicit idle
advancement is available, but calendar inflation is not inserted automatically.
New/revised data requires replay; incremental appending is not implemented.
Numeric custom/vector snapshots are supported. GAT training, graph updates and
temporal cross-fitting remain separate future work.

Verification passed **37 focused / 178 full tests**, including the official
tau=.5 example and permanent exact legacy comparisons. Independent replay of
380 PL matches produced 760 snapshots per result/corner stream, checked 38,000
lookup cells and exact saved-run feature parity. A delayed-release provenance
defect was fixed by the primary, including opponent-prior contributions. Default
solver precision differs from a high-precision reference by below 6.41e-6 public
units; a configured 1e-12 tolerance reduces this below 1.3e-11. These checks are
separate from exact equality with the retained legacy engine.

The [8-cell ratings notebook](notebooks/03_ratings_quickstart.ipynb) executed in
a fresh kernel; both earlier notebooks are unchanged. The wheel imports in an
isolated interpreter without installs. All 17 package sources were stable through
final verification, and the source population/records and legacy utility are
preserved. See [ratings usage and scales](docs/analytics/ratings.md) and
[evidence/saved runs](docs/analytics/ratings_check.json). These current choices
supersede earlier open proposals for the first rating interface below.

## Confirmed goals and organizing map

The user has wanted to build this system for a long time. Their experience with Alpaca for financial time-series forecasting inspired a newer direction. The separate pipelines scaffold resembles that style of work and can inform the design, while leaving room for a football-specific approach.

The intended system should support formal scientific experiments: feature construction and selection, model training and testing, proper cross-validation, and strategy backtests. Football betting may be an eventual application. A paper is an aspiration, with neither publication nor novelty established by the current planning work.

The reusable-library goal connects configurable feature definitions and window types/lengths, multi-season round-based splits, selectable prediction cutoffs, model choices, saved results/provenance, and the visualization/inspection module. The corner-count pilot should exercise and refine these reusable components while the broader system remains open to other targets and model families. (D)

**Proposed design implication:** expose a small, consistent experiment interface or configuration that composes supported pieces, applies clear defaults and validation, and returns inspectable, reusable results. This is a proposed way to make experiment setup and comparison easier. No exact configuration syntax, package layout/name, API, GUI, distribution method, or implementation has been selected. Defaults remain to be defined, and future feature/model combinations may still require adaptation. (D)

The main conceptual roadblock is how to handle irregular football histories: seasons and match weeks, changing home/away roles, rolling calculations over the appropriate past, and reshaping the resulting features into model inputs. The user also wants to retain the original artifact concept, especially snapshots for Glicko and any rating systems developed later. (D)

**Confirmed immediate priority:** make reshaping, table management, feature extraction, and Glicko state integration concrete in the design. The team-history/snapshot/match-feature flow is the present focus, with Glicko as a challenging example; remaining experiment settings need not all be finalized first. The reusable library remains the end goal, and architecture beyond confirmed choices remains open. (D)

**Current implementation direction:** loading, observed histories, initial features and result/statistic ratings are verified. First discuss the league-history/LOO and optional season-transition proposals in the latest design-priority section above; no implementation is requested by this discussion. Then return to the shared dataset contract, generic execution skeleton and reusable labels, assembly, temporal splits, model adapters and both reporter stages before configuring the corner experiment. Do not select a pilot recipe, tune a model or launch fitting at this stage. Existing league/season and eventual Lasso/corner choices remain context. (D)

The generic skeleton connects persistent state preparation, feature and label evaluation, dataset assembly, temporal splitting, training-only fitted transformations/model adapters, and pre-training/post-training reporting. Feature evaluation should consume prepared rating states so reports, models and folds can reuse them. The orchestration must support multiple labels, models and prediction layouts without hard-coding corners, Lasso or one row format. Detailed APIs remain to be designed; this is the implementation direction, not a claim that these layers already exist. (D)

**Confirmed first experiment:** predict **total corner kicks in a match** with **Lasso regression**. Prepare lags 1-5 plus rolling mean/std/Z-score for corners earned/conceded and shots for/against, for each upcoming home and away team, alongside match-result Glicko. Individual lags yield 40 candidate columns. The latest clarification makes these a selectable feature library, not a requirement to include every candidate in every experiment. Earlier normalized-standings and season-entry context remain available to specify. **Use available eligible samples, with no seasonal warm-start/blending in this first experiment.** Glicko still needs initialization and updates both teams from the same preceding states when a completed result becomes available. The user also requests reusable saved promotion/relegation preprocessing, pre/post-training reports and configurable bet-option/strategy evaluation. Lasso replaces the earlier unselected LightGBM/XGBoost pilot candidates; those remain possible later models. (D)

**Confirmed prediction-timing comparison:** make issuance timing selectable and compare both modes: separately before each match using that match's information cutoff, and all predictions for a competition round together before its first match using one frozen cutoff for that batch. Neither mode is the sole choice. (D)

Match-duration/shot-statistic definitions, prediction lead times, postponement conventions, historical data availability, evaluation metrics, baseline, Lasso regularization and Glicko initialization/time accounting remain open. The starting population is the selected 13 leagues across 2022/23–2024/25 above. Standard [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) uses squared error plus an L1 penalty; selecting it does not supply a predictive probability distribution or quantile forecasts. Bet-option outcomes remain separate from the count predictor. Prior/exponential blending stays outside this pilot. Source tables are available; feature-specific mappings still need explicit choices. The wider feature/target library remains extensible. (D)

**Desired model reuse:** the user wants to train a variety of models on existing data and reuse prepared data across model families. They have considered scikit-learn compatibility as a possible common interface and suggested skorch (spoken as “Scorch”) as a future PyTorch bridge, while reconsidering whether it is the best choice. Both remain proposed directions. (D)

**Working proposal:** prepare shared feature values, aligned targets, and match/team/time metadata, followed by small model-specific adapters for tabular arrays/DataFrames or PyTorch tensors/sequences as appropriate. Creating tensors alone does not guarantee universal estimator compatibility. Lasso is the selected first estimator family; the common adapter/backend contract remains to be implemented, and future neural input preparation stays separate. (D)

Skorch is an optional PyTorch training wrapper with a scikit-learn-compatible interface, as described in its [official documentation](https://skorch.readthedocs.io/en/stable/). It does not replace feature preparation or future temporal/multimodal assembly. Scikit-learn's Array API support is experimental and estimator-dependent, so it does not establish universal direct tensor compatibility. See the [official Array API documentation](https://scikit-learn.org/stable/modules/array_api.html). These documented capabilities inform the proposal without selecting an architecture.

**Confirmed ownership:** feature construction computes and manages season warm-up, including the relevant feature/state warm-start policy. Cross-validation defines training/validation membership, span, and stride. It must not compute feature warm-up or automatically discard early-season prediction targets because a feature uses that policy. (D)

**Confirmed initial history choice:** for the first experiment, just use available eligible samples, even when a requested window is only partially populated. This selects rolling-feature history handling, while concrete window settings, venue/competition scope, and the remaining statistical edge cases stay open. Windows can use previous-season matches. A previous-season prior followed by EMA-like updates remains a later candidate within the existing conceptual `warm_start_blend` mechanism. (D)

Feature window type and length must be selectable experiment settings. Cross-validation must support advancing by round with a configurable training span that can include several seasons, a configurable validation span, and a configurable advance step. Season boundaries must not force model training to reset. These decisions narrow the experimental requirements while leaving the architecture open. (D)

**Confirmed composability goal:** a feature/operator should be able to consume another feature's output, inspired by the user's company workflow. The football example is measuring the stability of average goals conceded. The precise stability metric remains open; the capability discussion below distinguishes current arithmetic composition from the proposed temporal extension. (D)

**Starter composition:** prepare rolling Z-scores for the corner and shot series from rolling means/stds. Feature definitions remain independently selectable. The historical numerator, baseline membership, endpoints and statistical edge policies need explicit configuration. The user now requests the initial reusable feature library; real-data validation remains pending. (D)

**Confirmed initial execution preference:** compute requested features on demand, with each composed feature obtaining its required components internally. Dependency evaluation determines what must be computed; output selection determines the columns returned as model inputs. A caller can request only a Z-score without separately requesting or exporting its rolling mean/std. Straightforward composition comes first; shared caching/deduplication is a possible later optimization driven by measured needs. (D)

**Confirmed future capability:** support feature outputs extracted from fitted/pretrained auxiliary models, for downstream models or composition. Direct extraction may suffice; model fitting need not be nested inside expressions. **This extension is excluded from the first experiment.** The initial selectable corner/shot lag and summary features, season context and match-result Glicko do not bring the future Poisson/model-extraction workflow into the pilot. (D)

The existing PyTorch Dataset is intended to support several neural-network architectures. **The initial tabular implementation excludes this PyTorch module. It is the lowest priority module to refactor for now.** Preserve the future idea and critical reuse assessment, and defer rework. The user believes no downstream models currently exist for it; no downstream model behavior has been established by this review. No Dataset rewrite or neural architecture has been selected for implementation. (D)

| Area | Questions and material it contains |
| --- | --- |
| Football data and context | Match and team identity, opponents, venue, leagues, seasons, rounds, actual times, and when source information becomes available. |
| Historical features | Team form, league baselines, scored and conceded statistics, opponent histories, head-to-head features, selectable window types/lengths, feature composition, lags, and warm starts. |
| Evolving team state | Ratings and their sequential updates, season transitions, promotion/demotion adjustments, and retained snapshots. |
| Prediction objectives | The confirmed first pilot predicts total match corner kicks as a numerical count. Broader ideas include team statistics, match totals/outcomes, betting-option labels, odds, and rankings, with their associated target definitions. |
| Research and evaluation | Chronological experiments, multi-season round-based CV, feature selection, model comparisons, backtests, and the original statistical hypotheses. |
| Data visualization and inspection | Bounded table/schema samples, changes before and after joins and pivots, and exploration of computed features. This reusable supporting module is a confirmed requirement; its interface remains open. |
| Configuration and saved results | Operators, templates, dependencies, audits, selected exports, transformations, and run provenance. |

## Three data artifacts plus run metadata

There are **three data artifacts: snapshots, team-match features, and targets**. **Run metadata** accompanies them as a fourth artifact category; it need not be a fourth data table. This distinction preserves H, pp. 1-2 and the user's clarification in D.

| Artifact | Original purpose | Relationship to the other artifacts |
| --- | --- | --- |
| **Snapshots** | Retain computed intermediate and final columns, lagged features, team ratings, flags, and warm-start coefficients. | A reusable record of calculations and evolving state; an explicit selection determines which columns become exported predictors. |
| **Team-match features** | Combine selected snapshot columns with match context using matching keys. | Join first, then pivot as needed for the requested long or wide export. |
| **Targets** | Retain target values and associated scaler columns with matching creation keys. | Preserve alignment with prediction records and enough transformation information to interpret model outputs. |
| **Run metadata** | Record the configuration that produced the artifacts. | Accompanies all three data artifacts and makes the feature/target recipe and evaluation setup inspectable. |

### Snapshots and ratings

The original snapshot key proposal is `[league, season_start, round, match_order, team_id]`, independent of final export shape. G and H, p. 1 propose retaining all necessary computed columns, including prerequisites that will never be exported. The explicit export selection is called a “white list” in the originals.

Snapshot content includes Glicko `mu`, `phi`, and `sigma` (written as Greek symbols in the sources), promoted/demoted flags, home/away context in G, and warm-start weights. Retaining ratings allows team state to be consulted and updated as subsequent match results arrive. This is the part of the artifact concept the user specifically wants to preserve. (G; H, pp. 1, 6; D)

**Confirmed initial rating meaning:** Glicko uses match win/draw/loss results first. Retained rating state and the subset of state columns exported as model inputs are separate decisions; the exported subset remains open. A later rating based on corner comparisons or another statistic would retain its own state and definition. (D)

**Working proposal:** distinguish the state available before a prediction from the state after a match result becomes available. A later update must not replace the historical state used to form an earlier prediction. Generalizing this representation to multiple rating algorithms, with configuration, version, and provenance, is a proposed refinement. There is no completed schema or implemented rating-state store implied here. (D)

H, p. 6 uses a shorter rating key, `[league, season_start, round, team_id]`, omitting `match_order`. That differs from the general snapshot key. The final identifiers, ordering, and representation of multiple updates are open.

### Features, targets, and export

The original team-match feature table uses the snapshot keys to join the selected columns and match context. H, pp. 1-2 explicitly places the long/wide pivot **after the join and before export**. The target table initially carries the same creation keys and contains target values and scaler columns. How match-level or ranking objectives ultimately map to their exported rows still depends on the prediction unit.

For the confirmed pilot, the target is the total corner count across both teams in a match. Its match-duration scope and mapping to the refactored collector's outputs remain open. This chosen count target leaves the broader target-library ideas intact. (D)

The on-demand preference distinguishes internal dependency calculations, returned model-input columns, optional intermediate persistence/auditing, and retained rating snapshots. Requesting only a Z-score need not expose its mean/std columns. Detailed persistence policy remains open: this preference neither requires permanently saving every temporary rolling component nor discards the snapshots, targets, and run-metadata artifact concept. Stateful Glicko snapshots remain available through rating orchestration. (D)

The note about cross-validation in H, p. 1 says the earlier scheme needs season and round columns to exist. Preserve those context columns as an original compatibility consideration. Their presence alone does not settle chronological validity for postponed games or different prediction cutoffs.

H, p. 2 proposes an inverter that retrieves a scaler by name and invokes its inverse method at inference. The durable intention is to retain the transformation definition and the parameters needed to interpret predictions. **Clarification:** lossy rank or bin transformations do not generally have an exact inverse. Quantile ranks, binning, LambdaRank, and pairwise ranking are mentioned as possible uses; the output interpretation or reconstruction rule for each remains to be defined.

Targets can be constructed retrospectively from outcomes. Keeping them in a separate artifact does not by itself prevent leakage: feature construction, scaling, and training still need explicit information cutoffs. (D; P, feature construction)

**Shared dataset contract — first orchestration layer:** aligned **X, y, metadata and declared prediction unit/layout** must travel together. Initial layouts are match and team-match: a row represents a match or one team's perspective, respectively. Metadata carries the relevant match/team identifiers, times and group identities, including paired-match or ranking groups. Assembly, splitting and reporting consume this declaration rather than inferring a layout from row count. Feature values and realized labels remain separate. (D)

**Extensible label families:** support statistic-derived quantities, perspective-aware win/draw/loss encoded **+1/0/-1**, configurable binary outcomes, and a **BetOption** family for outcome/threshold experiments. The label encoding is distinct from the rating engine's 1/.5/0 update scores. Binary mapping policies must be explicit: match draw, bet push, void and missing result remain distinct concepts and must not be silently collapsed. The design should allow future parlay composition without selecting bookmaker-specific settlement rules now. These general labels precede corner-experiment configuration; exact APIs and mapping choices remain open. (D)

### Run metadata

The original metadata list contains all template configurations, the selected exported features, target transformations and scalers, cross-validation configuration, export mode, and teacher-module configuration or pretraining weights when a teacher is used. (H, p. 2)

The current experiment configuration must distinguish feature window type/length, model-training span, validation horizon, advance step, prediction issuance mode/cutoff, model-refitting schedule, Glicko update trigger/uncertainty-time accounting, and season-start feature/rating policies so comparisons can identify which setting changed. (D)

**Working extension inspired by P and D:** retain the information needed to interpret and compare a run, including exact data/feature definitions, identifiers and split membership, information cutoffs, fitting scope, and rating/model provenance where applicable. The eventual storage format and persistence policy remain open. Recorded factory or algorithm names alone do not describe all fitted state.

## Working data model and information timing

This section distinguishes confirmed experiment settings from working data-shape and timing proposals. Together they address the user's conceptual roadblock without finalizing an architecture. (D)

### Collection dependency and deferred input contract

**Collection chronology:** the user deferred detailed collector-to-feature contracts while the refactor was unfinished and designated the refactorer's plan as the reference for intended outputs: [planning.md](C:/Users/luisi/Documents/Programming/Python/xDiyo/planning.md) and [scraping/REFACTOR_PROPOSALS.md](C:/Users/luisi/Documents/Programming/Python/xDiyo/scraping/REFACTOR_PROPOSALS.md). The owner of **Review Sofascore scraping refactors** now reports the initial collector refactor complete, including the match-date follow-up, with all **55 tests passing**. This reported completion does not finalize the analytical input mapping. (D)

**Confirmed compatibility choice:** legacy-format support is not a requirement. The user is happy to adapt to the refactored collector's new output format and considers collecting old data or feature inputs again if needed. Recollection remains a possibility; it has not been performed or scheduled by this planning update, and no legacy files are deleted. (D)

Detailed collector-to-feature container shapes, final input contracts, schema mappings, joins, and query implementation remain follow-up work; they have not been reviewed or finalized after the completion report. No detailed table-schema audit is part of this update. Neither present CSVs nor an inspected migrated export establishes the final feature input contract. The conceptual team histories, snapshots, features, and targets below remain in scope; their concrete mapping to collection outputs is still deferred. (D)

**Verified narrow date outcome:** `matches.kickoff_utc` already preserves provider `event.startTimestamp` as nullable floating-point UTC Unix seconds. It represents the scheduled/revised provider event start; the inspected evidence does not establish a separate actual first-whistle time. The collector added `event_metadata_observed_at` for the metadata response used in each match row, distinct from component `observed_at` fetch timestamps. Schema 2 documents these semantics in Parquet metadata and manifests. See the [match date/time contract](C:/Users/luisi/Documents/Programming/Python/xDiyo/scraping/sofascore/README.md:68).

The [date-validation report](C:/Users/luisi/Documents/Programming/Python/xDiyo/data/audits/match_datetime_validation/report.json) records three cached live events checked with **zero new API requests**. Rescheduled values preserve stable event IDs and observed raw versions. The current matches export is the latest observed schedule view; it does not reconstruct schedule knowledge from before collection began. Old undated imports remain unknown. This collection follow-up checked only the date contract and its evidence; the scraping owner's tests were not rerun and its code task remains separate.

### Proposed analytics loader plan (historical)

**Earlier planning component:** supply predictable tables to the reusable library's feature preparation. This is an analytics data loader, distinct from the deferred PyTorch Dataset/DataLoader. The following proposal preserves the earlier design discussion. The simple implemented API and completed population import above supersede its sample prerequisite and proposed diagnostics/API details. Feature construction and historical-cutoff validation remain future work. (D)

**Existing building blocks, inspected only:** `read_tables(manifest_path, *, names=None, verify_hashes=True)` resolves a CURRENT pointer or manifest, selects named tables, checks hashes by default, reads their Parquet files, and returns a dictionary of lists of row dictionaries. It does not currently provide competition/season queries, column projection, general schema-contract validation, or a DataFrame bundle. `merge_seasons` uses `legacy.load_season`, optional team-season flag joins, and a union of columns with missing observations preserved; it also normalizes standings and rounds by each loaded frame's maximum. Use the canonical export reader where suitable and keep statistical normalization in explicit feature policies, rather than adopting that compatibility helper wholesale. Prepared Parquet exports are versioned by competition/season, separate from raw archives; optional team-season flags have a separate reader. (L)

The following loader behavior is **proposed**:

1. **Boundary:** prepared collector exports → loader → team-history preparation → rolling features/Glicko orchestration → model input assembly. The loader selects, reads, and validates data. Historical joins/pivots, statistical scaling, feature calculations, CV, and model fitting belong to their respective later stages.
2. **Inputs:** accept a data root plus explicit competition/season selection, or explicit published manifest references, together with the experiment's required tables/statistics. Allow access to earlier history needed by requested features even when it lies outside the supervised training interval. Prefer explicit selection first; the exact function signature and automatic feature-dependency-to-query integration remain open.
3. **Version resolution:** resolve each requested partition to one published manifest and retain the concrete versions, schema/parser metadata, and provenance. Users supply a data directory, season and table; version IDs and SHA256 fingerprints are handled internally. An optional human-named season-selection record is saved after the first successful load and reused on later calls, with no fallback if its pinned files change or disappear. A later experiment runner can collect these records across its selected seasons. File discovery must not concatenate every historical export version. Keep shots and other model inputs separate, with explicit table/group selection. Current implementation and verification are tracked in `IMPLEMENTATION_PROGRESS.md`; projected/filtered reads and multi-season orchestration require later design.
4. **Output:** return a small named collection of consistent tables plus load provenance and diagnostics. DataFrames are a proposed starting representation for grouped tabular work; the class/API, schemas, dtypes, and keys await the sample. Candidate pilot inputs are match records, long statistics, coverage/availability information, and available pregame standings/context, with optional team-season flags. Exact fields, statistic keys, periods, and join grains are unverified.
5. **Validation:** check required tables/fields, supported schema versions, stable IDs, each table's proper row grain, and cross-table references. Preserve source missingness and time semantics; report incomplete coverage. Proposed handling is a clear failure for an unusable export/schema or absent indispensable table, while incomplete match-level observations retain diagnostics instead of silent zero-filling or removal. Exact strictness and optional-input rules remain reviewable. A frozen export does not establish historical point-in-time availability: feature construction still enforces cutoffs using supported source information. Verified kickoff/observation timestamps do not establish actual match-end or original publication times.
6. **Inspection:** expose row/column summaries, coverage, and bounded samples for the planned visualization/inspection module. No UI is required for this planning step.

**First implementation sequence after sample collection; future acceptance checks only:**

1. Inspect one representative prepared export and its manifest.
2. Confirm the minimal pilot field mappings, statistic periods, and table grains.
3. Implement explicit selection, manifest pinning, reading, validation, and return of the table collection using the existing reader where suitable.
4. Check multi-season concatenation, single-version selection per partition, stable IDs/references, and preservation/reporting of missing values.
5. Connect team-history preparation and a small season-boundary example to verify that the loader supplies predictable inputs. Feature calculations and rating orchestration follow this loader stage.

These checks have not been executed. This plan does not require legacy CSV support, change collector schemas, or finalize feature policies. Discussion has resumed; implementation remains a subsequent step at the user's direction, after inspecting the representative export. (D)

### Team histories across home and away appearances

Represent a historical match as two team-appearance records. Each record would identify the match, focal team, opponent, venue/role, and football context. Team identity anchors the history across changing home/away roles.

Illustration only:

| Match | Focal team | Opponent | Role |
| --- | --- | --- | --- |
| M1 | A | B | Home |
| M1 | B | A | Away |
| M2, a later match | C | A | Home |
| M2, a later match | A | C | Away |

Team A's history table includes its own appearances in both M1 and M2. Keeping both appearances in the table does not select which ones enter a feature window. Pooling home and away history versus conditioning windows on venue remains open; the upcoming match's home/away labels do not resolve this choice or change the team's identity.

Compute eligible historical features along these histories. For a match-level prediction, retrieve the home and away teams' feature snapshots as of the same prediction cutoff, join match context, and assemble one match record. Team-level targets and rankings remain in scope, so a match-wide row is one possible final representation. The intermediate two-row representation is a data-shape proposal within this workflow; it does not add a fourth required data artifact.

Realized statistics from a match can enter later historical calculations once available. They cannot become inputs to a prediction issued before that match merely because the retrospective source row contains them.

### Pilot features and snapshot assembly

**Initial feature-library scope:** prepare the following for **each upcoming home and away team**, with experiment recipes selecting which definitions to include. The Lasso pilot is the first test, not a fixed schema for the whole library. (D)

| Feature for each team | Intended content |
| --- | --- |
| Lags 1-5 | Separate previous-match columns for corners earned/conceded and shots for/against: 20 per team, 40 for a match. Missing observations do not renumber lag positions. |
| Rolling mean and standard deviation | Reusable summaries for all four historical series; a five-match horizon is the working pilot interpretation, independently configurable in the library. |
| Composed rolling Z-score | Latest eligible historical value minus its trailing-window mean, divided by the same window's std; the latest value is included and zero spread gives missing. |
| Match-result Glicko | All retained rating-state components; provisionally `mu`, `phi`, `sigma` per team based on the existing record. Additional derived outputs need their own definitions. |
| Season context | Normalized standings now use full-population team counts before filtering and a zero fallback. Promotion/relegation inclusion and any rating-adjustment role remain to specify. |

Lag, mean and std definitions are available even when a recipe uses only some of them. A five-match mean repeats information already in five complete lag columns; this affects interpretation of Lasso's sparse selection, not the usefulness of providing both feature families. Partial eligible windows remain allowed, with no fabricated history or seasonal blend. The separately mentioned plain "mean" has no distinct population selected yet. The first evaluator defaults to team+competition across venues/seasons, with explicit eligibility and missing-value rules; exact shot statistics and recipe overrides remain selectable. Standings use distinct full-population team counts, not an observed whole-season maximum position.

**Working table-flow proposal:**

1. Load historical matches and saved season-entry context; prepare team histories with eligible corner and shot observations, stable identity and information timing.
2. Evaluate the requested lag/mean/std/composed-feature recipe from eligible history. Allow partial summaries; retain missing lag positions for an explicit model-input policy. Feature preparation has no seasonal warm-start/blending in this pilot.
3. As soon as a match is completed and its result is available, compute both teams' Glicko updates from the same pair of immediately preceding eligible rating states. Retain both resulting snapshots with availability/provenance; neither team is updated against the other team's already changed state.
4. At each prediction cutoff, attach eligible feature and rating snapshots for the upcoming home and away teams, then assemble the proposed match-feature record with its aligned total-corner target and metadata. Retaining rating state does not require exporting every state field to the model.

The cutoff remains either the individual match cutoff or the frozen pre-round batch cutoff. Production source mappings, final join/storage contracts and rating time-unit/replay policies remain to validate. Basic feature-library implementation is now requested, independently of the real-data adapter. The original general Z-score/template examples remain preserved below. (D)

### Calendar time, match time, and rounds

Actual match times and source-information availability determine what can be known at a prediction cutoff. Season and competition round supply football context. A postponed match can be played after matches bearing higher round numbers, so sorting by round alone is insufficient.

A stable match identifier and a chronological ordering serve different purposes. An arbitrary tie-breaker can provide deterministic storage order, but it cannot make a simultaneous or unavailable result eligible history. Standings, ratings, and other context also need the version available at the cutoff.

### Selectable prediction issuance timing

**Confirmed requirement:** support and compare **both prediction timings** through a selectable experiment setting. (D)

| Mode | When predictions are issued | Information cutoff |
| --- | --- | --- |
| **Before each match** | Issue a separate prediction before each match. | Use only information available by that match's cutoff. Earlier results or publications may enter later predictions once available. |
| **Before each competition round** | Issue all predictions for that round together before its first match. | Use one frozen information cutoff for the entire round batch. A later match in that batch does not gain information published after the batch cutoff. |

Each prediction can use only information available at its own cutoff, excluding results from later matches and later publications. Exact lead times before kickoff, round schedule/postponement conventions, data-availability details, and implementation remain open. A competition round is distinct from a calendar week; the confirmed batch mode is organized by competition round.

For evaluation over several rounds, each later-round batch has its own cutoff and may use earlier-round results already available then. The round-batch mode does not freeze every validation round at the start of the fold. Freezing an entire fold at one origin would be a separate policy requiring a deliberate experiment choice. Changing the issuance cutoff does not itself require model refitting after every match or transfer seasonal warm-up from features to CV.

### Selectable feature window type and length

**Confirmed requirement:** window type and length remain selectable across experiments. The first feature recipe requests lags 1-5, with five-match rolling summaries as the working interpretation; this does not set a library-wide default. Four calendar weeks remains an illustrative elapsed-time setting. (D)

| Possible policy | What the lookback counts | Consequence |
| --- | --- | --- |
| Last N team appearances | Eligible played matches of the focal team | The elapsed calendar duration varies between teams and periods. |
| Elapsed calendar time, such as four calendar weeks | Eligible observations in the configured elapsed-time interval | The number of included matches varies, including gaps and congested schedules. |
| Previous N competition rounds | A competition's round labels under an explicit inclusion policy | Postponements and unequal schedules require a rule connecting rounds to availability. |

The preferred starting approach is one long team-appearance table with native grouped rolling for ordinary match-count and time windows. Round windows may use explicit round membership or aggregation. A custom pandas indexer is an implementation option only when its contiguous positional bounds correctly express the intended eligible history. A round-selected history need not satisfy that condition. This is a design preference, not a claim that these policies have been implemented. (D)

A no-match week contributes no match observation. It does not automatically create a zero-valued observation. The pilot uses available eligible observations in a partial window; zero observations, missing values, and statistic-specific minimum conditions still need explicit handling.

**Confirmed history direction:** keep a team's chronological history available with season and competition-round context. Each rolling feature's configurable window selects its eligible matches and can include previous-season matches. Actual chronological order and information availability still constrain membership. Venue-specific versus all-venue histories and competition inclusion remain open scope choices; selecting a window does not decide them. (D)

**Proposed season context:** retain an explicit season-start or season-transition flag to identify the boundary/jump for feature-owned warm-start/blending logic. Its exact definition and placement remain open. A boundary does not automatically reset all history, force a model/CV reset, or select a blend formula. The pilot continues to use available samples; flags can support later transition strategies within `warm_start_blend`. (D)

**Feature lookback and model-training lookback are independent.** A short feature window can be used in a model trained on several seasons. Conversely, a model fitted on a recent interval may need earlier eligible history to form its lagged predictors. Access to that history does not authorize fitting preprocessing or model parameters on every historical row.

### Initial partial-history policy and future prior alternatives

**Confirmed pilot choice:** use the available eligible samples within the requested window and chosen history scope. For example, a requested last-five-matches mean can use two eligible earlier matches when only two exist. A partial window is sufficient for the first experiment when the statistic has valid inputs and satisfies its mathematical minimum conditions. Do not require the full requested length merely because the scaffold currently does, fabricate padding matches, or reach outside the selected window to fill it. Window settings remain selectable; the first evaluator defaults to team+competition across venues/seasons, with custom grouping supported. Season boundaries do not automatically reset history, and CV does not own warm-up or automatically exclude early-season rows. (D)

**Implemented first-feature edge policies:** a mean can use one finite observation; zero eligible values produce missing. Windows count matches, while reductions skip missing/nonfinite values inside the chosen window and lags do not skip positions. Default std uses ddof=1 and requires n > ddof; min_periods is configurable and defaults to 1. Z-score standardizes the latest eligible value against a shared baseline including that value; missing-latest or zero-spread cases produce missing. Sample-count/flag outputs remain a diagnostic proposal, not a global imputation rule. The standing-specific zero fallback is described above. (D)

**Earlier discussion within `warm_start_blend`, outside the pilot default:** initialize a feature from a previous-season average/prior, then gradually update it with new observations using an EMA-like procedure. This is a possible strategy within the existing conceptual feature mechanism, not a new architectural module. The original template illustrates `LinearWarmStart` with an `alpha_schedule` from round 1 to round 6; that historical linear example is distinct from the possible exponential strategy and does not establish an implemented or equivalent recurrence. The user retained this as one possible strategy and then chose available samples for the first experiment. That discussion left prior source (team, league, or another source), smoothing/decay, time unit, seasonal transitions and any handoff to fixed rolling windows open; it also identified the need for prior variability to warm std/variance. The latest design-priority section now specifies selectable completed-round handoff durations and matching previous-season league variance priors, with remaining weighting/missing-prior details pending. No EMA prior is selected or implemented for the pilot. (D)

Glicko keeps its own initialization and update mathematics. Immediate completed-result updates are now chosen; raw-statistic EMA formulas do not replace them. Partial-history rolling summaries do not settle Glicko priors, uncertainty/volatility time units, idle-time accounting, or seasonal transitions. These alternatives remain planning material; no feature or experiment has been executed for this decision. (D)

## Cross-validation schemes and controls

### Confirmed requirements and separate controls

Cross-validation must be able to advance by **round**, with separately configurable training span, validation span, and advance step. The user's example was validation on the next four or five rounds. An earlier ten-training-round example was illustrative; it does not limit training to ten rounds or to one season. Training history must be able to include round blocks from **several seasons**, while matches in a new season use the selected feature-history policy and eligible rating state. A season boundary must not automatically reset the model's training window. (D)

The library should also offer selectable season-based and within-season evaluation schemes, plus retrospective grouped and combinatorial alternatives. Keep three settings separate: **feature history window definition/length**, **train/validation splitting**, and **prediction issuance cutoff**. Refitting has its own schedule; per-match prediction does not imply per-match refitting. Feature construction retains seasonal warm-start ownership. (D)

| Control | What it governs | What it does not determine |
| --- | --- | --- |
| Feature lookback type and length | The eligible history used to calculate a particular moving average, standard deviation, or other feature. | How many supervised examples train the model. |
| Split scheme and training-window behavior | Expanding or sliding chronological history, or an explicitly retrospective grouped/combinatorial scheme. | Feature initialization or prediction issuance; retrospective schemes do not become past-only through a cutoff on input features. |
| Model-training span | The eligible labeled training blocks, expressed in rounds, whole seasons, or portions of seasons; chronological schemes use earlier eligible observations. | The feature's own history length or season-start initialization. |
| Validation block/horizon | A full season, an early/later season segment, or a specified round interval; combinatorial schemes hold out combinations of blocks. | How far a chronological fold advances or when each prediction is issued. |
| Scoring segment | Which held-out rounds contribute to a reported score, including early-season versus later-season comparisons. | Whether unscored validation rows belong to training or when their results become available to causal features. |
| Advance step/stride | How far the next chronological fold's selection origin moves, in rounds or seasons. | The validation horizon, feature window length, or enumeration of retrospective fold combinations. |
| Prediction issuance mode/cutoff | Whether information is sampled before each match or frozen before the first match of each competition round. | Feature window definition/length, configured fold spans/stride, or automatic model refitting. |
| Refitting schedule | When models and fitted transformations are re-estimated within their eligible fitting scope. | The prediction issuance cutoff or feature warm-start ownership. |
| Glicko update timing | Update both teams from their shared pre-update state pair as soon as a completed match's result is available. | Feature windows, prediction issuance, CV spans/stride, or the model-refitting schedule; uncertainty-time accounting remains separate. |
| Season-start state handling | Feature construction owns initialization, warm starting, and shrinkage under the relevant feature/rating policy. | CV membership or automatic exclusion of early-season validation targets. |

Warm starting a feature or rating is different from including early-season matches as supervised training rows. **The user assigns season warm-up to feature construction, not to cross-validation.** A model trained on previous seasons can use available-sample rolling features and separately initialized rating state for early-season predictions, subject to the remaining edge policies. Partial-history handling does not itself require dropping those predictions from validation. Any deliberate inclusion in training or exclusion from validation needs its own evaluation policy and actual availability constraints. Ownership and the pilot's partial-window choice are settled; concrete window/scope settings, rating initialization, and future `warm_start_blend` prior/exponential settings remain open. (D)

### Selectable evaluation variations

The following list consolidates the requested variations. Some rows are configurations or scoring views of the same splitting mechanism, rather than independent algorithms. All counts are illustrations, not selected defaults; these are planning requirements, not implemented splitters. (D)

| Variation | Example or definition | Interpretation |
| --- | --- | --- |
| **Expanding chronological window** | Train on S1-S3 and validate on S4; the next fold trains on S1-S4 and validates on S5. An expanding origin may also advance by rounds. | Accumulate eligible earlier history. Whole-season train/validation membership is disjoint within each fold; earlier validation seasons can become training data in later folds. |
| **Sliding chronological window** | Keep a fixed recent span, such as 76 rounds or three seasons, and validate on the next configured block. | Discard older supervised training rows as the origin advances; history can span season boundaries. |
| **Within-season chronological split** | Train on two or three complete seasons plus the first half of the next season; validate on that season's remaining half. | Train and validation may share a season, but their matches and selected round blocks remain disjoint and ordered. |
| **Season-stage validation/scoring** | Train on preceding seasons and hold out a new season; evaluate its beginning, later part, or both separately. | Early-season scores probe initialization and warm-start effects. Holding out the whole season but scoring its second half is different from training on its first half. |
| **Season-grouped K-fold** | Hold out one season per fold and fit on the other selected seasons; this is a leave-one-season-out configuration. | Retrospective when training includes later seasons. Restricting training to earlier seasons instead gives a chronological scheme with a different training population. |
| **Combinatorial purged cross-validation (CPCV)** | Hold out combinations of chronological blocks and reconstruct multiple paths from held-out predictions. | Confirmed future option for robustness analysis of betting strategies based on model predictions; retain the caveats below and chronological evaluation alongside it. |

Define a season's half or stage using explicit competition-specific round boundaries, rather than assuming every season has the same length. Actual match/result availability still governs chronological eligibility, especially for postponements. To compare available-sample, shrinkage, or warm-start variants, use identical scored fixtures and report results by season stage. Mid-season validation includes more current-season history and may hide initialization effects. The future shrinkage/prior variants do not replace the pilot's available-sample policy. (D)

### Proposed round catalogue and an illustration

A minimal season-aware round catalogue, with an ordered coordinate across seasons, is a **working proposal** for selecting training and validation blocks. Within a specified competition calendar, the coordinate can continue from the last round of one season to the first round of the next. The catalogue is a selection aid; it does not establish when every match result or source value becomes available.

Illustration only: assume **one hypothetical competition with 38 rounds in each season**, no postponements, and all results from earlier round blocks available before the next fold's fitting cutoff. Choose a **76-round sliding training span**, a **four-round validation horizon**, and an **advance step of four rounds**. S1 and S2 are earlier seasons; S3 is the new season. These counts are examples, not selected settings or an assertion about any league.

| Fold | Candidate training blocks | Candidate validation block |
| --- | --- | --- |
| 1 | S1 rounds 1-38 plus S2 rounds 1-38: 76 rounds. | S3 rounds 1-4. |
| 2, advanced four rounds | S1 rounds 5-38 plus S2 rounds 1-38 plus S3 rounds 1-4: 34 + 38 + 4 = 76 rounds. | S3 rounds 5-8. |

The first fold illustrates early-season validation with training from two previous seasons. The second illustrates a sliding window that spans parts of three seasons; its formerly held-out rows can enter training only once their labels are eligible. This table describes round-block selection and can be used with either confirmed issuance mode. It does not settle lead times, a feature lookback, rating initialization, or future prior settings, and it does not authorize updates from an entire validation block.

For real data, actual prediction cutoffs and information availability must constrain the candidate membership. Postponed matches can violate catalogue order. Equal round numbers across leagues do not define a shared real-world cutoff. Multiple league calendars and postponement policy remain unresolved. (D)

### Validation and fitting scope

Chronological evaluation must remain tied to actual prediction opportunities and label-availability times, including when rounds select the folds. When the modeling unit uses team-level rows, both sides of one match must remain in the same fold. Using one side for fitting while scoring the other does not provide an independent held-out match.

Fit preprocessing, learned or supervised features, feature selection, tuning, and models within their eligible training/inner-validation scope. Supervised training features may need out-of-fold construction inside the training data. Outer test outcomes belong to evaluation; they must not guide choices subsequently reported as held-out performance. Labels become eligible for training when they are available, which may be later than the prediction or match timestamp. (D; P)

If validation features or ratings update after an earlier validation match's result arrives, that update must respect the selected issuance mode and actual availability. Per-match predictions may use newly available earlier results; each competition-round batch stays frozen at its own pre-round cutoff. Later-round batches within the same fold may use earlier-round results available by their cutoffs. There is no automatic permission to use the realized outcomes of a whole validation block, and causal state updates do not authorize fitting selection, transformations, tuning, or the predictive model on that fold's held-out labels. (D)

**Proposed comparison design:** evaluate the two issuance modes on the same fixtures and folds, with the same feature definitions, model configuration, and refitting schedule, varying only the available-information cutoff. Feature values may consequently differ because their eligible history differs. This is an experimental recommendation, not a mandated API. Apply the cutoff rule to every prediction and retain the distinction between a frozen round batch and a separately chosen whole-fold freeze. No exact schema, adapter, or prediction timestamps are implemented by these notes. (D)

The selectable variations above extend the confirmed round-based advancement and multi-season training requirements. Exact spans and stride, stage boundaries, gaps, retraining schedule, evaluation metrics, and weighting remain open. A strategy backtest also needs its own decision and outcome definitions. These are scientific design requirements, not implemented capabilities or a completed protocol.

### Future CPCV strategy evaluation and caveats

**Confirmed scope:** include CPCV in the selectable evaluation list for later analysis of betting strategies driven by model predictions. This is outside the first corner-count pilot and does not select a betting strategy or authorize implementation. The user also requested retaining the following methodological caveats. (D)

CPCV partitions ordered observations into blocks, tests combinations of held-out blocks, and reconstructs backtest paths from their predictions. In the author's illustration, six blocks with two held out per split yield 15 train/test combinations and five reconstructed paths. Standard CPCV can fit on later blocks while evaluating earlier ones. Purging removes training observations with overlapping label-information intervals; an embargo excludes training observations immediately following a test interval. These operations do not make the scheme past-only. See [Marcos Lopez de Prado, The 10 Reasons Most Machine Learning Funds Fail, PDF pages 12-14](https://smallake.kr/wp-content/uploads/2018/07/SSRN-id3104816.pdf). This source explains the financial method; the football adaptation below is a planning proposal, not a validated result.

- **Keep evaluation interpretations distinct.** Use chronological walk-forward evaluation as the primary simulation of what could have been done with information available at the time. CPCV supplies a complementary retrospective robustness analysis across training/held-out period combinations; it does not establish achievable historical deployment performance.
- **Evaluate the full prediction-to-strategy process.** Model fitting, preprocessing, feature selection, learned-feature fitting, and any strategy tuning must respect the relevant training/inner-validation scope. Applying CPCV only to predictions or returns produced by a model fitted on all matches cannot repair that leakage. If a strategy layer consumes previously generated predictions, their fitting and timing provenance must also be valid for the intended evaluation.
- **Define football-specific dependencies.** Keep both team rows and multiple bets on the same match together. Use relevant prediction, result-availability, label and settlement timing, together with shared outcome dependencies, to determine exclusions. Do not copy an arbitrary financial embargo percentage or assume round numbers establish chronological independence.
- **Audit rolling histories and persistent rating state.** Later training rows may carry held-out match outcomes through rolling summaries or Glicko updates. Split membership and a short time gap alone may not isolate that information, especially with recursive ratings. The policy for reconstructing or excluding affected histories and states remains to be designed. This is distinct from the already permitted causal use of earlier available validation results for later walk-forward predictions.
- **Use realistic betting inputs and accounting.** Retain odds available at the prediction/decision cutoff, define markets and settlement rules, include applicable costs, and process each reconstructed path chronologically for bankroll and overlapping commitments. A count prediction alone is not a completed betting decision rule.
- **Do not treat paths as independent histories.** Paths reuse historical matches and training data; their performance spread is a robustness diagnostic, not independent evidence from new seasons or a guarantee against selection overfitting. Keep a final untouched chronological test after model/strategy selection.

Exact block definitions, numbers of blocks and held-out blocks, dependency/purge/embargo rules, state reconstruction, path assembly, strategy tuning, and performance reporting remain open. Feature warm-up continues to belong to feature construction; CPCV does not manufacture missing past observations or settle season-start initialization. (D)

### Existing CV prototype: inspected, not run

The complete extensionless file C defines `SeasonRoundTimeSeriesSplit`. It was read as source only; neither this task nor the originating discussion ran or tested it. Its current masks and parameters describe a different splitting policy:

| Prototype component | Behavior visible in C |
| --- | --- |
| Training population | All seasons before the first validation season, plus optional early rounds from that first validation season. There is no configurable sliding last-N-round training span. |
| Fold advancement | `step_seasons` advances folds by positions in the season list, defaulting to `valid_seasons`. It does not advance by rounds. |
| Uncapped validation | With `maximum_validation_rounds=None`, use the selected `valid_seasons` season block, excluding warm-up rows from its first season. |
| `maximum_validation_rounds` | Restrict validation to `round <= N` in the first validation season, after excluding warm-up rows. N is an absolute early-round ceiling, not a rolling validation length starting at an arbitrary round. |
| `season_warmup` | Add early-round samples to training and remove them from validation in the first validation season, with optional capping intended to retain validation rounds. This is supervised row allocation, not feature/rating initialization or shrinkage. |
| Information/calendar handling | Build masks from season and round columns; the file does not sort by actual match time or check actual result/source availability. Its warm-up thresholds apply across leagues sharing those season/round labels, without resolving their different calendars. |

The prototype can include several earlier seasons because it includes all of them, but it does not implement the requested **sliding multi-season round window**, round-based stride, or actual availability safeguards. Its `season_warmup` parameter is legacy/prototype behavior that differs from the confirmed ownership decision. It must not be carried into the intended CV as the feature/rating warm-start policy. The prototype remains unchanged. (C; D)

## Data visualization and inspection module

**Confirmed requirement:** add a reusable supporting module for inspecting data and exploring computed features. The user wants to see table columns and indexes, inspect a bounded portion of a potentially large dataset, and understand how shapes, columns, and indexes change before and after joins and pivots. They also want to plot and analyze the resulting features. This adds a planning requirement; implementation has not been authorized. (D)

The module has two capabilities:

| Capability | Confirmed purpose | Proposed examples, with the interface still open |
| --- | --- | --- |
| **Table, schema, and transformation inspection** | View selected table samples and compare their structure before and after joins or pivots. Include team, season, calendar week, or competition round context where relevant. | Show row/column counts, data types, column names, index/key names, and before/after samples. Select a team, season, or time range to keep the displayed portion bounded. These are suggested inspection fields and selectors, not settled schemas or a sampling policy. |
| **Feature exploration** | Plot and analyze computed features to understand their behavior. | Feature-history plots and distributions are examples; the chart set and comparison controls remain open. |

Calendar weeks and competition rounds are distinct concepts and must remain distinguishable in selections and displays. Their eventual column or index representation depends on the completed collection contract. Inspecting a transformation does not settle its join keys or pivot schema in advance.

**Required reporter stages:** the pre-training reporter consumes the shared X/y/metadata/layout contract. It must distinguish match counts from team observations and respect paired-match/ranking groups instead of guessing the prediction unit from the number of rows. A post-training reporter is also required, consuming predictions and applicable model outputs with the same layout/group context. Both reporters belong to the reusable execution skeleton; defining those stages does not authorize model fitting now. (D)

No visualization framework, dashboard, GUI, notebook-only approach, sampling policy, or exact API has been selected. The original GUI and registry ideas remain historical design material; this additional module does not settle how those ideas will be delivered. Detailed integration with collection outputs remains follow-up work after the reported collector completion. (D)

## Original analytical workflow

### Orchestration and computation passes

G proposes a GUI for choosing feature and target templates, Pandas for feature calculations, and an orchestrator that resolves dependencies. H, pp. 5-6 provides the following three computation passes:

| Pass | Original proposal |
| --- | --- |
| **A: League** | Compute shifted league-season rolling baselines using the indicated partition. A proposed baseline table keyed by `[league, season_start, round]` is later reused for leave-one-out calculations or joined/broadcast to matches with those keys. |
| **B: Team** | Compute features on the stacked team table with the chosen partitions and league context. Preserve continuous team form; apply hierarchical shrinkage when league changes. Use Pass A baselines for team-relative-to-league features and leave-one-out calculations. |
| **C: Ratings** | Maintain rating snapshots representing state as of the previous match, with the season and league-transition adjustments described below. |

G says there are four passes, adds **Pass O** to resolve retrieval markers and construct a dependency graph, and then stops at the Pass A heading. H describes three computation passes. A possible reconciliation is one preparation pass plus A/B/C; this is a reading of the two sources, not a finalized execution specification.

Pass O would inspect families, output names, and requirements, reuse already requested/computed prerequisites, and arrange missing prerequisites before their consumers. The original main order is league, team, then ratings. How arbitrary dependency graphs, expressions, postprocessing, and teacher features fit into that order is not fully specified.

**Working timing qualification:** dependency order and chronological eligibility are separate concerns. A feature is not safe merely because its prerequisite was computed in an earlier pass. The proposed league-round baseline joins in particular need to respect each confirmed issuance mode and the still-open postponed-match policies.

### Season and league transitions for ratings

H, p. 6 proposes the following experiments/design choices:

- At a season boundary, inflate `phi` while preserving `mu`.
- For promoted or demoted teams, shrink `mu` toward a prior for the new league, optionally also inflating `phi`.
- Define the league-season `mu` prior before applying that shrinkage step.

These are historical proposals. The historical sources do not specify completed prior estimators, shrinkage/inflation, or inactivity handling. The current discussion now chooses updates when completed match results become available; uncertainty-time accounting and seasonal adjustments still need definition. D extends the motivation to future rating systems as well as Glicko. Snapshot timing still needs the explicit pre-prediction/post-result distinction described above.

### Scored, conceded, and the opponent's history

H, pp. 7-9 distinguishes three related ideas:

1. A focal team's **scored/produced** statistic comes from that team's performance in each of its historical matches.
2. Its **conceded/against** statistic comes from the opponents it faced in those matches, aligned along the focal team's history.
3. The upcoming opponent's **own historical feature** comes from that opponent's sequence of appearances and must be aligned to the current match.

For example, A's rolling mean of conceded corners summarizes corners conceded by A against several past opponents. B's rolling mean of corners produced summarizes B's own past matches. Relabeling the first column as “opponent form” would change the intended meaning.

The historical proposal is an operator called `PivotOpponent`, applied to already computed snapshot columns after the team pass. H, p. 8 sketches grouping by `team_id`, retaining the `[league, season_start, round, match_order]` tuple and opponent identifiers, retrieving counterpart values, and attaching the aligned result to the focal rows. A margin note questions whether the initial opponent-ID extraction is necessary. This is a lookup sketch, with grouping/index details still requiring resolution; it is not a validated implementation.

The intended result is to retrieve the current opponent's corresponding eligible feature by match/opponent identity. That opponent lookup is conceptually distinct from the final long/wide export pivot. H, p. 7 also suggests scored and conceded checkboxes in the GUI, with scored selected by default.

H, p. 9 gives a feature-builder example comparing the focal team's ratio of rolling scored to conceded means with the opponent's corresponding ratio. Expressed descriptively:

```text
team rolling scored mean / team rolling conceded mean
  minus
opponent's own rolling scored mean / opponent's own rolling conceded mean
```

Each mean has its own appropriate history/alignment and configured window, lag, and warm-start/shrinkage treatment. The sources do not finish the zero-denominator or missing-history behavior.

### Research hypotheses to preserve

| Original research idea | Status and remaining specification |
| --- | --- |
| Compare a league-mean prior with a James-Stein approach for the first m weeks. (H, p. 12) | An experiment proposal. The estimator, comparison criterion, eligible prior data, and meaning of “weeks” remain open. No preferred estimator or benefit has been established. |
| Investigate dependence involving home/away goals and corner counts. (H, p. 12) | Define the exact variables, pairings, conditioning context, sample, and tests. The wording does not establish a dependency structure. |
| Expose statistical tests as reusable operators, using home/away goals as an example. (H, p. 13) | An operator-library idea. Tests assess evidence about dependence under their assumptions; failure to reject a null does not prove independence. |

These ideas can inform the broader scientific experiments requested in D. There are no experiment results, backtest conclusions, or claims of novelty in these notes.

## Existing scaffold, rating core, and utility inventory

### Pipelines as inspiration

P describes a project-agnostic scaffold inspired by the high-level separation of responsibilities in Alpaca workflows. It separates data sources and targets, feature definitions and extraction, preprocessing and selection, training, pre/post-analysis reporting, and run artifacts. Its documentation explicitly distinguishes working defaults from extension skeletons.

The following documented ideas are relevant to the current discussion:

- `Observation` separates stable row identity, entity identity, and a prediction timestamp; source values need to be available by that time. Targets carry a separate `available_at` time.
- `Lag` and `RollingMean` count entity observations. `DurationMean` supplies an elapsed-time window. These address different window policies rather than choosing one for football.
- The default feature context exposes earlier rows of the focal entity. `RelatedEntityContext` and related calculations can supply eligible related/global histories; simultaneous observations do not become one another's history.
- Historical covariates can extend before the model-fitting interval to provide feature warm-up. Fitted transformations remain restricted to their permitted fitting population.
- Global feature selection discovers column choices in the first fold's eligible training set and freezes those names and their trace. Rolling selection repeats discovery per training fold. Both modes fit fresh feature builders, preprocessing, models, and baselines per fold.
- The ordinary runner evaluates sequential prediction opportunities. The documents also describe explicit fixed-origin request batches and compatible prediction providers; mapping the two confirmed football issuance modes onto these capabilities remains project work.
- Saved run/fold information includes configuration, split IDs, selection provenance, fitting scope, fingerprints, predictions, and reports. The documentation identifies model serialization, resume/caching, tuning, forecast providers, and deployment as areas with extension boundaries.

The working default runner is described for scalar regression/classification with numeric in-memory feature matrices. Football target families, ranking objectives, source joins, relationships, identities, and availability rules still need domain definitions/adapters. These documents do not establish that a football adapter exists, that every template idea is supported, or that the scaffold is production-ready. No scaffold example or test was run for this documentation task.

### Feature composition: present support and temporal extension

Composability is already present in part of the scaffold. Source inspection of PF shows the following boundaries:

| Component | Current support or limitation |
| --- | --- |
| `composition.py` | `BinaryFeature` takes left/right `Feature` objects; `Difference` and `Ratio` evaluate those children in the same context. `Shift(feature)` evaluates a child at an earlier context. Raw dependencies propagate; binary lookback uses the larger child requirement, and a shift adds its periods. |
| `time_series.py` and `base.py` | `RollingMean` and `RollingStd` inherit `RollingFeature`/`SourceFeature`. Their source is a named raw covariate string, read through `context.values(...)`; they do not directly accept another `Feature` as the series to roll over. |
| `extraction.py` | Each requested output is calculated against raw `Observation` histories. Original observations are appended to history; calculated outputs are not inserted as new historical covariates. There is no automatic dependency-stage materialization graph. |
| Feature library README | The reference engine is row-wise, scalar, and in memory, with no shared-subexpression optimization or automatic caching. Metadata and nested object descriptions do not supply those capabilities. |

**Working recommendation on records and trees:** use explicit dataclasses/feature objects as the nodes of an expression tree. A dataclass (the “struct” idea) is a record describing one operation, its settings, and its inputs; the tree describes how those input references connect, so the two structures work together. Existing `Difference`, `Ratio`, and `Shift` already form object trees. Temporal operators could extend this pattern, recipes could later be serialized to JSON, and an execution graph could reuse compatible shared calculations. This preserves the original AST intention as a design direction; it does not finalize an architecture or call for implementing a parser/framework now. (G; PF; D)

#### Initial execution policy: on-demand composition

**Confirmed preference:** compute a requested feature on demand and let it obtain/compute its required components internally. Dependency evaluation establishes the calculations needed for the result; output selection establishes the columns returned/exported as model inputs. If the caller requests only a Z-score, its rolling mean and rolling standard deviation are internal prerequisites and need not be listed separately or exposed in the final feature table. Either component can be returned as its own feature when explicitly requested. (D)

Prefer a straightforward implementation first. The user expects pandas operations to be adequate and does not want an early focus on computation time or optimization; no performance or benchmark result has been established. On-demand requests can operate on grouped/vectorized arrays or columns and do not imply a Python callback for every scalar row. Shared caching/deduplication can be considered later in response to measured needs. A first-phase DAG/cache project is not required by this preference; the original graph and dependency-reuse ideas remain available as later design material. (D)

Internal computation, returned columns, optional intermediate persistence/auditing, and stateful rating snapshots have separate roles. Detailed persistence/cache policy remains open, including which intermediate values to retain for inspection. The public request style preserves the existing snapshot/feature/target/run-metadata concepts. Named Glicko features still obtain chronological state through rating orchestration; requesting a scalar value should not trigger a cold replay of the full rating history. This is an execution preference for the planned local library, with no implementation authorized. (D)

#### Starter arithmetic composition: rolling Z-score

**User-proposed test:** form `z = (known input value - rolling mean) / rolling standard deviation`. A `Difference` combines the known-value feature and rolling mean; a `Ratio` divides by the matching rolling std. Corner and shot series are now selected as available feature definitions; the numerator observation and baseline membership remain to specify. (D)

**Proposed numerator:** standardize the latest already observed value of each corner/shot series against its eligible historical baseline. The exact observation remains a recipe choice. A pre-match feature cannot use the predicted match's realized statistics. Mean and std must share a baseline available at the cutoff. Including the latest observed value in that baseline versus excluding it remains explicit; either can be causal. A full N-match baseline excluding the numerator observation requires **N+1 available appearances**, but smaller eligible baselines remain allowed under the pilot's partial-window policy. (D)

**Inspected scaffold support:** `Difference` and `Ratio` compose child `Feature` objects, and `Lag`, `RollingMean`, and `RollingStd` already exist. These primitives can express the arithmetic pattern using their existing observation-count windows over raw history. This example does not require rolling over a computed child's historical outputs, and it does not implement or validate the broader temporal extension below. No new Z-score feature was implemented or executed. (PF)

`RollingStd` uses population standard deviation (`ddof=0`). The current rolling primitives require a complete, nonmissing window; unavailable windows produce `None`. `Ratio` produces `None` for missing inputs or an exactly zero denominator. These are inspected behaviors. The pilot's available-sample partial-window choice would require adapting the scaffold's full-window handling if these primitives are used; no code change has been made. Population/sample std and minimum-count choices, missing/zero-observation behavior, and zero-spread Z-score handling remain open alongside the input, shared baseline, window settings, and exact recipe. No normality assumption or predictive improvement has been established. (PF; D)

#### Deferred temporal composition: stability of average goals conceded

For the football example, **provisionally define stability as the rolling standard deviation of a series of rolling mean goals conceded**. This measures variation in estimated defensive form. A rolling standard deviation of raw goals conceded measures a different quantity. A low variation can coexist with a high conceding average, so stability alone does not establish defensive quality. The exact metric and whether to include this composed feature in an initial experiment remain open. (D)

| Stage | Calculation in this conceptual example | Configurable history |
| --- | --- | --- |
| Inner feature | Rolling mean of the focal team's eligible goals-conceded observations. | Its own window type and length, warm-start policy, and missing-history policy. |
| Outer feature | Rolling standard deviation of eligible historical values produced by the inner feature. | A separate window type and length, with explicit handling of missing or warm-started child values. |

This nested temporal calculation needs an extension. One **working proposal** is a temporal operator that evaluates its child at the appropriate historical contexts. Another is explicit causal materialization of the child's time series, followed by the outer calculation. Both fit the original prerequisites, retained snapshot intermediates, and expression-tree ideas; neither is existing generic nested-rolling support. (PF; G; D)

Historical child outputs must retain entity/opponent alignment, information cutoff, definition/version, and their warm-start/missing-history policy. The total history requirement can compound across temporal stages; it is not generally just the larger window. Define effective timestamps, availability, and interval endpoints for both stages before applying shifts. An already causal child snapshot should not receive an extra lag automatically. These are requirements for the proposed generalization, not completed implementation choices.

### Future model-derived features

**Confirmed scope and priority:** beyond the intended pilot's rating-state integration, the eventual system should support a general workflow for learned/model-derived features extracted from fitted or pretrained auxiliary models. The user's example is a small Poisson regression fitted to corner data, supplying a corner-strength estimate or expected corner count to downstream XGBoost. Expected count predictions and team-specific fitted strength parameters are different possible outputs; their meaning, granularity, and statistical model remain open. Poisson regression is an example, with no selected model or validated Poisson distribution for corner counts. **This auxiliary-model extension is future work, excluded from the first experiment; no implementation is authorized.** (D)

**Working proposal:** a fitted component emits named feature values with match/team/time identifiers and other needed metadata. Ordinary feature composition can subsequently consume those values. Keep fitting separate from applying frozen fitted state: direct extraction does not require AST nesting, training inside composition, or retraining at each scalar feature computation. The original `teacher`/pretrained-output template idea remains relevant without fixing this extension's API. (G; D)

Current-source findings establish an extension point, rather than a complete learned-feature workflow:

| Source | Observed boundary |
| --- | --- |
| `FeatureBuilder` contract (LF) | `fit` learns optional state from eligible training rows/targets; `transform` must preserve each row's information cutoff without accessing targets or updating learned state. |
| `FeatureExtractor` and `Feature.compute` (PF) | The concrete extractor is stateless: `fit` checks alignment and learns nothing. Ordinary feature computation is required to be deterministic and side-effect-free. |
| Fold runner (LF) | A fresh builder is fitted on outer-fold training data and then transforms that training data. This call sequence alone does not construct temporal cross-fitted training features from a supervised auxiliary model. |
| Feature-construction guidance (LF) | Learned embeddings/transforms can use the fitting hook; supervised training features may need out-of-fold construction within the training window. |

These are source observations, not runtime verification or an implementation request. The following temporal and provenance choices are **future design proposals**:

- For an auxiliary supervised model trained on this football dataset, build downstream training features through earlier-data walk-forward fitting/cross-fitting. Each row's generated values must exclude its own outcome and respect its prediction cutoff; excluding the row alone is insufficient if later information enters fitting. Keep validation/test outcomes out of fitting in the evaluation where they are held out, and apply or refit using only eligible prior information under an explicit schedule.
- An externally pretrained frozen model is reusable only if its training provenance and availability are admissible at the simulated cutoffs. The word “pretrained” alone does not establish this.
- Retain model versions/references, fitting scope and cutoff, input recipe, and output definitions alongside feature provenance and snapshots. Do not apply a later fitted model retrospectively to earlier rows and represent its outputs as values known then.

Detailed auxiliary-model APIs, output definitions, cross-fitting/burn-in and fitting/application schedules remain future work. Both prediction-timing modes retain their cutoff rules. This extension does not change the initial selectable corner/shot feature library, match-result Glicko, or feature-owned history policy. (D)

Scikit-learn's [StackingRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html) illustrates using model predictions as downstream features. Its default regression splitting uses ordinary KFold, and it warns about overfitting when prefit base models and the stacking model use the same training data. It is not established as a drop-in replacement for the proposed temporal workflow.

### Existing reshaping helper

Source inspection of R shows `to_team_match_long` converting match-wide home/away columns into two team records. It maps focal-team values to `team_*`, optionally maps selected opposing values to `opponent_*`, carries shared context, and sets `team_id` and `opponent_id`.

Its docstring explicitly says it **does not add or modify `is_home`**; an existing value is passed through as shared context. Its fallback creates `match_order` from input row order when no usable explicit match-ID column is supplied. That fallback does not establish a stable match identity across reordering or independent loads. The temporary side-order marker is not a retained venue field.

This helper is a possible starting point. It has been inspected, not executed or validated, and should not be treated as a certified solution to the team-history, venue, or identity questions.

### Existing Glicko core and snapshots

**Confirmed meaning and order:** first integrate Glicko based on **match win/draw/loss results**. The prediction target remains total match corner kicks. Adapting separate ratings to other statistics, including corners, is desired future extensibility rather than a second rating required in the pilot. (D)

**Confirmed update trigger:** update immediately once a match is completed and its result is available, without waiting for the competition round or validation fold to finish. Compute both teams' new ratings from the same pair of immediately preceding eligible states, then retain the resulting states/snapshots with availability and provenance. Do not update the second team against the first team's already changed rating. Future predictions retrieve state available at their own cutoffs; previously issued per-match predictions stay unchanged, and round-batch predictions remain frozen even while internal ratings continue updating. (D)

**Proposed named rating interfaces:** the user suggests dedicated library features for each application, illustratively `MatchResultGlicko` and `CornerGlicko`. These names are not a finalized API or implemented code. They describe local library interfaces; no new network API or service is requested. Match-result Glicko remains in the intended pilot, while corner Glicko remains a future possibility to validate. (D)

**Working architecture proposal:** reuse one Glicko calculation implementation beneath separate rating definitions/feature interfaces. Each definition specifies its outcome extraction rule and configuration and owns an independent stream of team states/snapshots. Match-result Glicko maps football win/draw/loss; a future corner definition could map more/equal/fewer corners to win/draw/loss. Public feature names identify what is requested, shared algorithm code provides the calculation, and per-definition state records that definition's history. Keep the histories separate without duplicating the mathematics. Separate configured engine instances are an optional internal implementation choice, compatible with both code reuse and isolated state. Retain all state required by the algorithm even when a model exports only selected fields. (D)

**Proposed feature-request flow:** an experiment requests a named rating feature; rating orchestration supplies the appropriate team snapshots, and feature assembly attaches eligible home/away values at each prediction cutoff. Scalar feature evaluation should extract the appropriate snapshot values, without silently replaying or refitting a full history on every call. History reconstruction and rating updates belong to explicit orchestration. Exact public names/API, cache/storage schema, snapshot keys, event ordering/replay, and scope remain open; no implementation is authorized. (D)

GR contains a frozen `Rating` record with `mu`, `phi`, and `sigma`; a `Glicko2` class; rating construction and scale-conversion helpers; and `rate` for a sequence of results. Public API ratings use the public scale for `mu` and `phi`, with defaults 1500 and 350. The internal conversion helpers use different coordinates. Matching field names must not silently equate a saved public-scale rating with standard internal Glicko-2 coordinates.

The one-match helpers compute both updated ratings using both original ratings. `rate_1vs1_result` accepts an explicit result from the first team's perspective, and `rate_1vs1_scores` accepts paired scores. The legacy `rate_1vs1` assumes its first argument wins unless `drawn=True`; a football adapter must not infer the winner merely from home/away argument order. `advance_periods` inflates uncertainty over effective idle periods, allowing fractional periods and an optional cap at the prior uncertainty.

**Working integration proposal:** use the rating core beneath retained team snapshots. Consume each completed match's win/draw/loss result when it becomes available. Retrieve the pair of immediately preceding eligible states, compute both updates against that same old pair, and retain the resulting states with effective/availability information and provenance. Attach eligible home and away snapshots at each prediction cutoff. The existing file does not orchestrate football periods, persistence, point-in-time joins, or league/season priors. The meaning of an effective idle period and prevention of double-counting periods when combining `advance_periods` with `rate`'s built-in period inflation remain open. Feature construction owns the relevant warm-start policy; model training may independently span several seasons. (GR; D)

The [official Glicko-2 specification](https://www.glicko.net/glicko/glicko2.pdf) groups games into rating periods. The intended football workflow now uses event-driven, match-by-match updates; its mapping to Glicko-2 uncertainty/volatility time units and idle-time inflation still needs a consistent definition without double counting. Immediate updates remain separate from feature windows, prediction issuance, CV, and model refitting. No Glicko algorithm validation was run during this discussion.

**Remaining availability/replay details:** ordering simultaneous or revised results, exact completion/result-availability fields, and replay policy remain open. The completed collector follow-up established scheduled/revised kickoff and observation times, not verified actual completion or first-whistle timestamps. These notes do not assume a historical exact match-end time was collected or request another collector change. (D)

**Proposed future corner adaptation:** compare the two teams' corner counts, mapping more corners to a comparison win (`1`), equal corners to a draw (`0.5`), and fewer corners to a loss (`0`) for the existing core. This would measure relative corner-comparison performance and discard the margin: 6-5 and 10-0 both count as wins. It is not an expected-total-corners estimator and has not been validated. Keep it separate from match-result ratings in both state and definition. A future corner-rating update must wait for its required corner observation to become available; match completion alone does not supply missing statistics. The exact future statistic mapping, strength/ranking interpretation, and alternatives remain open; this proposal neither adds a second pilot rating nor changes the target into a corner-winner classifier. (D)

`quality_1vs1` is documented as an even-match heuristic. It is not a calibrated win/draw/loss probability model. A specific source-level concern is that equal `phi` values make its two expected scores complementary, so its expression algebraically returns 1 even when `mu` values differ. Check this before relying on that heuristic. This observation is not a runtime test or an algorithm audit.

### Utility inventory and reuse assessment

The inventory covered **ten source files** in the saved-project `utils` directory at the earlier inspection. No nested source files were found. Generated `__pycache__` files were present and excluded. Each source was read sufficiently to assess its role; imports/callers were checked narrowly where relevant. No utility was imported, executed, benchmarked, edited, or deleted for this review. Assessments concern suitability for the intended scientific pipeline, not age alone. (U)

This inventory records an earlier inspection while **Review Sofascore scraping refactors** was changing the directory. At that inspection, `football_dataset_xDiyoFire.py` was modified and `data_loading.py`, `team_seasons.py`, and `tensor_inputs.py` were untracked additions. The dataset constructor changed and `tensor_inputs.py` appeared during that review. The initial collector refactor is now reported complete. Only the season-entry entries below have received a targeted current source refresh; the other entries and refactor-status labels retain the earlier snapshot.

**Season-entry refresh for the first pipeline:** promotion/relegation flags were already present in the original snapshot concept and as optional inputs in the loader plan. Current `team_seasons` code derives entry into a season from ID-based membership evidence, preserves unknown states, and checks many-to-one joins for both match sides. Ordinary promotion/relegation inference requires a complete predecessor membership for the destination competition and an observed origin in one adjacent tier of the same league system; explicit evidence-bearing overrides are supported. The [movement builder](C:/Users/luisi/Documents/Programming/Python/xDiyo/scraping/sofascore/movements.py:38) still gathers memberships from legacy CSVs, although it exports a versioned Parquet flag table with a hash-checked reader. Adapt membership input to the new prepared exports; do not reinstate a legacy-data requirement. Retrospective match appearances do not by themselves verify pregame availability. The new pipeline records this context step while predictor selection, timing and any rating adjustment remain open. No utility was executed or changed in this refresh. (U; D)

Utility-level contracts described in this inventory are observations of the inspected source. They do not establish the final collection-output schema or collector-to-feature mapping; those remain subject to the deferred collection dependency above.

| File | Purpose | Reuse assessment |
| --- | --- | --- |
| [data_loading.py](C:/Users/luisi/Documents/Programming/Python/xDiyo/utils/data_loading.py) | Merge legacy/versioned seasons, normalize context, and retain missingness information without importing Torch. | **Adapt, current refactor:** useful loading boundary; whole-season maximum normalization and prediction-time availability need explicit policies. |
| [football_dataset_xDiyoFire.py](C:/Users/luisi/Documents/Programming/Python/xDiyo/utils/football_dataset_xDiyoFire.py) | Build football histories, multimodal PyTorch samples, batches/loaders, league moments, and sample caches. | **Deferred rework candidate, actively modified:** excluded from the initial tabular implementation; lowest refactor priority for later neural models. |
| [football_features.py](C:/Users/luisi/Documents/Programming/Python/xDiyo/utils/football_features.py) | Empty feature-module placeholder. | **Retirement candidate:** zero bytes and no implementation to reuse; retaining the name for future work is also possible. |
| [glicko_rating.py](C:/Users/luisi/Documents/Programming/Python/xDiyo/utils/glicko_rating.py) | Rating records, Glicko-style updates, scale conversion, idle-period inflation, and a quality heuristic. | **Reuse candidate, needs validation:** a useful rating core beneath snapshots; period/timing integration and the heuristic require attention. |
| [promotions_demotions.py](C:/Users/luisi/Documents/Programming/Python/xDiyo/utils/promotions_demotions.py) | Compatibility entry point for the ID-based movement builder, explicit season-pinned seeds/evidence, and optional legacy CSV exports. | **Targeted refresh: retain as a compatibility wrapper.** The old name-set implementation is archived in `scraping/legacy`; reuse the underlying core with a new-export adapter rather than treating this wrapper as that old implementation. |
| [reshaping.py](C:/Users/luisi/Documents/Programming/Python/xDiyo/utils/reshaping.py) | Convert home/away match columns into two focal-team/opponent records. | **Adapt:** suitable starting shape; stable match identity and explicit venue handling still need work. |
| [team_seasons.py](C:/Users/luisi/Documents/Programming/Python/xDiyo/utils/team_seasons.py) | Derive season-entry movement from ID-based membership evidence and join nullable home/away flags to matches. | **Targeted refresh: reuse candidate.** Explicit evidence, unknown states, and checked joins fit season context; adapt membership inputs and establish prediction-time availability. Source inspected, not executed. |
| [tensor_inputs.py](C:/Users/luisi/Documents/Programming/Python/xDiyo/utils/tensor_inputs.py) | Convert absent/nonfinite numeric values to placeholders and carry masks for facts, padding, shots, and heatmaps. | **Reuse candidate, current refactor:** useful model-boundary missingness handling; it preserves the final label slot and does not enforce prediction cutoffs. |
| [utils_xDiyo.py](C:/Users/luisi/Documents/Programming/Python/xDiyo/utils/utils_xDiyo.py) | Shared-moment scaling, sequence alignment, attention, graph edges, threshold flags, and feature gating. | **Consolidation/retirement candidate:** substantial overlap with the league-aware variant; preserve distinct scaling APIs until caller compatibility is checked. |
| [utils_xDiyo_wleague.py](C:/Users/luisi/Documents/Programming/Python/xDiyo/utils/utils_xDiyo_wleague.py) | League-aware scaling/inversion, shared neural utilities, heatmap plots, and pickle caches. | **Adapt for later models:** components referenced in a notebook remain reuse candidates; fit moments within eligible folds and extract only needed pieces. |

The loading wrapper in the dataset now delegates to `data_loading.merge_seasons`. The current movement workflow imports `team_seasons`; the older promotion notebook imports `promotions_demotions`. The [historical heatmap notebook](C:/Users/luisi/Documents/Programming/Python/xDiyo/data/xDiyo_data_heatmaps/xDiyo_eyes_revamp_heatmaps_2.ipynb) contains dataset/league-utility imports and Dataset construction. Those references identify compatibility considerations; they do not establish how any downstream model currently uses the data. No downstream model behavior or historical leakage has been established. The `availability` masks in `tensor_inputs` describe missing/padded values, not the publication timestamps required for causal forecasting.

Overall, the directory contains useful building blocks, especially membership evidence, reshaping, missingness handling, and rating arithmetic. The loading normalization and model-specific data packing need adaptation to the experimental information rules. The empty feature placeholder does not implement the initial rolling means/stds or warm starts. Concrete future cleanup candidates are that placeholder and duplicated neural utility code after consolidation. The original name-based movement code is already archived; the current compatibility wrapper has the refreshed assessment above. None is a recommendation to delete files during this planning work.

### Dataset assessment for later neural models

**Confirmed priority:** Dataset rework is the lowest priority and is deferred. The initial tabular experiment does not need this PyTorch Dataset. The assessment below preserves context for a later revisit; it is not the next implementation task.

**Confirmed intent:** `FootballDataset` should be able to serve several neural-network architectures. The user deliberately appends the current/final match to each sequence so labels can be constructed by removing that final slice. This joint packaging is an established convention, not an accidental inclusion or evidence that previous experiments leaked. (D)

**Assessment:** substantial rework of the data-access structure is justified for that later goal, while extracting and validating useful football routines. The strongest source observations are:

- **Identity and chronology:** the latest constructor resets the frame index, so the previously observed non-default-index `.loc[idx]` mismatch no longer applies as described. Histories still use row positions as local match identifiers, accumulate in input order, and sort selected histories by those identifiers. That does not establish stable provider identity, actual match-time order, or result availability for CV subsets.
- **Deliberate label packaging:** retrieval includes the current match; `process_team` decrements the reported historical length while retaining the final scores/statistics/heatmaps, and the sequence mask marks the final slot valid. Companion utilities provide `split_current`. The architectural question is how every model consistently routes final observed statistics to labels while retaining any legitimately available prediction context. Missingness masks do not enforce that routing.
- **Repeated preparation:** history construction processes team/opponent heatmap views for both sides; each sample access scans histories and rebuilds many overlapping lists, pads modalities, and then collates/converts them into tensors. This suggests opportunities to share preparation across samples. There are no benchmarks or measured speedup claims.
- **Many responsibilities:** although loading now delegates to a separate boundary, the module still mixes football extraction, historical sequence construction, tensor formatting, batching/loader settings, and sample caching. Different architectures cannot easily request only the representations/modalities they need.
- **Cache alignment:** `precompute_data` writes overlapping expanded samples individually, logs errors and continues, and does not write a top-level stable match/cutoff/configuration manifest. `PrecomputedFootballDataset` sorts filenames lexicographically, so `sample_10.pt` precedes `sample_2.pt`. Enumeration therefore cannot serve as a dependable chronological match index.

**Proposed future direction:** prepare shared match/team data and causal snapshot/history indices; use a small Dataset to select prepared inputs together with separately identified targets and metadata; and let a collator/model adapter control modalities, padding, masks, and tensor shapes. The final-label split could be performed consistently at that boundary, or enforced by a shared splitter used by every model under the existing packaging convention. CV selection must retain stable match identity, and feature construction must retain ownership of season warm-up.

This recommendation remains deferred at the user's lowest refactor priority. The tabular corner-count pilot with available-sample rolling summaries, normalized standings, match-result Glicko, and feature-owned history/state policies takes precedence; no Dataset implementation work is authorized here. Retaining useful routines and existing experimental references does not require retaining every part of the current Dataset structure.

## Open decisions and source discrepancies

### Decisions for the continuing discussion

| Decision | What is still open |
| --- | --- |
| **Reusable library and experiment interface** | Discuss league histories/LOO and optional season transitions first, then return to orchestration before configuring the corner pilot. The shared X/y/metadata/prediction-unit contract remains the first orchestration layer, initially supporting match and team-match layouts and group identities. Exact APIs, defaults, GUI and distribution remain open; the skeleton must support different labels, models and layouts. |
| **Prediction issuance** | Both modes are confirmed as selectable settings to compare: before each match with its own cutoff, and all predictions for a competition round before its first match with a frozen cutoff for that batch. Exact lead times, round schedule/postponement conventions, data-availability details, and implementation remain open. |
| **First prediction experiment** | Total match corners with Lasso; selectable corner/shot lag and rolling features plus match-result Glicko, no seasonal blending, pre/post reports and configurable bet-option evaluation. The selected population is all 13 available leagues for 2022/23–2024/25. Feature subset, CV settings, regularization, baseline, metrics, strategy and statistical edge policies remain open. |
| **Starting-feature definitions** | Lags, means, stds, latest-included Z-scores, H2H, context and optional EMA are implemented and verified. Default scope is team+competition across venues/seasons, with custom groups supported. Full-population standings normalization and zero fallback are chosen. Exact shot statistic, recipe subset, separate plain-mean population and extra Glicko outputs remain open. |
| **Partial-history edge policies** | Implemented: match-count windows with finite-value reductions, non-skipping lags, min_periods=1, ddof=1 with n > ddof, and missing undefined/zero-spread Z-scores. Standing fallback is zero. Recipe overrides, sample-count/flag outputs and model-input missingness treatment remain separate decisions. |
| **Corner-count target definition** | Total match corners remains the Lasso regression target. Match-duration scope and collector mapping remain open. Configurable under/over options have separately derived evaluation outcomes; integer-line settlement and strategy rules remain open. No probability distribution or predictive-quantile output is supplied by ordinary Lasso. |
| **Collection inputs and queries** | Corrected publications in `data/xDiyo_data` are inspected and the chosen population imported with saved versions. Feature input fields, periods, groups/keys and historical availability still need explicit choices. Current exports alone do not finalize the feature contract; legacy-format support is not required. |
| **Analytics loader** | `load_seasons`, `load_season` and `inspect_season` are implemented and verified. Separate nullable tables retain source observations; multi-season output adds origin columns. The first import covers 39 publications, with selections saved outside the source. Automatic feature-dependency queries, joins and feature/rating integration remain future work. |
| **Model reuse and interfaces** | Lasso is the first model family; a model-adapter boundary should support other models without rewriting orchestration/reporting. Importance providers declare methods and named outputs; unsupported capabilities are explicit. Neural adapters and optional skorch remain later choices. |
| **Feature window policies** | Type/length remain selectable; the first catalog includes lags 1-5 and configurable rolling summaries. Eligible history may cross seasons. League design now fixes the window before LOO, without refill; incomplete rounds retain the last completed baseline, frozen for the target round. Observation units, endpoints, venue/competition scope and postponed-round completion rules remain explicit. |
| **Season and league transitions** | No-warm-up/available-history remains the default. Optional shared transition/prior helpers are proposed to apply each feature-specific policy once. Reuse ID-based movement evidence with new exports; freeze destination top/bottom priors before transition, excluding the mover, with configurable standings (preferred default) or rating ranking. Exact adjustments, state transfer and cross-league rating calibration remain open. Boundaries do not automatically reset models or CV. |
| **Optional prior/exponential strategy** | `warmup=None` remains the default. Optional seeded EMA uses explicit mean priors, blend and decay, then hands off after 1/2/3 completed rounds; one round when enabled is only a tentative recommendation. Variability uses matching prior-season league data (destination league for movers), with consistent mean/spread updates and no silent zero for missing variance. Weighting, sample correction and fallback remain open. The original `LinearWarmStart` remains illustrative; James-Stein is deferred. Glicko retains its own update rules. |
| **Identifiers and ordering** | Stable match/team identity, source mappings, chronological order, simultaneous events, and the role of the original compound keys. |
| **Rating-state timing** | Implemented: availability-ordered event replay, equal-release batches with shared prior states, cutoff-aware lookup, cumulative own/opponent-state provenance, competition scope across seasons and explicit optional idle advancement. Actual source completion/publication times, calendar-period mappings, revisions/incremental updates and league/season priors remain separate work. |
| **Named rating interfaces** | Implemented and verified: `MatchResultGlicko`, general `StatGlicko(Stat(...))`, `build_ratings`, `RatingRun.save/load` and `Rating(name)`. The unchanged legacy numerical engine is shared while each definition/period retains independent state. Full fields persist independently of model selection. Graph producers and broader orchestration remain future work. |
| **Statistic-based ratings** | Corner/other statistic comparisons now map greater/equal/less to1/.5/0, optionally reversed for lower-is-better; missing pairs skip updates and periods are separate. This comparative rating discards margins and is not a count forecast. Exact pilot inclusion and alternative rating definitions remain selectable. |
| **Historical grouping and alignment** | The first evaluator defaults to team+competition across venues/seasons; custom grouping is supported and H2H adds ordered opponent identity. Target identity and cutoff ties are excluded. Confirmed league design excludes focal-team contributions after selecting the window, with no refill, and recomputes pooled individual-observation moments/count; whole-fixture exclusion is optional. Exact observation units and wrapper APIs remain pending. Match-wide assembly remains later work. |
| **Availability and revisions** | When statistics, standings, odds/context, and labels are usable; how later source corrections affect reconstruction of earlier inputs. |
| **Evaluation design** | Selectable expanding/sliding chronological windows, season and within-season blocks, season-stage scoring, season-grouped K-fold, and future CPCV strategy evaluation are recorded in the cross-validation section. Retrospective schemes remain distinct from historical deployment simulation. Exact spans, stage boundaries, stride, calendar alignment, refitting, label maturity, inner validation, scoring, CPCV dependencies/state reconstruction, and betting rules remain open. |
| **Configuration and orchestration** | Return to the reusable skeleton after the league/transition design discussion: prepared states, features/labels, assembly, temporal splits, training-only fitted transforms/model adapters and both report stages. Reuse prepared rating states across reports/models/folds. Explicit output selection remains the preference; exact APIs and optional intermediate persistence/auditing remain open. Pilot configuration, tuning and fitting are deferred. |
| **Temporal feature composition** | Nested temporal expressions are implemented: each historical child uses that row's cutoff before an outer window consumes it. Concrete stability metrics, recipe window choices, persisted intermediates and rating/model-derived state remain separate decisions. |
| **Rolling Z-score composition test** | Verified: latest eligible observation is included in the trailing baseline, mean/std share its finite values, default ddof=1/min_periods=1, and missing-latest/zero-spread output is missing. General temporal nesting is also verified. Pilot measures and window overrides remain selectable. |
| **Future model-derived features** | General auxiliary-model extraction is a confirmed future capability, excluded from the pilot. Named outputs from fitted/pretrained models may be consumed directly or by composition; training stays separate from frozen-state application. Output meaning/granularity, models, APIs, temporal cross-fitting/burn-in policy, provenance, and fitting/application schedules remain open. The Poisson-to-XGBoost example selects neither a distribution nor a model. |
| **Target interpretation** | Extensible labels include statistic-derived quantities, perspective W/D/L +1/0/-1, configurable binary outcomes and BetOption outcome/threshold labels with future parlay composition. Preserve draw/push/void/missing distinctions. Exact binary/settlement policies, transformations, fitted scopes/fallbacks and rank/bin interpretation remain open. |
| **Data visualization and reporting** | Both pre-training and post-training reporters are required. They consume the declared dataset layout and metadata, distinguish matches from team observations and respect paired/ranking groups; post-training adds predictions/model outputs. Bounded inspection and model-independent importance interfaces remain required. Quantile-binned point diagnostics, empirical quantile comparison and predictive-quantile calibration remain distinct; methods/renderers are still selectable. |
| **Saved state and user interface** | Storage layout, provenance/versioning and fitted-state persistence; timing and scope of a GUI or registry editor. |
| **Future neural Dataset** | Excluded from the initial tabular implementation and the lowest refactor priority. Preserve the multi-architecture goal and later questions about shared preparation, stable identities, deliberate final-label routing, modalities, and caches. |

These decisions can remain open while the design conversation continues. They do not block creation or use of this reference.

### Meaningful discrepancies and incomplete notation

| Source issue | Treatment in these notes |
| --- | --- |
| G/H, p. 1 include `match_order` in snapshot keys; H, p. 6 omits it for ratings. | Both proposals are retained. A stable identity and state-time contract has not been chosen. |
| G adds Pass O and says four passes, then leaves Pass A unfinished; H, pp. 5-6 describes three computation passes. | Preparation plus three computations is a possible reconciliation, labeled as interpretation. |
| H, p. 4 places window/feature before the statistic in names; G uses `prefix_stat_(parameters)feature`. | Preserve both as historical alternatives. G also incorporates the general-parameters and optional-period refinements from H, p. 14. |
| Source examples mix `team`/`opponent`, `home`/`away`, and older `enemy` naming; a period-free example is `home_standing`. | The distinction between focal/opponent history and export venue is retained. Final aliases and parsing rules remain open. |
| G alternates `requires`/“requirements,” `parameters`/`params`, and feature `inputs`/target `input`; H uses labels such as `Op` and `Args`. | Reference fields follow G's main headings and describe their meaning. These spellings are not a finalized serialization contract. |
| G links season partitions to a warm-start choice; D distinguishes continuity, reset, and borrowing-strength policies. | The original partition examples are preserved below without silently equating those policies. |
| C uses `season_warmup` for supervised row allocation, advances by `step_seasons`, and includes all earlier seasons in training. | D assigns season warm-up to feature construction and requires configurable round-based CV with a sliding training window that can span several seasons. The prototype is preserved; its warm-up ownership is not the intended design. |
| G's feature-builder prose describes rolling means; its printed AST connects `RawFeature` nodes directly to ratios and contains incomplete pseudo-JSON syntax. | Preserve the intended composition and identify the mismatch. No executable corrected tree is manufactured. |
| H, p. 2 proposes inversion for examples including ranks and bins. | Preserve transformation provenance while recording that lossy transformations need an explicit interpretation/reconstruction policy. |
| Names are proposed as the means of identifying reusable prerequisites. | Whether names encode every relevant parameter, partition, lag, warm-start choice, fitting scope, and version is unresolved. Reuse compatibility cannot be assumed from a matching label. |

## Reference: templates, operators, and configuration

This section records the substantial template detail from G, with handwritten additions identified by page. The extensive library is retained as a future direction beyond the confirmed small first experiment. Examples describe intent and do not constitute an executable JSON schema, completed API, or mandatory framework.

### Libraries, registries, GUI, and placeholder conventions

The original design has three ingredients: a library of operators and operator building blocks, templates describing how to configure those operators, and an interface for selecting/filling templates. Separate registries could expose kernels, aggregations, partitions, and other building blocks. Operators expose metadata so an auditor can check that a requested combination is supported. Pandas and JSON are the original choices for calculations and template representation. (G; H, p. 3)

The GUI would choose existing templates and fill blank or selectable fields; G also discusses creating a template with a predefined structure. H, p. 3 treats creation/deletion of registered templates as an extension. H, p. 11 proposes operator metadata for supported options, parameters, and building blocks, a command returning a registry copy for local editing, and entries conceptually containing `(group, name, callable)`. None of those notes authorizes registry modifications during this documentation task.

| Original marker/convention | Intended behavior |
| --- | --- |
| `>parseable<` | Ask for a value using the field key as the UI label, then replace the marker with the supplied value. The example is a lookback parameter such as `weeks_back`. |
| `>selection<` | Offer checkboxes and pass the selected values as a list. |
| Alternatives separated by a vertical bar | Choose one value and replace the alternatives with that value. |
| `>retrieve_location` and examples such as `>retrieve_parameters_window` | Resolve a value from the indicated template location after the user's fields are filled. The example retrieves `15` from a parameters mapping containing `window: 15`. |

G intends retrieval resolution to precede dependency-graph construction. The marker grammar, value types, and complete parsing rules are unfinished.

### Feature template fields

| Field | Original meaning and examples |
| --- | --- |
| `name` | Capitalized words without spaces, such as `RollingMean`, `ExponentialMean`, or `StatRatioDifference`. |
| `family` | `league`, `team`, `rating`, `headtohead`, or `postprocessing`. Family and warm-start status influence partition choices. Shrinkage and warm-start operators are examples of postprocessing. |
| `kernel` | `rolling`, `expanding`, `snapshot`, `point`, `expression`, or `teacher`. Expression features use custom composition; teacher features use pretrained-model outputs. |
| `parameters` | A general mapping of operator settings, such as window length and smoothing. This incorporates H, p. 14's replacement of a window-only field with general parameters. |
| `partition` | Grouping that determines the history/population over which an operator acts. G describes Pandas grouping/vectorization and possible choices based on family and warm-start status. |
| `inputs` | Map argument keywords to input columns. A list applies a univariate operator separately to each column; multivariate lists are paired by position. Dependencies can supply these mappings directly. |
| `aggregation` | An available operation such as `sum`, `count`, `mean`, `var`, `std`, or `rate`; `custom` is mentioned for teacher modules. Expression kernels use an AST to describe composition. |
| `output` | A proposed generated column name encoding source/prefix, statistic, parameters, and feature name. See naming details below. |
| `requires` | Prerequisite feature descriptions, including the argument keyword receiving each computed result. Intermediate outputs may be retained in snapshots without being exported. |
| `is_loo` | Boolean declaring a leave-one-out feature. The unit excluded and eligible reference population need a separate definition. |
| `shift_policy` | `nolag` or positive integers indicating feature delay. Intended to prevent future leakage or use more distant history. The underlying step/time unit needs a policy. |
| `warm_start_blend` | The warm-start/shrinkage operator and settings. The example is `LinearWarmStart` with an `alpha_schedule` running from round 1 to round 6. Proposed outputs include weights and the adjusted feature. |
| `audit` | Consistency and validity settings such as `min_count: 3` and `allow_zero_var: False`. These are examples of configurable checks. |

The presence of `nolag` is not evidence that every unshifted input is available at prediction time. Teacher outputs similarly require eligible training/pretraining information and provenance; a pretrained model is not automatically safe for every historical evaluation. These are timing qualifications from D/P, rather than completed original template rules.

### Original partition examples

G gives the following choices, using its own labels:

| Family | “No warm-start” example | “Warm-start” example |
| --- | --- | --- |
| League | `[league]` | `[league, season_start]` |
| Team | `[team_id]` | `[team_id, season_start]` |
| Head-to-head | `[team_id, opponent_id]` | `[team_id, opponent_id, season_start]` |

The template may expose choices, and operator logic would choose according to the warm-start setting. More partitions were envisaged. These examples do not fully specify how previous-season information enters a season-partitioned calculation, or how league transitions interact with continuous team form. There is no complete partition rule for every listed family.

### Input argument pairing

For a univariate feature, G consistently uses the keyword `x`. Selecting `team_45_min_Goals` and `team_45_min_Cornerkicks` means one application/output for each selected column.

For a bivariate feature, the illustrated input lists are paired in their stated order:

| Application | Argument `x` | Argument `y` |
| --- | --- | --- |
| 1 | `team_45_min_Goals` | `opponent_45_min_Goals` |
| 2 | `team_45_min_Cornerkicks` | `opponent_45_min_Cornerkicks` |

This is positional pairing, not an instruction to evaluate every possible cross-pair. The source does not define behavior for incompatible list lengths. For a template requested through another template's requirements, the orchestrator would pass the argument mapping directly instead of asking for another GUI selection.

### Input and output names

G proposes input names of the form `prefix_stat`, with `team` or `opponent` as prefixes. The statistic may include a period segment such as `45_min`, `90_min`, or `total`, as in `opponent_45_min_Goalkeepersaves`. Some context/statistics omit a period; the GUI would need to recognize which sources require one.

The proposed output convention in G is `prefix_stat_(parameters)feature`. A `RollingMean` with a lookback of 15 would encode that value in the name. With several parameters, the source suggests collecting their values, such as `(15,20,8)`. Parameter ordering, disambiguation, and which additional settings belong in an identity are unfinished.

H, p. 4 instead explores names built from a home/away prefix, window and feature name, optional time/half, statistic, and optionally an explicit focal/opponent side. Examples concern a rolling mean with window marker 5 for a `45_min` shots-on-goal source, a rolling variance of offsides, and a raw player heatmap. A windowless raw feature is illustrated with a default window marker of 1. These are historical naming examples; they do not settle the window unit or establish a heatmap feature implementation.

Context listed on that page includes standings, FIFA score, promotion/demotion flags, normalized round, and league index. The meaning and source of “FIFA score” are not further specified. H, p. 14 considers handling period-free statistics separately as context or naming sources explicitly while making the UI aware of optional period labels. G already incorporates optional periods and general parameters, without settling all naming alternatives.

### Requirements, reuse, and auditing

G and H, pp. 9-10 describe requirements as a mapping from prerequisite calculations to the consuming operator's argument keywords. The z-score example uses a rolling mean for `mu` and a rolling standard deviation for `sigma` in `(x - mu) / sigma`.

Each prerequisite identifies the feature/operator, family, parameters, and other needed settings. Its source arguments can be explicit; otherwise G says they default to the parent feature's input. Retrieval markers can inherit settings such as the parent's window. H, pp. 9-10 emphasizes carrying grouping/alignment and warm-start/shrinkage choices, including whether the supplied moments are already adjusted.

The proposed orchestration sequence is:

1. Resolve the required calculation and the output name it would produce.
2. Check whether an equivalent prerequisite is already supplied by another selected/computed feature.
3. Reuse it when compatible; otherwise compute its requirements and then the missing prerequisite.
4. Store the result in snapshots and pass it under the requested argument keyword.
5. Export it only if the explicit feature selection includes it.

The originals frame the lookup primarily in terms of output names. Compatibility needs further definition, as recorded in the discrepancy table; a mean from a different partition, time scope, or configuration cannot automatically substitute for the requested mean. Operator metadata is intended to let the auditor check whether templates request supported settings. It does not by itself establish statistical validity or information availability.

### Target template fields

| Field | Original meaning and examples |
| --- | --- |
| `name` | Capitalized words without spaces, such as `BettingOption`, `Winner`, `Rank`, `SingleStat`, `MultiStat`, `MatchTotal`, or `MultiTotals`. |
| `family` | `team`, `match`, `bets`, `odds`, or `rank`; the family helps determine the target's grouping/partition. |
| `input` | Input/argument selection analogous to feature-template inputs. G uses the singular field name here. |
| `scaling` | Examples: `None`, `league_zscore`, `league_minmax`, `team_zscore`, `team_minmax`, and `quantiles`. Scalers are envisioned as library operators that can reuse already computed compatible moments. |
| `scaler_partition` | Population used for scaler calculations. When warm starting, adjust scaler moments/priors rather than the raw target statistic. |
| `scaler_window` | Scaler lookback. G proposes aligning it with feature-engineering windows; the relationship to multiple feature windows and fitted scope is unresolved. |
| `scaler_shift` | G gives `nolag` for debugging or `1` as alternatives intended to avoid future information in the scaler. Label creation and scaler estimation still need separate timing rules. |
| `transform` | Optional library operation applied to the raw target before scaling; default `None`. |
| `export_mode` | `stacked` for long output or `match_wide` for wide output. |
| `audit` | Consistency and fallback settings. The text mentions an unscaled target or global prior when early-round variance collapses, and sketches `min_std` with `fallback: use_mean_only`. Exact behavior is unfinished. |

| Target family | Intended objective in G |
| --- | --- |
| `team` | A statistic or group of statistics for one team in a match. |
| `match` | Pooled statistics of both teams, such as home corners plus away corners. |
| `bets` | Classification labels for betting options, such as more than 1.5 goals, more than 2.5 goals, or less than 2.5 goals. |
| `odds` | Construct the system's own odds, with logit models mentioned as a possible pairing. |
| `rank` | Relative team positions and tournament-winner objectives. |

The target families describe possibilities. They do not specify market rules, probability-to-odds conversion, a ranking training objective, or a selected modeling approach. The inverse/interpretation issue and requirement to retain transformation information are recorded in the artifact section.

### Expression feature builder

G reserves the `expression` kernel for composition using an Abstract Syntax Tree (AST), with operations drawn from the operator library. The intended example is a difference between two ratios of rolling means, corresponding to the scored/conceded comparison above. Raw-source leaves are labeled `RawFeature`, allowing the GUI/parser to collect input names from the tree.

The listed inputs are `team_corners_infavor`, `team_corners_against`, `opponent_corners_infavor`, and `opponent_corners_against`. H, pp. 7-9 adds the crucial opponent-history alignment via `PivotOpponent`. The printed AST in G does not fully express the rolling operations described in its prose and is not valid finished JSON. Operator argument types, nesting, prerequisite resolution, output naming, and missing/zero-value handling remain to be specified.

## Handwritten source navigation

The consolidated sections above retain the substance needed for ordinary discussion. This map supports targeted source checks if an exact handwritten detail later matters.

| PDF page | Content retained here |
| --- | --- |
| 1 | Three data artifacts plus metadata; snapshot keys/columns; explicit export selection; feature-context joins; season/round CV note. |
| 2 | Pivot after joining; target keys and scaler columns; proposed inversion; run configuration and teacher metadata. |
| 3 | Metadata-based template compatibility checks; GUI selection and completion; template registry editing as an extension. |
| 4 | Alternative naming conventions, raw/rolling examples, optional side/time labels, and context fields. |
| 5 | League and team passes; shifted baselines; leave-one-out reuse; continuous form and league-change shrinkage. |
| 6 | Rating snapshots and differing key; season-boundary uncertainty inflation; promotion/demotion prior adjustment. |
| 7 | Scored/conceded distinction, the opponent's own history, GUI choices, and `PivotOpponent`. |
| 8 | Proposed grouped/multi-index opponent lookup and reattachment; unresolved implementation details. |
| 9 | Difference of scored/conceded rolling-mean ratios; prerequisite grouping/alignment and adjusted moments. |
| 10 | Requirement argument mapping, operator settings, dependency reuse, and retained intermediate columns. |
| 11 | Registry metadata, local registry copies, and conceptual `(group, name, callable)` entries. |
| 12 | League-mean versus James-Stein prior experiment; dependence questions involving home/away goals and corners. |
| 13 | Statistical tests as possible reusable operators. |
| 14 | Period-free naming alternatives and replacement of a window-only field with general parameters. |
