# Historical goals scored and conceded

`MatchScore(score_field="current", side="for")` is an observed match-score
source. It reads `goals_for` / `goals_against` from `build_team_history`, without
requiring a statistics table. Use it inside a historical operator:

```python
from xdiyo_analytics.features import MatchScore, RollingMean, evaluate_features

definitions = {
    "goals_for_r20": RollingMean(
        MatchScore(score_field="current", side="for"),
        window=20, min_periods=1, venue="all"),
    "goals_against_r20": RollingMean(
        MatchScore(score_field="current", side="against"),
        window=20, min_periods=1, venue="all"),
}
features = evaluate_features(history, definitions, keyed=True)
```

Match-layout assembly creates exactly these four predictors:

- `home::goals_for_r20`
- `home::goals_against_r20`
- `away::goals_for_r20`
- `away::goals_against_r20`

Adding these to an unchanged 31-column baseline gives 35 columns before fitted
missingness indicators. This extension adds no other derived predictors or odds.

## Meaning and missing observations

`current` is currently the only supported score basis. It means the native
`home_score_current` / `away_score_current` values, reversed when the focal team
played away. It is **not universally regulation time**. The leaf does not sum
halves, add extra time or shootouts, use display scores, or substitute xG.
Unsupported bases raise `ValueError`. Existing Outcome and Glicko score semantics
are unchanged. `history.attrs['score_columns']` records the current-score basis;
feature metadata and serialized expressions retain the explicit constructor.

Missing native fields become missing history values. Halves and penalty fields
are irrelevant to this resolution, even if they disagree with current. Nonfinite
scores become missing during temporal evaluation. Missing measures still consume
a match-window slot. Reductions require `min_periods` finite values; no eligible
history yields NaN. Imputation belongs to training-fold preprocessing.

`side="against"` means goals conceded by the focal team; `side="both"` emits
separate `::goals_for` and `::goals_against` outputs. `ForAgainst(MatchScore(), ... )`
can also select this perspective. Missing manually supplied history columns raise
`KeyError`; the evaluator never performs a hidden lookup or fallback.

## Temporal boundaries

Lag, rolling mean/std/Z-score and EMA reuse the normal eligibility engine.
By default history crosses seasons and venues, grouping by team and competition.
Sources must be finished, kick off strictly before the cutoff, be available no
later than it, and have a different match identity from the target.

For a frozen round protocol, pass existing explicit cutoff/availability inputs:

```python
import pandas as pd

round_keys = ["competition_id", "season_id", "tournament_id", "round"]
cutoffs = history.groupby(round_keys)["kickoff_at"].transform("min") - pd.Timedelta(hours=1)
available_at = history["kickoff_at"] + pd.Timedelta(hours=3)
features = evaluate_features(
    history, definitions, keyed=True, cutoffs=cutoffs, available_at=available_at)
```

The three-hour delay is an explicit proxy, not verified publication timing.
The common cutoff includes postponed fixtures and remains frozen for their round.
This example requires complete stage-round identities. Defaults remain the
documented earlier-finished-kickoff assumption when no explicit times are supplied.

Raw scores cannot be prediction features, including inside arithmetic, H2H or
WarmStart. They also cannot serve as a current-match Z-score reference. Historical
arithmetic and references such as `Lag(MatchScore())` are supported. H2H works
with historical score operators; `WarmStart(..., policy=None)` is a pass-through.
This compact recipe does not add or configure league populations or seeded priors;
direct `League(MatchScore())` and `SeededEMA` on a direct MatchScore source remain
outside this addition's supported interfaces.

## Builder and portable recipes

In **Features & ratings**, add **Rolling Mean**, set its source to **Match Score**,
select **Score basis: current**, **Side: for**, **Window: 20**, and
**Min periods: 1**. Repeat with **Side: against**. Use the two names above.
The venue control remains specific to heatmaps in the builder; scores default
to all venues. Score sources are separate from statistics bundles.

The native recipe fragment is:

```json
{
  "goals_for_r20": {
    "component": "features.RollingMean",
    "params": {
      "source": {"component": "features.MatchScore", "params": {"score_field": "current", "side": "for"}},
      "window": 20, "min_periods": 1, "venue": "all"
    }
  },
  "goals_against_r20": {
    "component": "features.RollingMean",
    "params": {
      "source": {"component": "features.MatchScore", "params": {"score_field": "current", "side": "against"}},
      "window": 20, "min_periods": 1, "venue": "all"
    }
  }
}
```

Merge into the existing recipe's `features` mapping, preserving its original
predictors, cohort, explicit timing, model settings and odds configuration.
Native JSON encoding, preparation and Python/notebook exports need no custom
registration or auxiliary score matrix.

## Research handoff boundary

No historical study or model fit was run for this implementation. Carry forward
the study's regulation-time cohort verification, settlement exceptions and
post-hoc disclosure; do not infer them from this feature name. In particular,
the handoff reports an Aston Villa–Southampton native 3–4 score with inconsistent
halves and 6,152 rows without complete half verification. This implementation
does not repair that record or exclude those rows. Those handoff findings were
not independently re-audited here. The existing 0.80/0.90 exclusion thresholds,
inclusive equality, reviewed odds pins and previously examined 2025/26 status
remain matters for the separate study; 2026/27 outcomes were not inspected.

## Verification

From the repository root, using the project Python environment:

```powershell
$env:PYTHONPATH='.;src;tests/analytics'
python -m pytest tests/analytics/test_match_score.py tests/analytics/test_ui_match_score.py tests/analytics/test_features.py tests/analytics/test_team_history.py tests/analytics/test_keyed_features.py tests/analytics/test_datasets.py tests/analytics/test_labels.py tests/analytics/test_ratings.py tests/analytics/test_rating_transitions.py tests/analytics/test_warmup.py tests/analytics/test_league.py tests/analytics/test_ui_recipe_cutoffs.py tests/analytics/test_ui_feature_discovery.py tests/analytics/test_ui_feature_bundles.py tests/analytics/test_ui_inventory.py::test_saved_inventory_has_widgets_help_and_packaged_file tests/analytics/test_ui_inventory.py::test_known_form_schemas_use_saved_inventory_without_signature_inspection tests/analytics/test_ui_inventory.py::test_schema_reads_are_isolated_from_persistent_defaults tests/analytics/test_ui_workflow.py::test_catalog_json_defaults_and_all_constructor_parameters tests/analytics/test_ui_workflow.py::test_numeric_rolling_windows_have_no_temporal_split_choices tests/analytics/test_ui_workflow.py::test_recursive_ast_tuple_defaults_remain_hashable tests/analytics/test_ui_workflow.py::test_recipe_exports_roundtrip_without_execution tests/analytics/test_ui_workflow.py::test_named_ratings_and_warmup_prepare_from_recipe tests/analytics/test_feature_composition.py::test_temporal_arithmetic_keeps_cutoffs_and_rejects_raw_or_multicolumn_operands -q -p no:cacheprovider --tb=short
```

Result on 2 October 2026: **405 passed, 0 failed**, in 17.01 seconds. One existing
pandas FutureWarning concerns concatenating all-missing history columns. The
rendered Chromium test selected a score perspective, saved and exported both
recipe formats, checked browser errors and captured a screenshot that was
visually inspected. No prepare/run/predict UI job was launched. Separate Python
and notebook export tests executed preparation only against synthetic match data.

No separate lint/type-check or aggregate-check command is configured in
`pyproject.toml`. The targeted aggregate above and `git diff --check` passed.
The full repository suite was not run: it includes model-fitting and unrelated
collector work. Existing `forms.js` and `feature-bundles.js` required no changes:
ordinary source choices use the catalog, while statistics bundles intentionally
continue to require a Stat source.
