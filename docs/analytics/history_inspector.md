# Resetting historical features after data gaps

`HistoryInspector` is an optional wrapper around a team-history reducer. It keeps all
fixtures in the timeline. Apply it to a specific statistic with intermittent coverage,
rather than removing seasons or changing globally how a model assembles its samples.

```python
from xdiyo_analytics.features import EMA, HistoryInspector, RollingMean, Stat

corners = Stat("ALL", "Match overview", "cornerKicks")
features = {
    "corners_ema": HistoryInspector(EMA(corners, span=8)),
    "corners_last_5": HistoryInspector(RollingMean(corners, window=5, min_periods=2),
                                       gap_weeks=26),
}
```

## The cutoff and reset rule

1. Start with the wrapped operator's finished fixtures, in kickoff order. Its
   team/competition grouping, venue, H2H scope, availability and cutoff rules still
   apply. The current fixture itself cannot contribute. If no explicit publication
   availability is supplied, the usual earlier-finished-kickoff proxy still applies.
   For observed Stat and MatchScore leaves, including arithmetic on those leaves,
   the guard also honors any `available_at::<source column>` timestamps. An
   explicit missing release is treated as unknown, not as permission to fall
   back to kickoff. Unguarded nested producers that consume those leaves cannot
   be repaired retrospectively: guard the producer nearest the observation first.
2. Within those candidates, find the usable observations for each emitted source
   column. Ordinary reducers use finite observations. The weighted mean needs a
   finite value and a finite, strictly positive weight, known by the cutoff under
   any side-specific release settings on the weighted reducer.
3. If the latest usable kickoff is **strictly more** than `gap_weeks` (default:
   26 elapsed UTC weeks) behind the cutoff, emit missing. If there is no usable
   observation, also emit missing. Otherwise begin the active regime at the first
   usable observation after the **last** kickoff gap strictly greater than that
   threshold. A gap of exactly 26 weeks does not reset or expire the default.
4. Run the original reducer on that regime and apply its match-window,
   `min_periods`, and other options. Missing slots after a regime begins still
   occupy match-window slots. Lag does not reach past a missing slot; an EMA uses
   its normal missing-value handling. Without a break, an EMA can retain influence
   from observations older than 26 weeks. This is not a maximum-age window and
   is not a season-boundary reset.

For example, after a two-season outage in coverage, the first fixture with a new
statistic still cannot use its own observation. Once the observation is released,
a subsequent fixture can use it, but never averages it with pre-outage observations
unless new point-in-time evidence bridges the entire gap. A late release can change
which regime is known at a later cutoff, not at an earlier one.

## Browser authoring, recipe JSON and evidence

In **Named features**, choose **History Inspector**, then choose the reducer and
its source. For head-to-head features, use **H2H** around the guarded expression
or on the reducer's source. Configure **Gap weeks** and **Evidence**.
The equivalent recipe value is:

```json
{
  "component": "features.HistoryInspector",
  "params": {
    "gap_weeks": 26,
    "evidence": "diagnostics",
    "source": {
      "component": "features.RollingMean",
      "params": {
        "window": 5,
        "min_periods": 2,
        "source": {
          "component": "features.Stat",
          "params": {"period": "ALL", "group": "Match overview", "key": "cornerKicks"}
        }
      }
    }
  }
}
```

Existing saved recipes have no wrapper and are unchanged. Recipes containing the
wrapper require a version that registers it. Source capability discovery for
named datasets still walks the wrapped expression.

Evidence defaults to **diagnostics**, which writes an identity-keyed audit in
`features.attrs["history_inspection"]` and assembled dataset definitions.
Preparation Results show a bounded preview; it describes the evaluated fixture
history, not just fitted rows or scored fixtures. **Off** turns off the audit,
not the reset. **Predictors** explicitly adds five numeric evidence columns per
source column to a named root feature (optionally enclosed in H2H), named
`<name>::inspection::<source column>::<field>` before layout prefixes:

| Field | Meaning |
| --- | --- |
| `eligible_fixture_count` | Fixture slots retained in the active regime and reducer window, including missing slots. Zero if no active regime. |
| `usable_observation_count` | Usable observations in those slots, even when the reducer's `min_periods` is unmet. |
| `latest_age_days` | Elapsed UTC days from the latest known usable kickoff to the cutoff. Missing when no usable value is known. |
| `expired` | Latest known usable kickoff is older than the configured gap. |
| `restarted` | A gap was observed between usable kickoffs at this cutoff. This can persist across multiple fixtures, and is independent of expiration. |

The audit alone also records the pre-guard eligible fixture count. For a
multi-column source, including separate perspectives or periods, one column's
coverage never extends another's.

Supported reducers are Lag, Rolling mean/weighted mean/std/skewness/z-score,
and EMA. League populations, ratings, explicit warm-up wrappers, and standalone
spatial-history producers are not supported as directly wrapped operators.
The guard applies to the named reducer's input; it does not rewrite the
internal history of a nested source. Evidence mode does not change the
reducer's main output.

This wrapper neither imputes nor drops training rows. Use a separate fit-row
policy if a chosen model should train only on explicit complete constructed
features. Leave the validation/test/prediction population fixed when comparing
coverage policies.
