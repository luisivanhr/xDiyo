# Glicko engine choice

The analytics ratings module wraps the existing `utils.glicko_rating.Glicko2`.
There is one numerical implementation; the existing utility remains unchanged.
The analytics layer adds named numeric state, replay, snapshots and feature lookup.

## Comparison and decision

The initially separate implementations agreed within about 3e-11 over 1,500
randomized rating periods when configured identically. Their defaults differed:
the utility used tau=1.0 while the new implementation initially used tau=0.5.
The adapter now retains the utility's tau=1.0 default; tau remains configurable.

The retained engine also supplies explicit inactivity advancement. The adapter's
`advance_periods(state, periods, cap_to_prior=True)` reproduces that behavior.
Football replay does not insert idle periods automatically or change the chosen
event-driven update policy.

The utility's `Rating.mu` and `Rating.phi` names refer to public rating and RD.
The analytics state names these `rating` and `rd`; its `mu` and `phi` are the
standard internal coordinates `(rating - 1500) / 173.7178` and `rd / 173.7178`.
Volatility is `sigma` in both interfaces. Compare corresponding scales.

## Numerical evidence

Using a fixed seed, the adapter exactly matched the existing engine's public
rating, RD and volatility for all 24,500 comparisons:

- 1,500 randomized periods, including empty and multi-opponent periods, custom
  initial-rating centers and tau values.
- 15,000 inactivity comparisons with fractional/integer periods and both cap modes.
- Both team updates in a 4,000-match sequence across 20 teams.

The [official Glicko-2 worked example](https://www.glicko.net/glicko/glicko2.pdf),
with tau explicitly set to 0.5, produced rating 1464.0506705393013,
RD 151.51652412385727 and volatility 0.059995984286488495, consistent with the
paper's rounded intermediate calculations.

See [machine-readable comparison evidence](glicko_engine_comparison.json).
This establishes numerical parity for the comparisons above; broader replay,
snapshot and feature integration verification is recorded separately in the
project tracker. Graph/embedding model training remains future work.
