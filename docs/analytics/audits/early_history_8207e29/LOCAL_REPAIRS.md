# Local maintenance follow-up

The three bounded API findings in `REPORT.md` are repaired. See
`../../historical_result_reviews.md` for their semantics and compatibility limits.
The original report and frozen calibration specification remain source evidence;
no research calibration, feature-bank rebuild, betting run or dataset rewrite
was performed in this follow-up.

Validation on Python runtimes with pandas 2.2 and pandas 3:

- pandas 2.2: 379 passed across boundary repairs, Glicko ratings/transitions,
  labels, probability/SVC calibration, historical publication, temporal splits,
  dataset assembly and Bayesian replay.
- pandas 3: 220 passed across the first seven groups above, followed by all
  17 final boundary regressions passing (including two additional explicit
  missing-release/equality cases).

The Glicko transition regression now expects the valid earlier result at the
exact release boundary, while its saved transition state remains unchanged.
The native rating save/load, statistical ratings, full simultaneous updates,
source-row ordering, paired metadata, base/calibration fit populations and
legacy all-null-bound behavior are covered. Source archive identity is recorded
in `sources.json`; its binary copy is not duplicated in this repository.
