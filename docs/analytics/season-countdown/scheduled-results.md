# Scheduled-round verification results

Implementation: `b298dbdcf6d9fdfd4b38bb5a89dcfea866a89c5b`.
Comparison baseline: `643c20297cf4e2233c50d91a62bf097d18d7f985` (majority-started).

- 354 regression tests passed, including 59 countdown tests. This is the relevant targeted suite, not the full application suite.
- Independent exhaustive-cluster verification passed for all 289 publications / 92,983 fixtures: 92,855 eligible finite outputs, 128 intentionally excluded Belgian playoff fixtures, zero unexplained missing outputs, and 14 independently specified effective-round checks.
- 1,213 fixtures differ: 693 move to a higher effective round (lower countdown), 520 move to a lower effective round (higher countdown). The complete identity-level changes are in `scheduled-value-changes.csv`; publication counts are in `scheduled-change-counts.csv`.
- Every source/test/script hash in the verification receipt was checked against exact committed bytes. Generated receipt text is canonical LF. Previous receipts were preserved.
- No dataset edits, model fits, frozen experiment reruns or feature-artifact rewrites.

## Representative differences

Chelsea-Tottenham (12226774, Premier League 2023/24) moves from effective round 35 to 36. Its May 2 18:30 UTC kickoff is 24.5 hours before round 36's main interval, compared with 99 hours after round 35's endpoint. The upcoming round is valid scheduled context before any of its matches starts.

FC Tokyo-Hiroshima (7003326, J1 2016) moves from 15 to 16. Its June 15 10:00 UTC kickoff is 72 hours before round 16 and 96 hours after round 15. Its original first-stage round-10 identity remains unchanged; rescheduling does not convert it into a second-stage fixture.

Bristol City-Watford (3434240) changes from 28 to 29, Cerezo-Nagoya (8016558) from 31 to 32, and Millwall-Watford (9266059) from 25 to 26. Each lies inside the selected main interval of the new effective round. The former rule lagged because the required majority had not started.

The opposite direction is also expected. Yeovil-Millwall (4919948, Championship 2013/14) returns from 30 to its assigned round 27: it lies inside round 27's main cluster. Another round having started cannot override an ordinary fixture's own interval.

## Limits

Historical exports may contain revised schedules; this implementation does not reconstruct unavailable historical schedule knowledge. Missing dates/rounds remain missing inputs, and sparse schedule coverage can prevent recognition of a displaced fixture. Provide the same full schedule to query subsets. There are no batch overrides or timing-policy switches.

See `scheduled-verification.json` for exact commands, environment, source and artifact checksums, `scheduled-tests.xml` for test results, and `scheduled-publication-audit.csv` for publication-level verification.
