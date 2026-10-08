# Historical result reviews

The five exceptions reviewed on 8 October 2026 are recorded with source links in
[the evidence register](data/historical_result_reviews_20261008.json). These are
corrections to provider metadata, independent of promotion/relegation enrichment.

| Event | Fixture | Treatment |
| --- | --- | --- |
| 2893007 | Cagliariâ€“Roma, 23 September 2012 | Unplayed, awarded 0â€“3. No played score. |
| 1399769 | St Pauliâ€“Schalke, 1 April 2011 | Abandoned late at 0â€“2 and administratively settled 0â€“2. This is not a completed match. |
| 3944327 | Nantesâ€“Bastia, 10 August 2013 | Played 2â€“0, subsequently declared lost by penalty. Preserve played score; flag the ruling separately. |
| 1861771 | Padovaâ€“Torino | Started 3 December 2011, completed 14 December, played 1â€“0. The temporary later administrative reversal was overturned; the retained outcome is the played result. |
| 1795067 | Granadaâ€“Mallorca | Started 20 November 2011, completed 7 December, played 2â€“2. |

## Fields and default exclusion

`is_awarded` is the effective reviewed exclusion flag. `provider_is_awarded`
preserves the provider flag. The first three events are excluded by the existing
`load_season(..., include_awarded=False)` default, together with their event-linked
statistics and other requested rows. Opt-in loading retains them for inspection.
The two resumed events remain ordinary played results (`is_awarded=False`).

`status`, `status_code`, `*_score_current`, `*_score_display`, original kickoff and
round remain provider values. `play_status` distinguishes `not_played`,
`abandoned`, `completed` and `resumed` for reviewed rows. `home_score_played` and
`away_score_played` retain the actual score, with nulls for the unplayed match;
for an abandoned match these are the score at abandonment. `*_score_awarded`
contains an administrative score only when evidenced. Nantes' penalty is flagged
without inferring an administrative goal score from the points sanction.
`result_classification` distinguishes unplayed/abandoned awards, later forfeits and
resumed completions. `result_review_json` retains the record and sources. New review fields are null
for unreviewed rows; null does not certify that a fixture was ordinary.

## Completion and availability

`completion_date` is the reviewed local date. `result_available_at` is sparse UTC
datetime metadata imposing a **conservative availability bound**, not an exact
whistle or publication timestamp. With only a completion day verified, the bound
is the following local midnight:

- Padovaâ€“Torino: **2011-12-14 23:00 UTC** (15 December 00:00 Europe/Rome).
- Granadaâ€“Mallorca: **2011-12-07 23:00 UTC** (8 December 00:00 Europe/Madrid).

Historical feature eligibility, league/LOO history, Glicko/Bayesian replay and
Bayesian training populations honor these bounds. They are carried through team
history, labels and match/team dataset metadata into temporal splitting, CPCV,
temporal SVC calibration and chronological composition. Effective availability is
the later of the configured release policy and the reviewed bound. This includes
an explicit `kickoff + 3h` proxy. A later explicit release remains later; a missing
explicit release remains unknown. No bound on an ordinary row changes its
existing availability policy. Custom downstream code must retain and honor this
metadata too; deriving availability independently from kickoff discards the repair.

The original kickoff and round determine chronology; completion determines when
the final observation may enter history. Full-match statistics of a resumed game
are withheld with its result until the bound. The repair does not reconstruct
what partial statistics were available during the interruption.

## Publication and reproducibility

Run `PYTHONPATH=src;. python examples/repair_historical_results_20261008.py` for
read-only validation, or add `--apply` to publish. The operation validates event
identity and provider score, preserves Arrow fields and nested observations,
writes ZSTD, updates actual native/nested hashes, and retains a resumable journal
and before/after fingerprint audit under `data/xDiyo_data/_result_reviews/`.
Other tables, movement flags, fixture IDs and row ordering are unchanged.

The prior extension verification is a historical snapshot of its publication,
not verification of the corrected bytes. Existing source-pinned recipes need the
native recipe source-refresh helper; previous calibration artifacts are not
silently relabelled as using corrected data. No experiment or calibration is run
by this repair.

Charleroi–Cercle Brugge (1378121) remains unchanged: the superseded award is not
a reason to flag its completed 23 March 2011 match. Provider half scores and
numeric status codes are preserved, including the audit's 310 half-score-sum
exceptions and 65 unusual status codes. No score reconstruction is performed.
The no-awards loader retains 23,655 of the 23,658 early fixtures, including the two
resumed matches with completion bounds. Ayre's stricter five-case quarantine
contains 23,653 candidates; that separate research mask remains valid. Native
availability remains inclusive at the bound; the audit's stricter `< cutoff`
research protocol still requires its explicit prefilter.


## Re-audit of 8207e29 and API boundary repairs

Ayre cleared the five-fixture repair hold for the specifically audited native
pooled-Bayesian workflow with explicit kickoff +3h, clamped to reviewed bounds.
The eligible early-history count is 23,655. This is scoped clearance, not approval
of every research route. The original audit, frozen pre-fit specification and
65-partition counts are retained under `audits/early_history_8207e29/`, with source
hashes. The original commit pins in that specification are historical evidence;
this maintenance patch does not run calibration or alter the frozen specification.

Three API findings from that re-audit are repaired:

- **Glicko exact-time release ties:** when a delayed prior result and a current
  kickoff-proxy result share a release, an additional boundary snapshot contains
  only the prior results, using the same pre-batch opponent states. Exact-boundary
  predictions use it; later predictions use the unchanged full simultaneous
  update. Season transitions still precede equal-time result releases. Boundary
  snapshots survive native save/load and do not change the ongoing replay state.
- **Probability calibration:** `predict_proba` now uses the guarded chronological
  partition when nonmissing reviewed bounds or an explicit availability policy
  are present. It purges whole matches whose labels are unavailable at the first
  calibration prediction boundary or outer issue time. Without explicit timing,
  legacy kickoff-proxy/zero-lead semantics remain. Explicit availability, missing
  releases, prediction groups and configured cutoffs are respected. With no bounds
  or explicit timing, legacy selection is unchanged. Direct guarded partitioning
  or refitting requires `issue_at` or fold `fit_at`; the runner supplies the outer
  test boundary. Missing issue time raises instead of assuming future knowledge.
- **Match-label metadata:** the latest known `result_available_at` bound from
  either team row survives match-level assembly. A sparse null bound is absence
  of an additional constraint; two null bounds remain null. Team-level metadata
  remains unchanged. This reduction does not impute explicitly unknown release
  times in the availability policy.

No fixture data, hashes, population selections, Bayesian parameters, research
recipes or previous experiment artifacts are changed by these API repairs.
