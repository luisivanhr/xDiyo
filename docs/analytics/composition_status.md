# Composition and staking implementation checkpoints

Baseline: `29994933902d213bccd13376bc91ab5af70a481b` (7 October 2026).
Branch: `codex/model-composition-stake-policy` in an isolated Git worktree.
The primary checkout, frozen experiments, data and collector work are untouched.

| Milestone | Status |
| --- | --- |
| Baseline and typed contracts | Implemented; 5 focused contract/golden tests passed |
| Fixed ensembles, gates, persistence and native integration | Pending |
| Optional whole-batch staking and closed bankroll ledger | Pending |
| Chronological stacks | Pending |
| Residual and learned extensions | Pending |
| Full regression, examples and final compatibility report | Pending |

Implementation is local only. No push, pull request or merge is authorized.

## Baseline evidence

Unchanged `2999493`: `python -m pytest tests/analytics -q -p no:cacheprovider --tb=short`
returned **4,349 passed, 5 skipped, 1 failed** in 335.66 seconds.
The existing `test_publishable_wheel_runs_without_editable_install` expects only
`xdiyo-ui`, whereas the committed package also declares `xdiyo-report`.
No packaging assertion was suppressed or changed. Log: local temporary
`xdiyo-composition-baseline.log`. The initial contract suite passed 5 tests.
