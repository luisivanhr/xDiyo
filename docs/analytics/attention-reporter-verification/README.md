# Attention reporter verification

The existing pooled panel remains the default (`pooled=True`, `per_fixture=False`).
Enable `per_fixture=True` to add fixture selection, or combine it with
`pooled=False` for fixture panels alone. Both switches are available in the builder
and JSON/Python exports; disabling both is rejected.

Each fold's fixture panel offers Home team and Away team selectors. A kickoff
selector appears when the same pairing occurs more than once. Each choice renders
both teams at that fixture's native By Round or One Day Ahead issuance, with the
fold's frozen model. Shared team/issuance queries and plotted views are deduplicated.
An explicit issuance restricts fixture choices. Teams without historical edges
remain selectable and show an explicit empty state.

The new `attention_fixtures`, `fixture_attention`, and `fixture_concentration`
tables retain identities, issuances, all edges and summaries. The charts show the
strongest configured edges in chronological order, newest first. These weights
describe incoming team messages at that issuance, not a causal attribution of a
specific matchup. No model architecture, weights, predictions or training changed.

The legend and badge occupy separate horizontal regions. During verification,
the fixed 720-pixel chart minimum was found to hide the right-hand badge inside
narrow report panes. The charts now fit the available width. Both charts resize
through the native viewer, including after collapsing and reopening a study.
Badges use a 32-pixel square containment box, recalculated after relayout so narrow
charts cannot squeeze them. Browser checks assert square geometry in both live
charts and SVG exports, as well as separation from legends.

## Checks

- Exact attention-table comparison with direct model inspection, independent
  fixture issuances, target/future-edge exclusion, unchanged model state, pooled
  defaults, shared queries, multiple folds, explicit issuance and empty histories.
- Recipe preparation, JSON/Python exports, saved-model and report recovery.
- Real Chromium checks for home/away and kickoff selection, both rendered teams,
  badges, narrow layouts, study collapse/reopen, and the full native HTML viewer.
- Legend/badge separation with 1, 8 and 24 attention heads; SVG download and
  offline embedded image checks. The native fixture panel also downloads its
  complete CSV and SVG through actual browser actions.

All fixtures are synthetic. No production experiment was retrained or refreshed.
Generating every selected issuance can increase reporting time and export size;
this verification does not claim a large-fold performance benchmark.

`verify.py` runs an isolated committed source tree plus exactly the reporter
changes. Receipts record runtime/source hashes, commands, package versions,
log/XML hashes and whether source remained unchanged during each run. Browser
checks require permission to launch installed Chromium and a localhost server;
no browser installation or external service is needed.

From the repository root:

```powershell
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' docs/analytics/attention-reporter-verification/verify.py broader
& 'C:/Users/luisi/Documents/Programming/Python/xDiyo/.venv/Scripts/python.exe' docs/analytics/attention-reporter-verification/verify.py focused
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' docs/analytics/attention-reporter-verification/verify.py browser
```

See the accompanying JSON receipts and test logs for the final results. The
publication check compares tested and staged source after CRLF/LF normalization;
verification scripts and evidence preserve exact bytes in Git.

Results: **207 passed, 7 skipped** in the broader Python 3.14 run; **61 passed**
in the Python 3.12 focused run; **9 passed, no skips** in real Chromium. The seven
skips require an explicitly idle GPU and are unrelated to the rendering change.
No GPU audit was claimed. All three receipts cover the final source and tests.

Screenshots from the final browser run: `fixture-attention.png`,
`fixture-attention-mobile.png`, and `legend-24-heads.png`. The red circular badge
is a synthetic rendering fixture, not a real team badge or real fitted model.
