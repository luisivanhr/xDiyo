# GAT integration and parity checks

Integration branch: codex/gat-main-integration. Base: 64861849d8b7044e0b209b9ac9db28080a92b1e4.
The user authorized implementation and parity checks on 2026-11-04. Existing worktrees, uncommitted research, frozen sources, and experiment artifacts are preserved.

Authority: GAT_RECONCILIATION_DECISIONS.md supersedes GAT_MERGE_PLAN.md; GAT_HANDOFF_INDEX_CORRECTED_20261104.md updates recovered archive locations. These are implementation requirements, not passing evidence.

## Gates

| Stage | State | Evidence |
| --- | --- | --- |
| Source/artifact inventory | Preflight verified before implementation | Uploaded inventories and live hash checks |
| Generic semantic foundation | Exact pinned-main CPU and CUDA comparisons passed | Six cases, three complete updates each; `main_cpu/receipt.json`, `main_cuda/receipt.json`. Broad regression fixes still under verification. |
| Original retained inference | Passed, all 18 | `retained_inference_cuda_v2/comparison.json`: all original outer fixtures, exact inputs/logits/probabilities/rates/identities and round trips. |
| Older legacy inference | Passed, 54 exports | `legacy_comparison_index.json`: 53 representative schemas plus recovered expanded-totals fold02; exact four-fixture probes and round trips, not full-population evaluation. |
| Public profile and recipe integration | Passed | Six sealed replay recipes prepare; 14 public/reconciliation tests cover JSON/Python, native fit/reuse/save/load, data-only recovery and current v2 production. |
| Retained training and same-source resume | Passed | `retained_training_cuda_v2/comparison.json`: all three families, complete fold2 fitting/inner populations, seed17, exact gradients/AdamW/RNG/selection/restoration. |
| Opt-in speed implementation | Pending reference parity | Five separate flags, combined check, full representative replay |
| Final preservation and review | Pending | No merge or push |

Evidence is written under experiments/gat_integration_20261104_01. Historical outputs remain immutable. Selected model exports do not establish optimizer continuation. Old source-specific native checkpoints must retain their source rejection boundary.

## Recovered archive evidence

Both ZIPs were retrieved from the user's Space Library through its download UI. Their SHA256 values match the corrected handoff index. They were extracted only under this integration evidence root. All eight recovered expanded-fold model/data members match the original isolated-copy receipt. The archive contains its source manifest rather than the source tree; the exact `11ffa446` Git tree was reconstructed separately and all 213 manifest members matched after the recorded CRLF-to-LF normalization.

The older four-study ZIP matches its archived ZIP hash. One member, `examples/gat_suite/reporting.py`, differs from its separate source manifest. The untouched live research file matches that manifest exactly, so a separate manifest-resolved runtime copy was made with a recorded per-file origin. The original ZIP and extracted copy were preserved. The rest/propagation runtime had seven unavailable frozen members; all seven were recovered from `9499d4af` with exact recorded hashes. These source recovery receipts establish bytes, not compatibility.

## Verification record so far

- Generic pinned-main comparisons: exact CPU and CUDA inputs, initial state, three updates, gradients, AdamW, predictions, embeddings, attention and identity in six cases (default snapshot, released-statistic legacy policy, memory, statistics, combined extensions, CNN).
- Broad initial corrected pass: 370 passed, 10 failed. Nine failures came from the independent pre-research test oracle including new inactive fields; its projection now asserts those fields are inactive before omitting them, and its full 23-case preparation suite subsequently passed. One CUDA completed-resume case ran while source edits changed its execution fingerprint; it needs a clean stable-source rerun.
- Public/reconciliation pass: 19 passed. This is additional to the all18 original-model comparison, not a replacement for it.
- Historical inference is compared to independently imported frozen source. Existing saved-output arithmetic alone is never used as the inference oracle.

The frozen retained backend was compiled from its archived class and exact numerical helper definitions. Explicit original protocol/path bindings avoid importing an unused reporting dependency absent from the archive; per-file SHA256 and selected definitions are recorded in each frozen reference receipt. No live research numerical code substitutes for the oracle.

Inventory review against an isolated regeneration of pinned M found no removed components, changed stages or metrics, or changed non-graph fields. Broad changes versus the committed JSON also occur with unmodified M and the installed libraries; `inventory_reference_comparison.json` separates those from the graph additions.

Public preparation also requires an explicit `round_stage_column` mapping when the native source names its stage differently. This optional field only copies supplied complete evidence and records it in store provenance; it invents no regular-stage constant and does not alter generic recipes or the native history producer. The archived ComponentStore recipes already contain their exact round_stage values.

CPU cold-process continuation passed for all three retained profiles; CUDA continuation passed for all three and the generic fixture-feature case. Interrupted and uninterrupted gradients, optimizer, restored parameters, RNG and predictions match exactly; reopening a completed checkpoint performs zero updates. These tests use an explicitly declared miniature population, not an all13 reproduction. Full-population training parity is the separate three-update gate above.

The retained contract identity uses JSON-stable competition lists. The original tuple/list mismatch was caught by actual checkpoint publication and fixed before acceptance. Public prepared phase contexts now use data-only states with recomputed source identity; native run recovery does not pickle a live lookup or silently rebuild it.

Preserved main regression coverage passed, including current standings/venue behavior, named selection and cutoff guards, aliases/migration, temporal provenance, reporter settings and legacy report records. Initial long-path Windows directory renames failed in several dataset fixtures; all affected test files passed (59 tests) in a fresh shorter temporary directory without an application change. Other relevant passes are retained in `main_regressions.xml` and `main_regressions_retry.xml`; final evidence aggregates by test identity rather than summing reruns.
