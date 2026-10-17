# Bounded GAT execution evidence

These are copies of small, immutable receipts from isolated experiments on
2026-10-17. Canonical data, full state captures, checkpoints and disposable
test directories remain external to the repository. Each benchmark receipt
records its runtime, source and input hashes, all repeats, phase timings,
memory peaks and execution-plan counters.

- `*-receipt.json`: full benchmark receipts. Initial and reverse native runs
  are both retained; no repeat is discarded.
- `*-comparison.json`: fixed-weight outputs, all loss components, gradients,
  parameters, AdamW state, near-zero differences and same-mode repeatability.
- `lean-isolation-comparison.json` and `packing-isolation-comparison.json`:
  incremental preparation/packing comparisons.
- `numerical-summary.json`: compact numerical values from those comparisons.
- `historical-adamw-investigation.json`: read-only reconstruction of the
  earlier 0.00162 one-update difference, including original file hashes.
- `final-source-{start,end}.json`: source guards surrounding the final
  packed and reverse Bayesian benchmarks.
- JUnit XML files: targeted regression, UI round-trip, exact cold-process
  resume and legacy compatibility evidence. GPU checks use PyTorch 2.9.0+cu130.

The Bayesian three-repeat native receipt precedes final changes to the
packing admission ledger, which native execution does not use. The matching
source reverse Bayesian receipt has one repeat. Its end-to-end duration must
not be compared with a three-repeat process as if workloads were equal.
Whole-process memory peaks decrease in these measurements; preparation-only
host peaks increase. Neither the receipts nor the memory estimates establish
a universal process memory cap.

Full predictive acceptance requires the separately frozen 24-fit matrix.
One-update diagnostics, smoke fits and these receipts cannot establish it.

See [implementation and validation report](../gat_dependency_packed_execution.md).
