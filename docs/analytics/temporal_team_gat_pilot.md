# Bounded temporal team GAT pilot, 11 October 2026

This is an implementation and feasibility example. The native rolling-goals
baseline performed better overall than either GAT in this short experiment.
There is no established predictive advantage for the graph model.

## Data and timing

Inputs were read-only publications in
`C:/Users/luisi/Documents/Programming/Python/xDiyo/data/xDiyo_data`.
All 30 Bundesliga and Bundesliga_2 seasons from 2010/11 through 2024/25 were
loaded with source hash verification. The default loader excluded awarded
fixtures; no padding or fabricated observations were added.

The event store contains **9,177 fixtures and 18,354 directed perspectives**,
from 2010-08-20 16:00 UTC through 2025-05-18 13:30 UTC. Team-season metadata
contains 435 retained, 73 promoted and 32 relegated entries. Both divisions
remain in each eligible graph, including historical opponents that are absent
from the target competition. There are no hard age limits.

The target is the observed score comparison, home/draw/away. Only finished
fixtures with both scores enter fitting/evaluation. Scores are the only numeric
match statistics; xG and other optional statistics are unused. Neutral venue
is absent and explicitly masked. Season entry is assumed known at July 1 local
midnight, using the reviewed movement table. This is a declared retrospective
context assumption, not a reconstruction of old schedule or announcement vintages.

Forecast policy: **By Round**, `Europe/Berlin`. Each round's first scheduled
kickoff is its issuance; history ends at the preceding local calendar day.
Actual match times measure age at issuance. The source has no later sparse
reviewed completion bounds in this population. The source fingerprint is
`1882f381af56a7035f47998371045ee548123f98f6eb5ab4e3172a25b89f88b8`.

| Population | Season | Targets | First issuance UTC | Last issuance UTC |
|---|---|---:|---|---|
| Encoder fitting | 2021/22 Bundesliga | 305 | 2021-08-13 18:30 | 2022-05-14 13:30 |
| Encoder validation | 2022/23 Bundesliga | 306 | 2022-08-05 18:30 | 2023-05-27 13:30 |
| Standalone test; later downstream fitting | 2023/24 Bundesliga | 306 | 2023-08-18 18:30 | 2024-05-18 13:30 |
| Downstream test | 2024/25 Bundesliga | 305 | 2024-08-23 18:30 | 2025-05-17 13:30 |

The imported 2021/22 target population contains 305 finished fixtures; the
example uses those available rows. The 2024/25 publication explicitly excludes
one awarded event through the normal loader contract. Target kickoff minima
and maxima equal the listed first/last issue endpoints. Individual delayed
fixtures retain their assigned round's earlier issuance.

Fitting labels are known through **2022-05-14 22:00 UTC**. Validation selection
extends the frozen parameter boundary through **2023-05-27 22:00 UTC**.
Both precede the relevant held-out issuance boundaries.

## Fixed compute and validation selection

A single seed (17), 32 dimensions, two attention heads per layer, rho=0.5,
kappa=5, 64-wide head, AdamW learning rate 0.001, weight decay 0.001 and prior
penalty 0.001 were fixed. Six fresh fits compared one/two layers crossed with
H in 90, 180 and 365 days. Each fit was capped at 12 full-population steps with
early-stopping patience 3 and best-checkpoint restoration. Every candidate
reached step 12 with its best validation loss at that final step. This is a
tight compute cap, not a convergence study.

| Layers | H, days | Validation log loss |
|---:|---:|---:|
| 1 | 90 | 1.046225 |
| 1 | 180 | 1.043266 |
| 1 | 365 | **1.041521** |
| 2 | 90 | 1.043063 |
| 2 | 180 | 1.041021 |
| 2 | 365 | **1.039897** |

H=365 was selected within each layer count using validation alone. Each selected
model was evaluated once on the 2023/24 test. No held-out-guided tuning followed.
Parameters remained frozen while newly eligible observations entered later
round snapshots.

The existing native `RollingMean(ForAgainst(MatchScore(...)))` expressions
provided the baseline's ten-match goals-for/against histories for both teams,
including cross-competition history. A fitting-only imputer/scaler and
multinomial logistic regression (C=1, max_iter=500) used those values and their
home-away differences. It trained on the same 305 target fixtures with the
same previous-day By Round cutoffs. Its hyperparameters were fixed.

## Held-out results

Brier is the sum over three classes, averaged over fixtures. ECE uses ten bins
of the maximum class probability and compares mean confidence with accuracy.
It is descriptive, not a calibration guarantee.

| Model | N | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|
| 1-layer GAT, H=365 | 306 | 1.048878 | 0.631237 | 0.114764 |
| 2-layer GAT, H=365 | 306 | 1.053191 | 0.634215 | 0.118270 |
| Native rolling-goals baseline | 306 | **0.994672** | **0.592410** | **0.030772** |

| Group | N | 1-layer log loss | 2-layer log loss | Baseline log loss |
|---|---:|---:|---:|---:|
| Rounds 1-5 | 45 | 1.010766 | 1.015175 | 0.920859 |
| At least one movement entrant | 66 | 1.074352 | 1.089435 | 1.084088 |
| Other fixtures | 240 | 1.041872 | 1.043224 | 0.970083 |

The movement target subgroup here consists of promoted top-division entrants.
Relegated teams' lower-division observations and contexts are present in the
historical graph, but this top-division evaluation does not establish a separate
relegated-target effect. Subgroup sizes are small; no significance or superiority
claim follows from their differences.

For three-block downstream reuse, the selected **same two-layer encoder** was
frozen after 2021/22 fitting and 2022/23 validation selection. A separate
imputer/scaler/logistic classifier fit its exported fixture embedding combinations
on 2023/24, then evaluated on 2024/25. Results: **N=305, log loss 1.128200,
Brier 0.670399, ECE 0.129263**. Embedding coordinates were never pooled from
independently trained encoders.

## Execution and retained artifacts

The measured run took **123.98 seconds** including six fits, baseline construction,
test evaluation and frozen-encoder exports. Hardware was an RTX 4070 SUPER,
using the isolated checkout's Python 3.12 / PyTorch 2.14.0+cu130 environment,
one CPU thread per model operation. This timing is specific to this population
and machine. Full-history tensors are retained per fitting/validation issuance;
larger archives can require substantially more memory.

Maintained executable:
[examples/temporal_team_gat.py](../../examples/temporal_team_gat.py).
The package guide is [temporal_team_gat.md](temporal_team_gat.md).

Temporary artifacts are outside tracked source and retained pending user validation:

- `C:/Users/luisi/AppData/Local/Temp/xdiyo-temporal-gat-20261011/`:
  empty directory from the first attempt, which exposed the Arrow score check.
- `C:/Users/luisi/AppData/Local/Temp/xdiyo-temporal-gat-20261011-run2/`:
  `results.json`, `progress.json`, `standalone_predictions.parquet`,
  `frozen_encoder_exports.parquet`, and `gat_1_layer/`, `gat_2_layer/`
  (each contains `graph.json` and `weights.pt`).
- `C:/Users/luisi/AppData/Local/Temp/xdiyo-temporal-gat-20261011-retag_versions.py`:
  one-time metadata upgrade that included full configuration in representation
  version hashes. It changed no weights and did not retrain or rescore held-out
  targets. The mapping is recorded in results.json; restored exact-cutoff
  embeddings were checked against retained exports.
- `C:/Users/luisi/AppData/Local/Temp/xdiyo-temporal-gat-20261011-packaging/`:
  built package wheel for packaging inspection.
- Worktree-local `build/` and `xdiyo_analytics.egg-info/`:
  ignored setuptools build metadata/output from editable installation and wheel
  validation. They contain no source datasets.

`progress.json` records intermediate candidate state before the metadata-only
version upgrade; `results.json` is the final result/provenance record.
The worktree's `.venv/` is the retained CUDA development environment, not a
pilot artifact scheduled for cleanup. Standard pip download caches are managed
by pip. Source datasets were read-only throughout.

Limitations include one country, one seed, a small fitting population, a short
optimization budget, retrospective schedule/context assumptions, and no external
holdout beyond the declared seasons. The results justify a working implementation
and an honest baseline comparison, not deployment or a claim of better forecasts.
