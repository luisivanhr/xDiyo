# GAT preparation: second optimization pass

## Result

On the same completed expanded-totals fold 02, two new disposable full updates
including the unchanged post-update loss replay took **39.48 and 41.98 seconds**.
The first-pass implementation rerun in this session took **132.66 and 148.90
seconds**. Median time fell from **140.78 to 40.73 seconds**, a **3.46x** speed
ratio and **71.1%** reduction beyond the first optimization pass.

These are local measurements with another Python process using the GPU and an
active desktop. The first-pass timings differ from its earlier 101–112 second
measurements. Do not multiply ratios across the two sessions or treat these
numbers as an isolated benchmark or a guaranteed research-run speedup. Both
implementations in this comparison used the same runtime and saved fold.

## Bottlenecks and changes

- **Repeated Arrow ID conversion:** snapshot construction iterated both team ID
  columns for every issuance. The store now assigns exact IDs dense integer codes
  once, then gathers local receiver/sender positions in source-row order. Original
  IDs remain Python integers, including values larger than uint64. Two fixed code
  arrays add 895,616 bytes on the supplied source, independent of issuance count.
- **Repeated grouping for rest days:** eligible edge receiver positions already
  identify teams. A maximum reduction finds their latest eligible kickoff without
  rebuilding a pandas groupby. Scalar timestamp arithmetic is retained because
  changing it to direct nanosecond division changes submicrosecond rest values.
- **Repeated provenance copies:** the real fixture metadata carries roughly 35 KB
  of nested attrs. Column reads were copying these attrs onto intermediate Series.
  Temporary frames without attrs provide read-only column access. Returned metadata
  retains its complete attrs and independent nested copies; nothing is discarded
  from artifacts or reporting metadata.

A proposed narrow-column shortcut during fitting was rejected after it slowed
preparation. No per-issuance graph cache, retained tensors, learned embeddings,
new persistent state, mathematical reduction changes, or skipped loss replay was
introduced. Model configuration, architecture, cutoff rules, graph contents,
weights, priors, optimizer steps and serialized model identity remain unchanged.

## Timings and memory

Each repeat starts from the saved model and a fresh disposable AdamW. Each covers
all 1,463 issuances / 18,888 fixtures and then recomputes post-update training loss.

| Implementation | Training | Metric replay | Total |
| --- | ---: | ---: | ---: |
| First pass, repeat 1 | 69.27 s | 63.39 s | 132.66 s |
| First pass, repeat 2 | 72.21 s | 76.68 s | 148.90 s |
| Second pass, repeat 1 | 23.29 s | 16.19 s | 39.48 s |
| Second pass, repeat 2 | 23.32 s | 18.66 s | 41.98 s |

Warm preparation includes tensor conversion and device transfer (median of three):

| Graph | Directed edges | First pass | Second pass |
| --- | ---: | ---: | ---: |
| Early | 0 | 11.09 ms | 4.36 ms |
| Middle | 18,406 | 44.67 ms | 7.26 ms |
| Late | 37,752 | 62.33 ms | 11.64 ms |

Sampled peak process RSS was **1.882 / 1.881 GB**. Both passes had peak live CUDA
allocation **175,799,296 bytes** and reserved **228,589,568 bytes**. The fixed
source-edge encoding cache remains 34.86 MiB within the existing 128 MiB limit.
The small RSS difference is not evidence of an additional memory reduction.

## Verification

**222 Python tests and 3 UI tests passed.** The three actual early/middle/late
graphs also passed bit-exact deterministic CUDA comparison of ordered inputs,
logits, losses, gradients, parameters, AdamW state and CPU/CUDA RNG against frozen
`11ffa446`. This is a three-graph diagnostic, not a full-fold deterministic run.

The early/middle/late prepared tensor hashes match the first pass and original
implementation exactly. New frozen-reference tests cover Arrow/nullable/object
integer IDs, values beyond uint64, tied kickoff ordering, multiple intervening
season boundaries, cold teams, empty populations, precise release/cutoff bounds,
missing statuses, nanosecond rest arithmetic, and metadata copy isolation.

Default-CUDA full-fold updates remain nondeterministic, as documented in the
[first-pass audit](gat_speed_11ffa446.md). Bit-exact default-CUDA full-fold training
is not claimed. Model commitments, support-bundle hashes, fitted preprocessing,
store/model identities and original input hashes were verified. Completed-fold
files and active experiments were not modified.

## Evidence and reproduction

- [First-pass rerun](data/gat_speed_second_pass/first_pass_gpu.json)
- [Second-pass rerun](data/gat_speed_second_pass/second_pass_gpu.json)
- [Summary and current implementation hashes](data/gat_speed_second_pass/summary.json)
- [Deterministic real-graph parity](data/gat_speed_second_pass/deterministic.json)
- [First-pass source snapshot and replay wrapper](data/gat_speed_second_pass/first_pass_source.zip)

Run `examples/benchmark_gat_speed.py --mode native --repeats 2 --fold <fold>
--evidence <support-zip> --output <external-output.json>` for the new implementation.
The source archive preserves the three first-pass files and the local wrapper;
adjust its ROOT and SAVED paths before replaying that baseline. The adapter is
unchanged between these passes. The wrapper retains the native harness's model
and hash verification checks. Write all outputs outside the completed fold.

Production changes in this second pass are limited to `graphs/store.py`,
`graphs/streaming.py` and `graphs/preprocessing.py`; new regression tests are
`test_graph_snapshot_speed.py` and `test_graph_streaming_speed.py`.
