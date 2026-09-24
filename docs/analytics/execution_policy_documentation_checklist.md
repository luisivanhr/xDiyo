# Execution/device verification coverage

| Contract | Evidence |
| --- | --- |
| Policy values, worker counts and conservative auto/CUDA caps | option boundary tests |
| Real overlapping subprocesses, selected-fold order and completion replay | PID, parent PID, time intervals and observer checks |
| Seeded serial/parallel parity and nested estimator thread limits | RandomForest pipelines, both layouts |
| Exact uint64 identities, nullable metadata and missing test targets | shuffled layout comparisons |
| Validation and feature-selection training scope | parent-scoped selectors and worker context checks |
| Unpicklable observer stays parent; serial events remain live | explicit nonserializable observer and real iterative backend |
| Candidates and outer folds sequential; no nested pools | recorded parent pool creation and fit intervals |
| Parent-only trial/final publication, failure recording | store call PID assertions |
| Cancellation and usable pool after worker failure | coordinated failing/waiting jobs |
| Experiment/trial reuse, policy identity and inherited final fit | separate recovery cases |
| Native interruption restores optimizer/RNG/cursor/history | preserved checkpoints and uninterrupted-reference parity |
| Checkpoint keys separate device and thread settings | explicit namespace and pointer populations |
| Fitted model/refit execution metadata roundtrip | saved pipeline and numerical recovery |
| Supported/unavailable/malformed device requests | simulated adapters and plain sklearn |
| Real CUDA parameters and input tensors | tiny installed-Torch synthetic CPU/CUDA comparison |
| All APIs/options and minimal CPU notebook | exact signatures, three executable guide examples, notebook 16 |
| Packaging and saved reports | isolated wheel process execution, offline browser checks |
| Earlier work preserved | fifteen notebooks, earlier artifacts, prepared inputs and assets |

No genuine implementation failure has been found in this frozen batch. Existing
tests and library source remain unchanged. Real CUDA evidence is bounded to one
observed device and the tiny adapter; simulated routing does not establish other
framework or multi-GPU support. Fresh notebook execution and saved iframe behavior
are checked; the live Jupyter frontend remains unverified.

[Final evidence](execution_policy_check.json).

## Known notebook shutdown limitation

On the verified Windows environment (Python 3.14.0, joblib 1.5.2), notebook cells
complete correctly, but kernel shutdown emits loky resource_tracker KeyError
tracebacks for temporary joblib_memmapping_folder paths. A one-cell plain-joblib
notebook without xDiyo imports reproduces this. max_nbytes=None also reproduces
it, so disabling automatic memmapping did not resolve the symptom.

The captured notebook run has four cleanup tracebacks; each plain-joblib control
has two. All return exit status 0 with correct numerical results. This evidence
does not establish clean notebook shutdown. Standalone guide, test and isolated
wheel computations pass. The library and installed packages were left unchanged;
no workaround or dependency upgrade is claimed. See the runtime and cleanup
control logs linked from the verification record.
