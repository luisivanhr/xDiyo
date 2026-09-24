# Training documentation and verification coverage

Status: verified on synthetic data, 18 September 2026. All **914 analytics tests**
pass, including **90 new training cases**. Nine guide examples, a fresh executed
nine-cell/four-code-cell notebook and isolated wheel imports pass. All 15 LaTeX
expressions render without errors or overflow; 14 documentation sections and four
saved-notebook sections were visually reviewed. No real pilot fit or package
installation is included.

Final evidence: [training_check.json](training_check.json).

## Public API coverage

| API | Contract | Practical use |
| --- | --- | --- |
| `TrainingRunner` | [Configuration/execution](training_reference.md#trainingrunner-configuration-and-execution) | [Fresh fixed pipeline](training.md#put-fitted-preprocessing-inside-a-fresh-model) |
| `fit_predict` | [One-fold validation](training_reference.md#fit_predict-and-fold-validation) | [One fold and fixed columns](training.md#choose-fixed-columns-or-consume-a-training-selection) |
| `ModelAdapter` | [Structural protocol](training_reference.md#modeladapter-protocol) | [Custom model](training.md#implement-a-custom-adapter) |
| `EstimatorAdapter` | [Settings/shapes/identities](training_reference.md#estimatoradapter) | [Preprocessing](training.md#put-fitted-preprocessing-inside-a-fresh-model), [probabilities](training.md#retain-labels-and-class-probabilities) |
| `PredictionContext` | [Fields and exact groups](training_reference.md#predictioncontext-and-fitcontext) | [Custom adapter](training.md#implement-a-custom-adapter) |
| `FitContext` | [Inherited fields and y](training_reference.md#predictioncontext-and-fitcontext) | [Custom adapter](training.md#implement-a-custom-adapter) |
| `FoldResult` | [All fields/ownership](training_reference.md#foldresult) | [Prediction versus scoring](training.md#distinguish-prediction-from-scoring) |
| `TrainingResult` | [Fields/accessors](training_reference.md#trainingresult-and-prediction-accessors) | [Scoring](training.md#distinguish-prediction-from-scoring), [CPCV](training.md#reuse-numeric-predictions-in-cpcv-paths) |

The [helper inventory](training_reference.md#internal-helpers-and-validation-boundaries)
also covers name/position/frame validation and the nested factory wrapper.

## Acceptance checks

- [x] One fit/predict per selected fold, fresh adapter and wrapped estimator.
- [x] Original numbering/order, reordered indexes, exact uint64 identities and groups.
- [x] Whole-match train/test/score membership; all test rows retained with empty score allowed.
- [x] Single/multi-target shapes, missing test targets and selected missing-train-target rejection.
- [x] Training-only median/scaling/Ridge checked against independent normal equations in both layouts.
- [x] Class vocabularies, absent-class missingness and real multi-output probability lists.
- [x] Optional stored selectors, exact precedence, copied selections and train-scope containment.
- [x] Context/output copies, dataset/plan isolation, exception propagation and invalid outputs.
- [x] Framework-independent adapter, empty fold selection and numeric CPCV reconstruction.
- [x] Full existing analytics regression suite on stable final source.
- [x] Execute guide examples and a new minimal ninth notebook in the existing kernel.
- [x] Isolated wheel/package/import check without installation.
- [x] Render/review all LaTeX, tables, local links and API/helper coverage.
- [x] Final source/data/evidence preservation fingerprints and published manifest.

The final audit retains 45 tested source hashes, 478 unchanged earlier artifacts,
eight unchanged prior notebooks and four archived continuity documents. Sixteen
pinned data/selection files remain unchanged. Four live publication manifests
point to newer versions, with modification times before this batch; the evidence
retains their previous/current hashes and verifies the saved selections' pinned
versions. No data was modified by this worker or used for a real training fit.
The wheel check uses a 91-file staging copy and verifies all four training modules,
all eight public exports, the optional training extra and a generic model without
importing scikit-learn. No implementation defect, worker library edit or callback
was needed. Four document links in the saved notebook export were adjusted for
its output directory; all notebook cells/outputs and four review images remain
unchanged.

## Scope and environment limits

Synthetic checks validate orchestration and estimator integration; they do not
measure predictive quality. No real football model fit, search, scheduled refit or
post-training reporting is performed. Feature timing and fold-aware histories stay
upstream. Live Jupyter validation is not required for this increment; the earlier
batch documented that a frontend is unavailable in the checked runtimes. The new
notebook was executed with the existing kernel and its saved output reviewed.
