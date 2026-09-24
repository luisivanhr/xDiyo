# Prediction/persistence coverage

| Contract | Evidence |
| --- | --- |
| Explicit awarded exclusion across all event-linked tables; unknown flags retained | awarded loader tests |
| Hidden match reads versus include_awarded=True selected-only behavior | awarded loader tests |
| Exact signed/unsigned IDs, nullable/empty child rows | membership and public loader tests |
| Metadata survives history/labels/assembly | both layout tests |
| Ordinary and daily-named roots yield the same populations | identical synthetic publications |
| Completed early-season history survives status/round prediction selection | full feature/label/assembly tests |
| Load/select/align integer, list, mapping, empty and unpublished rounds | both layouts and empty-schema tests |
| UTC as_of, missing time/round values, invalid requests | selection boundary tests |
| Exact event/source/team alignment with unknown target labels | both layouts; mismatched/missing/duplicated row tests |
| Saved Ridge/classification preprocessing, feature order and class probabilities | real estimator roundtrips with fit methods disabled after save |
| Native serializer, format matching, wrapper unwrapping and fold selection | model persistence tests |
| Atomic new destinations, failed writer, corruption and contained paths | artifact boundary tests |
| Published fixture → features → saved model prediction | end-to-end synthetic check |
| All APIs/options/helpers | reference signatures and guide |
| Package and notebook behavior | isolated wheel, fresh notebook 15, offline reports |
| Preservation | source/artifact/14-notebook/input/asset final audit |

The primary repaired Arrow uint64 membership overflow in statistics validation
and awarded exclusion after one genuine failure callback. The first failing log
and all source freezes remain preserved. Subsequent test-side scalar comparisons
were corrected to use exact Python integer masks; no library change was needed
for those assertions. The notebook explicitly displays the test prediction
partition, since unknown outcomes have no scoring rows; the initial empty
display and its browser evidence remain archived. No old tests required
adjustment for the deliberate default.

All fits and publications used for execution are synthetic. No source export,
prior notebook, installation, scheduled work or live model experiment was changed.
Fresh-kernel execution and saved offline iframe behavior are checked; live Jupyter
frontend remains unverified. [Final evidence](prediction_persistence_check.json).
