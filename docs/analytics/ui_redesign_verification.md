# UI functional rework verification — 19 September 2026

The existing dark visual style is retained. These checks concern the controls,
their data sources, conditional behavior, and the existing pipeline they invoke.

## Browser checks

Checked in a separate local builder so the previously open user session remains
available. All eleven sections were inspected. Specific interactions verified:

- Discovery found 135 published files and 141 statistic identities for the
  selected 22_23–24_25 population. Season inspection omitted collection bookkeeping.
- Labels select statistics and available periods while preserving group identity
  internally. The group is not a separately editable field.
- Temporal, match K-fold, group K-fold and CPCV show distinct explanations.
  Switching schemes changes their parameter forms.
- A named rating stream exposes Glicko strength, uncertainty, volatility and tau.
  Disabled optional settings hide their nested forms.
- Correlation MCC settings appear only with MCC. F1 exposes class averaging;
  its positive-label setting appears only with binary averaging. MSE has neither.
- Search parameters come from the selected estimator. Lasso alpha uses a numeric
  candidate list. Search remains optional.
- Model preprocessing, execution/device, checkpoint, refit, post-report and future
  fixture controls retain their independent pipeline responsibilities.
- Match reports default to pooled evaluation with available local badges enabled.
- Preparation from the real selected season files completed with **13,971 labeled
  matches**, four example input columns, labels and fold counts. Fixture metadata
  is collapsed separately. This is a preparation check, not a new predictive-quality trial.
- No browser console warnings or errors were recorded during these interactions.

Known timing behavior: job inputs are snapshots. Changing preparation settings
invalidates discovered prepared columns/fold choices; an older job cannot replace
the choices for a different current configuration. Drafts are saved locally as
controls change; Save recipe creates an explicit reusable file.

## Numerical and integration checks

See [inventory evidence](ui_inventory.md) for the 163-test regression pass,
15 focused inventory/iterative checks, and the final 13-test inventory audit.
See [scaling evidence](ui_scaling_verification.md) for train-only fitting,
held-out transformation and target inverse-transformation checks.

JavaScript syntax checks passed for both app.js and forms.js. Existing arbitrary
custom callbacks still require registration in Python. This batch does not claim
new physical CUDA or third-party checkpoint-recovery verification.
