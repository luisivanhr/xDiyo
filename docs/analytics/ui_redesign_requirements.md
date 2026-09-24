# Experiment builder redesign

User requirements recorded on 2026-09-19. This preserves the six surviving
browser comments and the accompanying instructions; it does not claim to
reconstruct the missing 33 comments verbatim.

1. Remove all user-facing Value type controls throughout the builder. A maintained
   structured inventory owns widget types, choices, defaults and dependencies.
2. Select leagues, seasons, tables, statistic fields, metrics and named outputs
   from the available catalog. Text inputs remain for names, paths and genuinely
   free-form values; numeric parameters use appropriately sized numeric controls.
3. Inspect prediction-relevant tables and statistics. Do not show collection
   coverage, hashes, source keys or duplicated schema dumps as model inputs.
4. Statistic selection uses a discovered identity; hide the underlying group
   field and retain it internally to distinguish otherwise ambiguous statistics.
5. Explain every parameter beside its control, including examples where helpful.
   Explain each CV scheme, its purpose, and the effect of its parameters.
6. Reporter metrics are selectable; expose only configuration applicable to the
   selected metric. MCC categories/thresholds must not appear for Pearson alone.
7. Post-training results should open in pooled evaluation scope, with per-fold
   detail optional. Do not relabel internal validation as held-out test data.
8. Connect available local team badges to match reports with names as fallback.
9. Show the main report once; underlying data should remain downloadable without
   duplicated expanded tables. Improve surviving-coefficient presentation.
10. Make wins/statistic ratings and their parameters understandable, including
    initial rating, uncertainty, volatility, tau, and optional transitions.
11. Verify that preprocessing fits only on training rows. X is transformed before
    prediction; transformed y predictions are inverse-transformed afterward.
12. Preserve existing recipes and all independent library APIs. Verification and
    documentation may be delegated under the user's existing workflow preference.

Implementation and verification status is recorded in IMPLEMENTATION_PROGRESS.md.
