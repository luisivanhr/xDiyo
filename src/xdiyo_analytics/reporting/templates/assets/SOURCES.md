# Preserved ticket format

Ticket CSS extracted without edits from Ayre's supplied
`xDiyo_report_template_handoff/reference_sources/tickets/viewer.html`.

- Original file SHA256: `cc4f5cf357028d4c12d8028bb376d4f785b42fe9761b09e11e17a512e31efd84`
- Extracted stylesheet SHA256: `e7c2513eceb34b0bc1370457501bb54de3974ce09884323f10953524638cb508`
- Handoff date: 2026-10-03
- Source indicated by handoff: `historical-tickets-20261003/scripts/viewer.html`

The general viewer preserves the classes, card hierarchy, badge placement,
filters, accounting blocks and pagination. Data loading, field mappings and
text are generalized; draw-specific claims and probability/EV reconstruction
are removed. The explicitly named legacy adapter reads an audited flat CSV.
It does not execute old builders or decode their uint16 compressed archive.

Native curve/portfolio chrome uses public AnalysisReport contracts. Saved
figure JSON and the matching supplied Plotly runtime are copied without
altering the original trace definitions or runtime copyright/license header.
Producer revision remains per figure; it is not replaced with the renderer's
revision. Full original studies and team assets are not redistributed with
this module. For bundles that embed badges, callers must supply attribution
and license files, which are copied alongside the output.
