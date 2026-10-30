# Fixture attention SVG provenance repair

Base: `71d4da569c18cdcbccd0e1653a279baed8124b66`.

Each fixture chart now exports its home/away side, selected team and fold in
its title. A chart annotation records both fixture teams, exact competition,
season and event identities, kickoff and inference issuance. The filename
contains side/team, fold, exact fixture identity and both timestamps. Selection
changes update both chart metadata and export filename. Pooled behavior remains
unchanged. Existing HTML reports must be regenerated.

## Verification

- Python 3.14: 61 focused reporter, integration, UI and recovery tests passed.
- Python 3.12: the same 61 tests passed.
- Chromium / Python 3.14: all 11 browser tests passed, none skipped.
- New regressions downloaded eight actual SVGs: both sides of two same-pairing
  fixtures, at desktop (1000px) and mobile (390px) widths, with badges disabled.
  They assert exact titles, annotations and unique filenames, including IDs above
  signed 64-bit range and escaped team names. Each SVG was reopened offline;
  provenance text was checked inside the exported canvas. The downloaded files
  are retained under `exports-*` with SHA-256 checksums.
- Existing browser checks still cover fixture selection, empty-edge states,
  badges, wrapping legends, CSV export and embedding controls.

Reproduce with either recorded interpreter:

```text
python docs/analytics/attention-export-provenance/verify.py focused
python docs/analytics/attention-export-provenance/verify.py browser
```

JSON receipts contain exact commands, environment, source/runtime hashes and
log/XML checksums. Runtime source is isolated HEAD plus the owned patch; source
hashes remained unchanged throughout each run. All fixtures are synthetic.
No model mathematics, training, predictions, GPU runs or collectors were changed.
