"""Promote checked documentation statuses after the successful read-only audit."""
from support import ROOT, guard
guard()
changes = {
    'IMPLEMENTATION_PROGRESS.md': [
        ('Splits and CV — runtime/documentation checks passed; final audit pending', 'Splits and CV — implemented and verified'),
        ('The final stable-source/preservation audit is pending.', 'The final audit confirms 33 stable source files, 164 unchanged earlier files, 20 unchanged prepared-source/selection files and 58 valid local links. All six earlier notebooks and the read-only user reference notebook are unchanged. Six original documents are archived under `docs/analytics/archive/pre_splits_implementation/`.'),
        ('Temporal splits, model adapters, reports and general orchestration remain later work.', 'The newer split/CV batch covers fold membership; model adapters, reports and general orchestration remain later work.'),
    ],
    'football_analytics_working_notes.md': [
        ('The new splits/CV layer has passed runtime, real-data and documentation checks; final preservation audit is pending.', 'The new splits/CV layer is implemented and verified, including runtime, real-data, documentation and preservation checks.'),
        ('Splits and CV — implementation checked,', 'Splits and CV — implemented and verified,'),
        ('Final preservation audit is pending.', 'The final audit confirms 33 stable source files, 164 unchanged earlier files,\n20 unchanged prepared-source/selection files and 58 valid local links. All six\nearlier notebooks and the read-only reference notebook are unchanged.'),
        ('[verification evidence](docs/analytics/labels_check.json). Final X/y assembly,\nsplitting, training and reporting orchestration remain later work. (D)', '[verification evidence](docs/analytics/labels_check.json). The newer assembly\nand split/CV batches consume these labels; training and reporting orchestration\nremain later work. (D)'),
    ],
    'coaching_history.md': [
        ('Final preservation audit is pending.', 'Final preservation checks pass: 33 stable source files, 164 unchanged earlier files, 20 unchanged prepared-source/selection files, six prior notebooks and the read-only reference notebook.'),
    ],
    'src/README.md': [
        ('rendered equations pass; final preservation audit is pending.', 'rendered equations and the final preservation audit pass.'),
    ],
}
pending = {}
for name, replacements in changes.items():
    text = (ROOT/name).read_text(encoding='utf-8-sig')
    for old,new in replacements:
        assert text.count(old)==1, ('Expected one status fragment',name,old)
        text = text.replace(old,new)
    pending[name] = text
for name,text in pending.items():(ROOT/name).write_text(text,encoding='utf-8')
assert guard()
print({'updated':list(pending)})
