"""Verify the scheduled-cluster policy from exact committed bytes; no model fits."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TESTS = ['test_season_countdown', 'test_season_progress', 'test_ui_workflow',
    'test_ui_inventory', 'test_fixture_assembly', 'test_feature_presets_context',
    'test_ui_catalog_compatibility', 'test_derived_standings', 'test_team_history',
    'test_prediction_fixtures', 'test_labels']


def main():
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    paths = ['src', 'tests/analytics', str(HERE.relative_to(ROOT))]
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--', *paths], cwd=ROOT), 'Commit source before verification'
    # Byte equality is intentional: git diff alone ignores checkout CRLF changes.
    evidence_source = ['src/xdiyo_analytics/features/countdown.py',
        'src/xdiyo_analytics/ui/build_inventory.py', 'src/xdiyo_analytics/ui/inventory.json',
        'src/xdiyo_analytics/ui/countdown_migration.py', 'src/xdiyo_analytics/ui/recipe.py',
        'src/xdiyo_analytics/labels/creation.py', 'tests/analytics/test_season_countdown.py',
        *[f'docs/analytics/season-countdown/{name}' for name in (
            'audit_scheduled.py', 'compare_scheduled.py', 'verify_scheduled.py',
            'scheduled-effective-round-checks.json')]]
    hashes = {}
    for path in evidence_source:
        blob = subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)
        assert (ROOT/path).read_bytes() == blob, f'Execute exact committed bytes: {path}'
        hashes[path] = hashlib.sha256(blob).hexdigest()
    # Preserve every previous receipt. New verification uses only scheduled-*.
    historical = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir()
        if p.is_file() and not p.name.startswith('scheduled-') and p.name not in
        ('README.md', 'audit_scheduled.py', 'compare_scheduled.py', 'verify_scheduled.py')}
    commands = [[sys.executable, '-m', 'pytest',
        *[f'tests/analytics/{name}.py' for name in TESTS], '-q', '--disable-warnings', '--tb=short',
        '--junitxml=docs/analytics/season-countdown/scheduled-tests.xml'],
        [sys.executable, str(HERE/'audit_scheduled.py')],
        [sys.executable, str(HERE/'compare_scheduled.py')]]
    for name, command in zip(['tests','audit','comparison'], commands):
        with (HERE/f'scheduled-{name}.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'{name}: passed', flush=True)
    for p, digest in historical.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, f'Historical evidence changed: {p}'
    artifacts = [*HERE.glob('scheduled-*.csv'), *HERE.glob('scheduled-*.log'),
        HERE/'scheduled-tests.xml', HERE/'scheduled-comparison.json']
    # Canonical LF publication prevents a second checkout-vs-Git hash mismatch.
    for p in artifacts:
        p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n'))
        hashes[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    suite = ET.parse(HERE/'scheduled-tests.xml').getroot().find('testsuite')
    audit = pd.read_csv(HERE/'scheduled-publication-audit.csv')
    comparison = json.loads((HERE/'scheduled-comparison.json').read_text(encoding='utf-8'))
    receipt = dict(source_commit=commit, python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
        tests=suite.attrib, countdown_tests=sum('test_season_countdown' in t.get('classname','') for t in suite.findall('testcase')),
        publication_audit=dict(publications=len(audit), fixtures=int(audit.matches.sum()), eligible=int(audit.eligible.sum()),
            excluded=int(audit.excluded.sum()), unexplained_missing=int(audit.unexplained_missing.sum()),
            independent_reference=bool(audit.independent_reference.all()), hand_verified_rounds=int(audit.hand_verified.sum())),
        comparison=comparison, commands=commands, sha256=hashes,
        hash_basis='Verified source bytes equal Git blobs; generated artifacts use canonical LF bytes.',
        previous_receipts_preserved=True,
        limitations=['Targeted regression suite, not full application suite.',
            'Historical exports may contain revised dates; unavailable historical schedule knowledge is not reconstructed.',
            'Sparse schedules may not identify a displaced fixture; full schedule context is recommended.',
            'No batch-round override, schedule_mode, model training or frozen experiment updates.'])
    (HERE/'scheduled-verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(source_commit=commit,tests=suite.get('tests'),changed=comparison['changed'],audited=int(audit.matches.sum()))))


if __name__ == '__main__':
    main()
