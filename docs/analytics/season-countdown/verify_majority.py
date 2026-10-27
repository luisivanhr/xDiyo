"""Regenerate majority-policy evidence from committed source; no model fits."""
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
    'test_ui_catalog_compatibility', 'test_derived_standings', 'test_team_history']


def main():
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    # Do not certify source that changed after the recorded commit.
    paths = ['src', 'tests/analytics/test_season_countdown.py',
        str(HERE.relative_to(ROOT) / 'audit.py'),
        str(HERE.relative_to(ROOT) / 'compare_majority.py'),
        str(HERE.relative_to(ROOT) / 'verify_majority.py')]
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--', *paths], cwd=ROOT), 'Commit source before verification'
    commands = [[sys.executable, '-m', 'pytest',
        *[f'tests/analytics/{name}.py' for name in TESTS], '-q', '--disable-warnings', '--tb=short',
        '--junitxml=docs/analytics/season-countdown/majority-tests.xml'],
        [sys.executable, str(HERE / 'audit.py')],
        [sys.executable, str(HERE / 'compare_majority.py')]]
    for name, command in zip(['tests', 'audit', 'comparison'], commands):
        with (HERE / f'majority-{name}.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'{name}: passed', flush=True)
    suite = ET.parse(HERE / 'majority-tests.xml').getroot().find('testsuite')
    audit = pd.read_csv(HERE / 'majority-publication-audit.csv')
    comparison = json.loads((HERE / 'majority-comparison.json').read_text(encoding='utf-8'))
    tracked = subprocess.check_output(['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', source_commit], cwd=ROOT, text=True).splitlines()
    artifacts = [*HERE.glob('majority-*.csv'), *HERE.glob('majority-*.log'), HERE / 'majority-tests.xml', HERE / 'majority-comparison.json']
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in tracked if (ROOT / p).is_file()}
    hashes.update({str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts})
    receipt = dict(source_commit=source_commit, python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
        tests=suite.attrib, countdown_tests=sum('test_season_countdown' in t.get('classname', '') for t in suite.findall('testcase')),
        publication_audit=dict(publications=len(audit), fixtures=int(audit.matches.sum()), eligible=int(audit.eligible.sum()),
            excluded=int(audit.excluded.sum()), unexplained_missing=int(audit.unexplained_missing.sum()),
            independent_reference=bool(audit.independent_reference.all()), hand_verified_rounds=int(audit.hand_verified.sum())),
        comparison=comparison, commands=commands, sha256=hashes,
        browser='Native browser confirmed fresh Season countdown has format/total/team controls and no schedule_mode. See majority-ui.png. Temporary tab/server closed.',
        limitations=['Targeted regression suite, not full application suite.',
            'Observed progress assumes truthful actual kickoff/status and complete season membership.',
            'Legacy schedule_mode recipes require explicit semantic migration; no frozen experiments rerun.'])
    (HERE / 'majority-verification.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(source_commit=source_commit, tests=suite.get('tests'), changed=comparison['changed'], audited=int(audit.matches.sum()))))


if __name__ == '__main__':
    main()
