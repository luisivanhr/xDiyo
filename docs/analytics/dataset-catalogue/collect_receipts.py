"""Collect final local audit outputs; no original or derived data is copied."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET


def sha(path, normalize=False):
    raw=Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b'\r\n',b'\n') if normalize else raw).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--private-root',required=True);p.add_argument('--scratch',default='.pytest_tmp');a=p.parse_args()
    out=Path(__file__).resolve().parent;scratch=Path(a.scratch)
    first=json.loads((scratch/'named-published-first.json').read_text(encoding='utf-8'))
    repeated=json.loads((Path(a.private_root)/'import-receipt.json').read_text(encoding='utf-8'))
    assert first['versions']==repeated['versions']
    assert first['coverage']==repeated['coverage']
    assert first['input_checksums']==repeated['input_checksums']
    (out/'import-repeat.json').write_text(json.dumps(dict(first=first,repeated=repeated,identical_versions_and_coverage=True),indent=2)+'\n',encoding='utf-8')
    audit=json.loads((scratch/'named-real-verification.json').read_text(encoding='utf-8'))
    assert audit['versions']==repeated['versions'] and audit['frozen_membership_and_split_equal']
    shutil.copyfile(scratch/'named-real-verification.json',out/'input-verification.json')
    suites={}
    for name in ['named-regression-final','named-pandas3-final','named-browser-final']:
        for suffix in ['.xml','.log']:shutil.copyfile(scratch/(name+suffix),out/(name+suffix))
        xml=ET.parse(out/(name+'.xml')).getroot().find('testsuite')
        assert int(xml.attrib['errors'])==int(xml.attrib['failures'])==int(xml.attrib['skipped'])==0
        suites[name]={k:xml.attrib[k] for k in ['tests','failures','errors','skipped','time','timestamp']}
    files=[*Path('src/xdiyo_analytics/data/catalogue').glob('*.py'),
        *map(Path,['src/xdiyo_analytics/data/__init__.py','src/xdiyo_analytics/datasets/assembly.py',
        'src/xdiyo_analytics/features/evaluation.py','src/xdiyo_analytics/ui/recipe.py','src/xdiyo_analytics/ui/server.py',
        'src/xdiyo_analytics/ui/static/app.js','src/xdiyo_analytics/ui/static/named-data.js','src/xdiyo_analytics/ui/static/style.css',
        'examples/import_named_datasets.py','examples/named_dataset_recipe.py','tests/analytics/test_named_datasets.py',
        'tests/analytics/test_named_dataset_browser.py','docs/analytics/dataset-catalogue/verify_local.py','docs/analytics/dataset-catalogue/collect_receipts.py'])]
    envs={}
    for py in ['C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe','C:/Users/luisi/Documents/Programming/Python/xDiyo/.venv/Scripts/python.exe']:
        envs[py]=json.loads(subprocess.check_output([py,'-c',"import sys,json,pandas,numpy,pyarrow,sklearn;print(json.dumps(dict(python=sys.version,pandas=pandas.__version__,numpy=numpy.__version__,pyarrow=pyarrow.__version__,sklearn=sklearn.__version__)))"]))
    tests='tests/analytics/test_named_datasets.py tests/analytics/test_ui_workflow.py tests/analytics/test_loading.py tests/analytics/test_team_history.py tests/analytics/test_fixture_assembly.py tests/analytics/test_feature_composition.py tests/analytics/test_datasets.py tests/analytics/test_ui_recipe_cutoffs.py tests/analytics/test_recipe_refresh.py tests/analytics/test_venue_history.py'
    receipt=dict(created_at=datetime.now(timezone.utc).isoformat(),base_commit='a0f466712e2bc4f44eea86d382d56658a9cc1d9b',
        source_hash_contract='SHA256 of UTF-8 source bytes with CRLF normalized to LF; receipts retain exact bytes through .gitattributes',
        source_hashes={f.as_posix():sha(f,True) for f in sorted(files)},environments=envs,tests=suites,
        commands={'regression':'PYTHONPATH=src <each Python above> -m pytest '+tests+' -q --tb=short --junitxml=<receipt>.xml',
          'browser':'PYTHONPATH=src <Python 3.14 above> -m pytest tests/analytics/test_named_dataset_browser.py -q --tb=short --junitxml=<receipt>.xml',
          'import':'PYTHONPATH=src <Python 3.14 above> examples/import_named_datasets.py --legacy-root data/xDiyo_data --results-txt <unchanged v2> --workbook <unchanged workbook> --output '+a.private_root,
          'verification':'PYTHONPATH=src <Python 3.14 above> docs/analytics/dataset-catalogue/verify_local.py --catalogue '+a.private_root+' --v1 <unchanged v1> --v2 <unchanged v2> --output .pytest_tmp/named-real-verification.json'},
        browser={'engine':'Playwright Chromium','plugin_fallback':'Browser plugin unavailable','viewports':[1200,390],
          'checks':['preview before acceptance','frozen selection stored without JS integer rounding','selected team discovery','legacy restore','no console/page errors','no horizontal overflow'],
          'screenshots':'Local .pytest_tmp/deb75866424c408d8585ac7722258721/test_named_dataset_preview_acc{0,1}/named-data-{1200,390}.png'},
        notes=['Two sandbox-only hard-link permission failures passed outside the sandbox; no publication semantics were weakened.',
          'Fresh import exposed an accented reserve-name encoding issue; fixed and covered by a native classification regression before final publication.',
          'No frozen or real-data model experiment was trained. Regression tests include small native estimator fits.',
          'Input files and existing dataset publications were not modified; only new ignored catalogue outputs were written.',
          'Unresolved identities/classifications remain quarantined; coverage is the reviewed initial subset.',
          'No original or derived dataset Parquets are committed; redistribution rights are unconfirmed.'],
        receipt_hashes={f.name:sha(f) for f in sorted(out.iterdir()) if f.suffix in ('.xml','.log','.json') and f.name!='verification.json'})
    (out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'tests':suites,'versions':repeated['versions']},indent=2))

if __name__=='__main__':main()
