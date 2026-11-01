"""Collect repair evidence without copying private datasets or models."""
from datetime import datetime,timezone
import hashlib,json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent


def sha(path,normalize=False):
    data=Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n",b"\n") if normalize else data).hexdigest()


def main():
    tests={}
    for name in ('python314','python312','additional314','additional312','browser'):
        suite=ET.parse(OUT/(name+'.xml')).getroot().find('testsuite')
        assert int(suite.attrib['errors'])==int(suite.attrib['failures'])==int(suite.attrib['skipped'])==0
        tests[name]={k:suite.attrib[k] for k in ('tests','errors','failures','skipped','time','timestamp')}
    py314='C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe'
    py312='C:/Users/luisi/Documents/Programming/Python/xDiyo/.venv/Scripts/python.exe'
    envs={p:json.loads(subprocess.check_output([p,'-c',"import sys,json,pandas,numpy,pyarrow;print(json.dumps(dict(python=sys.version,pandas=pandas.__version__,numpy=numpy.__version__,pyarrow=pyarrow.__version__)))"])) for p in (py314,py312)}
    sources=[*ROOT.glob('src/xdiyo_analytics/data/catalogue/*.py'),ROOT/'pyproject.toml',ROOT/'src/xdiyo_analytics/ui/recipe.py',ROOT/'src/xdiyo_analytics/ui/static/named-data.js',
        *[ROOT/'tests/analytics'/n for n in ('test_named_datasets.py','test_named_dataset_repairs.py','test_recipe_dataset_migration.py','test_named_dataset_browser.py')],
        Path(__file__),OUT/'verify_private.py']
    files=' '.join('tests/analytics/'+n+'.py' for n in ('test_named_datasets','test_named_dataset_repairs','test_recipe_dataset_migration','test_ui_workflow','test_loading','test_team_history','test_fixture_assembly','test_feature_composition','test_datasets','test_ui_recipe_cutoffs','test_recipe_refresh','test_venue_history'))
    private=json.loads((OUT/'private-verification.json').read_text(encoding='utf-8'));assert private['private_files_unchanged'] and private['models_fitted']==0
    images=sorted((ROOT/'.pytest_tmp').rglob('named-data-*.png'),key=lambda p:p.stat().st_mtime)
    evidence=dict(created_at=datetime.now(timezone.utc).isoformat(),base_commit='d21eb1344cd28dcd761b97bcf6d369bef137c64b',
        environments=envs,tests=tests,source_hash_contract='SHA256 with CRLF normalized to LF',
        source_hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p,True) for p in sorted(sources)},
        commands=dict(broader='PYTHONPATH=src <python314 or python312> -m pytest '+files+' -q --tb=short --junitxml=<receipt>.xml',
            pandas3_qualification='Same suite, -k "not installed_wheel"; wheel test ran under Python 3.14 to avoid concurrent build-directory mutation.',
            supplemental='The two final tests test_reviewed_date_correction_cannot_cross_train_test and test_invalid_json_object_is_a_batch_blocker ran separately in each environment; their receipts are additional314/312.xml. Re-running the final files collects 294/293 cases.',
            browser='PYTHONPATH=src <python314> -m pytest tests/analytics/test_named_dataset_browser.py -q --tb=short --junitxml=<receipt>.xml',
            private='PYTHONPATH=src <python314> docs/analytics/dataset-catalogue/repair-d21eb134/verify_private.py --catalogue <unchanged private catalogue> --output docs/analytics/dataset-catalogue/repair-d21eb134/private-verification.json'),
        browser=dict(engine='Playwright Chromium',reason='Browser plugin unavailable',viewports=[1200,390],
            checks=['native authenticated page identity','meaningful rendered content','no page or console errors','no horizontal overflow','accept and legacy restore','delayed input/root cancellation','newest response wins','large IDs preserved as decimal strings'],
            screenshots=[dict(path=str(p),sha256=sha(p)) for p in images[-2:]]),
        limitations=['No reviewed coverage expansion or source reimport; existing union and saved selections resolved read-only.',
            'Migration blocks unsupported timing/history/graph/membership dependencies instead of guessing.',
            'Retrospective evidence remains conditional; no new historical availability facts were invented.',
            'Only Chromium desktop/mobile tested; no production fitting or saved experiment rebinding. Broader tests include small synthetic estimator fits.',
            'Windows hard-link tests ran outside the sandbox; no atomic publication guarantee was weakened.'],
        receipt_hashes={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.suffix in ('.json','.xml','.log') and p.name!='verification.json'})
    (OUT/'verification.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(tests=tests,screenshots=evidence['browser']['screenshots']),indent=2))

if __name__=='__main__':main()
