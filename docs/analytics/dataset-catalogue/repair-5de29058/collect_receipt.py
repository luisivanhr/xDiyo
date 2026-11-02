"""Collect immutable audit-repair receipts without reading private catalogue data."""
from datetime import datetime,timezone
import hashlib,json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent


def sha(p,normalize=False):
    raw=Path(p).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n",b"\n") if normalize else raw).hexdigest()


def main():
    suites={}
    for name in ('python314','python312'):
        s=ET.parse(OUT/(name+'.xml')).getroot().find('testsuite')
        assert int(s.attrib['tests'])=={'python314':307,'python312':306}[name], 'Incomplete or stale receipt'
        assert all(int(s.attrib[k])==0 for k in ('errors','failures','skipped'))
        suites[name]={k:s.attrib[k] for k in ('tests','errors','failures','skipped','time','timestamp')}
    interpreters=['C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe','C:/Users/luisi/Documents/Programming/Python/xDiyo/.venv/Scripts/python.exe']
    environments={p:json.loads(subprocess.check_output([p,'-c',"import json,sys,pandas,numpy,pyarrow;print(json.dumps(dict(python=sys.version,pandas=pandas.__version__,numpy=numpy.__version__,pyarrow=pyarrow.__version__)))"])) for p in interpreters}
    names=('test_native_dated_aliases','test_named_datasets','test_named_dataset_repairs','test_recipe_dataset_migration','test_ui_workflow','test_loading','test_team_history','test_fixture_assembly','test_feature_composition','test_datasets','test_ui_recipe_cutoffs','test_recipe_refresh','test_venue_history')
    paths=[*ROOT.glob('src/xdiyo_analytics/data/catalogue/*.py'),ROOT/'pyproject.toml',*[ROOT/'tests/analytics'/(name+'.py') for name in names],Path(__file__)]
    result=dict(base_commit='5de290583e975de044e362655a9f1272cb2a1593',created_at=datetime.now(timezone.utc).isoformat(),tests=suites,environments=environments,
        source_hash_contract='SHA256, CRLF normalized to LF',source_hashes={p.relative_to(ROOT).as_posix():sha(p,True) for p in sorted(paths)},
        command='PYTHONPATH=src <interpreter> -m pytest '+' '.join('tests/analytics/'+n+'.py' for n in names)+' -q --tb=short --junitxml=<receipt>.xml',
        pandas3_qualification='Add -k "not installed_wheel". The installed-wheel test is run only in Python 3.14; it is not a missing-package skip.',
        boundaries=['No private data reimport or mutation; no production fitting or collector changes.','No browser changes or new browser claim.','Source before repair reproduced canonical home ID 1 instead of expected 22 in the September case.'],
        receipt_hashes={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.suffix in ('.xml','.log')})
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(suites,indent=2))

if __name__=='__main__':main()
