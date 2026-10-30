"""Reproduce bounded contract/integration receipts, never production experiments."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import io
import zipfile
import tempfile
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
OWNED=['.gitattributes', 'src/xdiyo_analytics/reporting/attention_view.py', 'src/xdiyo_analytics/reporting/graphs.py', 'src/xdiyo_analytics/reporting/viewer.py', 'src/xdiyo_analytics/reporting/fixture_attention.py', 'src/xdiyo_analytics/ui/graph_inventory.py', 'src/xdiyo_analytics/ui/inventory.json', 'tests/analytics/test_graph_integration.py', 'tests/analytics/test_graph_reporter_browser.py', 'tests/analytics/test_fixture_attention.py']
SUITES={'focused': ['fixture_attention', 'graph_integration', 'reporting_viewer', 'ui_catalog_compatibility', 'report_refresh_recovery', 'table_attribute_recovery'], 'broader': ['fixture_attention', 'graph_integration', 'reporting_viewer', 'ui_catalog_compatibility', 'report_refresh_recovery', 'table_attribute_recovery', 'temporal_graph', 'graph_extensions', 'graph_integration_gates', 'composition_review_graph', 'ui_workflow'], 'browser': ['graph_reporter_browser']}

def hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [*OWNED,str(Path(__file__).relative_to(ROOT)).replace('\\','/')]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('suite',choices=SUITES);args=parser.parse_args()
    tag=f'{args.suite}-py{sys.version_info.major}{sys.version_info.minor}'
    log=OUT/(tag+'.log');xml=OUT/(tag+'.xml')
    command=[sys.executable,'-m','pytest',*[f'tests/analytics/test_{name}.py' for name in SUITES[args.suite]],
             '-q','--tb=short','--maxfail=8','-o','faulthandler_timeout=90','--junitxml='+str(xml)]
    before=hashes();started=time.time()
    # Test an isolated HEAD plus the owned reporter patch.
    scratch=ROOT/'.pytest_tmp';scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='xdiyo-temporal-verify-',dir=scratch) as directory:
        runtime=Path(directory).resolve()
        assert runtime.parent==scratch.resolve()
        archive=subprocess.check_output(['git','archive','--format=zip','HEAD','src','pyproject.toml','docs/analytics/team_assets'],cwd=ROOT)
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:bundle.extractall(runtime)
        for name in OWNED:
            if name.startswith('src/'):
                target=runtime/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/name,target)
        runtime_hashes={str(p.relative_to(runtime)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (runtime/'src').rglob('*') if p.is_file()}
        env=dict(os.environ,PYTHONPATH=str(runtime/'src'))
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    elapsed=time.time()-started;after=hashes()
    counts={}
    if xml.exists():
        suites=ET.parse(xml).getroot().findall('testsuite')
        counts={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ['tests','failures','errors','skipped']}
    receipt=dict(command=command,cwd=str(ROOT),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        python=sys.version,executable=sys.executable,packages={name:importlib.metadata.version(name) for name in ['pandas','numpy','scikit-learn','pytest']},
        elapsed_seconds=elapsed,exit_code=result.returncode,counts=counts,source_hashes=before,source_unchanged=before==after,runtime_source_hashes=runtime_hashes,
        runtime_scope="Isolated HEAD src plus owned patch; unrelated working-tree source edits excluded.",
        receipt_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [log,xml] if p.exists()},
        scope='Synthetic regression fixtures only; no real-data experiment or collector changes.')
    (OUT/(tag+'-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ['counts','elapsed_seconds','exit_code','source_unchanged']}),flush=True)
    return result.returncode if before==after else 2

if __name__=='__main__':raise SystemExit(main())
