"""Reproducible synthetic verification of the isolated publication repair."""
import argparse,hashlib,io,json,os,shutil,subprocess,sys,tempfile,time,zipfile
from pathlib import Path
import importlib.metadata as metadata
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
OWNED=['.gitattributes','src/xdiyo_analytics/features/standings.py','src/xdiyo_analytics/ratings/standings.py','src/xdiyo_analytics/ui/build_inventory.py','src/xdiyo_analytics/ui/inventory.json','tests/analytics/test_derived_standings.py','tests/analytics/test_venue_history.py','tests/analytics/test_standings_serialization.py','tests/analytics/test_standings_missing_fixture.py','tests/analytics/test_standings_available_publication.py','examples/validate_derived_standings.py']
REPLACEMENTS=[('All ranks stay initial until every first-round result is available; no cross-season history.','Publishes eligible results immediately at each cutoff; unknown result release times do not accrue points. No cross-season history.'),('Optional saved exact-cutoff standings snapshots. No reconstruction or fitting during lookup.','Optional current-policy exact-cutoff standings snapshots. Legacy opening-gate artifacts require rebuilding from source. No fitting during lookup.')]
FOCUSED=['derived_standings','standings_available_publication','standings_serialization','standings_missing_fixture','venue_history','standings_publication']
BROADER=FOCUSED+['features','feature_composition','keyed_features','feature_presets_context','ui_recipe_cutoffs','ui_catalog_compatibility','execution_recovery','table_attribute_recovery','venue_spatial','temporal_provenance']

def candidate(name):
    if name=='src/xdiyo_analytics/ui/inventory.json':
        raw=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode()
        for old,new in REPLACEMENTS:raw=raw.replace(old,new)
        return raw.encode()
    return (ROOT/name).read_bytes()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('suite',choices=['focused','broader']);args=ap.parse_args()
    names=FOCUSED if args.suite=='focused' else BROADER
    tag=args.suite+'-py'+str(sys.version_info.major)+str(sys.version_info.minor)
    files=OWNED+[str(Path(__file__).relative_to(ROOT)).replace('\\','/')]
    before={n:hashlib.sha256(candidate(n)).hexdigest() for n in files}
    log=OUT/(tag+'.log');xml=OUT/(tag+'.xml')
    cmd=[sys.executable,'-m','pytest',*[f'tests/analytics/test_{n}.py' for n in names],'-q','--tb=short','--maxfail=5','--junitxml='+str(xml)]
    start=time.perf_counter();base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    scratch=ROOT/'.pytest_tmp';scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='standings-publication-',dir=scratch) as td:
        runtime=Path(td)
        raw=subprocess.check_output(['git','archive','--format=zip','HEAD','src','pyproject.toml','docs/analytics/team_assets'],cwd=ROOT)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:z.extractall(runtime)
        for n in OWNED:
            if n.startswith('src/'):(runtime/n).write_bytes(candidate(n))
        runtime_hashes={str(p.relative_to(runtime)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in (runtime/'src').rglob('*') if p.is_file()}
        with log.open('w',encoding='utf-8') as stream:result=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(runtime/'src')),stdout=stream,stderr=subprocess.STDOUT)
    elapsed=time.perf_counter()-start
    after={n:hashlib.sha256(candidate(n)).hexdigest() for n in files}
    counts={k:sum(int(s.get(k,0)) for s in ET.parse(xml).getroot().findall('testsuite')) for k in ['tests','failures','errors','skipped']}
    receipt=dict(base_commit=base,command=cmd,python=sys.version,packages={n:metadata.version(n) for n in ['pandas','numpy','pytest']},source_hashes=before,runtime_source_hashes=runtime_hashes,source_unchanged=before==after,counts=counts,exit_code=result.returncode,elapsed_seconds=elapsed,receipt_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [log,xml]},scope='HEAD plus owned patch. Only two own inventory help strings overlaid; unrelated reporter edits excluded. Synthetic tests; no experiment retraining.')
    (OUT/(tag+'-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['counts','exit_code','elapsed_seconds','source_unchanged']}),flush=True)
    return result.returncode if before==after else 2
if __name__=='__main__':raise SystemExit(main())
