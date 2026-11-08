"""Read frozen artifacts and emit a compact, reproducible validation receipt."""
from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd

REPO=Path(__file__).resolve().parents[2]

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify(run,output):
    run=Path(run).resolve();output=Path(output)
    source=[]
    for path in (REPO/'src').rglob('*.py'):
        frozen=run/path.relative_to(REPO)
        if sha(path)!=sha(frozen):raise ValueError('Frozen source differs: '+str(path))
        source.append(path.relative_to(REPO).as_posix())
    for name in ('run.py','experiment_inputs.py','prepare_data.py','prepare_odds.py','preflight.py','analyze_results.py','report_new.py'):
        if sha(REPO/'examples/workbook_actual_date'/name)!=sha(run/name):raise ValueError('Active run script differs: '+name)
    preflight=json.loads((run/'results/PREFLIGHT.json').read_text())
    assert preflight['status']=='PASS' and len(preflight['smoke'])==9
    for item in preflight['smoke']:
        assert sha(item['path'])==item['sha256']
        fit=json.loads(Path(item['path']).read_text())
        assert fit['summary']['retained_step']==1 and fit['cuda_peak_allocated']>0
    unit=json.loads((run/'results/RUNNER_UNIT_TEST.json').read_text());assert unit['status']=='PASS'
    seal=json.loads((run/'results/SOURCE_SEAL.json').read_text())
    for path,digest in seal.items():
        if sha(path)!=digest:raise ValueError('Seal differs: '+path)
    populations=json.loads((run/'data/MANIFEST.json').read_text())
    for inventory in json.loads((run/'data/source_inventory.json').read_text()):
        assert sha(inventory['path'])==inventory['sha256']
    shifts=[]
    for stage in populations:
        schedule=pd.read_parquet(run/'data'/stage/'schedule.parquet')
        known=schedule.loc[schedule['round'].notna() & schedule.round_stage.notna()]
        keys=['competition_id','season_id','round_stage','round']
        anchors=known.groupby(keys).kickoff_at.transform('min').dt.tz_convert('Europe/Berlin').dt.normalize().dt.tz_convert('UTC')
        delta=(known.issuance_at-anchors).dt.total_seconds()/86400
        for i,year in enumerate(range(2022,2026),1):
            dates=known.kickoff_at.dt.tz_convert('Europe/Berlin')
            inside=dates.ge(pd.Timestamp(f'{year}-07-01',tz='Europe/Berlin'))&dates.lt(pd.Timestamp(f'{year+1}-07-01',tz='Europe/Berlin'))
            shifts.append(dict(stage=stage,fold=i,comparable_numeric_round_fixtures=int(inside.sum()),
                later_playing_date_cutoff_than_same_round_first_played_date=int((inside&delta.gt(0)).sum()),
                more_than_seven_days=int((inside&delta.gt(7)).sum()),
                caveat='This compares reconstructed group anchors and includes ordinary within-round spread; it is not a count of proven postponements.'))
    result=dict(status='PASS',run=str(run),source_files_verified=len(source),sealed_files=len(seal),
        source_seal_sha256=sha(run/'results/SOURCE_SEAL.json'),preflight=preflight,runner_recovery=unit,
        stages={k:{field:v[field] for field in ('fixtures','teams','leagues','ambiguous_clocks','phase_coverage','populations')} for k,v in populations.items()},
        original13_test_cohort=18736,opening_prices_valid=18708,opening_prices_missing=28,
        original13_mapping_commit='bda89fec1906a32557b4df799289e1d06430004a',
        v59_manifest_sha256=sha(REPO/'data/fixture_rounds/v59_20261108/manifest.json'),
        original13_manifest_sha256=sha(REPO/'data/fixture_rounds/original13_native_20261108/manifest.json'),
        timing_comparisons=shifts,limitations=[
            'Countdown missing throughout; B/CD_DW active context is window-10 form rank.',
            'Observed-member form ranks differ from earlier full-roster native standings.',
            'Unknown workbook timezone: declared Berlin civil-clock convention; 29 repeated hours use later bound.',
            '799 original13 annotation holds, including 367 in the experiment context window, remain null.',
            'Opening quote capture time and bookmaker are unavailable.',
            'Older native-source research differs in source, timing and context; this is a new comparison.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',sealed_files=len(seal),source_files=len(source),source_seal_sha256=result['source_seal_sha256']),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('output');verify(**vars(p.parse_args()))
