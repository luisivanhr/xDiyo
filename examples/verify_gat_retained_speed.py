"""Standalone, sealed speed acceptance queue; does not run a research grid."""
from pathlib import Path
import argparse,subprocess,sys,os,json,hashlib,traceback,threading
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--research',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
flags=('omit_diagnostic_replay','pack_validation','cache_indices','borrow_inputs','omit_unused','all')
jobs=[dict(mode='short',family=f,flag=flag,name=f.replace('/','__')+'/'+flag)
      for flag in flags for f in ('time_B/N','time_C/E1','time_B/CD_DW')]
jobs.append(dict(mode='short',family='time_C/E1',flag='all',budget=1024**2,name='low_budget'))
jobs.extend([dict(mode='full',family='time_C/E1',flag='all',name='full_103'),dict(mode='timing',family='time_C/E1',flag='all',name='timing')])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source={p.relative_to(ROOT).as_posix():sha(p) for folder in ('src/xdiyo_analytics/graphs','src/xdiyo_analytics/training') for p in sorted((ROOT/folder).rglob('*.py'))}
for name in ('examples/gat_retained/reference_inputs.json','examples/gat_retained/replay.py','tests/analytics/gat_retained_speed_worker.py'):
    source[name]=sha(ROOT/name)
S=a.research/'experiments/gat_speed_residuals_20261102_01/00_speedup/parity'
refs={}
for family in ('time_B_N','time_C_E1','time_B_CD_DW'):
    for name in ('001.pt','002.pt','003.pt','inputs.json','PASS.json'):
        file=S/family/'unchanged_a'/name;refs[file.relative_to(a.research).as_posix()]=sha(file)
for file in sorted((S/'full_C_E1_fold_2_seed_17/reference').iterdir()):
    if file.is_file():refs[file.relative_to(a.research).as_posix()]=sha(file)
protocol=dict(schema=1,short_check_workers=2,timing_workers=1,source=source,reference_hashes=refs,jobs=jobs,tolerance='exact',
              evaluation='Frozen-population engineering parity only; no new outer predictions or scores.',
              timing='Only after all short and full gates pass; ABBA BAAB, 3 complete updates per trial; cold/warm separate.')
path=a.output/'PROTOCOL.json'
if path.exists():
    assert json.loads(path.read_text())==protocol,'Candidate source or archive changed; do not resume this sealed queue.'
else:path.write_text(json.dumps(protocol,indent=2))
env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT/'examples'),PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
completed=[];active=[];lock=threading.Lock()
def progress():
    path=a.output/'PROGRESS.tmp'
    path.write_text(json.dumps(dict(completed=len(completed),total=len(jobs),active=active),indent=2))
    path.replace(a.output/'PROGRESS.json')

def execute(job):
    assert all(sha(ROOT/name)==digest for name,digest in source.items()),'Candidate source changed during the sealed queue.'
    folder=a.output/job['name'];folder.mkdir(parents=True,exist_ok=True)
    if not (folder/'PASS.json').exists():
        with lock:active.append(job);progress()
        cmd=[sys.executable,str(ROOT/'tests/analytics/gat_retained_speed_worker.py'),'--research',str(a.research),'--output',str(folder),
             '--mode',job['mode'],'--family',job['family'],'--flag',job['flag']]
        if 'budget' in job:cmd+=['--budget',str(job['budget'])]
        with (folder/'worker.log').open('a') as log:
            run=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        with lock:active.remove(job)
        if run.returncode:raise RuntimeError('Verification failed: '+str(folder/'worker.log'))
    with lock:
        completed.append(job['name']);progress()
        print('passed',len(completed),'/',len(jobs),job['name'],flush=True)

try:
    # Parallel short checks compare exact values only. Their runtimes are not
    # used as speed measurements. Full acceptance and ABBA timing are serial.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(execute,job) for job in jobs if job['mode']=='short']
        for future in as_completed(futures):future.result()
    for job in jobs:
        if job['mode']!='short':execute(job)
    (a.output/'COMPLETE.json').write_text(json.dumps(dict(completed=completed,passed=True),indent=2))
except BaseException as error:
    (a.output/'FAILED.json').write_text(json.dumps(dict(completed=completed,error=repr(error),traceback=traceback.format_exc()),indent=2));raise
