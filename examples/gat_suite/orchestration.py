"""Manifest-only sequential workers, bounded engineering retries and durable status."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import psutil
import pandas as pd
from . import core
from compare_gat_temporal import write_json


def status(run,**values):
    path=run/'PROGRESS.json'
    current=json.loads(path.read_text()) if path.exists() else {}
    current.update(values,updated_at=pd.Timestamp.now(tz='UTC').isoformat())
    write_json(path,current)


def run(run):
    run=Path(run).resolve();p,_=core.load(run,frozen=True)
    manifest=pd.read_csv(run/'RUN_MANIFEST.csv')
    if set(manifest.job_id)!=set(j['job_id'] for j in p['matrix']):raise ValueError('Mutable ledger disagrees with frozen matrix.')
    previous=json.loads((run/'PROGRESS.json').read_text())
    worker=previous.get('worker')
    if worker and worker.get('pid'):
        try:
            proc=psutil.Process(worker['pid']);command=proc.cmdline()
            if 'run_gat_suite.py' in ' '.join(command) and str(run) in ' '.join(command):
                print(json.dumps(dict(stage='waiting_for_existing_worker',pid=proc.pid)),flush=True)
                while proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE:time.sleep(5)
        except psutil.NoSuchProcess:pass
    command=[sys.executable,str(core.ROOT/'examples/run_gat_suite.py')]
    environment=os.environ.copy();environment.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def child(arguments,log,stage,job=None):
        with log.open('w',encoding='utf-8') as stream:
            process=subprocess.Popen([*command,*arguments,'--run',str(run)],cwd=core.ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            status(run,stage=stage,worker=dict(pid=process.pid,command=arguments,job=job,log=str(log.relative_to(run))))
            code=process.wait()
        status(run,worker=None)
        return code
    if not (run/'shared/baselines/COMPLETE.json').exists():
        code=child(['baselines'],run/'shared/baselines/workers.log','baselines')
        if code:raise RuntimeError('Baseline worker failed; inspect shared/baselines/workers.log and resume after repair.')
    def publish():
        temp=run/'RUN_MANIFEST.csv.tmp';manifest.to_csv(temp,index=False);temp.replace(run/'RUN_MANIFEST.csv')
        status(run,planned=108,completed=int(manifest.status.eq('completed').sum()),reused=0,
            failed=int(manifest.status.eq('failed').sum()),blocked=int(manifest.status.eq('blocked').sum()))
    for study in range(1,5):
        for job in [j for j in p['matrix'] if j['study']==study]:
            folder=run/job['path'];folder.mkdir(parents=True,exist_ok=True);index=manifest.index[manifest.job_id.eq(job['job_id'])][0]
            if (folder/'result.json').exists():
                result=json.loads((folder/'result.json').read_text())
                if result['job']!=job:raise ValueError('Existing result identity mismatch.')
                manifest.loc[index,'status']='completed';publish();continue
            while len(list(folder.glob('attempt_*')))<3:
                attempt=len(list(folder.glob('attempt_*')))+1
                manifest.loc[index,'status']='running';publish()
                print(json.dumps(dict(study=study,job=job['job_id'],attempt=attempt,completed=int(manifest.status.eq('completed').sum()),planned=108)),flush=True)
                code=child(['fit','--job',job['job_id']],folder/f'worker_{attempt:02d}.log','fitting',job['job_id'])
                if code==0 and (folder/'result.json').exists():
                    manifest.loc[index,'status']='completed';publish();break
                manifest.loc[index,'status']='failed';publish()
                write_json(folder/f'worker_failure_{attempt:02d}.json',dict(exit_code=code,attempt=attempt,when=pd.Timestamp.now(tz='UTC').isoformat()))
                # A retry is exact continuation/restart with unchanged identities.
                # No hyperparameter or scientific intervention is changed.
                if not (folder/f'attempt_{attempt:02d}').exists():
                    (folder/f'attempt_{attempt:02d}').mkdir()
            if not (folder/'result.json').exists():manifest.loc[index,'status']='blocked';publish()
        needed=manifest[manifest.study<=study]
        if needed.status.eq('completed').all():
            code=child(['report','--study',str(study)],run/core.STUDIES[study-1]/'reporting.log','reporting')
            if code:raise RuntimeError(f'Study {study} report failed; preserve fits and repair reporting before continuing.')
            print(json.dumps(dict(study=study,status='verified_report_complete',completed=int(manifest.status.eq('completed').sum()))),flush=True)
    if not manifest.status.eq('completed').all():
        status(run,stage='incomplete',worker=None)
        write_json(run/'INCOMPLETE.json',dict(jobs=manifest[~manifest.status.eq('completed')].to_dict('records')))
        return
    if child(['report'],run/'summary.log','summary'):raise RuntimeError('Cross-study report failed.')
    if child(['audit'],run/'audit.log','completion_audit'):raise RuntimeError('Evidence audit failed.')
    status(run,stage='complete',worker=None,completed=108)
    print(json.dumps(dict(status='complete',new=108,reused=0,run=str(run))),flush=True)
