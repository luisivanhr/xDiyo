"""Recovery entry point: same fixed matrix, numerical scorer repair recorded separately."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import shutil
import psutil
import pandas as pd
from . import core
from compare_gat_temporal import write_json


def execute(run):
    run=Path(run).resolve();p,_=core.load(run,frozen=True)
    ledger=pd.read_csv(run/'RUN_MANIFEST.csv')
    assert set(ledger.job_id)=={j['job_id'] for j in p['matrix']}
    old=json.loads((run/'PROGRESS.json').read_text());pid=old.get('orchestrator_pid')
    if pid and pid!=os.getpid():
        try:
            command=' '.join(psutil.Process(pid).cmdline())
            if 'run_gat_cross_league_recovered.py run' in command:raise RuntimeError('Matching orchestrator already active')
        except psutil.NoSuchProcess:pass
    worker=old.get('worker')
    if worker:
        try:
            process=psutil.Process(worker['pid'])
            if 'run_gat_cross_league_recovered.py' in ' '.join(process.cmdline()):
                while process.is_running() and process.status()!=psutil.STATUS_ZOMBIE:time.sleep(5)
        except psutil.NoSuchProcess:pass
    core.status(run,orchestrator_pid=os.getpid(),worker=None)
    environment=os.environ.copy();environment.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def child(arguments,log,stage,job=None):
        with log.open('w',encoding='utf-8') as stream:
            process=subprocess.Popen([sys.executable,str(core.ROOT/'examples/run_gat_cross_league_recovered.py'),*arguments,'--run',str(run)],
                                     cwd=core.ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            core.status(run,stage=stage,worker=dict(pid=process.pid,job=job,log=str(log.relative_to(run))))
            code=process.wait()
        core.status(run,worker=None)
        return code
    def publish():
        temporary=run/'RUN_MANIFEST.csv.tmp';ledger.to_csv(temporary,index=False);temporary.replace(run/'RUN_MANIFEST.csv')
        core.status(run,completed=int(ledger.status.eq('completed').sum()),failed=int(ledger.status.eq('failed').sum()),
                    blocked=int(ledger.status.eq('blocked').sum()),retry_events=sum(max(0,len(list((run/j['path']).glob('attempt_*')))-1) for j in p['matrix']))
    if not (run/'03_baselines/COMPLETE.json').exists():
        if child(['baselines'],run/'03_baselines/worker.log','baselines'):
            raise RuntimeError('Baseline failure; preserve artifacts, repair and resume')
    for job in p['matrix']:
        folder=run/job['path'];folder.mkdir(parents=True,exist_ok=True)
        index=ledger.index[ledger.job_id.eq(job['job_id'])][0]
        if (folder/'result.json').exists():
            result=json.loads((folder/'result.json').read_text())
            assert result['job']==job and result['identity']['protocol']==core.digest(run/'PROTOCOL.json')
            ledger.loc[index,'status']='completed';publish();continue
        while len(list(folder.glob('attempt_*')))<3:
            # A completed-state resume can fail during export without writing a
            # new continuation file. Carry the latest valid earlier state forward
            # to avoid a fresh statistical fit on the next permitted retry.
            prior=sorted(folder.glob('attempt_*'))
            if prior and not any((prior[-1]/f'{name}.pt').exists() for name in ('completed_training','last_training')):
                for earlier in reversed(prior[:-1]):
                    available=next((name for name in ('completed_training','last_training') if (earlier/f'{name}.pt').exists()),None)
                    if available:
                        meta=json.loads((earlier/f'{available}.json').read_text())
                        assert core.digest(earlier/f'{available}.pt')==meta['sha256']
                        for suffix in ('pt','json'):shutil.copyfile(earlier/f'{available}.{suffix}',prior[-1]/f'{available}.{suffix}')
                        write_json(prior[-1]/'continuation_carried_forward.json',dict(source=str(earlier.relative_to(run)),state=available,sha256=meta['sha256']))
                        break
            attempt=len(list(folder.glob('attempt_*')))+1
            ledger.loc[index,'status']='running';publish()
            print(json.dumps(dict(job=job['job_id'],attempt=attempt,completed=int(ledger.status.eq('completed').sum()),planned=36)),flush=True)
            code=child(['fit','--job',job['job_id']],folder/f'worker_{attempt:02}.log','fitting',job['job_id'])
            if code==0 and (folder/'result.json').exists():
                ledger.loc[index,'status']='completed';publish();break
            ledger.loc[index,'status']='failed';publish()
            write_json(folder/f'failure_{attempt:02}.json',dict(exit_code=code,attempt=attempt))
            (folder/f'attempt_{attempt:02}').mkdir(exist_ok=True)
        if not (folder/'result.json').exists():ledger.loc[index,'status']='blocked';publish()
    if not ledger.status.eq('completed').all():
        core.status(run,stage='incomplete',orchestrator_pid=None);return
    if child(['report'],run/'report.log','reporting'):raise RuntimeError('Reporting failed; completed fits remain intact')
    if child(['audit'],run/'audit.log','audit'):raise RuntimeError('Final audit failed')
    core.status(run,stage='complete',worker=None,orchestrator_pid=None,completed=36)
    print(json.dumps(dict(status='complete',new_fits=36,run=str(run))),flush=True)
