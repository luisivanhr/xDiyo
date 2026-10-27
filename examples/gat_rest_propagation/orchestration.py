"""One local CUDA fitting worker; inner-only fits precede sealed outer evaluation."""
from pathlib import Path
import json,os,subprocess,sys
import psutil
import pandas as pd
from . import core,execution,selection

def _execute(run):
    run=Path(run).resolve();p,_=core.load(run,frozen=True)
    ledger=pd.read_csv(run/'RUN_MANIFEST.csv');assert set(ledger.job_id)=={j['job_id'] for j in p['matrix']}
    prior=json.loads((run/'PROGRESS.json').read_text())
    for pid in (prior.get('orchestrator_pid'),(prior.get('worker') or {}).get('pid')):
        if pid and pid!=os.getpid():
            try:
                command=' '.join(psutil.Process(pid).cmdline())
                if 'run_gat_rest_propagation.py' in command and str(run) in command:
                    raise RuntimeError(f'Matching worker already active: {pid}')
            except psutil.NoSuchProcess:pass
    core.status(run,orchestrator_pid=os.getpid(),worker=None)
    environment=os.environ.copy();environment.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def child(command,log,stage,job=None):
        with log.open('w',encoding='utf-8') as stream:
            args=[sys.executable,str(core.ROOT/'examples/run_gat_rest_propagation.py'),command,'--run',str(run)]
            if job:args+=['--job',job]
            process=subprocess.Popen(args,cwd=core.ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            core.status(run,stage=stage,worker=dict(pid=process.pid,job=job,log=log.relative_to(run).as_posix()))
            code=process.wait()
        core.status(run,worker=None)
        if code:raise RuntimeError(f'{stage} failed with code {code}; inspect {log}. Preserve completed fits and resume after engineering recovery.')
    def publish():
        temporary=run/'RUN_MANIFEST.csv.tmp';ledger.to_csv(temporary,index=False);temporary.replace(run/'RUN_MANIFEST.csv')
        core.status(run,completed=int(ledger.status.eq('completed').sum()),failed=int(ledger.status.eq('failed').sum()),blocked=int(ledger.status.eq('blocked').sum()))
    try:
        for job in p['matrix']:
            folder=run/job['path'];folder.mkdir(parents=True,exist_ok=True);index=ledger.index[ledger.job_id.eq(job['job_id'])][0]
            if (folder/'fit.json').exists():
                result=json.loads((folder/'fit.json').read_text());assert result['identity']==execution.identity(run,job)
                ledger.loc[index,'status']='completed';publish();continue
            attempt=len(list(folder.glob('attempt_*')))+1
            if attempt>3:
                ledger.loc[index,'status']='blocked';publish();raise RuntimeError('Engineering attempts exhausted')
            ledger.loc[index,'status']='running';publish()
            print(json.dumps(dict(stage='fit',job=job['job_id'],attempt=attempt,completed=int(ledger.status.eq('completed').sum()),planned=12)),flush=True)
            try:child('fit',folder/f'worker_{attempt:02}.log','fitting',job['job_id'])
            except BaseException:
                ledger.loc[index,'status']='failed';publish();raise
            assert (folder/'fit.json').exists();ledger.loc[index,'status']='completed';publish()
        assert ledger.status.eq('completed').all()
        selection.seal(run)
        child('evaluate',run/'evaluation.log','evaluation_after_selection')
        child('report',run/'report.log','reporting')
        child('audit',run/'audit.log','final_audit')
        core.status(run,stage='complete_reviewed',worker=None,orchestrator_pid=None,completed=12)
        print(json.dumps(dict(status='complete_reviewed',new_fits=12)),flush=True)
    except BaseException:
        core.status(run,stage='engineering_attention',orchestrator_pid=None,worker=None);raise

def execute(run):
    """Cross-process file lock also works when process inventories are sandboxed."""
    path=Path(run)/'execution.lock'
    with path.open('a+b') as handle:
        if handle.tell()==0:handle.write(b'0');handle.flush()
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as exc:raise RuntimeError('This experiment already has an active orchestrator') from exc
            try:return _execute(run)
            finally:handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise RuntimeError('This experiment already has an active orchestrator') from exc
            try:return _execute(run)
            finally:fcntl.flock(handle,fcntl.LOCK_UN)
