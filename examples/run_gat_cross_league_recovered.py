"""Recorded numerical-scoring repair; original training sources stay frozen."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from gat_cross_league import core,execution,stable_scoring
from compare_gat_temporal import digest,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['recover','run','fit','report','audit'])
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--job')
    args=parser.parse_args();lock=json.loads((args.run/'SCORING_REPAIR.json').read_text())
    for relative,expected in lock['source_hashes'].items():assert digest(ROOT/relative)==expected,relative
    assert digest(args.run/'FREEZE.json')==lock['original_freeze']
    stable_scoring.install()
    if args.command=='recover':
        from gat_cross_league.export_recovery import recover
        recover(args.run,args.job)
    elif args.command=='run':
        from gat_cross_league.recovered_orchestration import execute
        execute(args.run)
    elif args.command=='fit':
        execution.fit(args.run,args.job)
        p,_=core.load(args.run,frozen=True)
        job=next(j for j in p['matrix'] if j['job_id']==args.job)
        path=args.run/job['path']/'result.json';result=json.loads(path.read_text())
        result['scoring_repair']=digest(args.run/'SCORING_REPAIR.json');write_json(path,result)
    else:getattr(execution,args.command)(args.run)


if __name__=='__main__':main()
