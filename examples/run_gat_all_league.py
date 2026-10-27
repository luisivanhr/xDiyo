"""Local entry point for the declared 18-fit corrected-clock experiment."""
import os,sys,argparse
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','checks','profile','freeze','baselines','fit','select','evaluate','report','audit','run'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--job');parser.add_argument('--handoff',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        from gat_all_league.data import prepare
        prepare(args.run,args.handoff)
    elif args.command in ('checks','profile','freeze'):
        from gat_all_league import checks
        getattr(checks,args.command)(args.run)
    elif args.command=='baselines':
        from gat_all_league.baselines import prepare
        prepare(args.run)
    elif args.command=='fit':
        from gat_all_league.execution import fit
        fit(args.run,args.job)
    elif args.command=='select':
        from gat_all_league.selection import seal
        seal(args.run)
    elif args.command=='evaluate':
        from gat_all_league import baselines,execution
        baselines.evaluate(args.run);execution.evaluate(args.run)
    elif args.command in ('report','audit'):
        from gat_all_league import reporting
        getattr(reporting,args.command)(args.run)
    else:
        from gat_all_league.orchestration import execute
        execute(args.run)

if __name__=='__main__':main()
