"""Execute the local, fixed cross-league GAT experiment."""
import argparse
from pathlib import Path
import os
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'src')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','initialize','checks','profile','freeze','baselines','fit','run','report','audit'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--handoff',type=Path)
    parser.add_argument('--job')
    args=parser.parse_args()
    if args.command=='prepare':
        from gat_cross_league.data import prepare
        prepare(args.run,args.handoff)
    elif args.command=='initialize':
        from gat_cross_league.initialization import initialize
        initialize(args.run)
    else:
        from gat_cross_league import execution
        getattr(execution,args.command)(args.run,**({'job_id':args.job} if args.command=='fit' else {}))


if __name__=='__main__':
    main()
