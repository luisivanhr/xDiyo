"""Run the prespecified four-study suite; each main CUDA fit is a fresh process."""
import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','garbage_collection_threshold:0.6')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','freeze','profile','pairing','fit','run','baselines','report','audit'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--mode',choices=['snapshot','time_features','historical'])
    parser.add_argument('--job')
    parser.add_argument('--study',type=int)
    args=parser.parse_args()
    if args.command in ('prepare','freeze'):
        from gat_suite import core
        getattr(core,args.command)(args.run)
    elif args.command in ('fit','profile','pairing'):
        from gat_suite import fitting
        if args.command=='fit': fitting.fit(args.run,args.job)
        elif args.command=='profile': fitting.profile(args.run,args.mode)
        else: fitting.pairing(args.run)
    elif args.command=='baselines':
        from gat_suite.baselines import run_baselines
        run_baselines(args.run)
    elif args.command=='run':
        from gat_suite.orchestration import run
        run(args.run)
    else:
        from gat_suite import reporting
        if args.command=='audit': reporting.audit(args.run)
        else: reporting.report(args.run,args.study)


if __name__=='__main__': main()
