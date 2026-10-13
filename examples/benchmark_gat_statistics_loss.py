"""Bounded statistics-head forward/backward benchmark (not whole-GAT timing)."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import subprocess
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import torch
    from xdiyo_analytics.graphs import StatisticsForecast, StatisticTarget
    from xdiyo_analytics.graphs.statistics_network import StatisticsHead
    from xdiyo_analytics.graphs.supervision import prepare_observations
    torch.set_num_threads(1)
    torch.manual_seed(174)
    rng=np.random.default_rng(174)
    rows,groups,width=32,8,16
    indices=[np.arange(rows*i,rows*(i+1)) for i in range(groups)]
    descriptor=SimpleNamespace(plan=SimpleNamespace(descriptors=tuple((None,i) for i in indices)))
    records=[]
    def sync():
        if args.device.startswith('cuda'):torch.cuda.synchronize(args.device)
    for family in ('point','poisson','negative_binomial','inflated_beta','quantile'):
        prop=family=='inflated_beta'
        targets=tuple(StatisticTarget('ballPossession' if prop else 'cornerKicks',side,period,family) for side in ('home','away') for period in ('ALL','1ST','2ND'))
        head=StatisticsHead(StatisticsForecast(targets),width,32).to(args.device)
        x=torch.randn(rows*groups,width,device=args.device)
        raw=rng.integers(0,15,(rows*groups,len(targets))).astype(np.float32)
        if prop:raw/=14
        raw[rng.random(raw.shape)<.25]=np.nan
        y=torch.tensor(raw,device=args.device);w=torch.tensor(rng.uniform(.1,1,rows*groups),dtype=torch.float32,device=args.device)
        sync();start=time.perf_counter();den,observations=prepare_observations(descriptor,y,w);sync();preparation=time.perf_counter()-start
        def run(cached):
            head.zero_grad(set_to_none=True)
            denominator=den if cached else (torch.isfinite(y)*w[:,None]).sum(0)
            totals=[]
            for i,pos in enumerate(indices):
                f,_=head(x[pos])
                loss=head.losses(f,y[pos],w[pos],denominator,prepared=observations[i] if cached else None)
                loss.backward();totals.append(loss.detach())
            return torch.stack(totals)
        old=run(False);old_grads=[p.grad.clone() for p in head.parameters()]
        new=run(True)
        torch.testing.assert_close(old,new,rtol=0,atol=0)
        for a,p in zip(old_grads,head.parameters()):torch.testing.assert_close(a,p.grad,rtol=0,atol=0)
        times={False:[],True:[]}
        for trial in range(6):
            for cached in ([False,True] if trial%2==0 else [True,False]):
                sync();start=time.perf_counter();run(cached);sync()
                times[cached].append(time.perf_counter()-start)
        old_time,new_time=statistics.median(times[False]),statistics.median(times[True])
        records.append(dict(family=family,cache_preparation_seconds=preparation,uncached_seconds=times[False],cached_seconds=times[True],uncached_median=old_time,cached_median=new_time,speedup=old_time/new_time,losses_and_gradients_exact=True))
    receipt=dict(command=subprocess.list2cmdline([sys.executable,*sys.argv]),python=platform.python_version(),torch=torch.__version__,device=args.device,gpu=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,rows=rows*groups,groups=groups,targets=6,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/xdiyo_analytics/graphs').glob('*.py')},results=records,limitation='Synthetic fixed labels, isolated statistics-head forward/backward only; not whole-model speed or accuracy evidence. Alternating paired six repeats; no optimizer update.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    for r in records:print(f"{r['family']}: {r['speedup']:.3f}x")


if __name__=='__main__':main()
