"""Strict deterministic parity on a completed fold's three real graph sizes.

No study writes. Uses saved preprocessing/weights; disposable diagnostic updates
use only early/middle/late groups with the original global weight denominator.
This does not claim a full-fold deterministic training comparison.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/analytics')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(args.fold.resolve()):
        parser.error('Write verification evidence outside the completed fold.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import pandas as pd
    import torch
    from test_graph_speed import reference
    from test_graph_preparation_memory import assert_exact
    from xdiyo_analytics.graphs import store,adapter,preprocessing,streaming
    from xdiyo_analytics.graphs.config import TemporalGraphConfig,CLASS_ORDER
    from xdiyo_analytics.graphs.network import TemporalNetwork
    from xdiyo_analytics.training.contracts import FitContext
    from benchmark_gat_speed import digest
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    files=[args.fold/'train_metadata.parquet',*[args.fold/'model'/name for name in
           ('history.parquet','team_seasons.parquet','schedule.parquet','graph.json','weights.pt')]]
    hashes={str(p.relative_to(args.fold)):digest(p) for p in files}
    meta=json.loads((args.fold/'model/graph.json').read_text())
    cfg=TemporalGraphConfig(**meta['config'])
    saved=torch.load(args.fold/'model/weights.pt',map_location='cuda:0',weights_only=True)
    frame=pd.read_parquet(args.fold/'train_metadata.parquet')
    ctx=FitContext(X=pd.DataFrame(index=frame.index),metadata=frame,layout='match',match_columns=(),fold_id=2,
        y=pd.DataFrame({meta['target_name']:np.sign(frame.goals_for-frame.goals_against).astype(int)}))
    classes=[reference(),dict(store=store,adapter=adapter,preprocessing=preprocessing,streaming=streaming)]
    backends=[]
    for modules in classes:
        source=modules['store'].TemporalEventStore(pd.read_parquet(args.fold/'model/history.parquet'),
            pd.read_parquet(args.fold/'model/team_seasons.parquet'),schedule=pd.read_parquet(args.fold/'model/schedule.parquet'))
        assert source.version==meta['event_store_version']
        pre=modules['preprocessing'].GraphPreprocessor.from_state(cfg,meta['preprocessing'])
        plan=modules['streaming'].GroupPlan(source,cfg,ctx,source.issue_times(frame,cfg))
        groups=modules['streaming'].PreparedGroups(plan,pre,'cuda:0')
        selected=[groups[i] for i in (0,len(plan)//2,len(plan)-1)]
        backend=modules['adapter']._Backend(cfg,source,17,'cuda:0')
        backend.preprocessor=pre
        with torch.random.fork_rng(devices=[]):
            backend.model=TemporalNetwork(cfg,len(pre.leagues),2*len(pre.edge_columns)+len(pre.leagues)+2,
                2*len(pre.fixture_columns)+len(pre.leagues)+1).to('cuda:0')
        backend.model.load_state_dict(saved)
        backend.refresh_version(); assert backend.model_version==meta['model_version']
        backend.output_classes=CLASS_ORDER
        backend.train_targets,backend.train_weights=backend._targets(ctx)
        backend.train_groups=selected; backend.validation_groups=None
        backend.optimizer=torch.optim.AdamW(backend.model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
        backends.append(backend)
    a,b=backends
    assert_exact(a.train_groups,b.train_groups)
    with torch.no_grad():
        for x,y in zip(a.train_groups,b.train_groups):
            assert_exact(a.model(*x[1:]),b.model(*y[1:]))
    results=[]
    for backend in backends:
        before=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
        values=backend.step()
        values=backend.checkpoint_metrics('train_loss',values)
        assert_exact(before,(torch.get_rng_state(),torch.cuda.get_rng_state()))
        results.append(deepcopy((values,backend.model.state_dict(),backend.optimizer.state_dict(),
            [p.grad for p in backend.model.parameters()])))
    assert_exact(results[0],results[1])
    assert hashes=={str(p.relative_to(args.fold)):digest(p) for p in files}
    report=dict(status='PASS_BIT_EXACT_DETERMINISTIC_CUDA',groups=[len(g[1]['x']) for g in a.train_groups],
        checks=['ordered_inputs','logits','losses','gradients','parameters','AdamW_state','CPU_and_CUDA_RNG'],
        device=torch.cuda.get_device_name(),torch=torch.__version__,input_sha256=hashes,inputs_unchanged=True,
        scope='Three actual early/middle/late graphs, not a full-fold deterministic comparison')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
