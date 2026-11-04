"""Public native construction with immutable canonical replay inputs."""
from dataclasses import asdict
from pathlib import Path
from hashlib import sha256
import argparse
import json
import pandas as pd
from xdiyo_analytics.graphs import (TemporalGraphModel, PhaseRankLookup, RetainedGATTraining,
    GATResearchBNV1, GATResearchCE1V1, GATResearchBCDDWV1)
from xdiyo_analytics.graphs.research_training import MONITOR
from xdiyo_analytics.training import TrainingControl, EarlyStopping, CheckpointPolicy
from gat_cross_league.components import ComponentStore
from temporal_team_gat import make_context


def prepare(research_root, *, family, fold, seed, window):
    """Read/verify frozen inputs and return a model spec, store and contexts."""
    if type(seed) is not int or seed not in (17,42,73):
        raise ValueError('Choose an explicit retained seed: 17, 42 or 73.')
    root=Path(research_root).resolve()
    manifest=json.loads(Path(__file__).with_name('reference_inputs.json').read_text())
    record=manifest['jobs'][f'{family}/{fold}/{seed}']
    hashes={**manifest['shared'],**record['files']}
    for relative,digest in hashes.items():
        if sha256((root/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen replay input differs from the inventory: '+relative)
    c=root/'experiments/gat_corners_propagation_20261101_01'
    a=root/'experiments/gat_capacity_shots_for_20261031_01'
    phase=PhaseRankLookup.load(a/'shared/features')
    factories={'time_B/N':lambda:GATResearchBNV1(),
        'time_C/E1':lambda:GATResearchCE1V1(phase.source_id),
        'time_B/CD_DW':lambda:GATResearchBCDDWV1(phase.source_id,window)}
    cfg=factories[family]()
    if cfg.phase_rank_features is not None and phase.source_id!=record['source_id']:
        raise ValueError('Frozen phase source differs from the retained recipe.')
    if family!='time_B/CD_DW' and window is not None:
        raise ValueError('Only CD_DW accepts a window.')
    store=ComponentStore(pd.read_parquet(c/'shared/data/history_corners.parquet'),
        pd.read_parquet(a/'shared/data/memberships.parquet'),schedule=pd.read_parquet(a/'shared/data/schedule.parquet'),
        provenance=json.loads((a/'PROTOCOL.json').read_text())['provenance'],phase_context=phase)
    pre=c/'shared/preprocessing'/fold/family/'COR_FA.json'
    initial=c/'shared/initialization'/fold/('seed_'+str(seed))/family/'COR_FA.pt'
    fit=c/'shared/populations'/(fold+'_fit.parquet');inner=c/'shared/populations'/(fold+'_inner.parquet')
    digest=lambda p:hashes[p.relative_to(root).as_posix()]
    contract=RetainedGATTraining(preprocessing_path=str(pre),preprocessing_sha256=digest(pre),
        initial_state_path=str(initial),initial_state_sha256=digest(initial),
        event_store_version=record['event_store_version'],fitting_population_path=str(fit),
        fitting_population_sha256=digest(fit),validation_population_path=str(inner),validation_population_sha256=digest(inner))
    control=TrainingControl(max_steps=200,early_stopping=EarlyStopping(15,0),scheduler=None,
        restarts=0,seed=seed,monitor=MONITOR,direction='minimize',restore_best=True)
    spec=TemporalGraphModel(cfg,control=control,research_training=contract,
        phase_context_path=str(a/'shared/features') if cfg.phase_rank_features else None)
    train=make_context(pd.read_parquet(fit));train.validation=make_context(pd.read_parquet(inner))
    outer=make_context(pd.read_parquet(c/'shared/populations'/(fold+'_outer.parquet')),fitting=False)
    contract.validate_population(store,train)
    spec.build(store)  # Public construction and phase/cutoff checks, no fitting.
    return spec,store,train,outer


def main(recipe):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=(17,42,73),required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    spec,store,train,outer=recipe(args.research_root,seed=args.seed)
    if args.prepare_only:
        print(json.dumps(dict(prepared=True,fit=len(train.X),inner=len(train.validation.X),outer=len(outer.X),seed=args.seed)))
        return
    if args.output is None:parser.error('Fitting requires a new --output directory.')
    out=args.output.resolve();source=args.research_root.resolve()
    if out.is_relative_to(source):parser.error('Replay output must be outside the protected research checkout.')
    out.mkdir(parents=True,exist_ok=False)
    (out/'recipe.json').write_text(json.dumps(asdict(spec),indent=2))
    adapter=spec.build(store);adapter.configure_device(args.device)
    resources=json.loads((args.research_root/'experiments/gat_corners_propagation_20261101_01/PROTOCOL.json').read_text())['resources']
    adapter.configure_input_cache(resources['input_cache_bytes'],resources['headroom_bytes'])
    CheckpointPolicy(every=1).wrap(lambda:adapter,out/'checkpoints','retained-replay')().fit_controlled(train,spec.control)
    from xdiyo_analytics.graphs import TemporalGraphSerializer
    TemporalGraphSerializer().save(adapter,out/'model')
    adapter.training_history_.to_parquet(out/'history.parquet',index=False)
    (out/'summary.json').write_text(json.dumps(adapter.training_summary_,indent=2))
