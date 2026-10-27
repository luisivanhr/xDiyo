"""Semantic category expansion of the archived untrained canonical states."""
from copy import deepcopy
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pandas as pd
import torch
from . import core
from gat_suite.core import initialize_state
from gat_suite.fitting import predict
from compare_gat_temporal import digest,write_json
from compare_gat_goals import digest_state
from temporal_team_gat import make_context
from xdiyo_analytics.graphs import TemporalEventStore
from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
from xdiyo_analytics.graphs.streaming import GroupPlan,PreparedGroups
from xdiyo_analytics.graphs.network import TemporalNetwork


def feature_names(pre):
    categories=['competition:unknown',*[f'competition:{k}' for k,_ in sorted(pre.leagues.items(),key=lambda x:x[1])]]
    edge=[*[f'value:{k}' for k in pre.edge_columns],*[f'missing:{k}' for k in pre.edge_columns],
          'historical_home',*categories,*pre.temporal_columns]
    fixture=[*[f'value:{k}' for k in pre.fixture_columns],*[f'missing:{k}' for k in pre.fixture_columns],*categories]
    if pre.config.fixture_gap_encoding != 'legacy_standardized_gap_v1':
        fixture=list(pre.fixture_encoded_columns)
    joint=[f'{block}:{i}' for block in ('home','away','difference','product') for i in range(pre.config.embedding_dim)]+fixture
    assert len(edge)==pre.edge_dim and len(fixture)==pre.fixture_dim and len(edge)==len(set(edge)) and len(joint)==len(set(joint))
    return dict(edge=edge,fixture=fixture,joint=joint)


def expand_state(state,old_pre,new_pre,seed):
    """Copy common coordinates, zero new indicators, independently draw new priors."""
    cfg=new_pre.config
    model=initialize_state(cfg,new_pre,0)
    target=model.state_dict()
    old_names,new_names=feature_names(old_pre),feature_names(new_pre)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    for name,value in state.items():
        if name=='league.weight':
            target[name].zero_()
            target[name][0]=value[0]
            for league,position in new_pre.leagues.items():
                if league in old_pre.leagues:
                    target[name][position]=value[old_pre.leagues[league]]
                else:
                    target[name][position]=.001*torch.randn(value.shape[1],generator=generator)
        elif name.endswith(('.e.weight','.u.weight')) or name in ('head.0.weight','goal_head.weight'):
            kind='edge' if name.endswith(('.e.weight','.u.weight')) else 'joint'
            target[name].zero_()
            mapping={key:i for i,key in enumerate(new_names[kind])}
            for i,key in enumerate(old_names[kind]):
                target[name][:,mapping[key]]=value[:,i]
        else:
            assert target[name].shape==value.shape,name
            target[name]=value.clone()
    model.load_state_dict(target)
    return model.state_dict()


def initial_path(run,job):
    mode='snapshot' if job['configuration']=='snapshot_B' else 'time_features'
    return Path(run)/'shared/initialization'/job['fold']/f'seed_{job["seed"]}'/f'{mode}.pt'


def raw_predictions(store,frame,cfg,pre,state,device='cpu'):
    model=TemporalNetwork(cfg,len(pre.leagues),pre.edge_dim,len(feature_names(pre)['fixture'])).to(device)
    model.load_state_dict(state); model.eval()
    plan=GroupPlan(store,cfg,make_context(frame),store.issue_times(frame,cfg))
    groups=PreparedGroups(plan,pre,device)
    values=[np.empty((len(frame),3)),np.empty((len(frame),2))]
    with torch.no_grad():
        for i in range(len(groups)):
            pos,data,h,a,f=groups[i]
            logits,details=model(data,h,a,f,return_goal_details=True)
            values[0][pos]=logits.cpu().numpy();values[1][pos]=details['rates'].cpu().numpy()
    return values


def initialize(run):
    run=Path(run);p,store=core.load(run)
    assert not (run/'FREEZE.json').exists()
    torch.set_num_threads(1)
    old_p=json.loads((core.PARENT/'MASTER_PROTOCOL.json').read_text())
    old_data=core.PARENT/'shared/data'
    old_store=TemporalEventStore(pd.read_parquet(old_data/'history.parquet'),pd.read_parquet(old_data/'memberships.parquet'),
        schedule=pd.read_parquet(old_data/'schedule.parquet'),provenance=old_p['provenance'])
    checks=[]
    for fold in core.FOLDS:
        old_receipt=json.loads((core.PARENT/'shared/preprocessing'/f'{fold}.json').read_text())
        old_pre=GraphPreprocessor.from_state(core.config(),old_receipt['state'])
        state=deepcopy(old_receipt['state'])
        state['leagues']=[[league,i+1] for i,league in enumerate(p['competitions'])]
        new_pre=GraphPreprocessor.from_state(core.config(),state)
        write_json(run/'shared/preprocessing'/f'{fold}.json',dict(state=state,
            numeric_source_sha256=digest(core.PARENT/'shared/preprocessing'/f'{fold}.json'),
            features=feature_names(new_pre),numeric_values_unchanged=True))
        targets=core.populations(run,fold,'local')['fit']
        issuance=store.issue_times(targets,core.config())
        probe=targets[issuance.isin([issuance.min(),issuance.max()])].reset_index(drop=True)
        for seed in core.SEEDS:
            original=core.PARENT/'shared/initialization'/fold/'snapshot'/f'seed_{seed}'/'canonical.pt'
            canonical=torch.load(original,map_location='cpu',weights_only=True)
            rng_seed=int.from_bytes(sha256(f'xdiyo-cross-league-priors-v1/{fold}/{seed}'.encode()).digest()[:8],'little')
            snapshot=expand_state(canonical,old_pre,new_pre,rng_seed)
            temporal_pre=GraphPreprocessor.from_state(core.config('time_B'),state)
            temporal=expand_state(snapshot,new_pre,temporal_pre,rng_seed)
            folder=run/'shared/initialization'/fold/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
            torch.save(snapshot,folder/'snapshot.pt');torch.save(temporal,folder/'time_features.pt')
            old_values=raw_predictions(old_store,probe,old_pre.config,old_pre,canonical)
            expanded=raw_predictions(old_store,probe,new_pre.config,new_pre,snapshot)
            new_values=raw_predictions(store,probe,new_pre.config,new_pre,snapshot)
            timed_values=raw_predictions(store,probe,temporal_pre.config,temporal_pre,temporal)
            for a,b in zip(old_values,expanded):np.testing.assert_allclose(a,b,atol=2e-7,rtol=2e-6)
            for a,b in zip(new_values,timed_values):np.testing.assert_allclose(a,b,atol=2e-7,rtol=2e-6)
            receipt=dict(fold=fold,seed=seed,prior_rng_seed=rng_seed,parent_sha256=digest(original),
                snapshot_sha256=digest_state(snapshot),temporal_sha256=digest_state(temporal),
                old_context_max_error=max(float(np.max(np.abs(a-b))) for a,b in zip(old_values,expanded)),
                common_context_temporal_max_error=max(float(np.max(np.abs(a-b))) for a,b in zip(new_values,timed_values)),
                probe_count=len(probe),atol=2e-7,rtol=2e-6)
            write_json(folder/'receipt.json',receipt);checks.append(receipt)
    p['initialization']=dict(source='Parent untrained canonical snapshot, never selected weights',
        mapping='Semantic values, masks, competition IDs, embedding blocks and temporal feature names',
        added_indicators='zero trainable',added_league_priors='normal std .001 from separate CPU torch.Generator',
        rng_derivation='little-endian first 8 SHA256 bytes of xdiyo-cross-league-priors-v1/{fold}/{seed}',
        time_columns='14 zero trainable',paired='One snapshot or temporal state per fold/seed; same full goal-head tensors in B/C',
        ordinary_scaling='Exact archived recent-Bundesliga numeric means/scales; extended categories only')
    write_json(run/'PROTOCOL.json',p)
    write_json(run/'shared/checks/canonical_mapping.json',checks)
    core.status(run,stage='canonical_initialization_verified')
    print(json.dumps(dict(stage='canonical_initialization_verified',pairs=len(checks),max_error=max(r['old_context_max_error'] for r in checks))))
