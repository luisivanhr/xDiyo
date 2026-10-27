"""Preserve the parent populations/scalers; map untrained columns by meaning."""
from dataclasses import asdict
from pathlib import Path
from copy import deepcopy
import json, shutil
import numpy as np
import pandas as pd
import torch
from . import core
from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
from xdiyo_analytics.graphs.network import TemporalNetwork
from compare_gat_goals import digest_state
from gat_cross_league.bootstrap import week_groups

def mapped_initial(old,legacy,corrected):
    """No random reinitialization of common parameters or changed columns."""
    result={k:v.clone() for k,v in old.items()}
    offset=4*corrected.config.embedding_dim
    aliases={f'{s}_no_eligible_history':f'{s}_rest_days__missing' for s in ('home','away')}
    mapping=[]
    for name in ('head.0.weight','goal_head.weight'):
        before=old[name];after=torch.zeros((before.shape[0],offset+corrected.fixture_dim),dtype=before.dtype)
        after[:,:offset]=before[:,:offset]
        for j,column in enumerate(corrected.fixture_encoded_columns):
            source=aliases.get(column,column)
            if source in legacy.fixture_encoded_columns:
                i=legacy.fixture_encoded_columns.index(source);after[:,offset+j]=before[:,offset+i]
            else: i=None
            if name=='head.0.weight':mapping.append(dict(new=column,new_index=offset+j,
                old=source if i is not None else None,old_index=offset+i if i is not None else None,
                initialization='copy' if i is not None else 'zero'))
        result[name]=after
    for name in old:
        if name not in ('head.0.weight','goal_head.weight'):torch.testing.assert_close(old[name],result[name],atol=0,rtol=0)
    return result,mapping

def prepare(run):
    run=Path(run)
    if (run/'FREEZE.json').exists() or (run/'PROTOCOL.json').exists():raise ValueError('Preparation cannot overwrite an existing protocol')
    parent=json.loads((core.PARENT/'PROTOCOL.json').read_text())
    for directory in ('data','populations','bootstrap'):
        shutil.copytree(core.PARENT/'shared'/directory,run/'shared'/directory,dirs_exist_ok=True)
    for directory in ('preprocessing','initialization','checks'):(run/'shared'/directory).mkdir(parents=True,exist_ok=True)
    cfg=core.config();legacy_cfg=core.config('archived_time_B');receipts=[]
    for fold in core.FOLDS:
        old_state=json.loads((core.PARENT/'shared/preprocessing'/f'{fold}.json').read_text())['state']
        legacy=GraphPreprocessor.from_state(legacy_cfg,old_state)
        corrected=GraphPreprocessor(cfg)
        corrected.leagues=deepcopy(legacy.leagues);corrected.edge_scale=deepcopy(legacy.edge_scale)
        corrected.fixture_scale=deepcopy(legacy.fixture_scale)
        state=corrected.state()
        assert state['edge_scale']==old_state['edge_scale'] and state['fixture_scale']==old_state['fixture_scale']
        assert (legacy.fixture_dim,corrected.fixture_dim)==(24,26)
        assert legacy.temporal_columns==corrected.temporal_columns and len(corrected.temporal_columns)==14
        core.write_json(run/'shared/preprocessing'/f'{fold}.json',dict(state=state,source=str(core.PARENT),
            policy='Exact parent scaler states; gap slots bypassed, all other columns unchanged'))
        for seed in core.SEEDS:
            source=core.PARENT/'shared/initialization'/fold/f'seed_{seed}'/'time_features.pt'
            old=torch.load(source,map_location='cpu',weights_only=True)
            new,mapping=mapped_initial(old,legacy,corrected)
            model=TemporalNetwork(cfg,len(corrected.leagues),corrected.edge_dim,corrected.fixture_dim)
            model.load_state_dict(new);assert sum(v.numel() for v in model.parameters())==35993
            path=core.initial_path(run,dict(fold=fold,seed=seed));path.parent.mkdir(parents=True,exist_ok=True);torch.save(new,path)
            receipts.append(dict(fold=fold,seed=seed,source=str(source),source_sha256=core.digest(source),
                mapped_sha256=core.digest(path),state_digest=digest_state(new),parameters=35993,
                shared_across_modes=True,column_mapping=mapping))
        for phase,frame in core.populations(run,fold).items():
            assert core.frame_hash(frame)==parent['folds'][fold][phase]['identity']
        frame=core.populations(run,fold)['outer'];weeks,positions=week_groups(pd.to_datetime(frame.round_issue_at,utc=True))
        base=run/'shared/bootstrap'/fold
        assert [v.isoformat() for v in weeks]==json.loads((base/'weeks.json').read_text())
        np.testing.assert_array_equal(positions,np.load(base/'fixture_week.npy'))
        counts=np.column_stack([np.bincount(positions,weights=frame.competition_id.eq(league),minlength=len(weeks))
                               for league in parent['competitions']]).astype(np.int64)
        np.testing.assert_array_equal(counts,np.load(base/'counts_by_league.npy'))
        for length in (2,4,8):
            weights=np.load(base/f'blocks_{length}.npy');assert weights.shape==(20000,len(weeks))
            assert core.digest(base/f'blocks_{length}.npy')==core.digest(core.PARENT/'shared/bootstrap'/fold/f'blocks_{length}.npy')
    core.write_json(run/'shared/checks/initialization.json',receipts)
    fields=('catalog','competitions','folds','store_version','provenance','history_start','stage_mapping','resources','training','bootstrap','metrics')
    p={k:deepcopy(parent[k]) for k in fields}
    p.update(schema=1,parent=str(core.PARENT),parent_completion_sha256=core.digest(core.PARENT/'COMPLETE.json'),
        assignment_sha256=core.digest(run/'HANDOFF.md'),matrix=core.matrix(),
        configurations={name:asdict(core.config(name)) for name in core.CONFIGS},
        configuration_selection=dict(population='own fold all13 inner',score='Mean selected raw inner micro CE across seeds',
            exact_tie_order=list(core.CONFIGS),outer_gate='Seal both fold choices before corrected outer scoring',single_recommendation='Latest fold inner winner'),
        reuse=dict(corrected_controls=0,fresh_fits=12,baselines='Exact archived LR and frequency predictions; no refitting',
                   archived_reference='time_B, descriptive entire-gap-repair contrast',bootstrap='Exact parent weights after key/week/mixture checks'),
        limitations=['Exploratory: previously inspected development periods',
            'Full-versus-archived contrast changes the entire gap encoding and reselected checkpoint; it does not isolate bounding',
            'Bootstrap intervals condition on fitted models and these three observed seeds'])
    p['training']['objective']='Global per-fixture outcome CE plus unchanged prior regularizer; dormant goal head beta=0'
    core.write_json(run/'PROTOCOL.json',p)
    pd.DataFrame(core.matrix()).assign(status='pending').to_csv(run/'RUN_MANIFEST.csv',index=False)
    core.status(run,stage='prepared',completed=0,planned=12,worker=None)
    print(json.dumps(dict(stage='prepared',fits=12,parameters=35993)),flush=True)
