from dataclasses import asdict, replace
from pathlib import Path
import json
import pandas as pd
from gat_all_league.core import ROOT, FOLDS, SEEDS, MONITOR, ROUND_KEYS, keys, truth, same_keys, populations
from gat_cross_league.components import ComponentStore
from compare_gat_temporal import digest, write_json, frame_hash
from xdiyo_analytics.graphs.config import MATCH_KEYS, TemporalGraphConfig
from xdiyo_analytics.graphs.extensions import GoalAuxiliary
from xdiyo_analytics.graphs.observed_gap import migrate_gap_config

PARENT=ROOT/'experiments/gat_all_league_validation_20261026_01'
FULL='corrected_full_time_B'
PRIOR='corrected_prior_context_time_B'
CONFIGS=(PRIOR,FULL)  # Exact ties prefer the simpler propagation control.

def decode_config(state):
    state=migrate_gap_config(state)
    if state.get('goal_auxiliary') is not None: state['goal_auxiliary']=GoalAuxiliary(**state['goal_auxiliary'])
    return TemporalGraphConfig(**state)

def config(name=FULL):
    parent=json.loads((PARENT/'PROTOCOL.json').read_text())
    cfg=decode_config(parent['configurations']['time_B'])
    if name=='archived_time_B': return cfg
    if name not in CONFIGS: raise ValueError(name)
    return replace(cfg,fixture_gap_encoding='bounded_observed_gap_v1',observed_gap_scale_days=7.,
                   observed_gap_recent_days=30.,sender_state_mode='updated' if name==FULL else 'layer_zero')

def matrix():
    return [dict(job_id=f'{name}__{fold}__{seed}',configuration=name,fold=fold,seed=seed,arm='pooled',
                 path=f'{stage}/{fold}/seed_{seed}')
            for name,stage in ((FULL,'01_rest_fix'),(PRIOR,'02_opponent_propagation'))
            for fold in FOLDS for seed in SEEDS]

def initial_path(run,job):
    return Path(run)/'shared/initialization'/job['fold']/f'seed_{job["seed"]}'/'time_features.pt'

def load(run,*,frozen=False):
    run=Path(run);p=json.loads((run/'PROTOCOL.json').read_text())
    if frozen:
        lock=json.loads((run/'FREEZE.json').read_text())
        assert digest(run/'PROTOCOL.json')==lock['protocol_sha256'],'Protocol changed'
        for relative,expected in lock['sources'].items(): assert digest(ROOT/relative)==expected,relative
        for relative,expected in lock['inputs'].items(): assert digest(run/relative)==expected,relative
        for relative,expected in lock['reused_parent'].items(): assert digest(PARENT/relative)==expected,relative
    root=run/'shared/data'
    store=ComponentStore(pd.read_parquet(root/'history.parquet'),pd.read_parquet(root/'memberships.parquet'),
                         schedule=pd.read_parquet(root/'schedule.parquet'),provenance=p['provenance'])
    assert store.version==p['store_version']
    return p,store

def status(run,**values):
    run=Path(run);path=run/'PROGRESS.json';state=json.loads(path.read_text()) if path.exists() else {}
    state.update(values,updated_at=pd.Timestamp.now(tz='UTC').isoformat());write_json(path,state)
    (run/'STATUS.md').write_text('# Bounded observed gap and opponent propagation\n\n'
        f"Stage: {state.get('stage')}. Completed {state.get('completed',0)}/12 fits. "
        f"Failed: {state.get('failed',0)}; blocked: {state.get('blocked',0)}.\n\n"
        f"Worker: {state.get('worker')}. Updated {state['updated_at']}.\n\n"
        'One CUDA worker. Full controls first, then prior-context controls. Own-fold inner CE selects checkpoints '
        'and configurations before corrected outer scoring. CE primary; RPS substantive secondary.\n',encoding='utf-8')
