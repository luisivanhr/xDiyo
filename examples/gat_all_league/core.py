from dataclasses import asdict,replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
from gat_cross_league.core import config as original_config
from gat_cross_league.components import ComponentStore
from compare_gat_temporal import digest,write_json,frame_hash
from xdiyo_analytics.graphs.config import MATCH_KEYS

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/'experiments/gat_cross_league_expansion_20261026_01'
CONFIGS=('snapshot_B','time_B','time_C')
FOLDS=('fold_1','fold_2')
SEEDS=(17,42,73)
MONITOR='all13_inner_raw_hda_ce_micro'
ROUND_KEYS=('competition_id','season_id','round_stage','round')

def config(name='snapshot_B'):
    return replace(original_config(name),round_keys=ROUND_KEYS)

def matrix():
    return [dict(job_id=f'{name}__{fold}__{seed}',configuration=name,fold=fold,seed=seed,arm='pooled',
        path=f'01_pooled/{name}/{fold}/seed_{seed}') for name in CONFIGS for fold in FOLDS for seed in SEEDS]

def keys(frame):return pd.MultiIndex.from_frame(frame[list(MATCH_KEYS)])
def truth(frame):return np.where(frame.goals_for>frame.goals_against,0,np.where(frame.goals_for==frame.goals_against,1,2))
def same_keys(left,right):return np.array_equal(left[list(MATCH_KEYS)].to_numpy(np.int64),right[list(MATCH_KEYS)].to_numpy(np.int64))
def populations(run,fold):
    return {phase:pd.read_parquet(Path(run)/'shared/populations'/f'{fold}_{phase}.parquet') for phase in ('fit','inner','outer')}
def initial_path(run,job):
    mode='snapshot' if job['configuration']=='snapshot_B' else 'time_features'
    return Path(run)/'shared/initialization'/job['fold']/f'seed_{job["seed"]}'/f'{mode}.pt'

def load(run,*,frozen=False):
    run=Path(run);protocol=json.loads((run/'PROTOCOL.json').read_text())
    if frozen:
        lock=json.loads((run/'FREEZE.json').read_text())
        assert digest(run/'PROTOCOL.json')==lock['protocol_sha256'],'Protocol changed'
        for path,expected in lock['sources'].items():assert digest(ROOT/path)==expected,path
        for path,expected in lock['inputs'].items():assert digest(run/path)==expected,path
    root=run/'shared/data'
    store=ComponentStore(pd.read_parquet(root/'history.parquet'),pd.read_parquet(root/'memberships.parquet'),
        schedule=pd.read_parquet(root/'schedule.parquet'),provenance=protocol['provenance'])
    assert store.version==protocol['store_version'],'Corrected store identity changed'
    return protocol,store

def status(run,**values):
    run=Path(run);path=run/'PROGRESS.json';state=json.loads(path.read_text()) if path.exists() else {}
    state.update(values,updated_at=pd.Timestamp.now(tz='UTC').isoformat());write_json(path,state)
    (run/'STATUS.md').write_text('# All-league validation experiment\n\n'
        f"Stage: {state.get('stage')}. Completed {state.get('completed',0)}/18 fresh pooled fits. "
        f"Failed: {state.get('failed',0)}; blocked: {state.get('blocked',0)}.\n\n"
        f"Worker: {state.get('worker')}. Updated {state['updated_at']}.\n\n"
        'Selection uses raw all-13 inner micro CE only. RPS is secondary. One CUDA worker. '
        'Configuration choices are fold-local and sealed before new outer scoring. Stop after reporting.\n',encoding='utf-8')
