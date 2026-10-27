"""Local protocol identities, shared populations, and native configurations."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from gat_suite.core import config as parent_config
from compare_gat_temporal import digest, write_json, frame_hash
from xdiyo_analytics.graphs import TemporalEventStore
from xdiyo_analytics.graphs.config import MATCH_KEYS

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'experiments/gat_four_study_suite_20261026_01'
CONFIGS = ('snapshot_B', 'time_B', 'time_C')
SEEDS = (17, 42, 73)
FOLDS = ('fold_1', 'fold_2')
ARMS = ('local', 'pooled')
DIRECTORIES = dict(local='01_local_supervision', pooled='02_all13_supervision')


def config(name='snapshot_B'):
    if name not in CONFIGS:
        raise ValueError(name)
    return parent_config('snapshot' if name == 'snapshot_B' else 'time_features',
                         'C_flexible' if name == 'time_C' else 'B_expanded')


def matrix():
    rows = []
    for name in CONFIGS:
        for fold in FOLDS:
            for seed in SEEDS:
                for arm in ARMS:
                    rows.append(dict(job_id=f'{name}__{fold}__{seed}__{arm}', configuration=name,
                        fold=fold, seed=seed, arm=arm,
                        path=f'{DIRECTORIES[arm]}/{name}/{fold}/seed_{seed}'))
    return rows


def populations(run, fold, arm='pooled'):
    root = Path(run)/'shared/populations'
    return {phase:pd.read_parquet(root/f'{fold}_{arm}_{phase}.parquet')
            for phase in ('fit', 'inner', 'outer')}


def load(run, *, frozen=False, components=None):
    run = Path(run)
    p = json.loads((run/'PROTOCOL.json').read_text())
    if frozen:
        lock = json.loads((run/'FREEZE.json').read_text())
        assert digest(run/'PROTOCOL.json') == lock['protocol_sha256'], 'Protocol changed'
        for path, expected in lock['source_hashes'].items():
            assert digest(ROOT/path) == expected, path
        for path, expected in lock['input_hashes'].items():
            assert digest(run/path) == expected, path
    store_class = TemporalEventStore
    if components is None:
        components = p['resources'].get('complete_components', False)
    if components:
        from .components import ComponentStore
        store_class = ComponentStore
    data = run/'shared/data'
    store = store_class(pd.read_parquet(data/'history.parquet'), pd.read_parquet(data/'memberships.parquet'),
        schedule=pd.read_parquet(data/'schedule.parquet'), provenance=p['provenance'])
    assert store.version == p['store_version'], 'Context identity changed'
    return p, store


def truth(frame):
    return np.where(frame.goals_for > frame.goals_against, 0,
                    np.where(frame.goals_for == frame.goals_against, 1, 2))


def keys(frame):
    return pd.MultiIndex.from_frame(frame[list(MATCH_KEYS)])


def status(run, **values):
    path = Path(run)/'PROGRESS.json'
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(values, updated_at=pd.Timestamp.now(tz='UTC').isoformat())
    write_json(path, old)
    (Path(run)/'STATUS.md').write_text(
        '# Cross-league experiment status\n\n'+
        f"Stage: {old.get('stage', 'preparing')}. Completed {old.get('completed', 0)}/36 new GAT fits. "
        f"Failed: {old.get('failed', 0)}; blocked: {old.get('blocked', 0)}.\n\n"
        f"Current worker: {old.get('worker')}. Updated {old['updated_at']}.\n\n"
        'The 36-fit specification was authorized directly by the user. Previous suite evidence is protected. '
        'One CUDA fitting worker at a time; no outcome-guided changes to the matrix.\n', encoding='utf-8')
