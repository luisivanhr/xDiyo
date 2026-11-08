"""Read the immutable, date-filtered stage snapshots built by prepare_data.py."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'examples')]


def load_stage(stage):
    import pandas as pd
    from gat_cross_league.components import ComponentStore
    from xdiyo_analytics.graphs import PhaseRankLookup
    if stage not in ('original13', 'first_divisions', 'first_and_second_divisions'):
        raise ValueError('Unknown dataset stage: ' + str(stage))
    directory = ROOT / 'data' / stage
    phase = PhaseRankLookup.load(directory / 'features')
    provenance = json.loads((directory / 'provenance.json').read_text(encoding='utf-8'))
    store = ComponentStore(pd.read_parquet(directory / 'history.parquet'),
        pd.read_parquet(directory / 'memberships.parquet'),
        schedule=pd.read_parquet(directory / 'schedule.parquet'),
        provenance=provenance, phase_context=phase)
    populations = directory / 'populations'
    def read(name):
        frame = pd.read_parquet(populations / (name + '.parquet')).reset_index(drop=True)
        if frame.empty or frame.event_id.duplicated().any():
            raise ValueError('Empty or duplicated population: ' + stage + '/' + name)
        return frame
    return dict(store=store, dev_fit=read('dev_fit'), dev_validation=read('dev_validation'),
        folds={f'fold_{i}': dict(train=read(f'fold_{i}_train'), test=read(f'fold_{i}_test')) for i in range(1, 5)})
