"""Independent complete ModelDataset fixtures for match-aware split contracts."""
from copy import deepcopy
import numpy as np
import pandas as pd
from xdiyo_analytics.datasets import ModelDataset

BASE = 2**63 + 101
MATCH_KEYS = ('source_league', 'source_season', 'competition_id', 'season_id', 'event_id')

def dataset(n=12, *, layout='match', shuffle=False, specs=None):
    specs = [{} for _ in range(n)] if specs is None else specs
    records = []
    for i, spec in enumerate(specs):
        competition, season = spec.get('competition', 17), spec.get('season', 2024)
        shared = dict(source_league=spec.get('league', f'L{competition}'), source_season=str(season), competition_id=competition, season_id=season, event_id=spec.get('event', BASE + i), kickoff_at=pd.Timestamp('2024-01-01', tz='UTC') + pd.Timedelta(days=spec.get('day', i)), round=spec.get('round', i + 1), stage=spec.get('stage', 'league'), case=i)
        home, away = BASE + 1000 + 2 * i, BASE + 1001 + 2 * i
        if layout == 'match':
            records.append(dict(shared, home_id=home, away_id=away))
        else:
            records += [dict(shared, team_id=home, opponent_id=away, side='home'), dict(shared, team_id=away, opponent_id=home, side='away')]
    metadata = pd.DataFrame(records)
    for name in ['event_id', 'home_id', 'away_id', 'team_id', 'opponent_id']:
        if name in metadata:
            metadata[name] = metadata[name].astype('uint64[pyarrow]')
    X = pd.DataFrame({'form': np.arange(len(metadata), dtype=float)})
    y = pd.DataFrame({'outcome': metadata.case.astype(float) + 1})
    if shuffle:
        order = np.random.default_rng(196).permutation(len(metadata))
        X, y, metadata = [frame.iloc[order].reset_index(drop=True) for frame in [X, y, metadata]]
    identities = (*MATCH_KEYS, 'team_id') if layout == 'team_match' else MATCH_KEYS
    return ModelDataset(X, y, metadata, layout, identities, MATCH_KEYS, 'team' if layout == 'team_match' else 'total', {'features': {'form': 'synthetic'}, 'label': {'kind': 'synthetic'}})

def cases(data, rows):
    return set(data.metadata.iloc[rows].case.tolist())

def rows_for(data, ids):
    return np.flatnonzero(data.metadata.case.isin(ids).to_numpy())

def whole_matches(data, fold):
    for selected in [fold.train, fold.test, fold.score]:
        assert isinstance(selected, np.ndarray) and selected.dtype.kind in 'iu'
        assert len(selected) == len(set(selected))
        assert all(0 <= row < len(data.X) for row in selected)
        assert selected.tolist() == sorted(selected.tolist())
        assert set(selected) == set(rows_for(data, cases(data, selected)))
    assert not set(fold.train) & set(fold.test)
    assert set(fold.score) <= set(fold.test)

def snapshot(data):
    return deepcopy(data)

def unchanged(data, before):
    for name in ['X', 'y', 'metadata']:
        pd.testing.assert_frame_equal(getattr(data, name), getattr(before, name))
    assert data.definitions == before.definitions
