"""Small datasets and structural reporters for independent reporting tests."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from split_samples import dataset, rows_for, BASE
from xdiyo_analytics.reporting import AnalysisContext, StudyResult
from xdiyo_analytics.splits import Fold, SplitPlan


def sample(*, layout='match', shuffle=False):
    data = dataset(specs=[{'day': i // 2, 'season': 2023 + (i >= 6)} for i in range(12)],
                   layout=layout, shuffle=shuffle)
    case = data.metadata.case.to_numpy()
    data.X = pd.DataFrame({'linear': case.astype(float), 'negative': -case.astype(float),
                          'cycle': case % 3, 'constant': np.ones(len(case)),
                          'missing': np.nan}, index=data.X.index)
    data.y = pd.DataFrame({'first': case + 1., 'second': (case % 3).astype(float)}, index=data.y.index)
    if layout == 'team_match':
        ids = [BASE + 500 + (side == 'away') for side in data.metadata.side]
        data.metadata['team_id'] = pd.Series(ids, dtype='uint64[pyarrow]')
    else:
        data.metadata['home_id'] = pd.Series([BASE + 500] * len(case), dtype='uint64[pyarrow]')
        data.metadata['away_id'] = pd.Series([BASE + 501] * len(case), dtype='uint64[pyarrow]')
    data.definitions = {'features': {'linear': {'description': 'synthetic'}}}
    return data


def plan(data):
    folds = [Fold(rows_for(data, range(4)), rows_for(data, [6, 7, 8]), rows_for(data, [7]), {'nested': [0]}),
             Fold(rows_for(data, range(7)), rows_for(data, [9, 10, 11]), np.array([], dtype=int), {'nested': [1]})]
    return SplitPlan(folds, len(data.X), np.arange(len(data.X)))


def context(features, targets=None):
    X = pd.DataFrame(features)
    y = pd.DataFrame({'target': np.arange(len(X))} if targets is None else targets)
    data = dataset(max(1, len(X)))
    data.metadata = data.metadata.iloc[:len(X)].copy()
    positions = np.arange(len(X))
    for frame in (X, y, data.metadata):
        frame.index = pd.Index(positions, name='row_position')
    return AnalysisContext(X, y, data.metadata, data.layout, data.match_columns, positions, 'overall', 'all')


@dataclass
class Custom:
    type: str
    partition: str
    action: object = None
    features_from: object = None
    supported_types = ('overall', 'per_fold', 'timeline')

    def run(self, ctx):
        if self.action is not None:
            return self.action(ctx)
        return StudyResult('Captured', tables={'X': ctx.X.copy(), 'y': ctx.y.copy(), 'metadata': ctx.metadata.copy()})
