"""Small recorded prediction populations with exact identities and hand values."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from xdiyo_analytics.training import FoldResult, TrainingResult
from xdiyo_analytics.reporting import PostTrainingContext, PredictionReporter, StudyResult


class RecordedModel:
    def fit(self, *args):
        raise AssertionError('Post-training reporting attempted to fit a model.')

    def predict(self, *args):
        raise AssertionError('Post-training reporting attempted to recompute predictions.')


def run_sample(layout='match'):
    case = np.repeat(np.arange(6), 2) if layout == 'team_match' else np.arange(6)
    metadata = pd.DataFrame({'case': case, 'competition_id': 17, 'season_id': 2025,
                             'event_id': pd.array([2**63+101+int(i) for i in case], dtype='uint64[pyarrow]'),
                             'kickoff_at': pd.to_datetime(['2025-01-%02d' % (int(i)+1) for i in case], utc=True)})
    match = ('competition_id', 'season_id', 'event_id')
    identity = match
    if layout == 'team_match':
        metadata['team_id'] = pd.array([2**63+31+int(i % 2) for i in range(len(case))], dtype='uint64[pyarrow]')
        metadata['side'] = ['home', 'away'] * 6
        identity += ('team_id',)
    y = pd.DataFrame({'count': np.array([0., 1., 3., 4., 7., 9.])[case]})
    y['other'] = 9-y['count']
    folds = []
    for fold_id, cases, values, scores in [(4, [4, 2, 0], [6., 3.5, -1.], [4, 0]),
                                            (9, [2, 1, 5], [2., 2., 10.], [2, 5])]:
        positions = np.concatenate([np.flatnonzero(case == i) for i in cases])
        scored = np.concatenate([np.flatnonzero(case == i) for i in scores])
        truth, meta = y.iloc[positions].copy(), metadata.iloc[positions].copy()
        truth.index = meta.index = pd.Index(positions, name='row_position')
        prediction = pd.DataFrame({'count': [values[cases.index(int(case[i]))] for i in positions]}, index=truth.index)
        prediction['other'] = 9-prediction['count']
        folds.append(FoldResult(fold_id, RecordedModel(), np.setdiff1d(np.arange(len(case)), positions),
                                positions, scored, ('synthetic',), tuple(y.columns), {'predict': prediction},
                                truth, meta, {'note': [fold_id]}))
    return TrainingResult(folds, layout, identity, match, 'total', {'nested': {'values': ['original']}})


def context(y, outputs, metadata=None, *, layout='match', identity_columns=('event_id',), fold_id=4):
    y = y.copy()
    y.index = pd.MultiIndex.from_arrays([[fold_id]*len(y), np.arange(len(y))], names=['fold_id', 'row_position'])
    predictions = {}
    for name, frame in outputs.items():
        predictions[name] = frame.copy()
        predictions[name].index = y.index
    if metadata is None:
        metadata = pd.DataFrame({'event_id': pd.array([2**63+101+i for i in range(len(y))], dtype='uint64[pyarrow]'),
                                 'kickoff_at': pd.date_range('2025-01-01', periods=len(y), tz='UTC')})
    metadata = metadata.copy()
    metadata.index = y.index
    return PostTrainingContext(y, predictions, metadata, layout, identity_columns, identity_columns,
                               'overall', 'test', fold_id, 'occurrences')


@dataclass(kw_only=True)
class Capture(PredictionReporter):
    contexts: list = field(default_factory=list)
    mutate: bool = False

    def run(self, ctx):
        self.contexts.append(ctx)
        if self.mutate:
            ctx.y.iloc[0, 0] = 999
            ctx.predictions['predict'].iloc[0, 0] = -999
            ctx.metadata.iloc[0, ctx.metadata.columns.get_loc('case')] = -999
            ctx.definitions['nested']['values'].append('mutation')
        return StudyResult('Capture', tables={'positions': ctx.y.index.to_frame(index=False)})
