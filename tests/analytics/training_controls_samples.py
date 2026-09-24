"""Scripted optimization traces; expected decisions are written in tests."""
from copy import deepcopy
import numpy as np
import pandas as pd
from xdiyo_analytics.training import FitContext, FoldResult, TrainingResult


def model_result(models, *, histories=None, summaries=None):
    folds = []
    empty = np.array([], dtype=int)
    index = pd.Index([], name='row_position', dtype=int)
    for i, model in enumerate(models):
        folds.append(FoldResult(i, model, np.array([0, 1]), empty.copy(), empty.copy(), ('x',), ('target',), {},
            pd.DataFrame(index=index), pd.DataFrame(index=index),
            training_history=pd.DataFrame() if histories is None else histories[i].copy(),
            training_summary={} if summaries is None else deepcopy(summaries[i])))
    return TrainingResult(folds, 'match', ('event_id',), ('event_id',), 'total')


def fit_context(*, validation=False, observer=None):
    def make(rows):
        index = pd.Index(rows, name='row_position')
        return FitContext(X=pd.DataFrame({'x': np.asarray(rows, dtype=float)}, index=index),
            y=pd.DataFrame({'target': np.asarray(rows, dtype=float) + 1}, index=index),
            metadata=pd.DataFrame({'event_id': [2**63 + int(i) for i in rows]}, index=index),
            layout='match', match_columns=('event_id',), fold_id=7,
            definitions={'nested': ['clean']}, fold_metadata={'scope': ['clean']})
    context = make([0, 1, 2])
    context.validation = make([3, 4]) if validation else None
    context.observer = observer
    return context


class ScriptBackend:
    def __init__(self, sequence, *, rate=1., mutate_inputs=False):
        self.sequence = sequence
        self.rate = rate
        self.rate_changes = []
        self.mutate_inputs = mutate_inputs

    def initialize(self, context):
        self.context = context
        self.before = deepcopy(context)
        self.position = 0
        self.target_columns = list(context.y)
        if self.mutate_inputs:
            context.X.iloc[0, 0] = 999
            context.y.iloc[0, 0] = 888
            context.metadata.iloc[0, 0] = 777
            context.definitions['nested'].append('changed')
            context.fold_metadata['scope'].append('changed')
            if context.validation is not None:
                context.validation.X.iloc[0, 0] = 666

    def step(self):
        item = self.sequence[self.position]
        self.position += 1
        if isinstance(item, Exception):
            raise item
        return item.copy() if isinstance(item, dict) else {'train_loss': item}

    def snapshot(self):
        return {'position': self.position, 'rate': self.rate}

    def restore(self, state):
        self.position, self.rate = state['position'], state['rate']

    def get_learning_rate(self):
        return self.rate

    def set_learning_rate(self, rate):
        self.rate = rate
        self.rate_changes.append(rate)

    def predict(self, context):
        return {'predict': pd.DataFrame(float(self.position), index=context.X.index, columns=self.target_columns)}
