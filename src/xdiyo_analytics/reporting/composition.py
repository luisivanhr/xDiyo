"""Read-only diagnostics for native fitted composition and correction adapters."""
from dataclasses import dataclass
from typing import ClassVar
import pandas as pd
from .contracts import StudyResult, Artifact


@dataclass(frozen=True)
class CompositionReporter:
    type: str
    partition: str
    supported_types: ClassVar[tuple] = ('per_fold', 'overall')

    def run(self, context):
        summaries, folds = [], []
        for fold_id, model in context.models.items():
            summary = getattr(model, 'training_summary_', {})
            if 'composition_version' not in summary and summary.get('mode') not in ('algorithmic_residual','honest_error_meta'):
                continue
            summaries.append(dict(fold_id=fold_id, **{k:v for k,v in summary.items() if k not in ('inner_folds','child_summaries')}))
            folds.extend(dict(outer_fold=fold_id, **record) for record in summary.get('inner_folds', []))
        if not summaries:
            raise ValueError('CompositionReporter requires retained fitted composition/correction models.')
        tables = {'composition_summary':pd.DataFrame(summaries), 'inner_fold_audit':pd.DataFrame(folds)}
        for name, values in context.predictions.items():
            if '/' in name:
                tables['child/' + name] = values.copy()
        return StudyResult('Composition diagnostics', tables=tables,
            artifacts=[Artifact('table', value, name.replace('_',' ')) for name,value in tables.items()],
            notes=['Inner audit keys refer to original dataset rows. OOF warm-up omissions and the deployment refit policy are explicit.'])
