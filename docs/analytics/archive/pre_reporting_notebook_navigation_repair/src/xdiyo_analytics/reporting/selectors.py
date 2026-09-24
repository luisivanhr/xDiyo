"""Common selector execution contract and reusable fold-winner aggregation."""

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
import pandas as pd

from .contracts import FeatureSelection, StudyResult


@dataclass(kw_only=True)
class FeatureSelector:
    """A Reporter that returns selected columns, with shared execution modes.

    Implement select(context) to calculate a selection for ONE supplied population.
    PreTrainingAnalysis provides each fold for type='per_fold', or the unique
    pooled population for type='overall'. With across_folds=True (overall only),
    this base runs select separately for each selected fold and calls
    combine(context, fold_results) to produce one consensus. Thus every selector
    uses the same scope rules, independently of its scoring algorithm.

    select/combine return StudyResult with a FeatureSelection. Custom selectors
    can reuse earlier tables via context.previous_results or calculate their own.
    combine defines the algorithm's aggregation policy; vote_selections supplies
    a reusable nomination-count policy without assuming correlation scores.
    Override for_fold() only when local configuration differs from the final
    configuration (e.g. number of local nominees versus final selected features).

    No input mutation, implicit global selection, model fitting or averaging of
    fold coefficients to approximate an overall analysis. A consensus built from
    these folds needs separate outer evaluation for an unbiased performance claim.
    """

    type: str
    partition: str
    features_from: object = None
    across_folds: bool = False
    supported_types: ClassVar[tuple] = ("per_fold", "overall")

    def select(self, context):
        raise NotImplementedError("Implement select(context) for this selector's local calculation.")

    def combine(self, context, fold_results):
        raise NotImplementedError("Implement combine(context, fold_results), optionally using vote_selections.")

    def for_fold(self):
        return replace(self, type="per_fold", across_folds=False)

    def run(self, context):
        if self.type not in self.supported_types:
            raise ValueError("FeatureSelector supports per_fold or overall execution.")
        if not self.across_folds:
            return self._selection_result(self.select(context), context)
        if self.type != "overall" or not context.fold_rows:
            raise ValueError("across_folds=True requires type=overall and selected folds.")
        results = {}
        population = set(context.row_positions)
        for fold_id, rows in context.fold_rows.items():
            if not set(rows) <= population:
                raise ValueError("Consensus context must contain every selected fold partition row.")
            child = replace(context, X=context.X.loc[rows].copy(), y=context.y.loc[rows].copy(),
                            metadata=context.metadata.loc[rows].copy(), row_positions=rows.copy(),
                            type="per_fold", fold_id=fold_id,
                            fold_metadata=deepcopy(context.fold_metadata_by_id.get(fold_id, {})),
                            definitions=deepcopy(context.definitions),
                            previous_results=deepcopy({name: views[fold_id]
                                                       for name, views in context.fold_results.items()
                                                       if fold_id in views}),
                            fold_rows={}, fold_results={}, fold_metadata_by_id={})
            results[fold_id] = self.for_fold().run(child)
        return self._selection_result(self.combine(context, results), context)

    @staticmethod
    def _selection_result(result, context):
        if not isinstance(result, StudyResult) or not isinstance(result.selection, FeatureSelection):
            raise TypeError("Selectors must return StudyResult with a FeatureSelection.")
        columns = result.selection.columns
        if len(set(columns)) != len(columns) or set(columns) - set(context.X.columns):
            raise ValueError("Selected columns must be distinct input feature names.")
        return result


def vote_selections(fold_results, *, feature_order, k, score_column=None):
    """Aggregate arbitrary selectors' nominated columns, not their raw inputs.

    Returns FeatureSelection. Each fold casts one vote per selected feature.
    Sort by votes, then optional mean finite score, then feature_order. Only
    nominated features are eligible. k may be None to retain all nominees.
    score_column=None needs no common numerical score. If specified, each
    selection.ranking must contain unique 'feature' plus that score column;
    scores must have a comparable meaning across folds. Undefined scores remain
    missing and do not affect vote counts. Repeated/overlapping folds still each
    cast votes, without implying statistical independence.
    """
    if not fold_results:
        raise ValueError("Voting needs at least one fold result.")
    if k is not None and (isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1):
        raise ValueError("k must be None or a positive integer.")
    order = list(feature_order)
    if len(set(order)) != len(order):
        raise ValueError("feature_order must contain distinct feature names.")
    ranking = pd.DataFrame(index=pd.Index(order, name="feature"))
    ranking["votes"] = 0
    scores = []
    for result in fold_results.values():
        selection = result.selection
        if len(set(selection.columns)) != len(selection.columns) or set(selection.columns) - set(order):
            raise ValueError("Fold nominations must be distinct names in feature_order.")
        ranking.loc[list(selection.columns), "votes"] += 1
        if score_column is not None:
            table = selection.ranking
            if "feature" not in table or score_column not in table or table.feature.duplicated().any():
                raise ValueError("Score aggregation needs unique feature names and the requested score column.")
            score = pd.to_numeric(table.set_index("feature")[score_column], errors="raise").reindex(order)
            scores.append(score.where(np.isfinite(score)))
    sort_columns = ["votes"]
    if score_column is not None:
        score_frame = pd.concat(scores, axis=1)
        ranking["mean_score"] = score_frame.mean(axis=1)
        ranking["scored_folds"] = score_frame.count(axis=1)
        sort_columns.append("mean_score")
    ranking["vote_fraction"] = ranking.votes / len(fold_results)
    # An explicit final key guarantees input-order ties for multi-key sorts.
    ranking["_input_order"] = np.arange(len(order))
    ranking = ranking.sort_values(sort_columns + ["_input_order"],
                                  ascending=[False] * len(sort_columns) + [True], na_position="last")
    nominees = ranking.index[ranking.votes > 0]
    selected = tuple(nominees if k is None else nominees[:k])
    ranking["selected"] = ranking.index.isin(selected)
    return FeatureSelection(selected, ranking.drop(columns="_input_order").reset_index())
