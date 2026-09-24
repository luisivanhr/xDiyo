"""Feature selection studies: reusable rankings plus explicit selected columns."""

from dataclasses import dataclass, replace

import pandas as pd

from .contracts import Artifact, FeatureSelection, StudyResult
from .studies import CorrelationAnalysis, _columns
from .selectors import FeatureSelector, vote_selections, resolve_selection_count


@dataclass(kw_only=True)
class TopKCorrelationSelector(FeatureSelector):
    """Select top-k features by summed absolute association with selected targets.

    Required type/partition make fitting scope explicit. For prospective model
    use, choose per_fold/train and keep selections separate for each fold.
    overall/all is an explicitly descriptive full-data selection, not an
    out-of-sample selection suitable for reusing in those data's test folds.

    source names an earlier CorrelationAnalysis result for identical rows,
    partition and fold. If None, calculate the chosen method internally. For
    configured MCC reuse a source with declared categories/thresholds; automatic
    MCC calculation without such configuration stays undefined as usual.
    features/targets select a subset of those available. If a source lacks any
    requested pair/method, raise instead of recomputing or silently skipping it.

    Ranking uses SUM absolute finite coefficients for the requested method across
    targets. All-missing features are ineligible; fewer than k eligible features
    returns fewer, explicitly noted. Ties follow requested/input feature order.
    features_from can restrict input to an earlier selector before ranking.
    Outputs are columns plus full ranking; input frames are never reduced in place.

    k and fold_k accept a count or a proportion strictly between zero and one.
    Proportions round upward against the requested input features, after any
    features_from restriction, before removing undefined scores. 1 means one
    feature. For example, k=0.8 requests 8 of 10 or 9 of 11 input features.

    across_folds=True requires type=overall and a split plan. Each selected fold
    nominates its top fold_k (None uses k); rank features by nominations, then mean
    finite pooled magnitude across folds, then input order. source can reuse a
    per-fold CorrelationAnalysis or None computes each fold's scores internally.
    This consensus is distinct from correlating pooled rows. Fold overlap counts
    as repeated votes intentionally; votes are not independent evidence. An outer
    untouched evaluation is needed to assess a consensus chosen using these folds.
    """

    k: int | float
    method: str = "spearman"
    source: object = None
    features: object = None
    targets: object = None
    fold_k: object = None

    def select(self, context):
        resolve_selection_count(self.k, 0)
        if self.method not in {"pearson", "spearman", "kendall", "mcc"}:
            raise ValueError("method must be pearson, spearman, kendall or mcc.")
        if not len(context.X.columns) and self.features is None:
            ranking = pd.DataFrame(columns=["feature", "pooled_magnitude", "selected"])
            return StudyResult("Feature selection", tables={"ranking": ranking},
                               notes=["No input features selected."],
                               selection=FeatureSelection((), ranking))
        features = _columns(context.X, self.features, "features")
        count = resolve_selection_count(self.k, len(features))
        targets = _columns(context.y, self.targets, "targets")
        if self.fold_k is not None:
            raise ValueError("fold_k only applies with across_folds=True.")
        if self.source is None:
            calculated = CorrelationAnalysis(type=self.type, partition=self.partition,
                                               features=features, targets=targets, methods=[self.method]).run(context)
        else:
            if self.source not in context.previous_results:
                raise ValueError(f"Correlation source {self.source!r} needs an earlier result with identical fold/partition/rows.")
            calculated = context.previous_results[self.source]
        if "coefficients" not in calculated.tables:
            raise ValueError("Correlation source must expose the coefficients table.")
        cells = calculated.tables["coefficients"]
        cells = cells[cells.feature.isin(features) & cells.target.isin(targets) & (cells.metric == self.method)].copy()
        actual = list(zip(cells.feature, cells.target))
        expected = {(feature, target) for feature in features for target in targets}
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Correlation source lacks a requested feature/target/method or has duplicate pairs.")
        grouped = cells.assign(magnitude=cells.value.abs()).groupby("feature", sort=False)
        ranking = grouped.magnitude.sum(min_count=1).reindex(features).rename("pooled_magnitude").to_frame()
        ranking["n_targets"] = grouped.value.count().reindex(features)
        ranking = ranking.sort_values("pooled_magnitude", ascending=False, kind="stable", na_position="last")
        chosen = tuple(ranking.index[ranking.pooled_magnitude.notna()][:count])
        ranking["selected"] = ranking.index.isin(chosen)
        ranking = ranking.rename_axis("feature").reset_index()
        notes = [f"Selected {len(chosen)} of {len(features)} input features using absolute {self.method} association.",
                 "Multiple targets contribute summed absolute coefficients; ties follow input feature order.",
                 f"Scores reused from {self.source!r}." if self.source is not None else "Scores calculated within this selection study."]
        if self.k < 1:
            notes.append(f"Requested {self.k:.1%} of {len(features)} input features, rounded up to {count}.")
        if len(chosen) < count:
            notes.append(f"Requested {count}; only {len(chosen)} features had defined scores.")
        if self.partition != "train" or self.type != "per_fold":
            notes.append("This selection describes its stated scope; it is not a per-fold training-only selection.")
        return StudyResult(
            title=f"Top {count} · {self.method}",
            artifacts=[Artifact("table", ranking, "Feature selection ranking")],
            tables={"ranking": ranking, "coefficients": cells}, notes=notes,
            selection=FeatureSelection(chosen, ranking.copy()))

    def for_fold(self):
        fold_k = self.k if self.fold_k is None else self.fold_k
        resolve_selection_count(fold_k, 0, name="fold_k")
        return replace(self, type="per_fold", k=fold_k, across_folds=False, fold_k=None)

    def combine(self, context, fold_results):
        features = _columns(context.X, self.features, "features") if len(context.X.columns) or self.features is not None else []
        count = resolve_selection_count(self.k, len(features))
        combined = vote_selections(fold_results, feature_order=features, k=self.k,
                                   score_column="pooled_magnitude")
        fold_tables, coefficient_tables = [], []
        for fold_id, local in fold_results.items():
            table = local.tables["ranking"].copy()
            table.insert(0, "fold_id", fold_id)
            fold_tables.append(table)
            coefficients = local.tables.get("coefficients", pd.DataFrame()).copy()
            coefficients.insert(0, "fold_id", fold_id)
            coefficient_tables.append(coefficients)
        per_fold = pd.concat(fold_tables, ignore_index=True)
        ranking = combined.ranking.rename(columns={"mean_score": "mean_magnitude"})
        chosen = combined.columns
        fold_k = self.k if self.fold_k is None else self.fold_k
        return StudyResult(
            title=f"Top {count} · {self.method} fold consensus",
            artifacts=[Artifact("table", ranking, "Winners across folds")],
            tables={"ranking": ranking, "fold_rankings": per_fold,
                    "coefficients": pd.concat(coefficient_tables, ignore_index=True)},
            notes=[f"Each of {len(context.fold_rows)} folds uses fold_k={fold_k}; proportions round up against that fold's input feature count.",
                   "Ordered by nomination count, then mean absolute pooled magnitude, then input feature order.",
                   f"Selected {len(chosen)} nominated features; requested {count} from {len(features)} inputs (k={self.k}).",
                   "Overlapping folds can repeat observations. Votes are descriptive, not independent evidence.",
                   "This consensus uses all stated fold partitions; evaluate it on separate outer test data."],
            selection=FeatureSelection(chosen, ranking.copy()))
