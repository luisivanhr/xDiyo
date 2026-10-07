"""Chronological inner populations, preserving original row keys at every fit."""
from copy import deepcopy
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from ..training.contracts import PredictionContext


def subset(context, rows, *, prediction=False):
    rows = np.asarray(rows, dtype=int)
    common = dict(X=context.X.iloc[rows].copy(), metadata=context.metadata.iloc[rows].copy(), layout=context.layout,
                  match_columns=context.match_columns, fold_id=context.fold_id, fold_metadata=deepcopy(context.fold_metadata),
                  definitions=deepcopy(context.definitions))
    if prediction:
        return PredictionContext(**common)
    from ..training.contracts import FitContext
    weights = context.sample_weight.iloc[rows].copy() if context.sample_weight is not None else None
    return FitContext(**common, y=context.y.iloc[rows].copy(), sample_weight=weights)


@dataclass(frozen=True)
class TrainingPlan:
    mode: str = 'parallel'
    n_splits: int = 3
    min_train_groups: int = 2
    group_by: tuple = ()
    time_column: str = 'kickoff_at'
    issue_column: str | None = None
    label_availability_column: str | None = None
    availability_delay: object = None
    feature_availability_column: str | None = None
    features_as_of_issue: bool = False
    embargo: str = '0h'
    uncovered: str = 'drop'
    repeated: str = 'error'
    deployment: str = 'refit_bases'
    max_fits: int = 1000

    def validate(self):
        if self.mode not in {'parallel', 'chronological', 'algorithmic_residual', 'honest_error_meta'}:
            raise ValueError('Unknown composition training mode.')
        if self.uncovered not in {'drop', 'error'} or self.repeated != 'error' or self.deployment != 'refit_bases':
            raise ValueError('Supported policies: uncovered drop/error, repeated error, deployment refit_bases.')
        for name in ('n_splits', 'min_train_groups', 'max_fits'):
            if isinstance(getattr(self,name), bool) or not isinstance(getattr(self,name), int) or getattr(self,name) < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if pd.Timedelta(self.embargo) < pd.Timedelta(0):
            raise ValueError('embargo cannot be negative.')

    def times(self, context):
        metadata = context.metadata
        kickoff = pd.to_datetime(metadata[self.time_column], utc=True, errors='raise')
        issue = pd.to_datetime(metadata[self.issue_column], utc=True, errors='raise') if self.issue_column else kickoff
        if kickoff.isna().any() or issue.isna().any() or (issue > kickoff).any():
            raise ValueError('Missing or invalid issue/kickoff timestamps.')
        if self.label_availability_column and self.availability_delay is not None:
            raise ValueError('Choose an availability column or an explicit delay proxy.')
        if self.label_availability_column:
            available = pd.to_datetime(metadata[self.label_availability_column], utc=True, errors='raise')
        elif self.availability_delay is not None:
            delay = pd.Timedelta(self.availability_delay)
            if delay < pd.Timedelta(0):
                raise ValueError('availability_delay cannot be negative.')
            available = kickoff + delay
        else:
            raise ValueError('Chronological composition needs explicit label availability or a declared delay proxy.')
        if (available.notna() & available.lt(kickoff)).any():
            raise ValueError('Outcome availability cannot precede fixture kickoff.')
        if self.feature_availability_column:
            feature_time = pd.to_datetime(metadata[self.feature_availability_column], utc=True, errors='raise')
            if feature_time.isna().any() or (feature_time > issue).any():
                raise ValueError('Features were unavailable at prediction issue time.')
        elif not self.features_as_of_issue:
            raise ValueError('Declare feature availability timestamps or features_as_of_issue=True for externally audited features.')
        return kickoff, issue, available

    def splits(self, context):
        self.validate()
        kickoff, issue, available = self.times(context)
        metadata = context.metadata
        keys = list(self.group_by or context.match_columns)
        if not keys or metadata[keys].isna().any().any():
            raise ValueError('Inner groups require complete identities.')
        codes = pd.factorize(pd.MultiIndex.from_frame(metadata[keys]), sort=False)[0]
        # A fixture cannot be divided even when a custom group declaration is used.
        check = pd.DataFrame({'group':codes},index=metadata.index)
        for key in context.match_columns:
            check[key] = metadata[key]
        if check.groupby(list(context.match_columns)).group.nunique().gt(1).any():
            raise ValueError('Prediction groups split fixture partners.')
        anchors = pd.Series(issue.to_numpy(), index=np.arange(len(issue))).groupby(codes).min()
        unique = sorted(anchors.unique())
        if len(unique) <= self.min_train_groups:
            raise ValueError('Not enough chronological groups for honest OOF features.')
        chunks = np.array_split(np.arange(self.min_train_groups, len(unique)), min(self.n_splits, len(unique)-self.min_train_groups))
        for chunk in chunks:
            selected = set(anchors.index[anchors.isin([unique[i] for i in chunk])])
            test = np.flatnonzero(np.isin(codes, list(selected)))
            boundary = issue.iloc[test].min() - pd.Timedelta(self.embargo)
            train_groups = []
            for code in sorted(set(codes)-selected):
                rows = np.flatnonzero(codes == code)
                # Whole groups; late released and overlapping rounds never leak.
                if (issue.iloc[rows] < boundary).all() and available.iloc[rows].notna().all() and (available.iloc[rows] <= boundary).all():
                    train_groups.append(code)
            train = np.flatnonzero(np.isin(codes, train_groups))
            if not len(train):
                raise ValueError('Availability/embargo leaves no eligible inner training rows.')
            yield train, test, boundary


def fresh(model):
    """Clone sklearn state; native adapters are deep-copied then fitted afresh."""
    from ..training.estimators import EstimatorAdapter
    from sklearn.base import clone
    if isinstance(model, EstimatorAdapter):
        return replace(model, estimator=clone(model.estimator))
    if hasattr(model, 'get_params'):
        return EstimatorAdapter(clone(model))
    if hasattr(model, 'build'):
        return model.build()
    if not callable(getattr(model, 'fit', None)) or not callable(getattr(model, 'predict', None)):
        raise TypeError('A graph model must implement the native ModelAdapter contract.')
    return deepcopy(model)


def fit_node(node, context):
    if node.frozen:
        if not node.artifact_id or node.artifact_vintage is None or node.trained_through is None:
            raise ValueError('Frozen children require artifact identity, vintage and trained-through cutoff.')
        vintage = pd.to_datetime(node.artifact_vintage, utc=True)
        cutoff = pd.to_datetime(node.trained_through, utc=True)
        if pd.isna(vintage) or pd.isna(cutoff) or cutoff > vintage:
            raise ValueError('Frozen provenance requires valid ordered cutoff/vintage timestamps.')
        expected = dict(artifact_id=node.artifact_id, artifact_vintage=str(node.artifact_vintage), trained_through=str(node.trained_through))
        if getattr(node.model, 'training_summary_', {}).get('frozen_provenance') != expected:
            raise ValueError('Frozen provenance does not match retained fitted artifact evidence.')
        return deepcopy(node.model)
    model = fresh(node.model)
    if node.calibration is None:
        if node.weighting is not None:
            context = replace(context, sample_weight=node.weighting.compute(context).aligned(context.X.index, list(context.y)))
        model.fit(context)
        return model
    from ..datasets import ModelDataset
    from ..training.calibration import CalibratedAdapter
    dataset = ModelDataset(context.X.reset_index(drop=True), context.y.reset_index(drop=True), context.metadata.reset_index(drop=True),
                           context.layout, tuple(context.match_columns), tuple(context.match_columns), 'declared', context.definitions)
    policy = deepcopy(node.calibration)
    fit, reserved, audit = policy.partition(dataset, np.arange(len(context.X)), context.fold_metadata)
    policy.validate_model(model, context.y.iloc[fit], context.y.iloc[reserved])
    fitting = subset(context,fit)
    if node.weighting is not None:
        fitting = replace(fitting, sample_weight=node.weighting.compute(fitting).aligned(fitting.X.index, list(fitting.y)))
    model.fit(fitting)
    prediction = subset(context,reserved,prediction=True)
    policy.fit(model.predict(prediction)[policy.response_method], context.y.iloc[reserved])
    audit['original_fit_keys'] = context.X.index[fit].tolist()
    audit['original_calibration_keys'] = context.X.index[reserved].tolist()
    policy.split_audit_ = audit
    return CalibratedAdapter(model,policy,context.X.index[reserved].to_numpy())
