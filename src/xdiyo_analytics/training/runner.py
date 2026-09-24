"""Single-configuration fitting over prepared, model-independent splits."""

from copy import deepcopy
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..datasets import ModelDataset
from ..splits import Fold, SplitPlan
from .contracts import FitContext, FoldResult, PredictionContext, TrainingResult


def _columns(requested, available, name):
    columns = list(available) if requested is None else (
        [requested] if isinstance(requested, str) else list(requested))
    if not columns or len(set(columns)) != len(columns):
        raise ValueError(f"{name} must select distinct, nonempty columns.")
    missing = set(columns) - set(available)
    if missing:
        raise KeyError(f"Unknown {name}: {sorted(missing)}")
    return columns


def _positions(values, n, name, *, allow_empty=False):
    rows = np.asarray(values)
    if rows.ndim != 1 or (rows.size and rows.dtype.kind not in "iu"):
        raise ValueError(f"{name} must contain integer row positions.")
    if (not allow_empty and not rows.size) or ((rows < 0) | (rows >= n)).any():
        raise ValueError(f"{name} is empty or contains positions outside the dataset.")
    if len(np.unique(rows)) != len(rows):
        raise ValueError(f"{name} contains repeated row positions.")
    return rows.astype(int, copy=True)


def _frame(frame, rows):
    result = frame.iloc[rows].copy(deep=True)
    result.index = pd.Index(rows, name="row_position")
    return result


def split_training_rows(dataset, train_positions, validation=None):
    """Return fitting/validation positions within an explicit training population.

    validation is None, original dataset positions, or an object such as
    ValidationTail exposing select(dataset, train_positions). Whole matches stay
    intact in both populations. This does not modify outer test/score membership.
    """
    train = _positions(train_positions, len(dataset.X), "train")
    chosen = validation.select(dataset, train.copy()) if hasattr(validation, "select") else validation
    valid = _positions([] if chosen is None else chosen, len(dataset.X), "validation", allow_empty=True)
    if not np.isin(valid, train).all():
        raise ValueError("Validation rows must be contained in the supplied training population.")
    fit = train[~np.isin(train, valid)]
    if not len(fit):
        raise ValueError("Validation must leave a nonempty fitting population.")
    codes, _ = pd.factorize(dataset.groups, sort=False)
    for name, rows in (("fitting", fit), ("validation", valid)):
        if np.isin(codes, codes[rows]).sum() != len(rows):
            raise ValueError(f"{name} must retain whole matches.")
    return fit, valid


def _fit_model(dataset, train, model_factory, features, targets, *, fold_id, fold_metadata,
               validation=None, control=None, observer=None, execution=None, execution_info=None):
    fit, valid = split_training_rows(dataset, train, validation)
    if dataset.y[targets].iloc[train].isna().any().any():
        raise ValueError("Training targets contain missing values; prepare eligible training matches upstream.")
    common = dict(layout=dataset.layout, match_columns=tuple(dataset.match_columns), fold_id=fold_id)

    def context(rows):
        return FitContext(X=_frame(dataset.X[features], rows), metadata=_frame(dataset.metadata, rows),
                          **common, fold_metadata=deepcopy(fold_metadata), definitions=deepcopy(dataset.definitions),
                          y=_frame(dataset.y[targets], rows))

    training = context(fit)
    training.validation = context(valid) if len(valid) else None
    training.observer = observer
    model = model_factory()
    if not callable(getattr(model, "fit", None)) or not callable(getattr(model, "predict", None)):
        raise TypeError("model_factory must return a fresh ModelAdapter with fit/predict methods.")
    from .execution import configure_model
    info = configure_model(model, execution)
    if execution_info is not None:
        execution_info.update(info)
    limits = nullcontext()
    if execution is not None:
        from threadpoolctl import threadpool_limits
        limits = threadpool_limits(limits=execution.inner_threads)
    with limits:
        if control is not None or len(valid):
            if not callable(getattr(model, "fit_controlled", None)):
                raise TypeError("This adapter does not support training controls/validation; use a capable adapter or omit them.")
            from .control import TrainingControl
            if control is not None and not isinstance(control, TrainingControl):
                raise TypeError("control must be TrainingControl or None.")
            model.fit_controlled(training, control)
        else:
            model.fit(training)
    return model, fit, valid


def fit_predict(dataset, fold, model_factory, *, fold_id=0,
                feature_columns=None, target_columns=None, validation=None, control=None, observer=None, execution=None):
    """Fit one fresh adapter and predict every test row of one supplied Fold.

    This is the reusable candidate-evaluation building block. The factory takes
    no arguments and returns a fresh ModelAdapter. No search, inner folds,
    scheduled refits or reporting are performed. Optional training controls use
    adapter.fit_controlled; validation comes only from fold.train. observer may
    receive model-supported TrainingEvent updates. Fitted preparation
    belongs inside the adapter; precomputed feature timing remains upstream.

    Selected training targets must be nonmissing; test targets may be missing.
    Features pass through as supplied, including NaNs. Whole matches must remain
    together in train/test/score. Empty score is valid, train/test are nonempty.
    Prediction DataFrames must preserve test row order/index exactly; positional
    arrays should be wrapped by an adapter such as EstimatorAdapter.
    """
    if not isinstance(dataset, ModelDataset) or not isinstance(fold, Fold):
        raise TypeError("fit_predict consumes ModelDataset and Fold.")
    if not (dataset.X.index.equals(dataset.y.index) and
            dataset.X.index.equals(dataset.metadata.index)):
        raise ValueError("Dataset X, y and metadata must retain their shared index/order.")
    if any(not frame.columns.is_unique for frame in (dataset.X, dataset.y, dataset.metadata)):
        raise ValueError("Dataset frames must retain distinct column names.")
    n = len(dataset.X)
    train = _positions(fold.train, n, "train")
    test = _positions(fold.test, n, "test")
    score = _positions(fold.score, n, "score", allow_empty=True)
    if np.intersect1d(train, test).size or not np.isin(score, test).all():
        raise ValueError("Training and test rows must be disjoint; score must be a test subset.")
    codes, _ = pd.factorize(dataset.groups, sort=False)
    for name, rows in (("train", train), ("test", test), ("score", score)):
        if np.isin(codes, codes[rows]).sum() != len(rows):
            raise ValueError(f"{name} must retain all rows of each selected match.")
    features = _columns(feature_columns, dataset.X.columns, "feature_columns")
    targets = _columns(target_columns, dataset.y.columns, "target_columns")
    common = dict(layout=dataset.layout, match_columns=tuple(dataset.match_columns), fold_id=fold_id)
    prediction = PredictionContext(
        X=_frame(dataset.X[features], test), metadata=_frame(dataset.metadata, test),
        **common, fold_metadata=deepcopy(fold.metadata), definitions=deepcopy(dataset.definitions))
    execution_info = {}
    try:
        model, fit, valid = _fit_model(dataset, train, model_factory, features, targets,
                                      fold_id=fold_id, fold_metadata=fold.metadata,
                                      validation=validation, control=control, observer=observer,
                                      execution=execution, execution_info=execution_info)
        outputs = model.predict(prediction)
    except Exception as error:
        error.add_note(f"Training/prediction failed in fold {fold_id}.")
        raise
    index = pd.Index(test, name="row_position")
    if not isinstance(outputs, dict) or not outputs:
        raise TypeError("Adapter predict must return a nonempty name -> DataFrame mapping.")
    for name, values in outputs.items():
        if not isinstance(name, str) or not name or not isinstance(values, pd.DataFrame):
            raise TypeError("Each prediction output needs a nonempty name and a DataFrame.")
        if not values.index.equals(index) or not values.columns.is_unique or not len(values.columns):
            raise ValueError("Prediction outputs need the exact test row index/order and distinct nonempty columns.")
    summary = deepcopy(getattr(model, "training_summary_", {}))
    if execution_info:
        summary["execution"] = execution_info
    return FoldResult(
        fold_id, model, train, test, score, tuple(features), tuple(targets),
        {name: values.copy(deep=True) for name, values in outputs.items()},
        _frame(dataset.y[targets], test), _frame(dataset.metadata, test), deepcopy(fold.metadata),
        fit_positions=fit, validation_positions=valid,
        training_history=getattr(model, "training_history_", pd.DataFrame()).copy(deep=True),
        training_summary=summary)


@dataclass
class TrainingRunner:
    """Run one fixed configuration, fitting a fresh model once per selected fold.

    model_factory creates an adapter (and its estimator/preprocessing) per fold.
    No final full-data fit, optional search or scheduled refitting is implicit.
    feature_columns/target_columns select fixed names; None uses all columns.

    run(..., analysis_report=report, features_from='selector') explicitly consumes
    a selector's columns. Same-fold selection takes precedence over an overall
    selection. Its calculation rows must be inside this fold's fitting rows,
    excluding any monitoring validation. selection_plan prepares that scope.
    Thus a shared selection is usable for an independent holdout, but a consensus
    influenced by a fold's test outcomes cannot be reused to evaluate that fold.
    Selections are not recalculated here. To learn them within each training
    scope, run PreTrainingAnalysis first with per_fold/partition='train'.
    """

    model_factory: object
    feature_columns: object = None
    target_columns: object = None
    control: object = None
    validation: object = None
    observer: object = None
    execution: object = None

    def _validation_for(self, fold_id):
        if isinstance(self.validation, dict):
            return self.validation[fold_id]
        return self.validation

    def selection_plan(self, dataset, split_plan):
        """A copy whose train rows exclude monitoring validation for selectors.

        Use this plan for train-scoped PreTrainingAnalysis feature selection,
        then run training against the ORIGINAL plan. This avoids using monitoring
        labels in a fitted feature selector. Descriptive reports can use either
        explicitly chosen population. With validation=None membership is unchanged.
        """
        folds = []
        for i, fold in enumerate(split_plan.folds):
            fit, _ = split_training_rows(dataset, fold.train, self._validation_for(i))
            folds.append(Fold(fit, fold.test.copy(), fold.score.copy(), deepcopy(fold.metadata)))
        return SplitPlan(folds, split_plan.n_rows, split_plan.row_order.copy(), deepcopy(split_plan.paths))

    def run(self, dataset, split_plan, *, fold_ids=None, analysis_report=None, features_from=None):
        if not isinstance(dataset, ModelDataset) or not isinstance(split_plan, SplitPlan):
            raise TypeError("TrainingRunner consumes ModelDataset and SplitPlan.")
        if split_plan.n_rows != len(dataset.X):
            raise ValueError("Split plan must refer to this dataset's original row count/order.")
        selected = list(range(len(split_plan.folds))) if fold_ids is None else list(fold_ids)
        if len(set(selected)) != len(selected) or any(
                isinstance(i, bool) or not isinstance(i, (int, np.integer)) or
                i < 0 or i >= len(split_plan.folds) for i in selected):
            raise ValueError("fold_ids must contain distinct valid zero-based fold numbers.")
        if features_from is not None and (analysis_report is None or self.feature_columns is not None):
            raise ValueError("features_from needs analysis_report and cannot be combined with feature_columns.")
        if analysis_report is not None and features_from is None:
            raise ValueError("Choose features_from explicitly when supplying an analysis_report.")
        models, estimators, target_states = [], [], []
        original_factory = self.model_factory

        def fresh_model():
            model = original_factory()
            # Factories own construction. Catch accidental state reuse across
            # folds without imposing cloning semantics on arbitrary frameworks.
            from .estimators import EstimatorAdapter
            if any(model is prior for prior in models):
                raise ValueError("model_factory reused an adapter; create a fresh model for every fold.")
            if isinstance(model, EstimatorAdapter):
                if any(model.estimator is prior for prior in estimators):
                    raise ValueError("model_factory reused an estimator; create a fresh estimator for every fold.")
                estimators.append(model.estimator)
            from .targets import TargetTransformAdapter
            if isinstance(model, TargetTransformAdapter):
                from ..selection.core import _fitted_objects
                objects = _fitted_objects(model)
                if any(value is previous for value in objects for previous in target_states):
                    raise ValueError("model_factory reused an estimator or target transformer; create fresh fitting state for every fold.")
                target_states.extend(objects)
            models.append(model)
            return model

        jobs = []
        for fold_id in selected:
            fold = split_plan.folds[fold_id]
            validation = self._validation_for(fold_id)
            features, selection = self.feature_columns, None
            if features_from is not None:
                candidates = [study for study in analysis_report.studies
                              if study.name == features_from and study.fold_id == fold_id]
                if not candidates:
                    candidates = [study for study in analysis_report.studies
                                  if study.name == features_from and study.fold_id is None]
                if len(candidates) != 1 or candidates[0].result.selection is None:
                    raise ValueError("features_from needs one selector result for this fold or overall scope.")
                selection = candidates[0]
                fitting_rows, _ = split_training_rows(dataset, fold.train, validation)
                if selection.layout != dataset.layout or not np.isin(selection.row_positions, fitting_rows).all():
                    raise ValueError("Consumed feature selection must use only this fold's fitting training rows and layout; use selection_plan when reserving validation.")
                features = selection.result.selection.columns
            jobs.append((fold_id, fold, features, validation, selection))
        targets, control, execution = self.target_columns, self.control, self.execution

        def fit_one(job, observer):
            fold_id, fold, features, validation, selection = job
            result = fit_predict(dataset, fold, fresh_model, fold_id=fold_id,
                                 feature_columns=features, target_columns=targets,
                                 validation=validation, control=control, observer=observer, execution=execution)
            result.selection = deepcopy(selection)
            return result

        from .execution import run_fold_jobs
        results = run_fold_jobs(fit_one, jobs, execution, self.observer)
        return TrainingResult(results, dataset.layout, tuple(dataset.identity_columns),
                              tuple(dataset.match_columns), dataset.target_perspective,
                              deepcopy(dataset.definitions))
