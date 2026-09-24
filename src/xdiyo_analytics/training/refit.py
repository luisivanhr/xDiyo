"""Explicit final fitting after configuration selection, without implicit search."""

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..datasets import ModelDataset
from .contracts import PredictionContext, FoldResult, TrainingResult
from .runner import _columns, _fit_model, _frame, _positions


@dataclass
class FittedModel:
    """Fresh final fit and its declared development population.

    predict returns named DataFrames. evaluate attaches observed targets and
    metadata for PostTrainingAnalysis, without fitting again. Features and output
    targets retain the selected names. Evaluation against the SAME dataset may
    not overlap development rows. For a different dataset the caller explicitly
    declares same_dataset=False and owns cross-dataset identity separation.
    """
    model: object
    train_positions: np.ndarray
    fit_positions: np.ndarray
    validation_positions: np.ndarray
    feature_columns: tuple
    target_columns: tuple
    layout: str
    identity_columns: tuple
    match_columns: tuple
    target_perspective: str
    definitions: dict
    execution: dict = field(default_factory=dict)

    def save(self, path, *, serializer=None):
        """Save this fitted model for future prediction; see save_model()."""
        from .persistence import save_model
        return save_model(self, path, serializer=serializer)

    def predict(self, dataset, *, positions=None):
        if self.model is None:
            raise RuntimeError("This recovered result has no live fitted model. Restore it with its native adapter or perform an explicit refit.")
        if dataset.layout != self.layout or tuple(dataset.match_columns) != self.match_columns:
            raise ValueError("Prediction dataset must match the fitted layout/match identities.")
        rows = _positions(np.arange(len(dataset.X)) if positions is None else positions,
                          len(dataset.X), "prediction")
        context = PredictionContext(_frame(dataset.X[list(self.feature_columns)], rows),
                                    _frame(dataset.metadata, rows), self.layout, self.match_columns,
                                    0, {"stage": "final_refit"}, deepcopy(self.definitions))
        outputs = self.model.predict(context)
        expected = pd.Index(rows, name="row_position")
        if not isinstance(outputs, dict) or not outputs:
            raise TypeError("Adapter predict must return named DataFrames.")
        for name, frame in outputs.items():
            if not isinstance(name, str) or not name or not isinstance(frame, pd.DataFrame):
                raise TypeError("Each output needs a name and DataFrame.")
            if not frame.index.equals(expected) or not frame.columns.is_unique or not len(frame.columns):
                raise ValueError("Predictions must preserve row positions and distinct nonempty columns.")
        return {name: frame.copy(deep=True) for name, frame in outputs.items()}

    def evaluate(self, dataset, *, test_positions, score_positions=None, same_dataset=True):
        """Retain held-out predictions/outcomes in the existing TrainingResult format."""
        test = _positions(test_positions, len(dataset.X), "test")
        score = test.copy() if score_positions is None else _positions(score_positions, len(dataset.X), "score", allow_empty=True)
        if not np.isin(score, test).all():
            raise ValueError("score_positions must be contained in test_positions.")
        if same_dataset and np.intersect1d(test, self.train_positions).size:
            raise ValueError("Final evaluation must exclude the development population used for fitting/monitoring.")
        codes, _ = pd.factorize(dataset.groups, sort=False)
        for rows in (test, score):
            if np.isin(codes, codes[rows]).sum() != len(rows):
                raise ValueError("Final evaluation partitions must retain whole matches.")
        fold = FoldResult(
            0, self.model, self.train_positions.copy() if same_dataset else np.array([], dtype=int), test, score,
            self.feature_columns, self.target_columns, self.predict(dataset, positions=test),
            _frame(dataset.y[list(self.target_columns)], test), _frame(dataset.metadata, test),
            {"stage": "final_refit", "same_dataset": same_dataset,
             "development_positions": self.train_positions.tolist()},
            fit_positions=self.fit_positions.copy() if same_dataset else np.array([], dtype=int),
            validation_positions=self.validation_positions.copy() if same_dataset else np.array([], dtype=int),
            training_history=getattr(self.model, "training_history_", pd.DataFrame()).copy(),
            training_summary=deepcopy(getattr(self.model, "training_summary_", {})))
        return TrainingResult([fold], self.layout, self.identity_columns, self.match_columns,
                              self.target_perspective, deepcopy(self.definitions))


def refit_model(dataset, model_factory, *, train_positions, feature_columns=None,
                target_columns=None, validation=None, control=None, observer=None, execution=None):
    """Fit a fresh selected configuration on explicitly supplied development rows.

    No automatic winner selection, held-out evaluation or retraining schedule.
    With validation=None all supplied rows are fitted. If a new monitoring subset
    is requested, it is held out as during ordinary training. For all-development
    iterative refits, a caller can reuse the selected step budget with stopping
    disabled and restore_best=False to retain the last step, rather than consult
    final test outcomes. Preprocessing is refitted
    by the new adapter. Fixed selected feature names must be supplied explicitly.
    """
    if not isinstance(dataset, ModelDataset):
        raise TypeError("refit_model requires ModelDataset.")
    if not dataset.X.index.equals(dataset.y.index) or not dataset.X.index.equals(dataset.metadata.index):
        raise ValueError("Dataset frames must remain aligned.")
    train = _positions(train_positions, len(dataset.X), "train")
    features = _columns(feature_columns, dataset.X.columns, "feature_columns")
    targets = _columns(target_columns, dataset.y.columns, "target_columns")
    execution_info = {}
    model, fit, valid = _fit_model(dataset, train, model_factory, features, targets, fold_id=0,
                                  fold_metadata={"stage": "final_refit"}, validation=validation,
                                  control=control, observer=observer, execution=execution, execution_info=execution_info)
    return FittedModel(model, train, fit, valid, tuple(features), tuple(targets), dataset.layout,
                       tuple(dataset.identity_columns), tuple(dataset.match_columns), dataset.target_perspective,
                       deepcopy(dataset.definitions), execution_info)
