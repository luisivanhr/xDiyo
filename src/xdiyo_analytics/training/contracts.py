"""Model-independent fitting inputs and reusable prediction results."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass
class PredictionContext:
    """A model's inputs, indexed by original dataset row_position.

    Metadata is context, never automatically appended to X. No evaluation
    targets are supplied to predict(). Custom adapters own any use of metadata.
    groups contains exact match identities, not ranker group sizes; rankers may
    reorder internally but must restore the original index in their outputs.
    """

    X: pd.DataFrame
    metadata: pd.DataFrame
    layout: str
    match_columns: tuple[str, ...]
    fold_id: int
    fold_metadata: dict = field(default_factory=dict)
    definitions: dict = field(default_factory=dict)

    @property
    def groups(self):
        keys = list(zip(*(self.metadata[name].tolist() for name in self.match_columns)))
        return pd.Series(keys, index=self.X.index, dtype=object, name="match_group")


@dataclass
class FitContext(PredictionContext):
    """Fitting inputs; y is always a DataFrame, including single targets.

    validation is a separate FitContext reserved from the training population.
    It is for monitoring, not preprocessing fit or optimization updates. observer
    is an optional callable receiving TrainingEvent objects from capable adapters.
    """

    y: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation: object = None
    observer: object = None


class ModelAdapter(Protocol):
    """An extensible model interface, with no required ML framework.

    fit updates this fresh adapter in place. predict returns named DataFrames
    (e.g. 'predict', 'predict_proba'), each indexed exactly like context.X.
    Output columns may describe targets, classes or other model-specific values.
    A factory supplied to the runner must create a fresh adapter on every call.
    """

    def fit(self, context: FitContext) -> None: ...

    def predict(self, context: PredictionContext) -> dict[str, pd.DataFrame]: ...


@dataclass
class FoldResult:
    """One initial fit and its held-out predictions, with original row positions.

    predictions/y_true/metadata cover ALL test rows, including missing targets
    and rows outside score_positions. No missing values are filled or dropped.
    model is the fitted adapter. selection optionally retains a consumed StudyRun.
    train_positions identifies the entire development population. fit_positions
    and validation_positions distinguish optimization from monitoring rows.
    Training frames are not duplicated; histories and summaries are retained.
    """

    fold_id: int
    model: object
    train_positions: np.ndarray
    test_positions: np.ndarray
    score_positions: np.ndarray
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    predictions: dict[str, pd.DataFrame]
    y_true: pd.DataFrame
    metadata: pd.DataFrame
    fold_metadata: dict = field(default_factory=dict)
    selection: object = None
    fit_positions: object = None
    validation_positions: object = None
    training_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    training_summary: dict = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Structured outputs for later metrics, reporters and candidate comparison.

    Repeated held-out rows remain distinct fold occurrences, never averaged.
    No scoring, search, scheduled refitting or final full-data fit is implicit.
    """

    folds: list[FoldResult]
    layout: str
    identity_columns: tuple[str, ...]
    match_columns: tuple[str, ...]
    target_perspective: str
    definitions: dict = field(default_factory=dict)
    fitted_report: object = None

    def predictions_by_fold(self, output="predict", *, scored_only=False):
        """Return copies indexed by row_position; numeric outputs suit CPCV paths.

        reconstruct_paths needs all test rows, so retain scored_only=False there.
        Different folds may have different probability-class columns; none are
        invented or silently zero-filled.
        """
        return {fold.fold_id: (fold.predictions[output].loc[fold.score_positions]
                              if scored_only else fold.predictions[output]).copy(deep=True)
                for fold in self.folds}

    def prediction_frame(self, output="predict", *, scored_only=False):
        """Concatenate predictions with (fold_id, row_position) index.

        Class columns absent from a fold stay missing when concatenating.
        """
        frames = self.predictions_by_fold(output, scored_only=scored_only)
        if not frames:
            return pd.DataFrame(index=pd.MultiIndex.from_arrays(
                [[], []], names=["fold_id", "row_position"]))
        return pd.concat(frames, names=["fold_id", "row_position"])
