"""Training-only target construction for a declared take/skip or allocation learner."""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from .training import fresh, prediction_context


@dataclass
class LearnedTargetAdapter:
    """Fit on supplied historical utility labels after availability; never expose them at predict.

    target kind must be take_skip or allocation. TargetSpec declares the objective;
    callers construct these training labels from historically issued selections.
    This adapter does not guess utility from ordinary match labels or odds.
    """
    model: object
    target_spec: object
    availability_column: str
    selection_id: str
    origin_column: str = 'prediction_origin'

    def fit(self,context):
        if self.target_spec.kind not in {'take_skip','allocation'} or not self.selection_id:
            raise ValueError('Declare take_skip/allocation labels and their selection identity.')
        if 'fit_at' not in context.fold_metadata:
            raise ValueError('Learned utility targets require fit_at.')
        time=pd.to_datetime(context.fold_metadata['fit_at'],utc=True)
        metadata=context.metadata
        needed={self.availability_column,self.origin_column,'selection_id','issued_at','trained_through'}
        if not needed<=set(metadata):
            raise ValueError('Learned targets require historical selection and timing provenance.')
        available=pd.to_datetime(metadata[self.availability_column],utc=True)
        issued=pd.to_datetime(metadata.issued_at,utc=True)
        trained=pd.to_datetime(metadata.trained_through,utc=True)
        if pd.isna(time) or available.isna().any() or issued.isna().any() or trained.isna().any() or (available>time).any() or (trained>=issued).any() or (issued>=available).any():
            raise ValueError('Training labels or learned selections were unavailable.')
        if not metadata.selection_id.eq(self.selection_id).all() or not metadata[self.origin_column].isin(['issued','chronological_oof','fixed_rule']).all():
            raise ValueError('Unknown selection definition or provenance.')
        from ..evaluation.decision_layer import _safe_columns
        _safe_columns(context.X.columns)
        if context.y.shape[1] != 1 or not np.isfinite(context.y.to_numpy(dtype=float)).all():
            raise ValueError('Utility adapters require one finite explicit training target.')
        values = context.y.iloc[:, 0]
        if self.target_spec.kind == 'allocation' and not values.between(0, 1).all():
            raise ValueError('Allocation training targets must be capital fractions in [0,1].')
        if self.target_spec.kind == 'take_skip' and not values.isin([0,1]).all():
            raise ValueError('Take/skip training targets must be binary 0/1.')
        self.model_=fresh(self.model)
        self.model_.fit(context)
        self.training_cutoff_=time
        self.training_summary_=dict(target=repr(self.target_spec),selection_id=self.selection_id,trained_through=str(time),rows=len(context.X))

    def predict(self,context):
        if not hasattr(self,'model_'):
            raise ValueError('Learned target adapter is not fitted.')
        return self.model_.predict(prediction_context(context))


@dataclass(frozen=True)
class UtilityTarget:
    """Explicit retrospective label rule, evaluated only after official availability.

    take_skip marks net return per unit above threshold. allocation clips positive
    excess unit return times scale to max_fraction. This is a declared research
    objective, not a claim that ex-post return is an optimal stake target.
    """
    kind: str = 'take_skip'
    threshold: float = 0.
    scale: float = .01
    max_fraction: float = .05
    name: str = 'utility'

    def build(self, outcomes, *, cutoff):
        if self.kind not in {'take_skip','allocation'} or not self.name or not outcomes.index.is_unique:
            raise ValueError('Declare a take_skip/allocation target and distinct outcome identities.')
        if not {'available_at','net_return_per_unit'} <= set(outcomes):
            raise ValueError('Utility labels need official availability and net return per unit.')
        if not np.isfinite([self.threshold,self.scale,self.max_fraction]).all() or self.scale <= 0 or not 0 <= self.max_fraction <= 1:
            raise ValueError('Invalid utility target threshold/scale/fraction.')
        time = pd.to_datetime(cutoff,utc=True)
        available = pd.to_datetime(outcomes.available_at,utc=True)
        if pd.isna(time) or available.isna().any() or (available > time).any():
            raise ValueError('Utility labels were unavailable at the training cutoff.')
        excess = pd.to_numeric(outcomes.net_return_per_unit,errors='raise') - self.threshold
        if not np.isfinite(excess).all():
            raise ValueError('Utility returns must be finite.')
        values = excess.gt(0).astype(int) if self.kind == 'take_skip' else (excess.clip(lower=0)*self.scale).clip(upper=self.max_fraction)
        return pd.DataFrame({self.name:values},index=outcomes.index)


@dataclass
class SavedUtilityModel:
    """Load an explicitly trusted, hash-pinned local utility artifact for decisions.

    Uses the existing trusted joblib loader; never fit or fetch a model. Pin the
    SHA256 of model.json, whose payload hashes are checked by load_model.
    """
    path: str
    manifest_sha256: str
    serializer: object = None

    def _model(self):
        if not hasattr(self, '_loaded'):
            from pathlib import Path
            from hashlib import sha256
            from ..training.persistence import load_model
            manifest = Path(self.path) / 'model.json'
            if sha256(manifest.read_bytes()).hexdigest() != self.manifest_sha256:
                raise ValueError('Saved utility artifact manifest differs from its pinned hash.')
            model = load_model(self.path, serializer=self.serializer).model
            if not isinstance(model, LearnedTargetAdapter) or not hasattr(model, 'training_cutoff_'):
                raise ValueError('Saved utility artifact must contain a fitted LearnedTargetAdapter.')
            self._loaded = model
        return self._loaded

    @property
    def target_spec(self):
        return self._model().target_spec

    @property
    def training_cutoff_(self):
        return self._model().training_cutoff_

    def predict(self, context):
        return self._model().predict(context)
