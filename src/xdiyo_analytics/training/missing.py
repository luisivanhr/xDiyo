"""Explicit, pre-imputation eligibility for optimization rows only."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MissingFitRows:
    """Drop fitting matches missing any explicitly required constructed feature.

    None (no policy) keeps all rows. Apply after reserving validation/calibration,
    never to raw history or evaluation. Team-layout pairs stay together: a missing
    required value on either side excludes both optimization rows. Missingness
    means pandas isna, not infinity. Required columns must be model inputs.
    """
    required_features: tuple[str, ...]

    def __post_init__(self):
        if isinstance(self.required_features, str) or self.required_features is None:
            raise ValueError("MissingFitRows requires explicit distinct nonempty required_features.")
        names = tuple(self.required_features)
        if not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
            raise ValueError('MissingFitRows requires explicit distinct nonempty required_features.')
        object.__setattr__(self, 'required_features', tuple(names))

    def select(self, dataset, rows, feature_columns=None):
        selected = dataset.X.columns if feature_columns is None else (
            [feature_columns] if isinstance(feature_columns, str) else feature_columns)
        unknown = set(self.required_features) - set(selected)
        if unknown:
            raise ValueError(f'MissingFitRows required_features must be selected model inputs: {sorted(unknown)}')
        values = dataset.X.loc[:, list(self.required_features)].iloc[rows]
        missing = values.isna()
        codes, _ = pd.factorize(dataset.groups, sort=False)
        dropped = np.isin(codes[rows], codes[rows[missing.any(axis=1).to_numpy()]])
        kept = rows[~dropped]
        if not len(kept):
            raise ValueError('MissingFitRows leaves no fitting rows.')
        return kept.copy(), dict(policy='drop_missing_required', required_features=list(self.required_features),
            input_rows=len(rows), retained_rows=len(kept), excluded_positions=rows[dropped].tolist(),
            missing_by_feature={str(k): int(v) for k, v in missing.sum().items()},
            exclusion_reason='missing_required_feature_or_paired_match')


def validate_fit_adapter(model):
    """Reject hidden validation splits: reserve monitoring before filtering."""
    from ..selection.core import _fitted_objects
    for obj in _fitted_objects(model):
        if getattr(obj, 'requires_inner_preparation', False):
            raise ValueError('MissingFitRows does not support chronological composition; configure child fit policies explicitly.')
        params = obj.get_params(deep=False) if callable(getattr(obj, 'get_params', None)) else {}
        early = params.get('early_stopping', False)
        if early not in (False, None) or params.get('n_iter_no_change') is not None:
            raise ValueError('MissingFitRows requires external validation reserved before filtering; disable estimator-owned early stopping.')
