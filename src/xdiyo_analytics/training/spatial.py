"""Opt-in shared map representations fitted inside the native model pipeline."""
from numbers import Integral
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class _SpatialEmbedding(TransformerMixin, BaseEstimator):
    """Common block handling; one fitted basis/codebook for all declared blocks."""
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _blocks(self, names):
        if self.blocks is None:
            return [np.arange(len(names))]
        if not self.blocks:
            raise ValueError('blocks must contain at least one ordered grid column list.')
        result = []
        for block in self.blocks:
            if isinstance(block, (str, bytes)) or not len(block):
                raise ValueError('Each block must be an ordered list of grid columns.')
            positions = []
            for col in block:
                if isinstance(col, Integral) and not isinstance(col, bool) and 0 <= col < len(names):
                    positions.append(int(col))
                elif isinstance(col, str) and col in names:
                    positions.append(names.index(col))
                else:
                    raise ValueError(f'Unknown spatial block column {col!r}.')
            result.append(np.asarray(positions))
        widths = {len(block) for block in result}
        if len(widths) != 1:
            raise ValueError('Shared spatial blocks must use identical grid resolution and cell ordering.')
        width = next(iter(widths))
        if int(np.sqrt(width))**2 != width or width < 4:
            raise ValueError('Each explicit spatial block must contain a complete square grid (at least 2x2).')
        flat = np.concatenate(result)
        if len(set(flat)) != len(flat):
            raise ValueError('Spatial blocks must contain distinct, nonoverlapping columns.')
        return result

    def fit(self, X, y=None):
        """Fit only supplied training rows; stack team maps into one shared basis."""
        self.__dict__.pop('encoder_', None)
        values = validate_data(self, X, ensure_all_finite='allow-nan', dtype=float)
        if not isinstance(self.scale, bool) or not isinstance(self.keep_other, bool):
            raise ValueError('scale and keep_other must be booleans.')
        names = list(getattr(self, 'feature_names_in_', [f'x{i}' for i in range(self.n_features_in_)]))
        self.blocks_ = self._blocks(names)
        self.input_names_ = np.asarray(names, dtype=object)
        selected = set(np.concatenate(self.blocks_))
        self.other_ = np.asarray([i for i in range(self.n_features_in_) if i not in selected and self.keep_other], dtype=int)
        sample = np.vstack([values[:, cols] for cols in self.blocks_])
        # An entirely absent training map cannot teach a basis or a median.
        sample = sample[~np.isnan(sample).all(axis=1)]
        if not len(sample) or np.isnan(sample).all(axis=0).any():
            raise ValueError('Spatial representation needs observed training values for every grid cell.')
        self.imputer_ = SimpleImputer(strategy='median', keep_empty_features=True).fit(sample)
        sample = self.imputer_.transform(sample)
        self.scaler_ = StandardScaler().fit(sample) if self.scale else None
        if self.scaler_ is not None:
            sample = self.scaler_.transform(sample)
        encoder = self._encoder()
        self.encoder_ = encoder.fit(sample)
        self.n_outputs_per_block_ = encoder.n_components_ if isinstance(encoder, PCA) else encoder.n_clusters
        self.n_training_maps_ = len(sample)
        return self

    def transform(self, X):
        check_is_fitted(self, 'encoder_')
        values = validate_data(self, X, reset=False, ensure_all_finite='allow-nan', dtype=float)
        outputs = [values[:, self.other_]]
        for columns in self.blocks_:
            sample = self.imputer_.transform(values[:, columns])
            if self.scaler_ is not None:
                sample = self.scaler_.transform(sample)
            outputs.append(self.encoder_.transform(sample))
        return np.hstack(outputs)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'encoder_')
        names = self.input_names_ if input_features is None else np.asarray(input_features, dtype=object)
        if len(names) != self.n_features_in_ or (hasattr(self, 'feature_names_in_') and not np.array_equal(names, self.feature_names_in_)):
            raise ValueError('input_features must match feature_names_in_.')
        kind = 'pc' if isinstance(self.encoder_, PCA) else 'cluster_distance'
        return np.asarray([*names[self.other_], *[f'spatial_{b}::{kind}{i+1}'
            for b in range(len(self.blocks_)) for i in range(self.n_outputs_per_block_)]], dtype=object)


class SpatialPCA(_SpatialEmbedding):
    """Shared PCA of causal historical grids, fitted per training fold.

    blocks lists ordered grid columns for e.g. home and away. None treats the
    entire input as one vector (use with ColumnTransformer). Run before generic
    array-only preprocessing when selecting columns by name. Missing cells use
    training-only medians; keep_other preserves unrelated columns for later steps.
    Grid construction and team-frame alignment happen before this transformer.
    """
    def __init__(self, n_components=3, *, blocks=None, scale=False, keep_other=True):
        self.n_components = n_components
        self.blocks = blocks
        self.scale = scale
        self.keep_other = keep_other

    def _encoder(self):
        return PCA(n_components=self.n_components, svd_solver='full')


class SpatialClusters(_SpatialEmbedding):
    """Shared training-only map codebook; output distances to fixed centers.

    These are distribution-shape clusters, not player positions or tactical
    labels. Shared home/away blocks use identical center ordering. No per-match
    independently relabeled clusters or fitted global pre-split state is used.
    """
    def __init__(self, n_clusters=3, *, blocks=None, scale=False, keep_other=True, random_state=0):
        self.n_clusters = n_clusters
        self.blocks = blocks
        self.scale = scale
        self.keep_other = keep_other
        self.random_state = random_state

    def _encoder(self):
        return KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
