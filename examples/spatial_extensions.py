"""Opt-in Phase C/D feature definitions. Importing this file never runs a fit."""
from xdiyo_analytics.features import (
    Heatmap, SpatialHistoricalDeviation, SpatialFixtureDistance, SpatialPointSummary,
    RollingMean, SpatialEntropy,
)


def feature_definitions(grid_size=6):
    source = Heatmap(grid_size=grid_size, kinds=('player',), orientation='team')
    return {
        'activity_deviation_from_prior_r20':RollingMean(SpatialHistoricalDeviation(source, window=20), window=20),
        'style_distance_r20':SpatialFixtureDistance(source),
        'home_context_distance_r20':SpatialFixtureDistance(source, comparison='home_context'),
        'away_context_distance_r20':SpatialFixtureDistance(source, comparison='away_context'),
    }


def keeper_definitions(grid_size=6):
    """Separate normalization and explicitly selected role; never an implicit mix."""
    return {
        'keeper_mean_x_r20':RollingMean(SpatialPointSummary('mean_x', kinds=('goalkeeper',)),20),
        'keeper_depth80_r20':RollingMean(SpatialPointSummary('depth_80', kinds=('goalkeeper',)),20),
        'keeper_entropy_r20':RollingMean(SpatialEntropy(Heatmap(grid_size=grid_size,
            kinds=('goalkeeper',), orientation='team')),20),
    }


def embedding_recipe_step(prepared, family, method='pca', dimensions=3):
    """Build a UI preprocessor node from an explicitly prepared grid family.

    The features must include RollingMean(Heatmap(..., orientation='team')).
    Place this returned node first in recipe['preprocessors']. Fitting then
    happens only inside each native training fold, and is saved with its model.
    """
    from xdiyo_analytics.ui.recipe import node
    specs = prepared.dataset.definitions['spatial_features']
    blocks = []
    selected = []
    for side in ('home','away'):
        matches = [s for s in specs.values() if s.get('family') == family and s.get('fixture_side') == side]
        if len(matches) != 1 or matches[0]['kind'] != 'grid' or matches[0]['orientation'] != 'team':
            raise ValueError('Select one complete team-oriented grid family with Home and Away blocks.')
        selected.append(matches[0]);blocks.append(matches[0]['columns'])
    contract = ('grid_size','normalization','method','kinds','use_weights','sigma','resolution')
    if any(selected[0].get(k) != selected[1].get(k) for k in contract):
        raise ValueError('Shared spatial blocks must have the same grid contract.')
    if method == 'pca':
        return node('training.SpatialPCA',blocks=blocks,n_components=dimensions)
    if method == 'clusters':
        return node('training.SpatialClusters',blocks=blocks,n_clusters=dimensions,random_state=0)
    raise ValueError('method must be pca or clusters.')
