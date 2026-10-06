"""Native opt-in recipes. Importing this module never loads data or fits a model."""
from xdiyo_analytics.features import (
    Difference, Heatmap, MatchScore, Product, Ratio, RegionMass,
    RollingMean, RollingWeightedMean, SpatialPointSummary, Stat,
)


def feature_definitions(window=20, min_periods=1, grid_size=6):
    common = dict(window=window, min_periods=min_periods, venue='all')
    corners = Stat(period='ALL', group='Match overview', key='cornerKicks', field='value')
    goals = MatchScore(score_field='current', side='for')
    point = SpatialPointSummary(field='mean_x', side='for', kinds=('player',))
    centered = Difference(Product(2., Ratio(point, 100.)), 1.)
    grid = Heatmap(grid_size=grid_size, normalization='mass', kinds=('player',),
                   side='for', orientation='team')
    territory = Difference(RegionMass(grid, 'opponent_half'), RegionMass(grid, 'own_half'))
    return {
        'corners_forward_interaction': RollingMean(Product(corners, centered), **common),
        'goals_territorial_interaction': RollingMean(Product(goals, territory), **common),
        'corner_weighted_position': RollingWeightedMean(point, weights=corners, **common),
        'goal_weighted_position': RollingWeightedMean(point, weights=goals, **common),
        'corner_weighted_axis_cos2': RollingWeightedMean(
            SpatialPointSummary('axis_cos2', side='for', kinds=('player',)), corners, **common),
        'corner_weighted_axis_sin2': RollingWeightedMean(
            SpatialPointSummary('axis_sin2', side='for', kinds=('player',)), corners, **common),
    }


def recipe_features(**kwargs):
    from xdiyo_analytics.ui import catalog_for_ui
    catalog = catalog_for_ui()
    return {name: catalog.encode(expression) for name, expression in feature_definitions(**kwargs).items()}
