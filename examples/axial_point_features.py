"""Native two-channel feature recipe; preparation only, no model fitting.

Merge recipe_features() into an existing recipe's features mapping. The original
recipe and existing features are not modified by this module.
"""
from xdiyo_analytics.features import RollingMean, SpatialPointSummary
from xdiyo_analytics.ui.recipe import node


def feature_specs():
    return {f'activity_{field}_r20': RollingMean(
        SpatialPointSummary(field, kinds=('player',), side='for', min_points=1),
        window=20, min_periods=1, venue='all')
        for field in ('axis_cos2', 'axis_sin2')}


def recipe_features():
    return {f'activity_{field}_r20': node('features.RollingMean',
        source=node('features.SpatialPointSummary', field=field, kinds=['player'],
                    side='for', min_points=1), window=20, min_periods=1, venue='all')
        for field in ('axis_cos2', 'axis_sin2')}


if __name__ == '__main__':
    import json
    print(json.dumps(recipe_features(), indent=2))
