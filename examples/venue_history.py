"""Native 18-predictor fixture recipe inspired by Berrar and Lopes (2024).

Prepare only; no model is fitted. Select a complete league-season for rankings.
"""
import argparse

from xdiyo_analytics.features import DerivedStandings, MatchScore, RollingMean
from xdiyo_analytics.ui.recipe import catalog_for_ui, prepare_recipe
from examples.derived_standings import recipe as base_recipe


def definitions():
    result = {}
    for venue, window in [('all', 6), ('home', 3), ('away', 3)]:
        for side in ('for', 'against'):
            result[f'goals_{side}_{venue}'] = RollingMean(
                MatchScore(side=side), window=window, min_periods=1, venue=venue)
        result[f'form_{venue}'] = DerivedStandings([window], venue=venue)
    return result


def recipe(data_root):
    spec = base_recipe(data_root)
    spec['name'] = 'Home and away historical goals and form'
    catalog = catalog_for_ui()
    spec['features'] = {name: catalog.encode(expr) for name, expr in definitions().items()}
    return spec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/xDiyo_data')
    args = parser.parse_args()
    print(prepare_recipe(recipe(args.data_root)).dataset.X.head())
