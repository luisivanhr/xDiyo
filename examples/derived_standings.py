"""Prepare native derived standings without fitting any model.

Run with --data-root /path/to/data/xDiyo_data. Uses completed 2022/23 only.
"""
import argparse
from xdiyo_analytics.data import load_seasons
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import DerivedStandings, evaluate_features
from xdiyo_analytics.ui.recipe import default_recipe, node, prepare_recipe


def recipe(data_root):
    spec = default_recipe(data_root)
    spec['name'] = 'Score-derived standings example'
    spec['data'].update(seasons=['22_23'], leagues=['Premier_League'], tables=['matches'])
    spec['features'] = {
        'cumulative': node('features.DerivedStandings', normalized_ranks=[], include_raw=True),
        'form': node('features.DerivedStandings', normalized_ranks=[3,6]),
    }
    spec['labels'] = {'outcome': node('labels.Outcome', perspective='home')}
    spec['target'] = 'outcome'
    spec['split'] = node('splits.TemporalSplit', train_size=20, test_size=5, unit='rounds')
    return spec


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',default='data/xDiyo_data')
    args=parser.parse_args()
    data=load_seasons(args.data_root,['22_23'],leagues=['Premier_League'],
                      tables=['matches','team_seasons'],include_awarded=True)
    history=build_team_history(data)
    features=evaluate_features(history,{
        'cumulative':DerivedStandings(None,include_raw=True),
        'form':DerivedStandings([3,6]),
    },team_seasons=data['team_seasons'],keyed=True)
    print(features.head())
    prepared=prepare_recipe(recipe(args.data_root))
    print(prepared.dataset.X.head())
