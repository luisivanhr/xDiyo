"""Export a portable native recipe only. Never load data, fit or predict.

Run with PYTHONPATH=src:
  python examples/all_combinations_draw.py --output <new-local-directory>
Supply your own reviewed odds mapping in the exported recipe before use.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from xdiyo_analytics.ui import default_recipe
from xdiyo_analytics.ui.recipe import node, export_python, export_notebook


def draw_recipe(*, threshold=0.5, snapshot='data/odds',
                crosswalk='data/odds/mappings/reviewed', snapshot_hash=None, crosswalk_hash=None):
    recipe = default_recipe()
    recipe['name'] = 'Binary draw whole-group combinations'
    option = node('labels.BetOption', source=node('labels.Outcome', perspective='home'), selection='draw')
    recipe['labels'] = {'draw': deepcopy(option)}
    recipe['target'] = 'draw'
    recipe['model'] = node('sklearn.linear_model.LogisticRegression', max_iter=1000)
    recipe['prediction_methods'] = ['predict', 'predict_proba']
    odds = node('input.OddsSeries', snapshot=snapshot, crosswalk=crosswalk,
                seasons=['22_23','23_24','24_25'], market='1x2', selection='draw', quote_type='opening',
                settlement_confirmed=True, snapshot_hash=snapshot_hash, crosswalk_hash=crosswalk_hash)
    recipe['post_reporters'] = {}
    for name, cutoff in [('Baseline', None), ('Filtered', threshold)]:
        recipe['post_reporters'][name] = node('reporting.BetOutcomeReporter', type='overall', partition='test',
            pooling='last', offers={'draw':node('evaluation.BetOffer', option=deepcopy(option), odds=deepcopy(odds))},
            policy=node('evaluation.BinaryDrawThreshold', max_non_draw_probability=cutoff),
            history={'ref':'history'}, catalog={'ref':'team_catalog'}, show_badges=True,
            composition=node('evaluation.BetSlip', tickets={
                title:node('evaluation.AllCombinations', legs=k, stage_column='tournament_id', stake=1., max_tickets=100000)
                for title,k in [('Pairs',2),('Triples',3),('Quads',4)]}))
        recipe['post_reporters'][name+' performance'] = node('reporting.BetPerformanceReporter',
            type='overall', partition='test', pooling='last', source=name)
    return recipe


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    recipe = draw_recipe()
    (args.output/'recipe.json').write_text(json.dumps(recipe, indent=2), encoding='utf-8')
    (args.output/'experiment.py').write_text(export_python(recipe), encoding='utf-8')
    (args.output/'experiment.ipynb').write_text(json.dumps(export_notebook(recipe), indent=2), encoding='utf-8')
    print('Recipe exported only. Nothing was fitted, predicted or evaluated.')
