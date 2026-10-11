"""Write a native UI recipe without fitting an experiment.

python examples/temporal_team_gat_recipe.py --data-root data/xDiyo_data \
    --league Bundesliga --seasons 22_23 23_24 24_25 --output gat-recipe.json
"""
import argparse
import json
from pathlib import Path

from xdiyo_analytics.ui.recipe import default_recipe, node, validate_recipe


def build_recipe(data_root, league, seasons, *, timezone='Europe/Berlin'):
    recipe = default_recipe(data_root)
    recipe.update(name='Temporal team GAT',
        features={'is_home': node('features.IsHome')},
        labels={'result': node('labels.Outcome', perspective='home')}, target='result',
        preprocessors=[], adapter=None, target_transformer=None,
        model=node('graphs.TemporalGraphModel',
            config=node('graphs.TemporalGraphConfig', timezone=timezone),
            control=node('training.TrainingControl', max_steps=100, seed=17),
            membership_policy='first_fixture'),
        prediction_methods=['predict', 'predict_proba'],
        post_reporters={
            'performance':node('reporting.PerformanceReporter', type='overall', partition='score', metrics=['log_loss','accuracy']),
            'loss':node('reporting.GATLossReporter', type='per_fold'),
            'embeddings':node('reporting.EmbeddingReporter', type='per_fold', partition='score', mode='tSNE', show_badges=True),
            'attention':node('reporting.GraphAttentionWeightReporter', type='per_fold', partition='score', show_badges=True),
        })
    recipe['data'].update(leagues=[league], seasons=list(seasons), tables=['matches','team_seasons'])
    return validate_recipe(recipe)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--league', required=True)
    parser.add_argument('--seasons', nargs='+', required=True)
    parser.add_argument('--timezone', default='Europe/Berlin')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    recipe = build_recipe(args.data_root, args.league, args.seasons, timezone=args.timezone)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(recipe, stream, indent=2)
