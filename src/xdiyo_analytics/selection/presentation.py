"""Readable model comparisons; machine identifiers remain in stored evidence."""

import json


def comparison_for_display(table, configurations=None):
    frame = table.copy()
    configurations = configurations or {}
    if 'run_id' in frame:
        frame.insert(1 if 'name' in frame else 0, 'parameters', [
            json.dumps(_parameters(configurations.get(key, {})), ensure_ascii=False, default=str, sort_keys=True)
            for key in frame.run_id])
    # Retain the statistical distinction without displaying opaque hashes.
    if 'comparison_group' in frame and frame.comparison_group.dropna().nunique() > 1:
        groups = {value: f'Population {i + 1}' for i, value in enumerate(frame.comparison_group.dropna().unique())}
        frame['comparison population'] = frame.comparison_group.map(groups)
    hidden = ['run_id', 'saved_run_id', 'config_hash', 'comparison_group', 'candidate_order']
    return frame.drop(columns=hidden, errors='ignore')


def _parameters(config):
    if config.get('grid_parameters'):
        return config['grid_parameters']
    model = config.get('model')
    if isinstance(model, dict) and 'params' in model:
        return model['params']
    return {k: v for k, v in config.items() if not k.endswith(('_id', '_hash'))}
