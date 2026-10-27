"""Explicit, copy-only migration to the majority-started countdown policy."""
from copy import deepcopy


def _nodes(value, path='$'):
    if isinstance(value, dict):
        if value.get('component') == 'features.SeasonCountdown':
            yield value, path
        for key, child in value.items():
            yield from _nodes(child, f'{path}.{key}')
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from _nodes(child, f'{path}[{i}]')


def reject_obsolete_countdown(recipe):
    for node, path in _nodes(recipe):
        if 'schedule_mode' in node.get('params', {}):
            raise ValueError(f'{path}: SeasonCountdown schedule_mode was removed. '
                'Use migrate_season_countdown_recipe explicitly; values now use the majority-started policy.')


def migrate_season_countdown_recipe(recipe):
    """Return (copied recipe, change records); never overwrite or prepare a run.

    Removing the old mode is a semantic migration, not old-value compatibility.
    Explicit schedule_history inputs are preserved: if they contain archived
    snapshots, replace them with one consistent fixture history before preparing.
    Legacy total_rounds-only custom sources also need verified membership or
    the existing team_count override to establish their majority threshold.
    """
    migrated = deepcopy(recipe)
    changes = []
    for node, path in _nodes(migrated):
        params = node.setdefault('params', {})
        if 'schedule_mode' in params:
            previous = params['schedule_mode']
            if previous not in ('as_of', 'retrospective'):
                raise ValueError(f'{path}: unrecognized legacy schedule_mode {previous!r}.')
            del params['schedule_mode']
            changes.append(dict(path=path, removed_schedule_mode=previous,
                policy='majority_started', values_may_change=True))
    return migrated, changes
