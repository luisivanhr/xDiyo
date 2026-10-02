"""Explicitly move a recipe's data pins to current published data."""

from copy import deepcopy
import json
from pathlib import Path


def refresh_recipe_data(recipe, output_path, *, workspace=None, data_root=None):
    """Save an updated recipe and fresh selection records without fitting.

    Accept a recipe dictionary or JSON path. All non-data settings and old
    selection records remain untouched. Paths resolve against workspace (the
    working directory by default), like the experiment builder. This migrates
    data pins, not saved results, auxiliary files or odds snapshots.
    """
    from .recipe import validate_recipe
    from ..data._source import _open_source

    base = Path(workspace or Path.cwd()).resolve()
    def resolve(p):
        p = Path(p)
        return p.resolve() if p.is_absolute() else (base / p).resolve()
    if not isinstance(recipe, dict):
        recipe = json.loads(resolve(recipe).read_text(encoding='utf-8-sig'))
    validate_recipe(recipe)
    updated = deepcopy(recipe)
    config = updated['data']
    root = resolve(data_root if data_root is not None else config['data_root'])
    output = resolve(output_path)
    records = output.with_name(output.stem + '_data_records')
    if output.exists() or records.exists():
        raise FileExistsError('Choose a new output recipe path; existing recipes and records are preserved.')
    if output.is_relative_to(root) or records.is_relative_to(root):
        raise ValueError('Save recipes and selection records outside the source data directory.')
    seasons = config['seasons']
    seasons = [seasons] if isinstance(seasons, str) else seasons
    leagues = config.get('leagues')
    leagues = [leagues] if isinstance(leagues, str) else leagues
    selected = []
    for path in sorted(root.glob('*.manifest.json')):
        stem = path.name.removesuffix('.manifest.json')
        league, a, b = stem.rsplit('_', 2)
        if f'{a}_{b}' in seasons and (leagues is None or league in leagues):
            selected.append((stem, league, f'{a}_{b}'))
    if set(seasons) - {r[2] for r in selected} or set(leagues or []) - {r[1] for r in selected}:
        raise ValueError('Some requested seasons or leagues have no current publication.')
    if not selected:
        raise ValueError('No matching current publications.')
    tables = config.get('tables', ['matches'])
    tables = [tables] if isinstance(tables, str) else tables
    pending, changes = [], []
    for stem, _, _ in selected:
        source = _open_source(root, stem)
        for table in tables:
            if table == 'team_seasons' and table not in source.manifest['tables']:
                from ..data.movements import _read_movements
                _read_movements(root, stem, source)
            elif not source.table_path(table).is_file():
                raise FileNotFoundError(source.table_path(table))
        previous = None
        if config.get('record_dir'):
            old_path = resolve(config['record_dir']) / f'{stem}.json'
            if old_path.exists():
                previous = json.loads(old_path.read_text(encoding='utf-8'))
                if previous.get('season_stem') != stem:
                    raise ValueError(f'Selection record belongs to another season: {old_path}')
        changes.append({'season': stem,
                        'previous_manifest_sha256': previous.get('manifest_sha256') if previous else None,
                        'current_manifest_sha256': source.record['manifest_sha256'],
                        'changed': previous != source.record})
        pending.append((stem, source.record))
    # Resolve everything before creating a migrated recipe. No old version files
    # are required: their pins may be stale precisely because data was replaced.
    config['data_root'] = str(root)
    config['record_dir'] = str(records)
    records.mkdir(parents=True)
    for stem, record in pending:
        (records / f'{stem}.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(updated, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    return {'recipe': updated, 'path': str(output), 'changes': changes}
