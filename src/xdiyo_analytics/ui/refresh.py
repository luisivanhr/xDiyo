"""Explicitly move a recipe's data pins to current published data."""

from copy import deepcopy
import json
import hashlib
import os
from tempfile import TemporaryDirectory
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
    from ..data.loading import _validate_season_options, _discover_seasons

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
    seasons, leagues, tables = _validate_season_options(config['seasons'], config.get('leagues'),
        config.get('tables', ['matches']), config.get('include_awarded', False))
    if not root.is_dir():
        raise FileNotFoundError(f'Data directory does not exist: {root}')
    try:
        selected = _discover_seasons(root, seasons, leagues)
    except ValueError as exc:
        raise ValueError('Some requested seasons or leagues have no current publication.') from exc
    pending, changes = [], []
    for _, _, stem in selected:
        source = _open_source(root, stem)
        for table in tables:
            if table == 'team_seasons' and table not in source.manifest['tables']:
                from ..data.movements import _read_movements
                _read_movements(root, stem, source)
            elif not source.table_path(table).is_file():
                raise FileNotFoundError(source.table_path(table))
            elif config.get('verify_hashes', False):
                with source.table_path(table).open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if digest != source.manifest['tables'][table]['sha256']:
                    raise ValueError(f'{stem}/{table}: fingerprint mismatch during recipe refresh.')
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
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.recipe-refresh-', dir=output.parent) as temporary:
        stage = Path(temporary)
        staged_records = stage / 'records'
        staged_records.mkdir()
        for stem, record in pending:
            (staged_records / f'{stem}.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        staged_recipe = stage / 'recipe.json'
        staged_recipe.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        installed = False
        try:
            staged_records.rename(records)
            installed = True
            # Atomic, no-overwrite publication on the same filesystem.
            os.link(staged_recipe, output)
        except BaseException:
            if installed:
                for stem, _ in pending:
                    (records / f'{stem}.json').unlink()
                records.rmdir()
            raise
    return {'recipe': updated, 'path': str(output), 'changes': changes}
