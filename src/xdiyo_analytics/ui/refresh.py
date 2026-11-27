"""Explicitly move a recipe's data pins to current published data."""

from copy import deepcopy
import json
import hashlib
import os
import base64
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path


def refresh_recipe_data(recipe, output_path, *, workspace=None, data_root=None,
                        previous_revision=None, dry_run=False):
    """Save an updated recipe and fresh selection records without fitting.

    Accept a recipe dictionary or JSON path. All non-data settings and old
    selection records remain untouched. Paths resolve against workspace (the
    working directory by default), like the experiment builder. This migrates
    data pins, not saved results, auxiliary files or odds snapshots.
    """
    from .recipe import validate_recipe
    from ..data._source import _open_source
    from ..data.loading import _validate_season_options, _discover_seasons, load_season

    base = Path(workspace or Path.cwd()).resolve()
    def resolve(p):
        p = Path(p)
        return p.resolve() if p.is_absolute() else (base / p).resolve()
    original_path = None if isinstance(recipe, dict) else resolve(recipe)
    if original_path is not None:
        recipe = json.loads(original_path.read_text(encoding='utf-8-sig'))
    validate_recipe(recipe)
    if recipe.get('data_selection') is not None:
        raise ValueError('This helper refreshes native data pins; use the named-dataset migration workflow for data_selection recipes.')
    updated = deepcopy(recipe)
    config = updated['data']
    root = resolve(data_root if data_root is not None else config['data_root'])
    output = resolve(output_path)
    records = output.with_name(output.stem + '_data_records')
    report_path = output.with_name(output.stem + '_refresh.json')
    if output.exists() or records.exists() or report_path.exists():
        raise FileExistsError('Choose a new output recipe path; existing recipes and records are preserved.')
    if output.is_relative_to(root) or records.is_relative_to(root):
        raise ValueError('Save recipes and selection records outside the source data directory.')
    if not root.is_dir():
        raise FileNotFoundError(f'Data directory does not exist: {root}')
    native_before = deepcopy(config.get('native_selection'))
    native_after = None
    previous_pins = None
    overlay = updated.get('statistics_overlay')
    if overlay is not None:
        from ..data.statistics_overlay import verify_overlay_selection
        verify_overlay_selection(overlay, base)
    allow_missing_stats = overlay is not None or config.get('allow_missing_statistics', False)
    if config.get('query') is not None or native_before is not None:
        from ..data.native_query import NativeLibrary, validate_selection_envelope
        if native_before is not None:
            validate_selection_envelope(native_before)
            previous_pins = {p['stem']: p['record'] for p in native_before['publications']}
        if config.get('seasons') is not None or config.get('leagues') is not None:
            raise ValueError('Choose native query or legacy seasons/leagues, not both')
        query = config.get('query')
        if query is None:
            query = native_before['query'] if native_before else {}
        library = NativeLibrary(root)
        native_after = library.select(**query)
        tables = config.get('tables',['matches'])
        tables = [tables] if isinstance(tables,str) else tables
        by_stem = {p['stem']:p for p in library.publications}
        selected = [(by_stem[p['stem']]['source_season'],by_stem[p['stem']]['source_league'],p['stem']) for p in native_after['publications']]
        config['native_selection'] = native_after
        # Validate filtered scope construction as well as each underlying file.
        library.load(query=query, tables=tables, verify_hashes=True, include_awarded=config.get('include_awarded', False),
                     allow_missing_statistics=allow_missing_stats)
    else:
        seasons, leagues, tables = _validate_season_options(config['seasons'], config.get('leagues'),
            config.get('tables', ['matches']), config.get('include_awarded', False))
        try:
            selected = _discover_seasons(root, seasons, leagues)
        except ValueError as exc:
            raise ValueError('Some requested seasons or leagues have no current publication.') from exc
    pending, changes = [], []
    def historical_manifest(record, revision):
        if not revision:
            return None
        # Resolve an exact commit, never interpolate shell text or fetch history.
        commit = subprocess.run(['git', 'rev-parse', '--verify', str(revision)+'^{commit}'],
            cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        repo = Path(subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=root,
            check=True, capture_output=True, text=True).stdout.strip()).resolve()
        publication = json.loads(base64.b64decode(record['publication_base64'], validate=True))
        path = (root / publication['manifest']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Previous manifest leaves dataset root')
        payload = subprocess.run(['git', 'show', commit+':'+path.relative_to(repo).as_posix()],
            cwd=root, check=True, capture_output=True).stdout
        if hashlib.sha256(payload).hexdigest() != record['manifest_sha256']:
            raise ValueError('Previous revision does not reproduce the saved manifest pin')
        return commit, json.loads(payload)
    for _, _, stem in selected:
        source = _open_source(root, stem)
        for table in tables:
            if table == 'statistics' and allow_missing_stats and table not in source.manifest['tables']:
                continue
            if table == 'team_seasons' and table not in source.manifest['tables']:
                from ..data.movements import _read_movements
                _read_movements(root, stem, source)
            elif not source.table_path(table).is_file():
                raise FileNotFoundError(source.table_path(table))
            else:
                with source.table_path(table).open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if digest != source.manifest['tables'][table]['sha256']:
                    raise ValueError(f'{stem}/{table}: fingerprint mismatch during recipe refresh.')
        previous = None
        if config.get('record_dir') and (previous_pins is None or stem in previous_pins):
            old_path = resolve(config['record_dir']) / f'{stem}.json'
            if not old_path.is_file():
                raise FileNotFoundError(f'Missing previous selection record: {old_path}')
            previous = json.loads(old_path.read_text(encoding='utf-8'))
            if previous_pins is not None and any(previous.get(field) != pin
                    for field, pin in previous_pins[stem].items()):
                raise ValueError(f'Previous record disagrees with the pinned native selection: {old_path}')
            if previous.get('season_stem') != stem:
                raise ValueError(f'Selection record belongs to another season: {old_path}')
            try:
                payload = base64.b64decode(previous['publication_base64'], validate=True)
                if hashlib.sha256(payload).hexdigest() != previous['publication_sha256']:
                    raise ValueError('Previous publication fingerprint mismatch')
                old_publication = json.loads(payload)
                if old_publication['season_file'] != stem+'.parquet':
                    raise ValueError('Previous publication identity mismatch')
                old_manifest_hash = previous['manifest_sha256']
                if len(old_manifest_hash) != 64 or any(c not in '0123456789abcdef' for c in old_manifest_hash):
                    raise ValueError('Invalid previous manifest fingerprint')
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f'Invalid previous selection: {old_path}') from exc
        # Validate readable tables, identities, calendar/rule metadata and joins,
        # independently of the old recipe's optional verify_hashes setting.
        load_season(root, stem, tables=tables, verify_hashes=True, allow_missing_statistics=allow_missing_stats,
                    include_awarded=config.get('include_awarded', False))
        if _open_source(root, stem).record != source.record:
            raise ValueError(f'Publication changed during refresh validation: {stem}')
        prior = historical_manifest(previous, previous_revision) if previous else None
        migration = source.manifest.get('native_integration', {}).get('previous_revision', {})
        revision = prior[0] if prior else (migration.get('git_commit')
            if previous and migration.get('manifest_sha256') == previous['manifest_sha256'] else None)
        old_tables = prior[1]['tables'] if prior else (migration.get('tables')
            if revision else None)
        table_changes = (sorted(n for n in set(old_tables) | set(source.manifest['tables'])
            if old_tables.get(n, {}).get('sha256') != source.manifest['tables'].get(n, {}).get('sha256'))
            if old_tables is not None else None)
        changed = previous != source.record
        changes.append({'season': stem,
                        'previous_manifest_sha256': previous.get('manifest_sha256') if previous else None,
                        'current_manifest_sha256': source.record['manifest_sha256'],
                        'previous_selection': previous, 'current_selection': source.record,
                        'previous_dataset_revision': revision,
                        'reproduction': 'restore_dataset_from_git_revision' if revision else 'previous_revision_unresolved',
                        'changed_tables': table_changes,
                        'change_kind': ('new_selection' if previous is None else 'unchanged' if not changed
                            else 'substantive' if table_changes else 'manifest_only' if table_changes == [] else 'unclassified'),
                        'changed': changed})
        pending.append((stem, source.record))
    # Resolve everything before creating a migrated recipe. No old version files
    # are required: their pins may be stale precisely because data was replaced.
    config['data_root'] = str(root)
    config['record_dir'] = str(records)
    report = dict(schema_version=1, original_recipe=recipe,
                  original_recipe_path=str(original_path) if original_path else None,
                  original_recipe_sha256=(hashlib.sha256(original_path.read_bytes()).hexdigest() if original_path else None),
                  output_recipe=str(output), changes=changes,
                  warning='Refresh creates a new input identity; previous experiment results are not migrated. Manifest-only changes can change feature semantics.')
    if overlay is not None:
        report['verified_statistics_overlay'] = deepcopy(overlay)
    if native_after is not None:
        report['native_selection'] = dict(previous=native_before,current=native_after,
            added=sorted({p['stem'] for p in native_after['publications']}-{p['stem'] for p in (native_before or {}).get('publications',[])}),
            removed=sorted({p['stem'] for p in (native_before or {}).get('publications',[])}-{p['stem'] for p in native_after['publications']}))
    if dry_run:
        return dict(recipe=updated, path=str(output), changes=changes, report=report, dry_run=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.recipe-refresh-', dir=output.parent) as temporary:
        stage = Path(temporary)
        staged_records = stage / 'records'
        staged_records.mkdir()
        for stem, record in pending:
            (staged_records / f'{stem}.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        staged_recipe = stage / 'recipe.json'
        staged_recipe.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        staged_report = stage / 'refresh.json'
        report['output_recipe_sha256'] = hashlib.sha256(staged_recipe.read_bytes()).hexdigest()
        staged_report.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        installed = False
        report_installed = False
        try:
            staged_records.rename(records)
            installed = True
            os.link(staged_report, report_path)
            report_installed = True
            # Atomic, no-overwrite publication on the same filesystem.
            os.link(staged_recipe, output)
        except BaseException:
            if report_installed:
                report_path.unlink()
            if installed:
                for stem, _ in pending:
                    (records / f'{stem}.json').unlink()
                records.rmdir()
            raise
    return {'recipe': updated, 'path': str(output), 'changes': changes, 'report_path': str(report_path)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe')
    parser.add_argument('--output', required=True)
    parser.add_argument('--workspace')
    parser.add_argument('--data-root')
    parser.add_argument('--previous-revision')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    result = refresh_recipe_data(args.recipe, args.output, workspace=args.workspace,
        data_root=args.data_root, previous_revision=args.previous_revision, dry_run=args.dry_run)
    print(json.dumps({k:v for k,v in result.items() if k not in ('recipe','report','changes')}, indent=2))
    for change in result['changes']:
        print(json.dumps({k:v for k,v in change.items() if k not in ('previous_selection','current_selection')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
