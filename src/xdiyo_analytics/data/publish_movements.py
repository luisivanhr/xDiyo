"""Materialize reviewed movement flags in the current dataset, in place."""

import hashlib
import json
from pathlib import Path


def materialize_movement_flags(data_root, season_stem, *, reviewed_flags=None, compression=None):
    """Update current native matches and nested season Parquet with real hashes.

    Does not keep duplicate dataset versions. Old source-selection records must
    be refreshed explicitly. Unrelated tables are untouched. Small audit records
    retain old/new fingerprints, not a second copy of the observations.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ._source import _open_source
    from .movements import _read_movements, attach_movement_flags

    root = Path(data_root).resolve()
    # A small journal makes interrupted multi-file replacement resumable without
    # retaining another copy of the full dataset.
    journal = root / '_team_seasons' / 'materialization' / f'{season_stem}.pending.json'
    def finish(plan):
        for item in plan['files']:
            path = (root / item['path']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Materialization path must stay inside the dataset')
            temporary = path.with_suffix('.movement-tmp')
            candidate = temporary if temporary.exists() else path
            if hashlib.sha256(candidate.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError(f'Interrupted materialization file changed: {candidate}')
            if temporary.exists():
                temporary.replace(path)
        journal.with_suffix('.audit.json').write_text(json.dumps(plan['audit'], indent=2), encoding='utf-8')
        journal.unlink()
        return plan['audit']
    if journal.exists():
        return finish(json.loads(journal.read_bytes()))
    source = _open_source(root, season_stem)
    enrichment = source.manifest.get('movement_enrichment', {})
    original_hash = enrichment.get('original_matches_sha256') or (
        Path(enrichment['evidence']['path']).parent.name if enrichment.get('materialized')
        else source.manifest['tables']['matches']['sha256'])
    if enrichment.get('materialized'):
        from .loading import load_season
        flags = load_season(root, season_stem, tables='team_seasons', verify_hashes=True)['team_seasons']
        evidence = enrichment['evidence']
    else:
        flags, evidence = _read_movements(root, season_stem, source)
    changed = False
    if reviewed_flags is not None:
        from .movements import assert_movement_parity
        scope = source.manifest['scope']
        selected = reviewed_flags.loc[reviewed_flags.competition_id.eq(scope['league_id']) &
                                      reviewed_flags.season_id.eq(scope['season_id'])].copy()
        if selected.empty:
            raise ValueError('Reviewed movement evidence does not cover this season.')
        try:
            assert_movement_parity(selected, flags, season=season_stem)
        except ValueError:
            changed = True
        flags = selected
    if enrichment.get('materialized') and not changed and compression is None:
        return {'season': season_stem, 'already_materialized': True}
    native = source.table_path('matches')
    publication_path = root / f'{season_stem}.manifest.json'
    publication = json.loads(publication_path.read_bytes())
    nested = root / publication['season_file']
    keys = ['competition_id', 'season_id', 'home_id', 'away_id']

    def write_preserving(table, path, temporary):
        original = pq.ParquetFile(path)
        codecs = {original.metadata.row_group(i).column(j).compression.lower()
                  for i in range(original.num_row_groups)
                  for j in range(original.metadata.row_group(i).num_columns)}
        if (compression is None or path != nested) and len(codecs) > 1:
            raise ValueError('Mixed Parquet compression requires an explicit compression choice.')
        codec = (compression if path == nested else None) or next(iter(codecs), 'zstd')
        if codec == 'uncompressed':
            codec = 'none'
        # Preserve existing list-child names rather than renaming item to element.
        pq.write_table(table, temporary, compression=codec, use_compliant_nested_type=False)
        written = pq.ParquetFile(temporary).read()
        if not written.schema.equals(table.schema, check_metadata=True) or not written.equals(table):
            raise ValueError(f'Parquet rewrite changed schema or values: {path}')

    def enrich(path):
        table = pq.ParquetFile(path).read()
        context = table.select(keys).to_pandas(types_mapper=__import__('pandas').ArrowDtype)
        expected_ids = set(context.home_id) | set(context.away_id)
        if set(flags.team_id) != expected_ids:
            raise ValueError('Reviewed movement evidence must cover exactly the season roster.')
        enriched = attach_movement_flags(context, flags)
        for name in enriched.columns:
            if name in keys:
                continue
            values = pa.array(enriched[name])
            if name in table.column_names:
                index = table.column_names.index(name)
                table = table.set_column(index, table.schema.field(index), values)
            else:
                table = table.append_column(name, values)
        # Existing Arrow fields (including nested observations) stay unchanged.
        temporary = path.with_suffix('.movement-tmp')
        write_preserving(table, path, temporary)
        payload = temporary.read_bytes()
        return temporary, dict(file=path.name, rows=table.num_rows, bytes=len(payload),
                                sha256=hashlib.sha256(payload).hexdigest())

    native_temp, native_info = enrich(native)
    nested_temp, nested_info = enrich(nested)
    flag_path = native.parent / 'team_seasons.parquet'
    flag_temp = flag_path.with_suffix('.movement-tmp')
    pq.write_table(pa.Table.from_pandas(flags, preserve_index=False), flag_temp)
    flag_payload = flag_temp.read_bytes()
    old_manifest_hash = hashlib.sha256(source.path.read_bytes()).hexdigest()
    old_publication_hash = hashlib.sha256(publication_path.read_bytes()).hexdigest()
    source.manifest['tables']['matches'] = native_info
    source.manifest['tables']['team_seasons'] = dict(file=flag_path.name, rows=len(flags),
        bytes=len(flag_payload), sha256=hashlib.sha256(flag_payload).hexdigest())
    source.manifest['movement_enrichment'] = dict(materialized=True,
        original_matches_sha256=original_hash,
        evidence={'path': str(flag_path.resolve()), 'sha256': source.manifest['tables']['team_seasons']['sha256']})
    publication.update(bytes=nested_info['bytes'], sha256=nested_info['sha256'])
    publication['movement_enrichment'] = 'Season-entry flags materialized in matches and season Parquet; team_seasons contains evidence.'
    reconciliation = publication.get('reconciliation', {})
    if reconciliation:
        reconciliation['promotion_relegation_flags'] = publication['movement_enrichment']
        reconciliation.get('feature_status_counts', {})['promotion_relegation_flags'] = {'materialized': native_info['rows']}
    files = [dict(path=str(path.relative_to(root)), sha256=info['sha256'])
             for path, info in ((native, native_info), (nested, nested_info),
                 (flag_path, {'sha256': hashlib.sha256(flag_payload).hexdigest()}))]
    for path, value in ((source.path, source.manifest), (publication_path, publication)):
        temporary = path.with_suffix('.movement-tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
        files.append(dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(temporary.read_bytes()).hexdigest()))
    audit = dict(season=season_stem, old_manifest_sha256=old_manifest_hash,
                old_publication_sha256=old_publication_hash,
                new_manifest_sha256=files[-2]['sha256'],
                new_publication_sha256=files[-1]['sha256'],
                matches_sha256=native_info['sha256'], season_sha256=nested_info['sha256'])
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps({'files': files, 'audit': audit}, indent=2), encoding='utf-8')
    return finish({'files': files, 'audit': audit})
