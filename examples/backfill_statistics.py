"""Audit all published seasons and optionally fill missing workbook statistics.

Run with --workbook PATH; inspect the audit, then rerun with --apply.
Source extracts and detailed audits stay in the ignored data/audits directory.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import re
from pathlib import Path
import time
import unicodedata
import xml.etree.ElementTree as ET

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from xdiyo_analytics.data._source import _open_source
from xdiyo_analytics.data.statistics_backfill import (
    STATISTICS, fill_missing_statistics, make_candidate, statistic_value, statistics_schema,
)
from xdiyo_analytics.odds.workbook import _Workbook, NS, LEAGUES, _season
from xdiyo_analytics.odds.crosswalk import load_mapping_rules


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def statistics_scope(row):
    """Resolve calendar seasons and the distinct Japanese transition tournament."""
    if (row['country'], row['league']) == ('Japan', 'J1 League'):
        year = str(row['season'])
        if re.fullmatch(r'20\d{2}', year):
            suffix = f'{int(year) % 100:02}'
            return ('J1_Transition' if year == '2026' else 'J1'), f'{suffix}_{suffix}'
        return 'J1', _season(year)
    return LEAGUES.get((row['country'], row['league'])), _season(row['season'])


def extract(workbook, directory):
    fingerprint = digest(workbook)
    # Extraction identity includes the mapping revision, not only workbook bytes.
    directory = directory / 'calendar-seasons-v2' / fingerprint[:20]
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'extract.json'
    if marker.exists():
        manifest = json.loads(marker.read_bytes())
        for item in manifest['files']:
            if digest(directory/item['file']) != item['sha256']:
                raise ValueError('Source extract changed')
        return directory, fingerprint
    book = _Workbook(workbook)
    try:
        # Decode shared strings once; stream the much larger worksheet XML.
        with book.archive.open('xl/sharedStrings.xml') as f:
            for _, element in ET.iterparse(f, events=('end',)):
                if element.tag == NS+'si':
                    book.strings[len(book.strings)] = ''.join(t.text or '' for t in element.iter(NS+'t'))
                    element.clear()
        files = []
        for sheet in ('Statistics_FT', 'Statistics_1H', 'Statistics_2H'):
            headers = book.headers(sheet)
            rows = []
            for number, cells in book.rows(sheet):
                if number == 1:
                    continue
                row = {name: book.decode(cells.get(col)) for col, name in headers.items()}
                league, season = statistics_scope(row)
                if league is None or season is None:
                    continue
                row.update(source_league=league, source_season=season, source_row=number)
                rows.append(row)
            frame = pd.DataFrame(rows)
            if frame.match_id.duplicated().any():
                raise ValueError(f'Duplicate source fixture IDs in {sheet}')
            out = directory / f'{sheet}.parquet'
            frame.to_parquet(out, compression='zstd', index=False)
            files.append(dict(file=out.name, sha256=digest(out), rows=len(frame)))
            print(f'Extracted {sheet}: {len(frame)} rows', flush=True)
        dump(marker, dict(workbook_sha256=fingerprint, files=files))
    finally:
        book.archive.close()
    return directory, fingerprint


def normalize(name):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(name)).casefold() if c.isalnum())


def match_fixtures(vendor, native, aliases, overrides):
    """Unique ordered-team/date joins only, with separately reviewed exceptions."""
    names = defaultdict(set)
    for row in native:
        for side in ('home', 'away'):
            names[row['source_league'], normalize(row[side+'_name'])].add(int(row[side+'_id']))
    def resolve(league, name):
        known = aliases.get((league, name))
        if known is not None:
            return int(known)
        ids = names[league, normalize(name)]
        return next(iter(ids)) if len(ids) == 1 else None
    groups = defaultdict(list)
    for row in native:
        groups[row['source_league'], row['source_season'], row['home_id'], row['away_id']].append(row)
    exceptional = {str(r['vendor_match_id']): r for r in overrides}
    matches, audit = {}, []
    for provider_id, copies in vendor.items():
        first = copies[0]
        league, season = first['source_league'], first['source_season']
        home, away = resolve(league, first['home_team']), resolve(league, first['away_team'])
        date = pd.to_datetime(float(first['start_datetime']), unit='D', origin='1899-12-30').round('s')
        record = dict(vendor_match_id=provider_id, source_league=league, source_season=season,
                      home=first['home_team'], away=first['away_team'], home_id=home, away_id=away,
                      vendor_date=str(date), status='unmatched', reason='no ordered-team/date match')
        identities = {(r['source_league'], r['source_season'], resolve(r['source_league'], r['home_team']),
                       resolve(r['source_league'], r['away_team']), r['start_datetime']) for r in copies}
        candidates = groups[league, season, home, away]
        selected = [r for r in candidates if pd.Timestamp(r['kickoff_utc'], unit='s', tz='UTC').date() == date.date()]
        exception = exceptional.get(provider_id)
        if len(identities) != 1:
            selected = []; record['reason'] = 'source sheets disagree'
        elif home is None or away is None:
            selected = []; record['reason'] = 'unknown team alias'
        elif not selected and exception is not None:
            selected = [r for r in candidates if str(r['event_id']) == str(exception['event_id'])
                        and str(exception['source_league']) == league and str(exception['source_season']) == season
                        and pd.Timestamp(exception['vendor_scheduled_at']) == date
                        and pd.Timestamp(exception['native_kickoff_at']) == pd.Timestamp(r['kickoff_utc'], unit='s', tz='UTC')]
            record['reason'] = 'reviewed schedule exception'
        if len(selected) == 1:
            match = selected[0]
            matches[provider_id] = match
            record.update(status='matched', event_id=match['event_id'], stem=match['stem'])
            if record['reason'] != 'reviewed schedule exception':
                record['reason'] = 'unique ordered-team/calendar-date match'
        elif len(selected) > 1:
            record['reason'] = 'ambiguous native fixtures'
        audit.append(record)
    reverse = Counter((r['stem'], r['event_id']) for r in matches.values())
    for row in audit:
        if row['status'] == 'matched' and reverse[row['stem'], row['event_id']] != 1:
            matches.pop(row['vendor_match_id'])
            row.update(status='unmatched', reason='multiple source IDs map to one fixture')
    return matches, audit


def write_table(table, original, temporary):
    source = pq.ParquetFile(original) if original.exists() else None
    codecs = ({source.metadata.row_group(i).column(j).compression.lower()
               for i in range(source.num_row_groups) for j in range(source.metadata.row_group(i).num_columns)}
              if source is not None else {'zstd'})
    if len(codecs) > 1 or 'snappy' in codecs:
        raise ValueError(f'Unexpected compression requires review: {original}: {codecs}')
    codec = next(iter(codecs), 'zstd')
    pq.write_table(table, temporary, compression='none' if codec == 'uncompressed' else codec,
                   use_compliant_nested_type=False)
    check = pq.ParquetFile(temporary).read()
    if not check.equals(table, check_metadata=True):
        raise ValueError(f'Parquet round-trip failed: {original}')
    return dict(file=original.name, rows=table.num_rows, bytes=temporary.stat().st_size, sha256=digest(temporary))


def finish(root, journal):
    plan = json.loads(journal.read_bytes())
    before = {Path(k).as_posix(): v for k, v in plan.get('before', {}).items()}
    # Validate the whole transaction before replacing its first file. A source
    # edited by another writer must not be overwritten during recovery.
    for entry in plan['files']:
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Publication path escaped dataset')
        allowed = {entry['sha256'], before.get(entry['path'])}
        if path.exists() and digest(path) not in allowed:
            raise ValueError(f'Publication changed since staging: {path}')
        temporary = path.with_suffix(path.suffix + '.statistics-tmp')
        if digest(temporary if temporary.exists() else path) != entry['sha256']:
            raise ValueError(f'Interrupted backfill changed: {path}')
    for entry in plan['files']:
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Publication path escaped dataset')
        temporary = path.with_suffix(path.suffix + '.statistics-tmp')
        candidate = temporary if temporary.exists() else path
        if digest(candidate) != entry['sha256']:
            raise ValueError(f'Interrupted backfill changed: {candidate}')
        if temporary.exists():
            temporary.replace(path)
    journal.replace(journal.with_name(journal.name.removesuffix('.pending.json')+'.done.json'))


def publish(root, source, pub_path, statistics, updates, audit_path, fingerprint):
    """Journal a coherent long-table/nested-table/manifest update with real hashes."""
    pub = json.loads(pub_path.read_bytes())
    nested_path = root/pub['season_file']
    if digest(nested_path) != pub['sha256']:
        raise ValueError(f'Nested season fingerprint changed: {nested_path}')
    nested = pq.ParquetFile(nested_path).read()
    stat_path = (source.table_path('statistics') if 'statistics' in source.manifest['tables']
                 else source.path.parent/'statistics.parquet')
    original_statistics = (pq.ParquetFile(stat_path).read() if stat_path.exists()
                           else pa.Table.from_pylist([], schema=statistics_schema()))
    original_rows, new_rows = original_statistics.to_pylist(), statistics.to_pylist()
    if len(new_rows) < len(original_rows):
        raise ValueError('Backfill cannot remove native statistic rows')
    def equal(a, b):
        return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
    for old, new in zip(original_rows, new_rows):
        for name, value in old.items():
            if equal(value, new[name]):
                continue
            empty = value is None or (isinstance(value, float) and math.isnan(value))
            if name == 'display':
                empty = empty or value in ('', '-', '—')
            if name not in ('value', 'display') or not empty:
                raise ValueError(f'Backfill attempted to alter a populated native field: {name}')
    original_by_event = defaultdict(list)
    for row in original_rows:
        original_by_event[row['event_id']].append(row)
    current = nested['statistics'].to_pylist() if 'statistics' in nested.column_names else [[] for _ in range(len(nested))]
    for i, event_id in enumerate(nested['event_id'].to_pylist()):
        if event_id in updates:
            # Never replace a nested observation that differs from its native
            # long-table copy. Both surfaces must agree before enrichment.
            before = pa.Table.from_pylist(current[i] or [], schema=original_statistics.schema)
            expected = pa.Table.from_pylist(original_by_event[event_id], schema=original_statistics.schema)
            if not before.equals(expected):
                raise ValueError(f'Nested/native statistics disagree before backfill: {event_id}')
            current[i] = updates[event_id]
    if 'statistics' in nested.column_names:
        field = nested.schema.field('statistics')
        changed = nested.set_column(nested.schema.get_field_index('statistics'), field, pa.array(current, type=field.type))
    else:
        changed = nested.append_column('statistics', pa.array(current, type=pa.list_(pa.struct(original_statistics.schema))))
    for name in nested.column_names:
        if name != 'statistics' and not nested[name].equals(changed[name]):
            raise AssertionError(f'Unrelated nested column changed: {name}')
    entries = []
    before = {path.relative_to(root).as_posix(): digest(path) for path in
              (stat_path, nested_path, source.path, pub_path) if path.exists()}
    for path, table in ((stat_path, statistics), (nested_path, changed)):
        temporary = path.with_suffix(path.suffix+'.statistics-tmp')
        info = write_table(table, path, temporary)
        entries.append(dict(path=path.relative_to(root).as_posix(), sha256=info['sha256']))
        if path == nested_path:
            pub.update(bytes=info['bytes'], sha256=info['sha256'])
        else:
            source.manifest['tables']['statistics'] = info
    enrichment = dict(policy='fill_missing_only', workbook_sha256=fingerprint,
                      audit=str(audit_path.resolve()), audit_sha256=digest(audit_path))
    source.manifest['statistics_enrichment'] = enrichment
    pub['statistics_enrichment'] = enrichment
    for path, contents in ((source.path, source.manifest), (pub_path, pub)):
        temporary = path.with_suffix(path.suffix+'.statistics-tmp')
        dump(temporary, contents)
        entries.append(dict(path=path.relative_to(root).as_posix(), sha256=digest(temporary)))
    journal = audit_path.with_suffix('.pending.json')
    dump(journal, dict(files=entries, before=before,
                      native_rows_preserved=len(original_rows),
                      appended_rows=len(new_rows)-len(original_rows),
                      unchanged_nested_columns=[n for n in nested.column_names if n != 'statistics']))
    finish(root, journal)


def run(args):
    root, audit_root = Path(args.data_root).resolve(), Path(args.audit_root).resolve()
    directory, fingerprint = extract(Path(args.workbook), audit_root/'source')
    run_dir = audit_root/fingerprint[:20]
    run_dir.mkdir(parents=True, exist_ok=True)
    for journal in run_dir.glob('*.pending.json'):
        if not args.apply:
            raise ValueError('Interrupted publication: rerun with --apply to finish')
        finish(root, journal)
    sources, native, pairs = {}, [], set()
    for path in sorted(root.glob('*.manifest.json')):
        stem = path.name.removesuffix('.manifest.json')
        league, a, b = stem.rsplit('_', 2)
        if getattr(args, 'leagues', None) and league not in args.leagues:
            continue
        source = _open_source(root, stem)
        match_path = source.table_path('matches')
        if digest(match_path) != source.manifest['tables']['matches']['sha256']:
            raise ValueError(f'Native matches changed: {stem}')
        league, a, b = stem.rsplit('_', 2)
        season = a+'_'+b
        pairs.add((league, season))
        sources[stem] = source
        for row in pq.ParquetFile(match_path).read().to_pylist():
            row.update(stem=stem, source_league=league, source_season=season)
            native.append(row)
    print(f'Loaded {len(native)} native fixtures in {len(sources)} publications', flush=True)
    frames, vendor = {}, defaultdict(list)
    for sheet in ('Statistics_FT', 'Statistics_1H', 'Statistics_2H'):
        frame = pd.read_parquet(directory/f'{sheet}.parquet')
        frame = frame[[tuple(v) in pairs for v in frame[['source_league', 'source_season']].itertuples(index=False, name=None)]]
        frames[sheet] = frame
        for row in frame.to_dict('records'):
            vendor[row['match_id']].append(row)
    aliases, overrides = load_mapping_rules(Path(args.mapping_root))
    if args.team_aliases:
        for item in json.loads(Path(args.team_aliases).read_bytes()):
            key = (item['league'], item['name'])
            if key in aliases and int(aliases[key]) != int(item['team_id']):
                raise ValueError(f'Conflicting reviewed team alias: {key}')
            aliases[key] = int(item['team_id'])
    mapped, mapping = match_fixtures(vendor, native, aliases, overrides)
    pd.DataFrame(mapping).to_csv(run_dir/'fixture_mapping.csv', index=False)
    print('Fixture mapping:', Counter(r['status'] for r in mapping), flush=True)
    candidates, rejected = defaultdict(list), []
    now = time.time()
    for sheet, frame in frames.items():
        period = sheet.removeprefix('Statistics_').lower()
        for row in frame.to_dict('records'):
            match = mapped.get(row['match_id'])
            if match is None or match.get('status_code') != 100 or match.get('is_awarded'):
                continue
            for stat in STATISTICS:
                try:
                    values = {side: statistic_value(row[f'{side}_{stat}_{period}'], stat) for side in ('home', 'away')}
                    if stat == 'ball_possession' and all(v is not None for v in values.values()) and abs(sum(values.values())-100) > 1:
                        raise ValueError('Possession pair does not total 100')
                except ValueError as exc:
                    rejected.append(dict(sheet=sheet, row=row['source_row'], match_id=row['match_id'], stat=stat, reason=str(exc)))
                    continue
                for side, value in values.items():
                    if value is None:
                        continue
                    evidence = dict(workbook_sha256=fingerprint, sheet=sheet, row=row['source_row'],
                                    column=f'{side}_{stat}_{period}', vendor_match_id=row['match_id'])
                    candidates[match['stem']].append(make_candidate(match, field=stat, period=period,
                        side=side, value=value, evidence=evidence, observed_at=now))
    dump(run_dir/'rejected_values.json', rejected)
    summary = []
    for stem, source in sources.items():
        if 'statistics' not in source.manifest['tables'] and not candidates[stem]:
            summary.append(dict(season=stem, source_candidates=0, filled_cells=0,
                                affected_matches=0, by_statistic={}, by_period={}))
            continue
        if 'statistics' in source.manifest['tables']:
            stat_path = source.table_path('statistics')
            if digest(stat_path) != source.manifest['tables']['statistics']['sha256']:
                raise ValueError(f'Native statistics changed: {stem}')
            original = pq.ParquetFile(stat_path).read()
        else:
            original = pa.Table.from_pylist([], schema=statistics_schema())
        rows = original.to_pylist()
        merged, audit = fill_missing_statistics(rows, candidates[stem])
        record = dict(season=stem, source_candidates=len(candidates[stem]), filled_cells=len(audit),
                      affected_matches=len({r['event_id'] for r in audit}),
                      by_statistic=dict(Counter(r['key'] for r in audit)),
                      by_period=dict(Counter(r['period'] for r in audit)))
        summary.append(record)
        if not audit:
            continue
        print(f'{stem}: {len(audit)} missing values across {record["affected_matches"]} matches', flush=True)
        audit_path = run_dir/f'{stem}.cells.json'
        if args.apply:
            if audit_path.exists():
                # Preserve earlier completed run provenance on repeated imports.
                previous = json.loads(audit_path.read_bytes())
                audit = previous + audit
            dump(audit_path, audit)
            changed_ids = {r['event_id'] for r in audit}
            updates = defaultdict(list)
            for row in merged:
                if row['event_id'] in changed_ids:
                    updates[row['event_id']].append(row)
            table = pa.Table.from_pylist(merged, schema=original.schema)
            publish(root, source, root/f'{stem}.manifest.json', table, updates, audit_path, fingerprint)
        else:
            dump(run_dir/f'{stem}.proposed.json', audit)
    report = dict(workbook_sha256=fingerprint, applied=args.apply, publications=len(sources),
                  mapping=dict(Counter(r['status'] for r in mapping)), rejected_values=len(rejected),
                  filled_cells=sum(r['filled_cells'] for r in summary),
                  affected_matches=sum(r['affected_matches'] for r in summary), seasons=summary)
    dump(run_dir/('applied.json' if args.apply else 'preflight.json'), report)
    print(json.dumps({k:v for k,v in report.items() if k!='seasons'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workbook', required=True)
    p.add_argument('--data-root', default='data/xDiyo_data')
    p.add_argument('--audit-root', default='data/audits/statistics_backfill')
    p.add_argument('--mapping-root', default='data/odds')
    p.add_argument('--team-aliases', default='docs/analytics/data/statistics_backfill_aliases.json')
    p.add_argument('--apply', action='store_true')
    p.add_argument('--leagues', nargs='+', help='Only inspect/publish these native league names.')
    run(p.parse_args())
