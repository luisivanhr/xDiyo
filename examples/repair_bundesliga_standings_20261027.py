"""Install the reviewed abandoned/awarded fixture on both published surfaces.

Default is a validated dry run. No collector records or unrelated tables change.
Old match columns/rows remain byte-for-value identical; new evidence is sparse.
Application is recoverable, not a simultaneous four-file atomic switch: exclude
concurrent readers during publication. A durable pending journal precedes the
first replacement; after interruption rerun the same command with --apply to
roll forward before manifest hash checks. Dry runs never recover or publish.
Changed/missing pending payloads or unexpected live bytes stop recovery. Do not
delete the journal or remaining .standings-tmp files while recovery is pending.
The OS lock excludes concurrent invocations of this repair, releases on process
exit, and leaves a reusable lock file. This covers process interruption, not a
hardware/power-loss durability guarantee across every filesystem.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from xdiyo_analytics.data.fixture_coverage import audit_fixture_coverage
from xdiyo_analytics.data.supplemental_fixtures import append_fixture_tables, supplemental_event_id

EVIDENCE = Path('docs/analytics/data/bundesliga_20220318_review.json')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


@contextmanager
def _publication_lock(root):
    """OS-owned lock: process termination releases it; retain the lock inode."""
    with (root / '.bundesliga-standings.lock').open('a+b') as lock:
        lock.seek(0, os.SEEK_END)
        if not lock.tell():
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another Bundesliga standings publication is active.') from exc
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == 'nt':
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _target(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Publication path must stay inside the dataset.')
    return path


def _finish_publication(root, journal, evidence_hash):
    """Roll forward only known before/after bytes; preflight the entire set."""
    plan = json.loads(journal.read_text(encoding='utf-8'))
    if plan.get('version') != 1 or plan.get('evidence_sha256') != evidence_hash:
        raise ValueError('Pending publication evidence/version differs; recovery stopped.')
    files = plan['receipt']['files']
    paths = [_target(root, item['path']) for item in files]
    if not files or len(set(paths)) != len(paths):
        raise ValueError('Pending publication has empty or duplicate targets.')
    for item, path in zip(files, paths):
        current = digest(path)
        temporary = path.with_suffix(path.suffix + '.standings-tmp')
        if current == item['after']:
            continue
        if current != item['before']:
            raise ValueError(f'Publication target changed outside the pending repair: {path}')
        if not temporary.exists() or digest(temporary) != item['after']:
            raise ValueError(f'Pending publication payload missing or changed: {temporary}')
    for item, path in zip(files, paths):
        current = digest(path)
        if current == item['after']:
            continue
        if current != item['before']:
            raise ValueError(f'Publication target changed during recovery: {path}')
        temporary = path.with_suffix(path.suffix + '.standings-tmp')
        if digest(temporary) != item['after']:
            raise ValueError(f'Pending publication payload changed during recovery: {temporary}')
        temporary.replace(path)
    if any(digest(path) != item['after'] for item, path in zip(files, paths)):
        raise ValueError('Publication verification failed; recovery journal retained.')
    journal.unlink()
    return plan['receipt']


def _publish(root, journal, pending, receipt):
    # Prepared Parquet/JSON files are flushed before the durable journal becomes
    # visible. Never consume a payload before recording how to recover it.
    for temporary, path in pending:
        with temporary.open('r+b') as stream:
            os.fsync(stream.fileno())
    for item in receipt['files']:
        if digest(_target(root, item['path'])) != item['before']:
            raise ValueError('Source changed during preparation; nothing published.')
    if journal.exists():
        raise ValueError('Recover the existing pending publication first.')
    draft = journal.with_suffix('.journal-tmp')
    plan = dict(version=1, evidence_sha256=receipt['evidence_sha256'], receipt=receipt)
    with draft.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    draft.replace(journal)
    return _finish_publication(root, journal, receipt['evidence_sha256'])


def repaired_tables(matches, statistics, nested, review):
    """Pure, schema-preserving append with explicit sparse ruling metadata."""
    original = matches
    fields = {
        'provider_is_awarded': pa.bool_(), 'play_status': pa.string(),
        'result_classification': pa.string(), 'result_review_json': pa.string(),
        'result_available_at': pa.timestamp('us', tz='UTC'),
        'standings_result_available_at': pa.timestamp('us', tz='UTC'),
        **{side + suffix: pa.float64() for side in ('home', 'away')
           for suffix in ('_score_played', '_score_awarded', '_standings_goals')},
    }
    for name, dtype in fields.items():
        for which in ('matches', 'nested'):
            table = matches if which == 'matches' else nested
            if name not in table.column_names:
                values = table['is_awarded'] if name == 'provider_is_awarded' else pa.nulls(len(table), type=dtype)
                table = table.append_column(name, values)
            if which == 'matches':
                matches = table
            else:
                nested = table
    rows = original.to_pylist()
    if {(r['competition_id'], r['season_id']) for r in rows} != {(review['competition_id'], review['season_id'])}:
        raise ValueError('Reviewed competition/season differs from source.')
    fixture = {name: None for name in matches.column_names}
    for name in ('competition_id', 'season_id', 'tournament_id', 'season_year'):
        values = {r[name] for r in rows}
        if len(values) != 1:
            raise ValueError('Ambiguous season identity.')
        fixture[name] = values.pop()
    if supplemental_event_id(review['source_match_id']) != review['event_id']:
        raise ValueError('Invalid supplemental identity.')
    release = pd.Timestamp(review['result_available_at'])
    fixture.update(event_id=review['event_id'], status='finished', status_code=100,
        is_awarded=True, provider_is_awarded=None, play_status=review['play_status'],
        result_classification=review['classification'], round=review['round'],
        stage_json=json.dumps({'round': review['round']}),
        kickoff_utc=pd.Timestamp(review['kickoff']).timestamp(),
        result_available_at=release, standings_result_available_at=release,
        result_review_json=json.dumps(review, sort_keys=True))
    for i, side in enumerate(('home', 'away')):
        tid = review[side + '_id']
        attributes = ('name', 'slug', 'got_promoted', 'got_demoted', 'season_entry')
        donors = {tuple(r[s + '_' + a] for a in attributes)
                  for r in rows for s in ('home', 'away') if r[s + '_id'] == tid}
        if len(donors) != 1:
            raise ValueError('Missing/conflicting team identity or movement evidence.')
        fixture[side + '_id'] = tid
        fixture.update({side + '_' + a: v for a, v in zip(attributes, donors.pop())})
        for field in ('current', 'display', 'awarded'):
            fixture[side + '_score_' + field] = float(review['awarded_score'][i])
        fixture[side + '_score_played'] = float(review['played_score'][i])
        fixture[side + '_standings_goals'] = float(review['awarded_score'][i])
    result = append_fixture_tables(matches, statistics, nested, fixture, [])
    if not original.equals(result[0].slice(0, len(original)).select(original.column_names), check_metadata=True):
        raise AssertionError('Existing match records changed.')
    return result


def repair(root, *, apply=False, evidence=EVIDENCE):
    root = Path(root).resolve()
    evidence = Path(evidence)
    if apply:
        with _publication_lock(root):
            return _repair(root, apply=True, evidence=evidence)
    return _repair(root, apply=False, evidence=evidence)


def _repair(root, *, apply, evidence):
    review = json.loads(evidence.read_text(encoding='utf-8'))
    journal = _target(root, review['season'] + '.standings-pending.json')
    if journal.exists():
        if not apply:
            raise RuntimeError('Interrupted standings publication: rerun with --apply to recover before reading manifests.')
        return _finish_publication(root, journal, digest(evidence))
    pub_path = root / (review['season'] + '.manifest.json')
    pub = json.loads(pub_path.read_text(encoding='utf-8'))
    manifest_path = root / pub['manifest']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    native = manifest_path.parent / manifest['tables']['matches']['file']
    nested_path = root / pub['season_file']
    stat_path = manifest_path.parent / manifest['tables']['statistics']['file']
    for path, info in ((native, manifest['tables']['matches']), (nested_path, pub),
                       (stat_path, manifest['tables']['statistics'])):
        if digest(path) != info['sha256']:
            raise ValueError(f'Source hash mismatch: {path}')
    matches, statistics, nested = [pq.ParquetFile(p).read() for p in (native, stat_path, nested_path)]
    if review['event_id'] in matches['event_id'].to_pylist():
        record = next(r for r in matches.to_pylist() if r['event_id'] == review['event_id'])
        if json.loads(record['result_review_json']) != review:
            raise ValueError('Existing repair evidence differs.')
        if not matches.equals(nested.select(matches.column_names), check_metadata=False):
            raise ValueError('Published surfaces disagree.')
        return {'status': 'already_applied', 'matches': len(matches)}
    changed, _, new_nested = repaired_tables(matches, statistics, nested, review)
    played_coverage = audit_fixture_coverage(changed.to_pylist(),
        team_ids=set(changed['home_id'].to_pylist()), meetings_per_pair=2,
        required=[dict(home_id=review['home_id'], away_id=review['away_id'], round=27,
                       date_utc='2022-03-18', score=[0, 2])])
    rows = changed.to_pylist()
    pairs = Counter(frozenset((r['home_id'], r['away_id'])) for r in rows)
    directed = Counter((r['home_id'], r['away_id']) for r in rows)
    if len(pairs) != 153 or set(pairs.values()) != {2} or set(directed.values()) != {1}:
        raise ValueError('Reviewed result schedule is not a complete double round robin.')
    coverage = dict(status='balanced_with_reviewed_award', expected_matches=306,
        saved_matches=len(rows), ordinary_matches=305, reviewed_awarded_matches=1,
        scope='Complete result schedule, including the separately reviewed abandoned/awarded fixture',
        ordinary_played_audit=played_coverage)
    receipt = dict(status='applied' if apply else 'validated_dry_run', original_matches=len(matches),
        matches=len(changed), original_rows_preserved=True, evidence_sha256=digest(evidence),
        coverage=coverage, files=[])
    if not apply:
        return receipt
    pending = []
    for path, table, info in ((native, changed, manifest['tables']['matches']), (nested_path, new_nested, pub)):
        temporary = path.with_suffix(path.suffix + '.standings-tmp')
        pq.write_table(table, temporary, compression='zstd', use_compliant_nested_type=False)
        if not pq.ParquetFile(temporary).read().equals(table, check_metadata=True):
            raise ValueError('Repaired Parquet failed exact round trip.')
        receipt['files'].append(dict(path=path.relative_to(root).as_posix(), before=digest(path), after=digest(temporary)))
        info.update(bytes=temporary.stat().st_size, sha256=digest(temporary))
        info['rows' if path == native else 'matches'] = len(table)
        pending.append((temporary, path))
    for metadata in (manifest, pub):
        metadata['fixture_coverage'] = coverage
        metadata.setdefault('result_reviews', {})[str(review['event_id'])] = review
        metadata['complete_scope'] = 'Collector cohort plus reviewed supplemental fixture; original capture/component coverage is retained separately.'
    manifest['match_awarded_contract']['is_awarded']['meaning'] = 'Effective awarded flag including reviewed supplements; provider_is_awarded preserves supplied source flags.'
    manifest['match_time_contract']['result_available_at'] = dict(
        meaning=review['availability_policy'], timezone='UTC', unit='Arrow timestamp')
    for path, content in ((manifest_path, manifest), (pub_path, pub)):
        temporary = path.with_suffix(path.suffix + '.standings-tmp')
        newline = '\r\n' if b'\r\n' in path.read_bytes() else '\n'
        temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline=newline)
        receipt['files'].append(dict(path=path.relative_to(root).as_posix(), before=digest(path), after=digest(temporary)))
        pending.append((temporary, path))
    return _publish(root, journal, pending, receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/xDiyo_data')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--receipt')
    args = parser.parse_args()
    result = repair(args.data_root, apply=args.apply)
    text = json.dumps(result, indent=2)+'\n'
    if args.receipt:
        Path(args.receipt).write_text(text, encoding='utf-8')
    print(text)
