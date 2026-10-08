"""Apply five reviewed historical result exceptions; default is read-only.

Preserves provider status/scores, kickoff, rounds, movement flags and unrelated
tables. Use --apply to update the native and nested publications with real hashes.
No collection, fitting, calibration or automatic recipe refresh.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


EVIDENCE = Path('docs/analytics/data/historical_result_reviews_20261008.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_table(table, review):
    ids = table['event_id'].to_pylist()
    if ids.count(review['event_id']) != 1:
        raise ValueError('Reviewed event must occur exactly once.')
    row = ids.index(review['event_id'])
    for name in ('home_id', 'away_id'):
        if table[name][row].as_py() != review[name]:
            raise ValueError('Reviewed fixture teams changed.')
    for side, expected in zip(('home', 'away'), review['expected_score']):
        if table[side + '_score_current'][row].as_py() != expected:
            raise ValueError('Reviewed provider score changed.')
    if 'provider_is_awarded' not in table.column_names:
        table = table.append_column('provider_is_awarded', table['is_awarded'])
    updates = {'is_awarded': (pa.bool_(), review['is_awarded']),
               'play_status': (pa.string(), review['play_status']),
               'result_classification': (pa.string(), review['classification']),
               'completion_date': (pa.string(), review['completion_date']),
               'result_available_at': (pa.timestamp('us', tz='UTC'),
                    pd.Timestamp(review['result_available_at']) if review['result_available_at'] else None),
               'result_review_json': (pa.string(), json.dumps(review, sort_keys=True))}
    for i, side in enumerate(('home', 'away')):
        updates[side + '_score_played'] = (pa.float64(), review['played_score'][i])
        updates[side + '_score_awarded'] = (pa.float64(), review['awarded_score'][i])
    for name, (dtype, value) in updates.items():
        if name in table.column_names:
            position = table.column_names.index(name)
            field = table.schema.field(position)
            values = table[name].to_pylist()
            values[row] = value
            table = table.set_column(position, field, pa.array(values, type=field.type))
        else:
            values = [None] * len(table)
            values[row] = value
            table = table.append_column(name, pa.array(values, type=dtype))
    return table


def repair(root=Path('data/xDiyo_data'), evidence=EVIDENCE, *, apply=False):
    root = root.resolve()
    journal = root / '_result_reviews' / 'pending.json'

    def finish(plan):
        for item in plan['files']:
            path = (root / item['path']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Review path escapes dataset.')
            temp = path.with_suffix(path.suffix + '.result-tmp')
            candidate = temp if temp.exists() else path
            if digest(candidate) != item['sha256']:
                raise ValueError(f'Review publication changed: {candidate}')
            if temp.exists():
                temp.replace(path)
        audit = root / '_result_reviews' / '20261008.audit.json'
        receipt = dict(plan)
        if audit.exists():
            previous = json.loads(audit.read_bytes())
            files = {item['path']: item for item in previous['files']}
            for item in plan['files']:
                files[item['path']] = {**item, 'previous_sha256':
                    files.get(item['path'], item)['previous_sha256']}
            receipt['files'] = list(files.values())
        audit.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        journal.unlink()
        return plan

    if journal.exists():
        if not apply:
            raise ValueError('Interrupted result review: resume with --apply.')
        return finish(json.loads(journal.read_bytes()))
    plan = dict(evidence_sha256=digest(evidence), files=[], seasons=[])
    # Validate and prepare every output before any publication is replaced.
    pending = []
    for review in json.loads(evidence.read_bytes())['records']:
        stem = review['season']
        pub_path = root / (stem + '.manifest.json')
        pub = json.loads(pub_path.read_bytes())
        manifest_path = root / pub['manifest']
        manifest = json.loads(manifest_path.read_bytes())
        native = manifest_path.parent / manifest['tables']['matches']['file']
        nested = root / pub['season_file']
        meaning = ('Effective awarded-result flag: provider value with sourced result_reviews overrides; '
                   'provider_is_awarded preserves the raw provider flag.')
        changed = manifest['match_awarded_contract']['is_awarded'].get('meaning') != meaning
        for path, expected in ((native, manifest['tables']['matches']), (nested, pub)):
            if digest(path) != expected['sha256']:
                raise ValueError(f'Publication fingerprint changed: {path}')
            original = pq.ParquetFile(path).read()
            updated = patch_table(original, review)
            if updated.equals(original, check_metadata=True):
                continue
            changed = True
            if not apply:
                continue
            temp = path.with_suffix(path.suffix + '.result-tmp')
            pq.write_table(updated, temp, compression='zstd', use_compliant_nested_type=False)
            if not pq.ParquetFile(temp).read().equals(updated, check_metadata=True):
                raise ValueError('Parquet round trip changed values or schema.')
            pending.append((path, temp))
            expected.update(bytes=temp.stat().st_size, sha256=digest(temp))
        if changed and apply:
            manifest.setdefault('result_reviews', {})[str(review['event_id'])] = review
            contract = manifest['match_awarded_contract']
            if 'provider_is_awarded' not in contract:
                contract['provider_is_awarded'] = dict(contract['is_awarded'])
                contract['provider_is_awarded'].pop('review_override', None)
            contract['is_awarded']['meaning'] = meaning
            manifest['match_awarded_contract']['is_awarded']['review_override'] = (
                'Effective exclusion flag includes sourced result_reviews; provider_is_awarded retains the original flag.')
            manifest['match_time_contract']['result_available_at'] = {
                'meaning': 'Sparse reviewed conservative availability bound; never an exact whistle time unless explicitly evidenced.',
                'timezone': 'UTC', 'unit': 'Arrow timestamp',
                'missing': 'No reviewed bound; caller availability policy applies.'}
            pub.setdefault('result_reviews', {})[str(review['event_id'])] = review
            for path, content in ((manifest_path, manifest), (pub_path, pub)):
                temp = path.with_suffix(path.suffix + '.result-tmp')
                temp.write_text(json.dumps(content, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
                pending.append((path, temp))
        plan['seasons'].append(dict(season=stem, event_id=review['event_id'], changed=changed))
    if apply and pending:
        plan['files'] = [dict(path=path.relative_to(root).as_posix(),
                             previous_sha256=digest(path), sha256=digest(temp)) for path, temp in pending]
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
        return finish(plan)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    print(json.dumps(repair(apply=parser.parse_args().apply), indent=2))
