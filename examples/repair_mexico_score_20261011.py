"""Correct the reviewed Tigres result in both publications; default is read-only."""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from examples.backfill_statistics import digest, dump, finish, write_table


def repair(*, apply=False):
    root = Path('data/xDiyo_data').resolve()
    evidence = Path('docs/analytics/data/mexico_score_review_20261010.json')
    review = json.loads(evidence.read_bytes())
    pub_path = root / 'Liga_MX_Apertura_23_24.manifest.json'
    pub = json.loads(pub_path.read_bytes())
    manifest_path = root / pub['manifest']
    manifest = json.loads(manifest_path.read_bytes())
    journal = root / '_result_reviews/mexico_20261011.pending.json'
    if journal.exists():
        if not apply:
            raise ValueError('Interrupted correction; resume with --apply.')
        finish(root, journal)
        return
    pending = []
    for path, entry in ((manifest_path.parent / manifest['tables']['matches']['file'],
                         manifest['tables']['matches']), (root / pub['season_file'], pub)):
        if digest(path) != entry['sha256']:
            raise ValueError(f'Publication fingerprint changed: {path}')
        original = pq.ParquetFile(path).read()
        ids = original['event_id'].to_pylist()
        if ids.count(11341818) != 1:
            raise ValueError('Expected exactly one reviewed fixture.')
        row = ids.index(11341818)
        for name, expected in dict(home_id=1940, away_id=1951, kickoff_utc=1693703100,
                                   home_score_period1=2, home_score_period2=3,
                                   away_score_current=0, away_score_display=0).items():
            if original[name][row].as_py() != expected:
                raise ValueError(f'Reviewed fixture changed: {name}')
        updated = original
        for name in ('home_score_current', 'home_score_display'):
            if original[name][row].as_py() not in (4, 5):
                raise ValueError(f'Unexpected score: {name}')
            values = original[name].to_pylist()
            values[row] = 5
            field = original.schema.field(name)
            updated = updated.set_column(original.column_names.index(name), field,
                                         pa.array(values, type=field.type))
        # Prove all other cells and schema metadata remain identical.
        restored = updated
        for name in ('home_score_current', 'home_score_display'):
            restored = restored.set_column(original.column_names.index(name),
                                            original.schema.field(name), original[name])
        assert restored.equals(original, check_metadata=True)
        if not apply or updated.equals(original, check_metadata=True):
            continue
        temporary = path.with_suffix(path.suffix + '.statistics-tmp')
        info = write_table(updated, path, temporary)
        entry.update(bytes=info['bytes'], sha256=info['sha256'])
        pending.append((path, temporary))
    if pending:
        for path, content in ((manifest_path, manifest), (pub_path, pub)):
            content.setdefault('result_reviews', {})['11341818'] = review
            temporary = path.with_suffix(path.suffix + '.statistics-tmp')
            dump(temporary, content)
            pending.append((path, temporary))
        plan = dict(evidence_sha256=digest(evidence), event_id=11341818,
                    before={p.relative_to(root).as_posix(): digest(p) for p, _ in pending},
                    files=[dict(path=p.relative_to(root).as_posix(), sha256=digest(t))
                           for p, t in pending])
        dump(journal, plan)
        finish(root, journal)
    print('Reviewed score 5-0; ' + ('publication verified.' if apply else 'read-only validation passed.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    repair(apply=parser.parse_args().apply)
