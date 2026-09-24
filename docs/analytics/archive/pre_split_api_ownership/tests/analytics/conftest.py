"""Small synthetic exports for public loader/discovery behavior."""

import base64
import hashlib
import json
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

STEM = "Premier_League_24_25"
EVENT, HOME, AWAY = 2**53 + 1, 2**53 + 3, 2**53 + 5


class Export:
    def __init__(self, root):
        self.root = root
        self.version = root / "_tables/competition=17/season=61627/versions/v1"
        self.manifest_path = self.version / "manifest.json"
        self.publication_path = root / (STEM + ".manifest.json")

    def manifest(self):
        return json.loads(self.manifest_path.read_bytes())

    def change_manifest(self, change):
        manifest = self.manifest()
        change(manifest)
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def table(self, name):
        with pq.ParquetFile(self.version / f"{name}.parquet") as parquet:
            return parquet.read()

    def replace(self, name, table):
        path = self.version / f"{name}.parquet"
        pq.write_table(table, path, compression="NONE")
        payload = path.read_bytes()
        self.change_manifest(lambda m: m['tables'].__setitem__(name, {
            'file': path.name, 'rows': table.num_rows, 'bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest(),
        }))

    def legacy_record(self, path):
        publication = self.publication_path.read_bytes()
        record = {'schema_version': 1, 'season_stem': STEM,
                  'publication_base64': base64.b64encode(publication).decode(),
                  'publication_sha256': hashlib.sha256(publication).hexdigest(),
                  'manifest_sha256': hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()}
        path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')

    def advance(self):
        newer = self.version.parent / 'v2'
        shutil.copytree(self.version, newer)
        manifest = self.manifest()
        manifest['version'] = 'v2'
        (newer / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        publication = json.loads(self.publication_path.read_bytes())
        publication['manifest'] = (newer / 'manifest.json').relative_to(self.root).as_posix()
        self.publication_path.write_text(json.dumps(publication), encoding='utf-8')

    def snapshot(self):
        return {p.relative_to(self.root).as_posix(): p.read_bytes()
                for p in self.root.rglob('*') if p.is_file()}


@pytest.fixture
def export(tmp_path):
    result = Export(tmp_path / 'exports')
    result.version.mkdir(parents=True)
    manifest = {'schema_version': '2', 'parser_version': '2', 'version': 'v1',
                'scope': {'league_id': 17, 'league_name': 'Premier_League', 'season_id': 61627,
                          'season_start': 2024, 'season_end': 2025}, 'tables': {}}
    result.manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    result.replace('matches', pa.table({
        'event_id': [EVENT, 7], 'home_id': [HOME, 42], 'away_id': [AWAY, 50],
        'competition_id': [17, 17], 'season_id': [61627, 61627],
        'kickoff_utc': [1735752600.0, 1736017200.0],
        'optional_flag': pa.array([None, False], type=pa.bool_()),
    }))
    result.replace('statistics', pa.table({
        'event_id': [EVENT, EVENT, EVENT, 7], 'period': ['ALL', 'ALL', 'ALL', '1ST'],
        'group_name': ['Match overview', 'Shots', 'Shots', 'Attack'],
        'key': ['totalShotsOnGoal', 'totalShotsOnGoal', 'totalShotsOnGoal', 'goals'],
        'side': ['home', 'home', 'away', 'home'],
        'team_id': pa.array([HOME, HOME, AWAY, None], type=pa.int64()),
        'value': pa.array([3.0, 3.0, 4.0, None], type=pa.float64()),
        'display': ['3', '3', '4', 'not supplied'],
    }))
    result.replace('pregame', pa.table({'event_id': [EVENT, 7], 'side': ['home', 'home'], 'position': [3, 2]}))
    result.replace('shots', pa.table({
        'event_id': [EVENT, EVENT, EVENT],
        'shot_id': pa.array([2**63 + 9, 2**63 + 10, 2**63 + 11], type=pa.uint64()),
        'xg': pa.array([0.0, None, float('nan')], type=pa.float64(), from_pandas=False),
        'flag': pa.array([False, None, True], type=pa.bool_()),
        'details': pa.array([[{'x': 1}], [], None], type=pa.list_(pa.struct([('x', pa.int64())]))),
    }))
    result.replace('future_context', pa.table({'details': pa.array([], type=pa.list_(pa.int64()))}))
    publication = {'schema_version': '2', 'complete': True, 'matches': 2,
                   'season_file': STEM + '.parquet',
                   'manifest': result.manifest_path.relative_to(result.root).as_posix()}
    result.publication_path.write_text(json.dumps(publication), encoding='utf-8')
    return result
