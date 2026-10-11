"""Refresh all published team identities while preserving acquired badge metadata."""
import json
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data/xDiyo_data'
HERE = Path(__file__).resolve().parent
catalog_path = HERE / 'catalog.json'
catalog = json.loads(catalog_path.read_text(encoding='utf-8')) if catalog_path.exists() else {}
exports = []
for publication in sorted(DATA.glob('*.manifest.json')):
    season = '_'.join(publication.name.removesuffix('.manifest.json').split('_')[-2:])
    manifest_path = DATA / json.loads(publication.read_text(encoding='utf-8'))['manifest']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    matches = pq.ParquetFile(manifest_path.parent / manifest['tables']['matches']['file']).read(columns=['home_id', 'home_name', 'away_id', 'away_name']).to_pandas()
    scope = manifest['scope']
    exports.append({'publication': publication.name, 'matches': len(matches), 'league': scope['league_name']})
    for side in ('home', 'away'):
        for team_id, name in matches[[f'{side}_id', f'{side}_name']].drop_duplicates().itertuples(index=False, name=None):
            if team_id is None or name is None:
                continue
            key = str(int(team_id))
            item = catalog.setdefault(key, {'name': name, 'aliases': [], 'leagues': [], 'seasons': []})
            item['name'] = name
            for field, value in [('aliases', name), ('leagues', scope['league_name']), ('seasons', season)]:
                if value not in item[field]:
                    item[field].append(value)
catalog = dict(sorted(catalog.items(), key=lambda item: int(item[0])))
(HERE / 'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
(HERE / 'inventory.json').write_text(json.dumps({'exports': exports, 'team_count': len(catalog)}, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'exports': len(exports), 'teams': len(catalog), 'leagues': sorted({x['league'] for x in exports})}))
