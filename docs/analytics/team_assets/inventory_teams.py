"""Rebuild name-only badge inventory from the three experiment-season match exports."""
import json
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data/xDiyo_data'
HERE = Path(__file__).resolve().parent
catalog = {}
exports = []
for season in ('22_23', '23_24', '24_25'):
    for publication in sorted(DATA.glob(f'*_{season}.manifest.json')):
        manifest_path = DATA / json.loads(publication.read_text(encoding='utf-8'))['manifest']
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        matches = pq.read_table(manifest_path.parent / manifest['tables']['matches']['file']).to_pandas()
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
