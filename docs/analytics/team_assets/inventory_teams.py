"""Refresh all published team identities without discarding acquired badge metadata."""
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data/xDiyo_data'
HERE = Path(__file__).resolve().parent


def inventory(data, previous):
    catalog = {key: dict(value) for key, value in previous.items()}
    exports, identities = [], {}
    for publication in sorted(Path(data).glob('*.manifest.json')):
        manifest_path = Path(data) / json.loads(publication.read_text(encoding='utf-8'))['manifest']
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        columns = ['home_id', 'home_name', 'away_id', 'away_name']
        matches = pq.ParquetFile(manifest_path.parent / manifest['tables']['matches']['file']).read(columns=columns).to_pandas()
        scope = manifest['scope']
        season = scope.get('season_name', scope.get('season'))
        if season is None:
            season = publication.name.removesuffix('.manifest.json').rsplit('_', 2)[-2:]
            season = '_'.join(season)
        exports.append(dict(publication=publication.name, matches=len(matches), league=scope['league_name'], season=season))
        for side in ('home', 'away'):
            for team_id, name in matches[[f'{side}_id', f'{side}_name']].drop_duplicates().itertuples(index=False, name=None):
                if pd.isna(team_id) or pd.isna(name):
                    continue
                key = str(int(team_id))
                item = identities.setdefault(key, dict(name=str(name), aliases=[], leagues=[], seasons=[]))
                item['name'] = str(name)
                for field, value in [('aliases', str(name)), ('leagues', scope['league_name']), ('seasons', season)]:
                    if value not in item[field]:
                        item[field].append(value)
    for key, identity in identities.items():
        catalog.setdefault(key, {}).update(identity)
    return dict(sorted(catalog.items(), key=lambda item: int(item[0]))), dict(exports=exports, team_count=len(identities), team_ids=sorted(identities, key=int))


if __name__ == '__main__':
    catalog_path = HERE / 'catalog.json'
    previous = json.loads(catalog_path.read_text(encoding='utf-8')) if catalog_path.exists() else {}
    catalog, summary = inventory(DATA, previous)
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (HERE/'inventory.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(exports=len(summary['exports']), teams=summary['team_count'])))
