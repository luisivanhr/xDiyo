"""Read-only inventory of native publications against saved SofaScore responses.

No collector imports, requests, or writes. Outputs evidence, not guessed names.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
from collections import defaultdict

from xdiyo_analytics.data._source import _open_source

ROOT = Path('data/xDiyo_data')
OUT = Path('docs/analytics/native_integration_20261116')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory():
    review = json.loads(Path('docs/analytics/data/library_movement_review_20261116.json').read_bytes())
    keys = {(s['competition_id'], s['season_id']) for s in review['seasons']}
    events, seasons = {}, {}
    rejected = []
    aliases = defaultdict(set)
    for db in sorted(Path('data/website_capture_v3/discovery').glob('*/discovery.sqlite')):
        with sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True) as connection:
            for endpoint, observed, body, sha in connection.execute("SELECT path,observed,body,sha256 FROM responses WHERE path LIKE '%/seasons'"):
                cid = int(endpoint.split('/')[2])
                if hashlib.sha256(body.encode()).hexdigest() != sha:
                    raise ValueError(f'Captured response digest mismatch: {db} {endpoint}')
                for season in json.loads(body).get('seasons', []):
                    key = cid, season['id']
                    if key in keys and observed > seasons.get(key, {}).get('observed_at', -1):
                        seasons[key] = dict(value=season, observed_at=observed, database=str(db), endpoint=endpoint, sha256=sha)
            for cid, sid, payload, observed in connection.execute('SELECT competition,season,payload,observed FROM events'):
                if (cid, sid) not in keys:
                    continue
                event = json.loads(payload)
                tournament = event.get('tournament', {})
                competition = tournament.get('uniqueTournament', {})
                if competition.get('id') != cid or event.get('season', {}).get('id') != sid:
                    rejected.append(dict(database=str(db), event_id=event.get('id'), indexed=[cid,sid],
                                         actual=[competition.get('id'), event.get('season', {}).get('id')]))
                    continue
                key = cid, sid
                evidence = dict(observed_at=observed, database=str(db), event_id=event['id'],
                                sha256=hashlib.sha256(payload.encode()).hexdigest())
                group = events.setdefault(key, dict(competition=None, tournaments={}, teams={}))
                def update(mapping, identity, value):
                    old = mapping.get(identity)
                    if old is None or observed > old['evidence']['observed_at']:
                        mapping[identity] = dict(value=value, evidence=evidence)
                update(group, 'competition', {k: competition.get(k) for k in ('id', 'name', 'slug')})
                update(group['tournaments'], str(tournament['id']), {k: tournament.get(k) for k in ('id', 'name', 'slug')})
                for side in ('homeTeam', 'awayTeam'):
                    team = event[side]
                    aliases[team['id']].add(team['name'])
                    update(group['teams'], str(team['id']), {k: team.get(k) for k in ('id', 'name', 'slug')})
    rows = []
    for s in review['seasons']:
        stem = s['stem']; src = _open_source(ROOT, stem)
        publication = json.loads((ROOT / (stem + '.manifest.json')).read_bytes())
        key = s['competition_id'], s['season_id']
        row = dict(stem=stem, new=s['new'], scope=src.manifest['scope'],
                   publication_sha256=digest(ROOT / (stem + '.manifest.json')),
                   manifest=str(src.path.relative_to(ROOT.resolve())), manifest_sha256=digest(src.path),
                   version=src.manifest['version'], calendar=src.manifest.get('season_calendar'),
                   tables=src.manifest['tables'], sofa_season=seasons.get(key), sofa=events.get(key),
                   source_teams=s['teams'], reconciliation=publication.get('reconciliation', {}).get('archive_cohort', {}))
        row['exceptions'] = []
        if not row['sofa_season']: row['exceptions'].append('No captured season-list evidence')
        if not row['sofa']: row['exceptions'].append('No captured competition/team evidence')
        elif set(s['teams']) - set(row['sofa']['teams']): row['exceptions'].append('Some teams lack captured identity evidence')
        rows.append(row)
    result = dict(schema_version=1, base_revision='fa2014db73336d7ec1a3d3432749ebe4f37e62ee',
                  publications=rows, rejected_capture_identities=rejected,
                  team_aliases={str(k): sorted(v) for k, v in aliases.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'inventory.json'
    if path.exists():
        raise FileExistsError('Inventory is immutable; retain the original baseline')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(publications=len(rows), exceptions=[dict(stem=r['stem'], issues=r['exceptions']) for r in rows if r['exceptions']]), ensure_ascii=True))


if __name__ == '__main__':
    inventory()
