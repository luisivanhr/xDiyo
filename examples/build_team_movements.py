"""Rebuild reviewed additive movement tables without touching season exports.

Run from the repository root with PYTHONPATH=.;src. New/changed match populations
need an updated roster review; this script refuses to infer from a stale review.
"""
import hashlib
import json
from pathlib import Path

from xdiyo_analytics.data import derive_movements, load_season, save_movement_enrichment
from xdiyo_analytics.data._source import _open_source


def build(root=Path('data/xDiyo_data'), evidence_root=Path('docs/analytics/data')):
    review = json.loads((evidence_root / 'team_movement_roster_review.json').read_text(encoding='utf-8'))
    boundaries = json.loads((evidence_root / 'team_movement_boundary_evidence.json').read_text(encoding='utf-8'))
    if isinstance(boundaries, dict):
        boundaries = boundaries['seasons']
    rosters, protected = [], {}
    for entry in review['seasons']:
        stem = entry['stem']
        source = _open_source(root, stem)
        original_hash = source.manifest['tables']['matches']['sha256']
        if source.manifest.get('movement_enrichment', {}).get('materialized'):
            original_hash = Path(source.manifest['movement_enrichment']['evidence']['path']).parent.name
        if original_hash != entry['matches_sha256']:
            raise ValueError(f'Review membership again for changed source: {stem}')
        matches = load_season(root, stem, include_awarded=True, verify_hashes=True).matches
        teams = {str(int(team)): name for side in ('home', 'away') for team, name in
                 matches[[f'{side}_id', f'{side}_name']].itertuples(index=False, name=None)}
        if sorted(map(int, teams)) != entry['team_ids']:
            raise ValueError(f'Roster changed: {stem}')
        scope = source.manifest['scope']
        rosters.append(dict(stem=stem, league=scope['league_name'], year=scope['season_start'],
                            competition_id=scope['league_id'], season_id=scope['season_id'], teams=teams))
        for path in (root / f'{stem}.manifest.json', source.path, source.table_path('matches')):
            protected[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = derive_movements(rosters, review['competitions'],
                              reviewed_seasons=[r['stem'] for r in review['seasons']], boundaries=boundaries)
    outputs = {}
    for r in rosters:
        source = _open_source(root, r['stem'])
        outputs[r['stem']] = (source.manifest['tables']['team_seasons']
            if source.manifest.get('movement_enrichment', {}).get('materialized')
            else save_movement_enrichment(root, r['stem'], frame))
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in protected.items())
    audit = dict(team_seasons=len(frame), seasons=len(rosters),
                 movement_counts=frame.movement.value_counts().to_dict(),
                 original_files_verified_unchanged=len(protected),
                 inferred_promotions=int(frame.evidence.str.startswith('Inferred entry').sum()),
                 outputs=outputs)
    (root / '_team_seasons' / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print({k: v for k, v in audit.items() if k != 'outputs'})
    return frame


if __name__ == '__main__':
    build()
