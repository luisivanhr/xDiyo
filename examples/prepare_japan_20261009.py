"""Validate and materialize reviewed Japanese season-entry flags (no fitting).

The 2026 transition and 2026/27 league share the entry changes from 2025.
They remain separate publications; neither is used as the other's predecessor.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from examples.backfill_statistics import digest, dump
from xdiyo_analytics.data import derive_movements, load_season, save_movement_enrichment
from xdiyo_analytics.data._source import _open_source
from xdiyo_analytics.data.movements import assert_movement_parity, validate_movement_flags
from xdiyo_analytics.data.publish_movements import materialize_movement_flags


REVIEW = Path('docs/analytics/data/japan_movement_review_20261009.json')


def derive_japan(rosters, boundaries):
    regular = [r for r in rosters if r['league'] != 'J1_Transition']
    flags = derive_movements(regular, {'J1': {'system': 'Japan', 'tier': 1},
                                     'J2': {'system': 'Japan', 'tier': 2}},
                             reviewed_seasons=[r['stem'] for r in regular], boundaries=boundaries)
    for transition in (r for r in rosters if r['league'] == 'J1_Transition'):
        reference = next(r for r in regular if r['league'] == 'J1' and r['year'] == transition['year'])
        if (set(reference['teams']) != set(transition['teams']) or
                reference['competition_id'] != transition['competition_id']):
            raise ValueError('Transition and regular J1 membership differ; review required')
        selected = flags.loc[flags.season_stem.eq(reference['stem'])].copy()
        selected['season_id'] = pd.array([transition['season_id']] * len(selected), dtype='UInt64')
        selected['season_stem'] = transition['stem']
        selected['evidence'] = (selected['evidence'] + '; 2026 special/2026-27 share 2025 entry changes: '
                               'https://www.jleague.jp/news/article/32852/; '
                               'no movement from special tournament: https://www.jleague.jp/news/article/31576/')
        flags = pd.concat([flags, selected], ignore_index=True)
    validate_movement_flags(flags)
    if flags.movement.eq('unknown').any():
        raise ValueError('Unresolved Japanese season entries')
    return flags


def inspect(root, review):
    rosters, protected, summary = [], {}, {}
    for entry in review['seasons']:
        stem = entry['stem']
        src = _open_source(root, stem)
        pub_path = root / f'{stem}.manifest.json'
        pub = json.loads(pub_path.read_bytes())
        scope = src.manifest['scope']
        if scope['league_id'] not in (196, 402):
            raise ValueError('Japan review includes another competition')
        matches = load_season(root, stem, include_awarded=True, verify_hashes=True).matches
        teams = {str(int(t)): n for side in ('home', 'away') for t, n in
                 matches[[side + '_id', side + '_name']].itertuples(index=False, name=None)}
        original = src.manifest.get('movement_enrichment', {}).get(
            'original_matches_sha256', src.manifest['tables']['matches']['sha256'])
        if original != entry['matches_sha256'] or sorted(map(int, teams)) != entry['team_ids']:
            raise ValueError(f'Reviewed population changed: {stem}')
        cohort = pub['reconciliation']['archive_cohort']
        if (not cohort['schedule_verified'] or cohort['validation_issues'] or
                cohort['selected_matches'] != len(matches) or len(teams) != entry['member_count']):
            raise ValueError(f'Collection/roster validation failed: {stem}')
        rosters.append(dict(stem=stem, league=scope['league_name'], year=scope['season_start'],
            competition_id=scope['league_id'], season_id=scope['season_id'], teams=teams))
        files = {name: (src.table_path(name), info) for name, info in src.manifest['tables'].items()}
        files['season_export'] = (root / pub['season_file'], dict(rows=len(matches), **pub))
        checked = {}
        for name, (path, info) in files.items():
            parquet = pq.ParquetFile(path)
            codecs = {parquet.metadata.row_group(i).column(j).compression
                      for i in range(parquet.num_row_groups)
                      for j in range(parquet.metadata.row_group(i).num_columns)}
            if (digest(path) != info['sha256'] or path.stat().st_size != info['bytes'] or
                    parquet.metadata.num_rows != info['rows'] or codecs - {'ZSTD'}):
                raise ValueError(f'Fingerprint, rows or ZSTD validation failed: {path}')
            protected[path] = info['sha256']
            checked[name] = dict(rows=info['rows'], sha256=info['sha256'], codecs=sorted(codecs))
        protected[src.path] = digest(src.path)
        protected[pub_path] = digest(pub_path)
        summary[stem] = dict(matches=len(matches), teams=len(teams), snapshot=cohort['snapshot'], files=checked)
    return rosters, protected, summary


def prepare(root=Path('data/xDiyo_data'), *, apply=False):
    root = Path(root).resolve()
    review = json.loads(REVIEW.read_bytes())
    rosters, protected, summary = inspect(root, review)
    flags = derive_japan(rosters, review['boundaries'])
    totals = dict(publications=len(rosters), matches=sum(s['matches'] for s in summary.values()),
                  team_seasons=len(flags), movements=flags.movement.value_counts().to_dict())
    if not apply:
        print(json.dumps(totals, indent=2))
        return totals
    if any(digest(p) != h for p, h in protected.items()):
        raise ValueError('Publication changed during preflight')
    for roster in rosters:
        stem = roster['stem']
        src = _open_source(root, stem)
        paths = [src.table_path('matches'), root / f'{stem}.parquet']
        before = [pq.ParquetFile(p).read() for p in paths]
        other = {n: digest(src.table_path(n)) for n in src.manifest['tables'] if n not in ('matches', 'team_seasons')}
        if not src.manifest.get('movement_enrichment', {}).get('materialized'):
            save_movement_enrichment(root, stem, flags)
        materialize_movement_flags(root, stem, reviewed_flags=flags)
        for path, original in zip(paths, before):
            result = pq.ParquetFile(path).read().select(original.column_names)
            if not original.equals(result, check_metadata=True):
                raise ValueError(f'Original observations changed: {path}')
        updated = _open_source(root, stem)
        if any(digest(updated.table_path(n)) != h for n, h in other.items()):
            raise ValueError(f'Unrelated table changed: {stem}')
        stored = load_season(root, stem, tables='team_seasons', verify_hashes=True)['team_seasons']
        assert_movement_parity(flags.loc[flags.season_stem.eq(stem)], stored, season=stem)
        print(f'{stem}: entry flags verified', flush=True)
    _, _, final = inspect(root, review)
    dump('docs/analytics/data/japan_movement_publication_20261009.json',
         dict(**totals, original_observations_preserved=True, seasons=final))
    return totals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--data-root', type=Path, default=Path('data/xDiyo_data'))
    args = parser.parse_args()
    prepare(args.data_root, apply=args.apply)
