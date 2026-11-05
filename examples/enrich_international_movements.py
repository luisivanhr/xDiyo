"""Publish reviewed season-entry evidence without changing frozen publications.

PYTHONPATH=src python examples/enrich_international_movements.py --help
Local licensed inputs remain local; the receipt contains hashes and counts.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from xdiyo_analytics.data import load_season
from xdiyo_analytics.data.movements import derive_movements, assert_movement_parity
from xdiyo_analytics.data.catalogue import load_snapshot, enrich_movements
from xdiyo_analytics.data.catalogue.movements import KEYS, MOVEMENT_COLUMNS
from xdiyo_analytics.data.catalogue.storage import file_hash

# Complete promotion lists for entry into each top-flight season. Reviewed
# against the Premier League's table (not inferred from partial imports).
SOURCE = 'https://www.premierleague.com/en/news/412265'
PROMOTED = {
    2000: [47, 17, 32], 2001: [43, 46, 5], 2002: [17, 8, 9],
    2003: [2, 31, 3], 2004: [263, 8, 7], 2005: [41, 49, 37],
    2006: [28, 15, 24], 2007: [41, 9, 27], 2008: [8, 29, 96],
    2009: [3, 9, 6],
}


def run(catalogue, version, native_root, receipt):
    name = 'international_dataset'
    tables, manifest, registry = load_snapshot(catalogue, name, version)
    matches = tables['matches']
    evidence, rosters, boundaries, sources = [], [], [], []
    for (competition, season, year), group in matches.groupby(['competition_id', 'season_id', 'season_year']):
        year = int(str(year)[:4])
        teams = {str(int(row[side+'_id'])): row[side+'_name'] for row in group.to_dict('records')
                 for side in ('home', 'away')}
        if year < 2010:
            if competition != 17 or year not in PROMOTED:
                raise ValueError(f'No reviewed boundary evidence: {competition}, {year}')
            stem = f'Premier_League_{year}_{year+1}'
            rosters.append(dict(league='Premier_League', year=year, stem=stem,
                                competition_id=int(competition), season_id=int(season), teams=teams))
            boundaries.append(dict(stem=stem, promoted_ids=[t for t in PROMOTED[year] if str(t) in teams],
                                   sources=[SOURCE], complete_review=True))
        else:
            league = {17: 'Premier_League', 18: 'Championship'}[int(competition)]
            stem = f'{league}_{year%100:02}_{(year+1)%100:02}'
            native = load_season(native_root, stem, tables=['team_seasons'], verify_hashes=True, include_awarded=True)
            flags = native['team_seasons']
            if not flags.competition_id.eq(competition).all() or not flags.season_id.eq(season).all():
                raise ValueError(f'Canonical/native scope mismatch: {stem}')
            if not set(teams) <= set(flags.team_id.astype(str)):
                raise ValueError(f'Native movement evidence is missing imported teams: {stem}')
            selected = flags.loc[flags.team_id.astype(str).isin(teams), list(MOVEMENT_COLUMNS)].copy()
            evidence.append(selected)
            source_manifest=Path(native.provenance['manifest_path'])
            source_info=json.loads(source_manifest.read_text(encoding='utf-8'))['tables']['team_seasons']
            sources.append(dict(stem=stem, version=native.provenance['version'],
                                manifest_sha256=file_hash(source_manifest),
                                table_sha256=source_info['sha256']))
    if rosters:
        evidence.append(derive_movements(rosters, {'Premier_League': dict(system='England', tier=1)}, boundaries=boundaries))
    flags = pd.concat(evidence, ignore_index=True)
    for col in KEYS + ['previous_competition_id', 'previous_season_id']:
        flags[col] = pd.array(flags[col], dtype='UInt64')
    for col in ('got_promoted', 'got_demoted'):
        flags[col] = flags[col].astype('boolean')
    if flags.movement.eq('unknown').any() or flags[['got_promoted','got_demoted']].isna().any().any():
        raise ValueError('Incomplete movement review; refuse full-coverage publication')
    new_version = enrich_movements(catalogue, name, version, flags,
                                  evidence_checksums=[s['table_sha256'] for s in sources])
    published, new_manifest, _ = load_snapshot(catalogue, name, new_version)
    assert_movement_parity(flags, published['team_seasons'])
    for table, original in tables.items():
        if table=='team_seasons':continue
        pd.testing.assert_frame_equal(original, published[table][original.columns], check_dtype=True)
    import pyarrow.parquet as pq
    codecs = {kind: sorted({pq.ParquetFile(Path(catalogue)/name/new_version/info['file']).metadata.row_group(g).column(c).compression
                           for g in range(pq.ParquetFile(Path(catalogue)/name/new_version/info['file']).metadata.num_row_groups)
                           for c in range(pq.ParquetFile(Path(catalogue)/name/new_version/info['file']).metadata.num_columns)})
              for kind, info in new_manifest['tables'].items()}
    result = dict(dataset=name, previous_version=version, version=new_version, matches=len(matches),
                  team_seasons=len(flags), movements=flags.movement.value_counts().to_dict(),
                  league_seasons=len(matches[['competition_id','season_id']].drop_duplicates()),
                  missing_flags=int(flags[['got_promoted','got_demoted']].isna().sum().sum()),
                  original_values_unchanged=True, compression=codecs, boundary_source=SOURCE,
                  boundary_promoted_ids=PROMOTED, native_sources=sources, script_sha256=file_hash(__file__),
                  tables=new_manifest['tables'])
    Path(receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(receipt).write_text(json.dumps(result, indent=2, default=str)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('native_sources','tables','boundary_promoted_ids')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('catalogue', 'version', 'native-root', 'receipt'):
        parser.add_argument('--'+arg, required=True)
    args = parser.parse_args()
    run(args.catalogue, args.version, args.native_root, args.receipt)
