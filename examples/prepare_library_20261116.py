"""Prepare the reviewed native expansion, preserving observations and source flags.

No collector calls, model fitting, synthetic fixtures or named-catalogue rebuild.
Run from the repository root with --apply only after the dry run succeeds.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from xdiyo_analytics.data import derive_movements, load_season, save_movement_enrichment
from xdiyo_analytics.data._source import _open_source
from xdiyo_analytics.data.movements import assert_movement_parity, validate_movement_flags
from xdiyo_analytics.data.publish_movements import materialize_movement_flags
from xdiyo_analytics.data.season_calendar import publish_calendar

REVIEW = Path('docs/analytics/data/library_movement_review_20261116.json')
RECEIPT = Path('docs/analytics/data/library_preparation_20261116.json')


def annual_league(league):
    return league.removesuffix('_Apertura').removesuffix('_Clausura')


def derive_annual(rosters, review):
    """Annual entry flags shared by phases; predecessor IDs remain native.

    Union membership permits a documented mid-year entrant (Colombia 2021).
    It does not create rows for that team in the earlier tournament. No history
    crosses a missing year or uses the closing phase as the next season.
    """
    groups = defaultdict(list)
    for r in rosters:
        groups[annual_league(r['league']), r['year']].append(r)
    annual, competitions, canonical = [], {}, {}
    for (league, year), members in sorted(groups.items()):
        members.sort(key=lambda r: ('Clausura' in r['league'], r['stem']))
        first = members[0]
        if len(members) > 1 and any(annual_league(r['league']) == r['league'] for r in members):
            raise ValueError(f'Ambiguous annual publications: {league} {year}')
        canonical.setdefault(league, first['competition_id'])
        competitions[league] = dict(system=first['country'], tier=first['tier'])
        teams = {t: n for r in members for t, n in r['teams'].items()}
        annual.append(dict(first, league=league, competition_id=canonical[league], teams=teams))
    boundaries = []
    for boundary in review['boundaries']:
        current = next(r for r in annual if (r['league'], r['year']) == (boundary['league'], boundary['year']))
        boundaries.append(dict(boundary, stem=current['stem']))
    for league, url in review['closed_leagues'].items():
        for current in (r for r in annual if r['league'] == league):
            prior = next((r for r in annual if r['league'] == league and r['year'] == current['year'] - 1), None)
            if prior is not None:
                boundaries.append(dict(stem=current['stem'], complete_review=True,
                    other_entry_ids=sorted(set(current['teams']) - set(prior['teams'])), sources=[url]))
    derived = derive_movements(annual, competitions,
        reviewed_seasons=[r['stem'] for r in annual], boundaries=boundaries)
    parts = []
    for (league, year), members in sorted(groups.items()):
        first = next(r for r in annual if (r['league'], r['year']) == (league, year))
        for current in members:
            if not current.get('new', False):
                continue
            part = derived.loc[derived.season_stem.eq(first['stem']) &
                               derived.team_id.isin([int(t) for t in current['teams']])].copy()
            part['competition_id'] = pd.array([current['competition_id']] * len(part), dtype='UInt64')
            part['season_id'] = pd.array([current['season_id']] * len(part), dtype='UInt64')
            part['season_stem'] = current['stem']
            part['team_name'] = [current['teams'][str(t)] for t in part.team_id]
            for i, row in part.iterrows():
                if row.movement == 'retained':
                    previous = [r for r in groups.get((league, year - 1), [])
                        if r['competition_id'] == current['competition_id'] and str(row.team_id) in r['teams']]
                    # Prefer the same tournament; for a combined publication use
                    # the most recent source phase, never an invented season ID.
                    same = [r for r in previous if r['league'] == current['league']]
                    prior = (same or previous)[-1] if previous else None
                    part.loc[i, 'previous_competition_id'] = prior['competition_id'] if prior else pd.NA
                    part.loc[i, 'previous_season_id'] = prior['season_id'] if prior else pd.NA
            parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    validate_movement_flags(result)
    if result.movement.eq('unknown').any():
        raise ValueError('Unresolved entries: ' + str(result.loc[result.movement.eq('unknown'), ['season_stem', 'team_id']].to_dict('records')))
    return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect(root, review):
    """Bind the review to exact source observations, rosters and native IDs."""
    for r in review['seasons']:
        src = _open_source(root, r['stem'])
        original = src.manifest.get('movement_enrichment', {}).get('original_matches_sha256',
            src.manifest['tables']['matches']['sha256'])
        # Existing enrichments were already materialized when the roster review
        # was captured. New ones bind to their pre-enrichment observation hash.
        actual = original if r['new'] else src.manifest['tables']['matches']['sha256']
        if actual != r['hash']:
            raise ValueError(f"Reviewed observation hash changed: {r['stem']}")
        loaded = load_season(root, r['stem'], verify_hashes=True, include_awarded=True)
        m = loaded.matches
        teams = {str(int(t)): n for side in ('home', 'away') for t, n in
                 m[[side + '_id', side + '_name']].itertuples(index=False, name=None)}
        if teams != r['teams'] or len(m) != r['matches']:
            raise ValueError(f"Reviewed population changed: {r['stem']}")
        pub = json.loads((root / (r['stem'] + '.manifest.json')).read_bytes())
        cohort = pub.get('reconciliation', {}).get('archive_cohort', {})
        if r['new'] and (not cohort.get('schedule_verified') or cohort.get('validation_issues')):
            raise ValueError(f"Unverified publication: {r['stem']}")
        if cohort.get('snapshot') != r['snapshot']:
            raise ValueError(f"Snapshot status changed: {r['stem']}")


def prepare(root=Path('data/xDiyo_data'), *, apply=False):
    root = Path(root).resolve()
    review = json.loads(REVIEW.read_bytes())
    # Finish only our pending, fingerprinted publications before reading them.
    if apply:
        for r in review['seasons']:
            pending = root / '_team_seasons' / 'materialization' / (r['stem'] + '.pending.json')
            if pending.exists():
                materialize_movement_flags(root, r['stem'])
            pending = root / '_team_seasons' / 'calendars' / (r['stem'] + '.pending.json')
            if pending.exists():
                publish_calendar(root, r['stem'], r['calendar'])
    inspect(root, review)
    # Existing upper-division rosters supply genuine predecessor identities for
    # the two newly collected lower divisions. Japan/Mexico flags are preserved.
    rosters = [r for r in review['seasons'] if r['new'] or r['competition_id'] in (36, 37)]
    flags = derive_annual(rosters, review)
    totals = dict(publications=len(review['seasons']), new_publications=sum(r['new'] for r in review['seasons']),
        matches=sum(r['matches'] for r in review['seasons']), new_team_seasons=len(flags),
        movements=flags.movement.value_counts().to_dict(), snapshots=sum(r['snapshot'] is True for r in review['seasons']))
    print(json.dumps(totals), flush=True)
    if not apply:
        return totals
    audits = []
    for index, r in enumerate(review['seasons'], 1):
        stem = r['stem']
        src = _open_source(root, stem)
        pub_path = root / (stem + '.manifest.json')
        pub = json.loads(pub_path.read_bytes())
        protected = {n: info['sha256'] for n, info in src.manifest['tables'].items()
                     if not r['new'] or n not in ('matches', 'team_seasons')}
        reconciliation = json.loads(json.dumps(pub.get('reconciliation', {})))
        # Only the declared movement enrichment status may change here.
        for value in (reconciliation,):
            value.pop('promotion_relegation_flags', None)
            value.get('feature_status_counts', {}).pop('promotion_relegation_flags', None)
        paths = [src.table_path('matches'), root / pub['season_file']]
        if r['new']:
            selected = flags.loc[flags.season_stem.eq(stem)].reset_index(drop=True)
            before = [pq.ParquetFile(p).read() for p in paths]
            if not src.manifest.get('movement_enrichment', {}).get('materialized'):
                save_movement_enrichment(root, stem, selected)
            materialize_movement_flags(root, stem, reviewed_flags=selected)
            for path, original in zip(paths, before):
                actual = pq.ParquetFile(path).read().select(original.column_names)
                if not original.equals(actual, check_metadata=True):
                    raise ValueError(f'Observation/schema/missingness changed: {path}')
            stored = load_season(root, stem, tables='team_seasons', verify_hashes=True)['team_seasons']
            assert_movement_parity(selected, stored, season=stem)
        publish_calendar(root, stem, r['calendar'])
        updated = _open_source(root, stem)
        if any(digest(updated.table_path(n)) != h for n, h in protected.items()):
            raise ValueError(f'Unrelated table changed: {stem}')
        after_pub = json.loads(pub_path.read_bytes())
        after_reconciliation = json.loads(json.dumps(after_pub.get('reconciliation', {})))
        after_reconciliation.pop('promotion_relegation_flags', None)
        after_reconciliation.get('feature_status_counts', {}).pop('promotion_relegation_flags', None)
        if after_reconciliation != reconciliation:
            raise ValueError(f'Source coverage/availability evidence changed: {stem}')
        checked = {}
        for name, info in updated.manifest['tables'].items():
            path = updated.table_path(name)
            p = pq.ParquetFile(path)
            codecs = {p.metadata.row_group(i).column(j).compression for i in range(p.num_row_groups)
                      for j in range(p.metadata.row_group(i).num_columns)}
            if (digest(path) != info['sha256'] or path.stat().st_size != info['bytes'] or
                    p.metadata.num_rows != info['rows'] or r['new'] and codecs - {'ZSTD'}):
                raise ValueError(f'Hash/rows/compression mismatch: {path}')
            checked[name] = dict(sha256=info['sha256'], rows=info['rows'], codecs=sorted(codecs))
        p = pq.ParquetFile(paths[1])
        codecs = {p.metadata.row_group(i).column(j).compression for i in range(p.num_row_groups)
                  for j in range(p.metadata.row_group(i).num_columns)}
        if digest(paths[1]) != after_pub['sha256'] or codecs - {'ZSTD'}:
            raise ValueError(f'Nested publication mismatch: {stem}')
        if not r['new'] and after_pub['sha256'] != pub['sha256']:
            raise ValueError(f'Existing nested observations changed: {stem}')
        loaded = load_season(root, stem, tables=['matches', 'team_seasons'], verify_hashes=True, include_awarded=True)
        if loaded.matches.annual_season.unique().tolist() != [r['calendar']['annual_season']]:
            raise ValueError('Annual grouping did not round trip')
        if loaded['team_seasons'].movement.eq('unknown').any():
            raise ValueError(f'Unknown native movements: {stem}')
        audits.append(dict(stem=stem, matches=len(loaded.matches), snapshot=r['snapshot'],
            calendar=r['calendar'], source_manifest_sha256=digest(updated.path),
            publication_sha256=digest(pub_path), season_sha256=digest(paths[1]), tables=checked))
        if index % 10 == 0 or r['new']:
            print(f'{index}/{len(review["seasons"])} {stem}: verified', flush=True)
    inspect(root, review)
    receipt = dict(**totals, observations_preserved=True, source_coverage_preserved=True,
        review_sha256=digest(REVIEW), script_sha256=digest(Path(__file__)), seasons=audits)
    RECEIPT.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    return totals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--data-root', type=Path, default=Path('data/xDiyo_data'))
    args = parser.parse_args()
    prepare(args.data_root, apply=args.apply)
