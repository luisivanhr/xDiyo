"""Review/publish the 21 historical league-seasons requested on 5 October 2026.

No network collection, model fitting, or discovery of additional seasons.
Default is read-only preflight. Use --apply after reviewing its summary.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

import pandas as pd
import pyarrow.parquet as pq

from xdiyo_analytics.data import derive_movements, load_season, save_movement_enrichment
from xdiyo_analytics.data._source import _open_source
from xdiyo_analytics.data.movements import assert_movement_parity
from xdiyo_analytics.data.publish_movements import materialize_movement_flags
from xdiyo_analytics.histories import build_team_history


SELECTION = {
    2015: ['Pro_League', 'Premiership', 'Championship', 'La_Liga_2', 'Bundesliga_2', 'Ligue_2', 'Serie_B'],
    2016: ['Pro_League', 'Premiership', 'Eredivisie', 'Bundesliga_2', 'Ligue_2', 'Serie_B'],
    2017: ['Pro_League', 'Premiership', 'Serie_B'],
    2018: ['Pro_League', 'Premiership', 'Serie_B'],
    2019: ['Pro_League', 'Premiership'],
}
TARGETS = [f'{league}_{year % 100:02}_{(year + 1) % 100:02}'
           for year, leagues in SELECTION.items() for league in leagues]
MEMBERS = {'Pro_League': 16, 'Premiership': 12, 'Championship': 24,
           'La_Liga_2': 22, 'Bundesliga_2': 18, 'Ligue_2': 20, 'Serie_B': 22, 'Eredivisie': 18}

# Web-reviewed season-entry lists, mapped to the IDs observed in native matches.
# [P]/[R] in RSSSF tables refer to entry into that season, not its final outcome.
BOUNDARIES = {
    'Pro_League_15_16': ([2895, 2918], [], [
        ('https://www.rsssf.org/tablesb/belg2016.html', 'RSSSF Belgium 2015/16',
         'Sint-Truiden and OH Leuven entered as promoted clubs.', 'secondary')], []),
    'Premiership_15_16': ([2353], [], [
        ('https://www.rsssf.org/tabless/scot2016.html', 'RSSSF Scotland 2015/16',
         'Hearts was the only promoted Premiership entrant.', 'secondary'),
        ('https://spfl.co.uk/news/championship-roll-of-honour', 'SPFL Championship roll of honour',
         'Hearts won the 2014/15 second-tier championship.', 'primary')], []),
    'Championship_15_16': ([58, 4, 21], [6, 96, 1], [
        ('https://www.rsssf.org/tablese/eng2015.html', 'RSSSF England 2014/15',
         'Bristol City and MK Dons won automatic promotion; Preston won the playoff final.', 'secondary'),
        ('https://www.rsssf.org/tablese/eng2016.html', 'RSSSF England 2015/16',
         'Burnley, Hull and QPR entered the Championship from the Premier League.', 'secondary')], []),
    'La_Liga_2_15_16': ([2851, 2840, 24265, 24324], [2850, 2858, 2846], [
        ('https://www.rsssf.org/tabless/span2016.html', 'RSSSF Spain 2015/16',
         'Oviedo, Gimnastic, Huesca and Bilbao Athletic arrived from below; Cordoba, Almeria and Elche from above.', 'secondary'),
        ('https://en.wikipedia.org/wiki/2015%E2%80%9316_Segunda_Divisi%C3%B3n', '2015/16 Segunda Division',
         'Pre-season promotion/relegation and Elche administrative relegation.', 'secondary')],
        ['Elche entered from the upper division following administrative relegation.',
         'Athletic Club B U21 is the current source name for historical Bilbao Athletic.']),
    'Bundesliga_2_15_16': ([2540, 2543], [2538, 2561], [
        ('https://www.rsssf.org/tablesd/duit2016.html', 'RSSSF Germany 2015/16',
         'Bielefeld and Duisburg arrived from below; Freiburg and Paderborn from the Bundesliga.', 'secondary')], []),
    'Ligue_2_15_16': ([52874, 6070, 35056], [1651, 1648, 21800], [
        ('https://www.rsssf.org/tablesf/fran2016.html', 'RSSSF France 2015/16',
         'Red Star, Paris FC and Bourg promoted; Metz, Lens and Evian relegated into Ligue 2.', 'secondary')], []),
    'Serie_B_15_16': ([2767, 2710, 2704, 2707], [2719, 2742], [
        ('https://www.rsssf.org/tablesi/ital2016.html', 'RSSSF Italy 2015/16',
         'Novara, Salernitana, Como and Ascoli entered from below; Cagliari and Cesena from Serie A.', 'secondary'),
        ('https://www.lega-pro.com/com/1516-143L.pdf', 'Lega Pro official notice 143/L',
         'Explicitly records Ascoli promoted and Virtus Entella reinstated in Serie B.', 'primary'),
        ('https://www.figc.it/it/federazione/news/consiglio-federale-definiti-gli-organici-della-serie-b-e-della-lega-pro-s1jgrqf2',
         'FIGC: Serie B and Lega Pro membership',
         'Entella and Ascoli replaced Catania and Teramo following disciplinary decisions.', 'primary'),
        ('https://en.wikipedia.org/wiki/2015%E2%80%9316_Serie_B', '2015/16 Serie B',
         'Brescia and Entella readmissions; Parma did not enter Serie B after bankruptcy.', 'secondary')],
        ['Ascoli is promoted, following the explicit Lega Pro classification despite the administrative route.',
         'Brescia and Virtus Entella remain retained: both played in Serie B the previous season.',
         'Parma is absent from this roster and must not be fabricated as a relegated entrant.']),
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def codecs(path):
    file = pq.ParquetFile(path)
    return sorted({file.metadata.row_group(i).column(j).compression
                   for i in range(file.num_row_groups)
                   for j in range(file.metadata.row_group(i).num_columns)})


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def prepare(root=Path('data/xDiyo_data'), evidence=Path('docs/analytics/data'), *, apply=False):
    root, evidence = Path(root), Path(evidence)
    review_path = evidence / 'team_movement_roster_review.json'
    boundary_path = evidence / 'team_movement_boundary_evidence.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    boundaries = json.loads(boundary_path.read_text(encoding='utf-8'))
    entries = {entry['stem']: entry for entry in review['seasons']}
    rosters, observations, fingerprints = [], {}, {}
    for stem in sorted(set(entries) | set(TARGETS)):
        source = _open_source(root, stem)
        publication_path = root / f'{stem}.manifest.json'
        publication = json.loads(publication_path.read_bytes())
        scope = source.manifest['scope']
        matches = load_season(root, stem, include_awarded=True, verify_hashes=True).matches
        teams = {str(int(t)): n for side in ('home', 'away') for t, n in
                 matches[[side + '_id', side + '_name']].itertuples(index=False, name=None)}
        original = source.manifest.get('movement_enrichment', {}).get(
            'original_matches_sha256', source.manifest['tables']['matches']['sha256'])
        entry = dict(stem=stem, matches_sha256=original, team_ids=sorted(map(int, teams)))
        if stem in entries and entries[stem] != entry:
            raise ValueError(f'Reviewed membership changed: {stem}')
        entries[stem] = entry
        rosters.append(dict(stem=stem, league=scope['league_name'], year=scope['season_start'],
                            competition_id=scope['league_id'], season_id=scope['season_id'], teams=teams))
        fingerprints[stem] = {str(p): digest(p) for p in
                              (publication_path, source.path, source.table_path('matches'))}
        if stem not in TARGETS:
            continue
        expected_members = 19 if stem == 'Serie_B_18_19' else MEMBERS[scope['league_name']]
        if len(teams) != expected_members or not publication['complete']:
            raise ValueError(f'Unexpected membership or unpublished season: {stem}')
        if not source.manifest['requested_components_terminal']:
            raise ValueError(f'Collection still in progress: {stem}')
        nested = root / publication['season_file']
        if digest(nested) != publication['sha256'] or nested.stat().st_size != publication['bytes']:
            raise ValueError(f'Season export fingerprint mismatch: {stem}')
        table_summary = {}
        for name, info in source.manifest['tables'].items():
            path = source.table_path(name)
            if digest(path) != info['sha256'] or path.stat().st_size != info['bytes']:
                raise ValueError(f'Table fingerprint mismatch: {stem}/{name}')
            file = pq.ParquetFile(path)
            if file.metadata.num_rows != info['rows'] or set(codecs(path)) - {'ZSTD'}:
                raise ValueError(f'Unexpected rows or compression: {stem}/{name}')
            table_summary[name] = dict(rows=info['rows'], sha256=info['sha256'], codecs=codecs(path))
        if set(codecs(nested)) - {'ZSTD'}:
            raise ValueError(f'Unexpected season compression: {stem}')
        observations[stem] = dict(matches=len(matches), teams=len(teams),
            awarded=int(matches.is_awarded.fillna(False).sum()), tables=table_summary,
            export_sha256=publication['sha256'], coverage=source.manifest.get('coverage_summary', {}))

    for stem, (promoted, relegated, sources, notes) in BOUNDARIES.items():
        if any(b['stem'] == stem for b in boundaries):
            continue
        roster = next(r for r in rosters if r['stem'] == stem)
        boundaries.append(dict(stem=stem, promoted_ids=promoted, relegated_ids=relegated,
            other_entry_ids=[], complete_review=True, reviewed_on='2026-10-05', uncertainty=[],
            sources=[dict(url=u, title=t, scope=s, kind=k) for u, t, s, k in sources], notes=notes,
            observed_team_names={str(i): roster['teams'][str(i)] for i in promoted + relegated},
            observed_member_count=len(roster['teams'])))
    flags = derive_movements(rosters, review['competitions'], reviewed_seasons=entries, boundaries=boundaries)
    if flags.movement.eq('unknown').any():
        raise ValueError('Unresolved season entries remain; research them before publication.')
    changes = {}
    for roster in rosters:
        stem = roster['stem']
        source = _open_source(root, stem)
        selected = flags.loc[flags.season_stem.eq(stem)]
        if not source.manifest.get('movement_enrichment', {}).get('materialized'):
            if stem not in TARGETS:
                raise ValueError(f'Unexpected unmaterialized previous season: {stem}')
            changes[stem] = 'new movement enrichment'
            continue
        prior = load_season(root, stem, tables='team_seasons', verify_hashes=True)['team_seasons']
        try:
            assert_movement_parity(selected, prior, season=stem)
        except ValueError:
            classification = ['team_id', 'movement', 'got_promoted', 'got_demoted']
            pd.testing.assert_frame_equal(selected[classification].sort_values('team_id').reset_index(drop=True),
                prior[classification].sort_values('team_id').reset_index(drop=True), check_dtype=False)
            changes[stem] = 'newly available adjacent-season predecessors; classifications unchanged'
    summary = dict(requested_seasons=len(TARGETS), requested_matches=sum(r['matches'] for r in observations.values()),
        requested_team_seasons=int(flags.season_stem.isin(TARGETS).sum()), reviewed_seasons=len(rosters),
        reviewed_team_seasons=len(flags), requested_movement_counts=flags.loc[flags.season_stem.isin(TARGETS), 'movement'].value_counts().to_dict(),
        movement_counts=flags.movement.value_counts().to_dict(), changes=changes)
    print(json.dumps(summary, indent=2), flush=True)
    if not apply:
        return summary

    # Recheck source fingerprints before the first mutation, including old rosters.
    if any(digest(p) != h for files in fingerprints.values() for p, h in files.items()):
        raise ValueError('Sources changed during review; rerun preflight.')
    published = {}
    for stem in changes:
        source = _open_source(root, stem)
        publication = json.loads((root / f'{stem}.manifest.json').read_bytes())
        paths = [source.table_path('matches'), root / publication['season_file']]
        before = [pq.ParquetFile(p).read() for p in paths]
        before_hashes = [digest(p) for p in paths]
        unchanged = {name: digest(source.table_path(name)) for name in source.manifest['tables']
                     if name not in ('matches', 'team_seasons')}
        if not source.manifest.get('movement_enrichment', {}).get('materialized'):
            save_movement_enrichment(root, stem, flags)
        materialize_movement_flags(root, stem, reviewed_flags=flags)
        for p, original in zip(paths, before):
            after = pq.ParquetFile(p).read().select(original.column_names)
            if not original.equals(after, check_metadata=True):
                raise ValueError(f'Existing observation values/schema changed: {p}')
        source = _open_source(root, stem)
        if any(digest(source.table_path(n)) != h for n, h in unchanged.items()):
            raise ValueError(f'Unrelated native table changed: {stem}')
        loaded = load_season(root, stem, tables=['matches', 'team_seasons'], include_awarded=True, verify_hashes=True)
        assert_movement_parity(flags.loc[flags.season_stem.eq(stem)], loaded['team_seasons'], season=stem)
        history = build_team_history(loaded)
        if len(history) != 2 * len(loaded.matches) or history.team_got_promoted.isna().any() or history.team_got_demoted.isna().any():
            raise ValueError(f'History flags missing: {stem}')
        publication = json.loads((root / f'{stem}.manifest.json').read_bytes())
        if digest(paths[1]) != publication['sha256']:
            raise ValueError(f'Updated export fingerprint mismatch: {stem}')
        check_codecs = [*paths, source.table_path('team_seasons')] if stem in TARGETS else [source.table_path('team_seasons')]
        if any(set(codecs(p)) - {'ZSTD'} for p in check_codecs):
            raise ValueError(f'Publication must use ZSTD: {stem}')
        if stem not in TARGETS and [digest(p) for p in paths] != before_hashes:
            raise ValueError(f'Predecessor-only update changed observation files: {stem}')
        published[stem] = dict(matches_sha256=source.manifest['tables']['matches']['sha256'],
            season_sha256=publication['sha256'], team_seasons_sha256=source.manifest['tables']['team_seasons']['sha256'],
            preserved_observation_tables=len(unchanged), history_rows=len(history))
        print(f'Published and verified {stem}', flush=True)
    for stem, files in fingerprints.items():
        if stem not in changes and any(digest(p) != h for p, h in files.items()):
            raise ValueError(f'Unrelated reviewed publication changed: {stem}')
    review['seasons'] = [entries[k] for k in sorted(entries)]
    dump(review_path, review)
    dump(boundary_path, sorted(boundaries, key=lambda b: b['stem']))
    # Standard builder verifies all persisted evidence and refreshes the audit.
    build = runpy.run_path(str(Path(__file__).with_name('build_team_movements.py')))['build']
    build(root, evidence)
    audit = dict(reviewed_on='2026-10-05', **summary, inputs=observations, outputs=published,
        verification='All source table/export hashes and codecs checked; existing observation columns and unrelated tables preserved.',
        coverage_policy='Missing/empty optional source tables are retained as coverage limitations; no outcomes or statistics fabricated.')
    audit_path = evidence / 'season_extension_20261005.json'
    if published or not audit_path.exists():
        dump(audit_path, audit)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    prepare(apply=args.apply)
