"""Review and enrich the 65 European league-seasons from 2010/11 to 2014/15.

Run with PYTHONPATH=.;src. Default is a read-only preflight; --apply publishes
reviewed entry flags with the existing movement writer. No collection, model
fitting, statistical backfill, or rewriting of unrelated source tables.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from examples.build_team_movements import build
from examples.prepare_season_extension_20261005 import codecs, digest, dump
from xdiyo_analytics.data import derive_movements, load_season, save_movement_enrichment
from xdiyo_analytics.data._source import _open_source
from xdiyo_analytics.data.movements import assert_movement_parity
from xdiyo_analytics.data.publish_movements import materialize_movement_flags
from xdiyo_analytics.histories import build_team_history


MEMBERS = {
    'Bundesliga': 18, 'Bundesliga_2': 18, 'Championship': 24,
    'Premier_League': 20, 'La_Liga': 20, 'La_Liga_2': 22,
    'Ligue_1': 20, 'Ligue_2': 20, 'Serie_A': 20, 'Serie_B': 22,
    'Eredivisie': 18, 'Premiership': 12, 'Pro_League': 16,
}
TARGETS = sorted(f'{league}_{y:02}_{y + 1:02}' for league in MEMBERS for y in range(10, 15))
UEFA_2010 = ('https://editorial.uefa.com/resources/025d-0f8426e39406-2c0d6ce46092-1000/'
             '2010_11_clubs_de_premiere_division_en_europe.pdf')


def source(url, title, scope, kind='secondary'):
    return dict(url=url, title=title, scope=scope, kind=kind)


def uefa(page, scope):
    return source(UEFA_2010, 'UEFA first division clubs 2010/11', f'Page {page}: {scope}', 'primary')


# These lists describe entry INTO 2010/11, never relegation at its end.
# IDs are from the actual native match rosters; source names are kept intact.
BOUNDARIES = {
    'Bundesliga_10_11': ([2675, 2526], [], [uefa(58, 'Kaiserslautern and St Pauli promoted.')], []),
    'Bundesliga_2_10_11': ([2581, 2566, 5880], [2542, 2528], [
        source('https://www.rsssf.org/tablesd/duit2010.html', 'RSSSF Germany 2009/10',
               'Osnabruck and Aue promoted directly; Ingolstadt won the promotion playoff.'),
        uefa(58, 'Bochum and Hertha relegated; Nurnberg survived the playoff.')], []),
    'Championship_10_11': ([263, 34, 25], [6, 96, 2], [
        source('https://en.wikipedia.org/wiki/2010%E2%80%9311_Football_League_Championship',
               '2010/11 Championship: changes from last season',
               'Norwich, Leeds and Millwall arrived from League One; Burnley, Hull and Portsmouth from above.'),
        source('https://www.rsssf.org/tablese/eng2010.html', 'RSSSF England 2009/10',
               'Previous-season standings and promotion playoffs.')], []),
    'Premier_League_10_11': ([39, 8, 67], [], [uefa(43, 'Newcastle, West Brom and Blackpool promoted.')], []),
    'La_Liga_10_11': ([2824, 2877, 2849], [], [uefa(136, 'Real Sociedad, Hercules and Levante promoted.')], []),
    'La_Liga_2_10_11': ([33779, 6195, 24343, 24322], [2831, 2822, 2838], [
        source('https://en.wikipedia.org/wiki/2010%E2%80%9311_Segunda_Divisi%C3%B3n',
               '2010/11 Segunda Division: teams',
               'Granada, Ponferradina, Barcelona B and Alcorcon entered from below.'),
        uefa(136, 'Valladolid, Tenerife and Xerez relegated.')], []),
    'Ligue_1_10_11': ([1667, 1715, 7721], [], [uefa(53, 'Caen, Brest and Arles-Avignon promoted.')], []),
    'Ligue_2_10_11': ([21800, 1652, 1682], [1672, 1711, 1671], [
        source('https://www.rsssf.org/tablesf/fran2010.html', 'RSSSF France 2009/10',
               'Evian, Troyes and Reims promoted from National.'),
        uefa(53, 'Le Mans, Boulogne and Grenoble relegated.')], []),
    'Serie_A_10_11': ([2689, 2742, 2691], [], [uefa(73, 'Lecce, Cesena and Brescia promoted.')], []),
    'Serie_B_10_11': ([2767, 5316, 2743, 7935], [2686, 170650, 2726], [
        source('https://en.wikipedia.org/wiki/2010%E2%80%9311_Serie_B', '2010/11 Serie B: teams',
               'Novara, Portogruaro, Varese and Pescara promoted; Atalanta, Siena and Livorno relegated. Triestina reinstated.')],
        ['Triestina is retained: it already played Serie B in 2009/10 and was reinstated after Ancona exclusion.',
         'Modern provider names and IDs (including Siena and Reggina) are preserved; no identity remapping.']),
    'Eredivisie_10_11': ([2949, 2967], [], [uefa(99, 'De Graafschap and Excelsior promoted.')], []),
    'Premiership_10_11': ([2362], [], [uefa(125, 'Inverness promoted.')], []),
    'Pro_League_10_11': ([116075, 2926], [], [uefa(22, 'Lierse and Eupen promoted.')],
        ['Provider historical club IDs/names are preserved, including Lierse and Beerschot.']),
}


def boundary_additions(rosters):
    by_stem = {r['stem']: r for r in rosters}
    result = []
    for stem, (promoted, relegated, sources, notes) in BOUNDARIES.items():
        result.append(dict(stem=stem, promoted_ids=promoted, relegated_ids=relegated,
                           other_entry_ids=[], complete_review=True, sources=sources, notes=notes))
    # Keep official promotion terminology for Dundee, with its administrative
    # route explicit, as in the existing Ascoli 2015 review. Vicenza's ripescaggio
    # is an administrative admission, consistent with Paris FC 2017.
    result.append(dict(stem='Premiership_12_13', promoted_ids=[2357], relegated_ids=[],
        other_entry_ids=[], complete_review=False, sources=[source(
            'https://editorial.uefa.com/resources/0201-0f842817135c-7850e9be42b3-1000/first_division_clubs_in_europe_2012_13.pdf',
            'UEFA first division clubs 2012/13',
            'Scotland lists Dundee and Ross County as promoted clubs; Rangers voted out of SPL.', 'primary')],
        notes=['Dundee entered by invitation following Rangers exclusion; UEFA explicitly classifies it as promoted.']))
    result.append(dict(stem='Serie_B_14_15', promoted_ids=[], relegated_ids=[],
        other_entry_ids=[2722], complete_review=False, sources=[source(
            'https://www.ansa.it/sito/notizie/sport/2014/08/29/vicenza-ripescato-in-serie-b_acfbe120-f343-419d-aeb3-e792a5c64160.html',
            'ANSA: Vicenza ripescato in Serie B, 29 August 2014',
            'Vicenza admitted administratively to replace Siena after the latter failed to register.'), source(
            'https://files.figc.it/version/c%3AMWI2ZDhkMjMtNTczNC00%3AODU0MTM0MDgtODZmOC00/C_2_ContenutoGenerico_2525403_DettaglioAreaStampa_lstAllegati_0_upfAllegato.pdf',
            'FIGC youth sector notice, September 2014',
            'Confirms Vicenza inclusion in Serie B by federal notice 59/A of 29 August 2014.', 'primary')],
        notes=['Vicenza did not win sporting promotion in 2013/14: classify the 2014 ripescaggio as other_entry.',
               'Its 2012/13 entry remains retained, since it played Serie B in 2011/12.']))
    for entry in result:
        roster = by_stem[entry['stem']]
        marked = entry['promoted_ids'] + entry['relegated_ids'] + entry['other_entry_ids']
        entry.update(reviewed_on='2026-10-08', uncertainty=[],
            observed_team_names={str(i): roster['teams'][str(i)] for i in marked},
            observed_member_count=len(roster['teams']))
    return result


def prepare(root=Path('data/xDiyo_data'), evidence=Path('docs/analytics/data'), *, apply=False):
    root, evidence = Path(root), Path(evidence)
    review_path = evidence / 'team_movement_roster_review.json'
    boundary_path = evidence / 'team_movement_boundary_evidence.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    boundaries = json.loads(boundary_path.read_text(encoding='utf-8'))
    entries = {e['stem']: e for e in review['seasons']}
    protected = {str(p): digest(p) for p in (review_path, boundary_path)}
    rosters, observations = [], {}
    for stem in sorted(set(entries) | set(TARGETS)):
        src = _open_source(root, stem)
        publication_path = root / f'{stem}.manifest.json'
        publication = json.loads(publication_path.read_bytes())
        scope = src.manifest['scope']
        matches = load_season(root, stem, include_awarded=True, verify_hashes=True).matches
        teams = {str(int(t)): n for side in ('home', 'away') for t, n in
                 matches[[side + '_id', side + '_name']].itertuples(index=False, name=None)}
        original = src.manifest.get('movement_enrichment', {}).get(
            'original_matches_sha256', src.manifest['tables']['matches']['sha256'])
        entry = dict(stem=stem, matches_sha256=original, team_ids=sorted(map(int, teams)))
        if stem in entries and entries[stem] != entry:
            raise ValueError(f'Reviewed membership changed: {stem}')
        entries[stem] = entry
        rosters.append(dict(stem=stem, league=scope['league_name'], year=scope['season_start'],
            competition_id=scope['league_id'], season_id=scope['season_id'], teams=teams))
        for p in (publication_path, src.path, src.table_path('matches')):
            protected[str(p)] = digest(p)
        if stem not in TARGETS:
            continue
        cohort = publication.get('reconciliation', {}).get('archive_cohort', {})
        if (len(teams) != MEMBERS[scope['league_name']] or
                not src.manifest.get('requested_components_terminal') or
                not cohort.get('schedule_verified') or cohort.get('validation_issues') or
                cohort.get('selected_matches') != len(matches)):
            raise ValueError(f'Unreviewed membership or unfinished collection: {stem}')
        tables = {}
        for name, info in src.manifest['tables'].items():
            path = src.table_path(name)
            if (digest(path) != info['sha256'] or path.stat().st_size != info['bytes'] or
                    pq.ParquetFile(path).metadata.num_rows != info['rows'] or set(codecs(path)) - {'ZSTD'}):
                raise ValueError(f'Table fingerprint/rows/compression mismatch: {stem}/{name}')
            protected[str(path)] = info['sha256']
            tables[name] = dict(rows=info['rows'], sha256=info['sha256'], codecs=codecs(path))
        nested = root / publication['season_file']
        if (digest(nested) != publication['sha256'] or nested.stat().st_size != publication['bytes'] or
                pq.ParquetFile(nested).metadata.num_rows != len(matches) or set(codecs(nested)) - {'ZSTD'}):
            raise ValueError(f'Season export fingerprint/rows/compression mismatch: {stem}')
        protected[str(nested)] = publication['sha256']
        observations[stem] = dict(matches=len(matches), teams=len(teams), tables=tables,
            awarded=int(matches.is_awarded.fillna(False).sum()), export_sha256=publication['sha256'],
            schedule_statuses=cohort.get('statuses'), regular_matches=cohort.get('regular_matches'),
            coverage=src.manifest.get('coverage_summary', {}))
    prior_boundaries = {b['stem']: b for b in boundaries}
    for entry in boundary_additions(rosters):
        if entry['stem'] in prior_boundaries:
            if prior_boundaries[entry['stem']] != entry:
                raise ValueError(f'Boundary review changed: {entry["stem"]}')
        else:
            boundaries.append(entry)
    flags = derive_movements(rosters, review['competitions'], reviewed_seasons=entries, boundaries=boundaries)
    if flags.movement.eq('unknown').any():
        raise ValueError('Unresolved season entries remain; research before publishing.')
    changes = {}
    for roster in rosters:
        stem = roster['stem']
        src = _open_source(root, stem)
        if not src.manifest.get('movement_enrichment', {}).get('materialized'):
            if stem not in TARGETS:
                raise ValueError(f'Unexpected unmaterialized previous season: {stem}')
            changes[stem] = 'new movement enrichment'
            continue
        selected = flags.loc[flags.season_stem.eq(stem)]
        prior = load_season(root, stem, tables='team_seasons', verify_hashes=True)['team_seasons']
        try:
            assert_movement_parity(selected, prior, season=stem)
        except ValueError:
            # Earlier observations may supply previously unavailable predecessor
            # IDs, but may never silently change published classifications.
            cols = ['team_id', 'movement', 'got_promoted', 'got_demoted']
            pd.testing.assert_frame_equal(selected[cols].sort_values('team_id').reset_index(drop=True),
                prior[cols].sort_values('team_id').reset_index(drop=True), check_dtype=False)
            changes[stem] = 'adjacent predecessor evidence; classifications unchanged'
    summary = dict(requested_seasons=len(TARGETS), requested_matches=sum(o['matches'] for o in observations.values()),
        requested_team_seasons=int(flags.season_stem.isin(TARGETS).sum()), reviewed_seasons=len(rosters),
        reviewed_team_seasons=len(flags), requested_movement_counts=flags.loc[flags.season_stem.isin(TARGETS), 'movement'].value_counts().to_dict(),
        movement_counts=flags.movement.value_counts().to_dict(), changes=changes)
    print(json.dumps(summary, indent=2), flush=True)
    if not apply:
        return summary
    if any(digest(p) != h for p, h in protected.items()):
        raise ValueError('Sources changed during preflight; rerun review.')
    outputs, modified = {}, {str(review_path), str(boundary_path)}
    for stem in changes:
        src = _open_source(root, stem)
        publication_path = root / f'{stem}.manifest.json'
        paths = [src.table_path('matches'), root / f'{stem}.parquet']
        before = [pq.ParquetFile(p).read() for p in paths]
        hashes = [digest(p) for p in paths]
        unchanged = {n: digest(src.table_path(n)) for n in src.manifest['tables'] if n not in ('matches', 'team_seasons')}
        modified.update(str(p) for p in [*paths, src.path, publication_path, src.path.parent / 'team_seasons.parquet'])
        if not src.manifest.get('movement_enrichment', {}).get('materialized'):
            save_movement_enrichment(root, stem, flags)
        materialize_movement_flags(root, stem, reviewed_flags=flags)
        for p, original in zip(paths, before):
            if not original.equals(pq.ParquetFile(p).read().select(original.column_names), check_metadata=True):
                raise ValueError(f'Existing observation values/schema changed: {p}')
        src = _open_source(root, stem)
        if any(digest(src.table_path(n)) != h for n, h in unchanged.items()):
            raise ValueError(f'Unrelated native table changed: {stem}')
        loaded = load_season(root, stem, tables=['matches', 'team_seasons'], include_awarded=True, verify_hashes=True)
        assert_movement_parity(flags.loc[flags.season_stem.eq(stem)], loaded['team_seasons'], season=stem)
        history = build_team_history(loaded)
        if len(history) != 2 * len(loaded.matches) or history.team_got_promoted.isna().any() or history.team_got_demoted.isna().any():
            raise ValueError(f'History flags missing: {stem}')
        publication = json.loads(publication_path.read_bytes())
        if digest(paths[1]) != publication['sha256'] or paths[1].stat().st_size != publication['bytes']:
            raise ValueError(f'Updated season export fingerprint mismatch: {stem}')
        # Older exports may have a legacy codec; a predecessor-only correction
        # must not rewrite those observation files. All new data must use ZSTD.
        codec_paths = [*paths, src.table_path('team_seasons')] if stem in TARGETS else [src.table_path('team_seasons')]
        if any(set(codecs(p)) - {'ZSTD'} for p in codec_paths):
            raise ValueError(f'Publication must use ZSTD: {stem}')
        if stem not in TARGETS and [digest(p) for p in paths] != hashes:
            raise ValueError(f'Predecessor-only update changed observation files: {stem}')
        outputs[stem] = dict(matches_sha256=src.manifest['tables']['matches']['sha256'], season_sha256=publication['sha256'],
            team_seasons_sha256=src.manifest['tables']['team_seasons']['sha256'], history_rows=len(history),
            preserved_observation_tables=len(unchanged))
        print(f'Published and verified {stem}', flush=True)
    if any(digest(p) != h for p, h in protected.items() if p not in modified):
        raise ValueError('Unrelated reviewed source changed during publication.')
    review['seasons'] = [entries[k] for k in sorted(entries)]
    dump(review_path, review)
    dump(boundary_path, sorted(boundaries, key=lambda b: b['stem']))
    build(root, evidence)
    audit = dict(reviewed_on='2026-10-08', **summary, inputs=observations, outputs=outputs,
        verification='All target table/export hashes and ZSTD codecs checked; existing observation values/schema and unrelated tables preserved.',
        coverage_policy='Missing/empty optional tables and excluded abandoned fixtures are preserved; no statistics or outcomes invented.')
    audit_path = evidence / 'season_extension_20261008.json'
    if outputs or not audit_path.exists():
        dump(audit_path, audit)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    prepare(apply=args.apply)
