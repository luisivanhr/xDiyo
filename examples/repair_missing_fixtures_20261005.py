"""Apply the two externally reviewed 2015/16 fixture omissions.

Requires the verified workbook result extract and statistics backfill cache.
Run without --apply to stage/audit; --apply installs the validated transaction.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pandas as pd
import pyarrow.parquet as pq

from backfill_statistics import digest, dump, finish
from xdiyo_analytics.data._source import _open_source
from xdiyo_analytics.data.fixture_coverage import audit_fixture_coverage
from xdiyo_analytics.data.statistics_backfill import STATISTICS, make_candidate, statistic_value
from xdiyo_analytics.data.supplemental_fixtures import append_fixture_tables, supplemental_event_id

ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = 'a91c03e33820453892708d08baa3a070a8e6ddc74e4ced23296a6ba8542dbf0f'
REPAIRS = {'Premiership_15_16': ('72241', '2015-09-15T18:45:00+00:00'),
           'Ligue_1_15_16': ('18849', '2015-08-08T19:00:00+00:00')}


def write_table(table, original, temporary):
    # Some old match-only tables still use Snappy. All repaired outputs use
    # the current dataset's ZSTD convention, preserving the Arrow schema.
    pq.write_table(table, temporary, compression='zstd', use_compliant_nested_type=False)
    if not pq.ParquetFile(temporary).read().equals(table, check_metadata=True):
        raise ValueError(f'Parquet round-trip changed: {original}')
    return dict(file=original.name, rows=len(table), bytes=temporary.stat().st_size, sha256=digest(temporary))


def run(apply=False):
    root = ROOT/'data/xDiyo_data'
    audit = ROOT/'data/audits/missing_fixtures_20261005'
    audit.mkdir(exist_ok=True, parents=True)
    reviewed = json.loads((ROOT/'docs/analytics/data/fixture_coverage_review_20261005.json').read_bytes())
    results = {r['match_id']: r for r in json.loads((audit/'workbook_results.json').read_bytes())}
    cache = ROOT/'data/audits/statistics_backfill/source'/FINGERPRINT[:20]
    extract = json.loads((cache/'extract.json').read_bytes())
    assert extract['workbook_sha256'] == FINGERPRINT
    for f in extract['files']:
        assert digest(cache/f['file']) == f['sha256']
    reports = {}
    for stem, (source_id, kickoff) in REPAIRS.items():
        journal = audit/(stem+'.pending.json')
        if journal.exists():
            if apply:
                finish(root, journal)
            else:
                receipt = json.loads(journal.read_bytes())
                for entry in receipt['files']:
                    path = root/entry['path']
                    assert digest(path.with_suffix(path.suffix+'.statistics-tmp')) == entry['sha256']
                    assert digest(path) == receipt['before'][entry['path']]
                reports[stem] = json.loads((audit/(stem+'.review.json')).read_bytes())
                continue
        done = audit/(stem+'.done.json')
        if done.exists():
            receipt = json.loads(done.read_bytes())
            for entry in receipt['files']:
                assert digest(root/entry['path']) == entry['sha256']
            reports[stem] = json.loads((audit/(stem+'.review.json')).read_bytes())
            continue
        source = _open_source(root, stem)
        pub_path = root/(stem+'.manifest.json')
        pub = json.loads(pub_path.read_bytes())
        paths = [source.table_path('matches'), source.table_path('statistics'), root/pub['season_file']]
        for name, path in zip(('matches', 'statistics'), paths):
            assert digest(path) == source.manifest['tables'][name]['sha256']
        assert digest(paths[2]) == pub['sha256']
        matches, statistics, nested = [pq.ParquetFile(p).read() for p in paths]
        specification = reviewed[stem]
        target = specification['required'][0]
        donor = results[source_id]
        assert donor['season'] == '2015/2016'
        assert [int(donor[s+'_goals_ft']) for s in ('home', 'away')] == target['score']
        # Workbook clocks are Europe/Paris here; independently reviewed kickoff
        # times above agree with UK 19:45 and France 21:00 summer local time.
        clock = pd.Timestamp('1899-12-30') + pd.to_timedelta(float(donor['start_datetime']), unit='D')
        assert abs((clock.tz_localize('Europe/Paris').tz_convert('UTC')-pd.Timestamp(kickoff)).total_seconds()) < 1
        rows = matches.to_pylist()
        fixture = {field.name: None for field in matches.schema}
        for key in ('competition_id', 'season_id', 'tournament_id', 'season_year'):
            fixture[key] = rows[0][key]
        fixture.update(event_id=supplemental_event_id(int(source_id)), status='finished', status_code=100,
                       round=target['round'], stage_json=json.dumps({'round':target['round']}), is_awarded=False,
                       kickoff_utc=datetime.fromisoformat(kickoff).timestamp(), event_metadata_observed_at=time.time())
        for side in ('home', 'away'):
            tid = target[side+'_id']
            identities = []
            for old in rows:
                for old_side in ('home', 'away'):
                    if old[old_side+'_id'] == tid:
                        identities.append(tuple(old[old_side+'_'+k] for k in ('name', 'slug', 'got_promoted', 'got_demoted', 'season_entry')))
            assert identities and len(set(identities)) == 1
            fixture[side+'_id'] = tid
            for key, value in zip(('name', 'slug', 'got_promoted', 'got_demoted', 'season_entry'), identities[0]):
                fixture[side+'_'+key] = value
            for native, excel in [('current','ft'), ('display','ft'), ('period1','1h'), ('period2','2h')]:
                fixture[f'{side}_score_{native}'] = float(donor[f'{side}_goals_{excel}'])
        additions = []
        for sheet in ('Statistics_FT', 'Statistics_1H', 'Statistics_2H'):
            frame = pd.read_parquet(cache/(sheet+'.parquet'))
            selected = frame[frame.match_id == source_id]
            assert len(selected) == 1
            stat = selected.iloc[0].to_dict()
            assert stat['home_team'] == donor['home_team'] and stat['away_team'] == donor['away_team']
            assert stat['source_league'] == stem.rsplit('_', 2)[0] and stat['source_season'] == '15_16'
            period = sheet.removeprefix('Statistics_').lower()
            for field in STATISTICS:
                values = [statistic_value(stat[f'{s}_{field}_{period}'], field) for s in ('home','away')]
                if field == 'ball_possession' and all(v is not None for v in values):
                    assert abs(sum(values)-100) <= 1
                for side, value in zip(('home','away'), values):
                    if value is None:
                        continue
                    evidence = dict(workbook_sha256=FINGERPRINT, sheet=sheet, row=int(stat['source_row']),
                                    column=f'{side}_{field}_{period}', vendor_match_id=source_id)
                    candidate = make_candidate(fixture, field=field, period=period, side=side, value=value,
                                               evidence=evidence, observed_at=fixture['event_metadata_observed_at'])
                    candidate.pop('evidence')
                    additions.append(candidate)
        changed = append_fixture_tables(matches, statistics, nested, fixture, additions)
        teams = set(matches['home_id'].to_pylist()) | set(matches['away_id'].to_pylist())
        check = lambda m: audit_fixture_coverage(m, team_ids=teams, meetings_per_pair=specification['meetings_per_pair'], required=specification['required'])
        review = dict(before=check(rows), after=check(changed[0].to_pylist()), fixture=fixture,
                      statistics_added=len(additions), workbook_sha256=FINGERPRINT, source_match_id=source_id,
                      source_sheet='Matches_Results', source_row=donor['source_row'], native_played_event_id=None,
                      provenance='Reviewed workbook supplement, not a native provider capture',
                      sources=[target['source'], target['corroboration']],
                      unavailable_components='Lineups, incidents, heatmaps, pregame and other uncaptured components remain empty')
        assert review['after']['status'] == 'balanced'
        dump(audit/(stem+'.review.json'), review)
        before = {p.relative_to(root).as_posix(): digest(p) for p in [*paths, source.path, pub_path]}
        entries = []
        for i, (path, table) in enumerate(zip(paths, changed)):
            info = write_table(table, path, path.with_suffix(path.suffix+'.statistics-tmp'))
            entries.append(dict(path=path.relative_to(root).as_posix(), sha256=info['sha256']))
            if i < 2:
                source.manifest['tables'][('matches','statistics')[i]] = info
            else:
                pub.update(bytes=info['bytes'], sha256=info['sha256'], matches=len(table))
        for metadata in (source.manifest, pub):
            metadata['fixture_coverage'] = review['after']
            metadata['fixture_enrichment'] = {k:v for k,v in review.items() if k not in ('before','after','fixture')}
            metadata['fixture_enrichment']['event_id'] = fixture['event_id']
        pub['complete_scope'] = 'Saved collector cohort plus reviewed supplemental fixtures; component coverage is separate'
        for path, contents in ((source.path, source.manifest), (pub_path, pub)):
            tmp = path.with_suffix(path.suffix+'.statistics-tmp')
            dump(tmp, contents)
            entries.append(dict(path=path.relative_to(root).as_posix(), sha256=digest(tmp)))
        dump(journal, dict(files=entries, before=before, original_match_rows_preserved=len(matches),
                           original_statistic_rows_preserved=len(statistics), original_nested_rows_preserved=len(nested)))
        if apply:
            finish(root, journal)
        reports[stem] = review
    dump(audit/'summary.json', reports)
    print(json.dumps({s:dict(before=r['before']['saved_matches'], after=r['after']['saved_matches'],
                            statistics_added=r['statistics_added']) for s,r in reports.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    run(parser.parse_args().apply)
