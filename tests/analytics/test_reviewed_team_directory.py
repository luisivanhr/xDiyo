from copy import deepcopy
import json

import pytest

from xdiyo_analytics.data.catalogue import (
    Registry, apply_reviewed_team_directory, load_reviewed_team_directory,
)
from xdiyo_analytics.data.catalogue.team_identities import identity_digest
from xdiyo_analytics.reporting import TeamCatalog


def document(*rows):
    result = dict(schema=1, policy='supplied_country_team_directory_v1',
                  input_sha256='a'*64, teams=list(rows))
    result['sha256'] = identity_digest(result)
    return result


def row(identity=123, country='England', name='United'):
    return dict(team_id=identity, canonical_name=name, country=country,
                names=[dict(name=name, start='2016-08-01', end='2026-05-01')],
                profile_url=f'https://www.sofascore.com/football/team/united/{identity}',
                evidence_url=f'https://www.sofascore.com/football/team/united/{identity}',
                review_status='verified_web_profile')


def registry():
    result = Registry()
    for identity, country in [(17, 'England'), (35, 'Germany'), (196, 'J1')]:
        result.seed('competition', identity, name=f'League {identity}', region=country, entity_kind='club_league')
    return result


def test_country_dates_missing_artwork_and_roundtrip(tmp_path):
    old = registry(); before = deepcopy(old.payload)
    review = document(row(), row(456, 'Germany'), row(789, 'Japan'))
    new = apply_reviewed_team_directory(old, review, catalog=TeamCatalog({}))
    assert old.payload == before
    for comp, identity in [(17, 123), (35, 456), (196, 789)]:
        assert new.resolve('team', 'United', comp, '2017-01-01') == identity
        assert new.resolve('team', 'United', comp, '2016-07-31') is None
        assert new.resolve('team', 'United', comp, '2026-05-02') is None
        assert new.squad(identity, '2017-01-01') == 'needs_review'
    assert 'fixture' not in new.payload['entities']
    assert new.payload['entities']['competition'] == old.payload['entities']['competition']
    assert apply_reviewed_team_directory(new, review, catalog=TeamCatalog({})).payload == new.payload
    new.save(tmp_path/'registry.json')
    assert Registry.load(tmp_path/'registry.json').payload == new.payload
    (tmp_path/'review.json').write_text(json.dumps(review), encoding='utf-8')
    assert load_reviewed_team_directory(tmp_path/'review.json') == review


def test_unknown_country_does_not_admit_competitions_or_fixtures():
    old = registry()
    review = document(row(country='Argentina'))
    new = apply_reviewed_team_directory(old, review, catalog=TeamCatalog({}))
    assert '123' in new.payload['entities']['team']
    assert not new.payload['aliases']
    assert new.payload['entities']['competition'] == old.payload['entities']['competition']


def test_conflicts_fail_without_mutating_registry():
    old = registry(); old.seed('team', 123, name='Other club', entity_kind='club')
    before = deepcopy(old.payload)
    with pytest.raises(ValueError, match='Conflicting canonical team'):
        apply_reviewed_team_directory(old, document(row(456), row()), catalog=TeamCatalog({}))
    assert old.payload == before
    with pytest.raises(ValueError, match='catalogue identity conflicts'):
        apply_reviewed_team_directory(registry(), document(row()), catalog=TeamCatalog({123:'Other club'}))


def test_alias_conflict_is_not_silently_rebound():
    old = registry(); old.seed('team', 999, name='Original', entity_kind='club')
    old.alias('team', 999, 'United', scope=17, start='2017-01-01', end='2017-12-31', decision='prior review')
    before = deepcopy(old.payload)
    with pytest.raises(ValueError, match='conflicting dated alias'):
        apply_reviewed_team_directory(old, document(row()), catalog=TeamCatalog({}))
    assert old.payload == before


@pytest.mark.parametrize('field,value', [
    ('team_id', True), ('team_id', 123.0), ('team_id', '123'), ('team_id', -1),
    ('country', ''), ('profile_url', 'https://www.sofascore.com/football/team/united/999'),
    ('profile_url', 'https://www.sofascore.com.evil.test/football/team/united/123'),
    ('profile_url', ''), ('evidence_url', 'file:///local'), ('names', []),
])
def test_malformed_review_rejected(field, value):
    record = row(); record[field] = value
    with pytest.raises(ValueError):
        apply_reviewed_team_directory(registry(), document(record), catalog=TeamCatalog({}))


def test_content_tampering_duplicates_and_exact_large_ids():
    review = document(row()); review['teams'][0]['country'] = 'Germany'
    with pytest.raises(ValueError, match='SHA-256'):
        apply_reviewed_team_directory(registry(), review, catalog=TeamCatalog({}))
    with pytest.raises(ValueError, match='unique exact'):
        apply_reviewed_team_directory(registry(), document(row(), row()), catalog=TeamCatalog({}))
    identity = 2**53+1
    new = apply_reviewed_team_directory(registry(), document(row(identity)), catalog=TeamCatalog({}))
    assert new.resolve('team', 'United', 17, '2020-01-01') == identity


def test_localized_profiles_and_explicit_external_evidence():
    record = row(); record['profile_url'] = 'https://www.sofascore.com/pt-pt/football/team/united/123'
    apply_reviewed_team_directory(registry(), document(record), catalog=TeamCatalog({}))
    record.update(profile_url='', evidence_url='https://github.com/example/repo/blob/123/mapping.csv',
                  review_status='verified_external_mapping')
    apply_reviewed_team_directory(registry(), document(record), catalog=TeamCatalog({}))


def test_shipped_supplied_directory_and_order_invariance():
    review = load_reviewed_team_directory()
    assert len(review['teams']) == 1204
    assert len({r['team_id'] for r in review['teams']}) == 1204
    assert len({r['country'] for r in review['teams']}) == 38
    left = apply_reviewed_team_directory(registry(), document(row(), row(456, 'Germany')), catalog=TeamCatalog({}))
    right = apply_reviewed_team_directory(registry(), document(row(456, 'Germany'), row()), catalog=TeamCatalog({}))
    for comp in (17, 35):
        assert left.resolve('team', 'United', comp, '2020-01-01') == right.resolve('team', 'United', comp, '2020-01-01')


def test_supplied_directory_badges_are_packaged_and_content_typed(tmp_path):
    import hashlib
    from pathlib import Path
    review = load_reviewed_team_directory()
    catalog = TeamCatalog.builtin()
    identities = [row['team_id'] for row in review['teams']]
    path = catalog.export_bundle(tmp_path/'portable', team_ids=identities)
    portable = TeamCatalog.from_json(path)
    for identity in identities:
        name, image, error = portable.display(identity, badges=True)
        assert image and error is None
        assert (name, image, error) == catalog.display(identity, badges=True)
        entry = portable.entries[str(identity)]
        assert hashlib.sha256(Path(entry['badge_path']).read_bytes()).hexdigest() == entry['sha256']
    assert portable.display(36836, badges=True)[1].startswith('data:image/svg+xml;')
    assert portable.display(188441, badges=True)[1].startswith('data:image/png;')


def test_prepare_supplied_directory_matches_exact_source_ids(tmp_path):
    import csv
    import importlib.util
    from pathlib import Path
    from zipfile import ZipFile
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('prepare_directory', root/'examples/prepare_supplied_team_mapping.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    mapping = tmp_path/'mapping.csv'
    fields = ['footiqo_team_id', 'sofascore_team_id', 'team_name_footiqo', 'country',
              'sofascore_team_url', 'sofascore_source_url', 'sofascore_match_status']
    source_id = str(2**53+1)
    profile = 'https://www.sofascore.com/football/team/united/123'
    data = [source_id, '123', 'United', 'England', profile, profile, 'verified_web_profile']
    def write_mapping(entries):
        with mapping.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream); writer.writerow(fields); writer.writerows(entries)
    write_mapping([data])
    workbook = tmp_path/'workbook.xlsx'
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    columns = ['home_team_id','away_team_id','home_team','away_team','start_datetime','country']
    content = [columns, [source_id, '7', 'United', 'Other', '44075', 'England']]
    body = ''.join('<row r="'+str(i)+'">'+''.join(
        f'<c r="{chr(65+j)}{i}" t="inlineStr"><is><t>{value}</t></is></c>'
        for j,value in enumerate(values))+'</row>' for i,values in enumerate(content,1))
    with ZipFile(workbook, 'w') as archive:
        archive.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Matches_Results" sheetId="1" r:id="r1"/></sheets></workbook>')
        archive.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr('xl/sharedStrings.xml', f'<sst xmlns="{ns}"/>')
        archive.writestr('xl/worksheets/sheet1.xml', f'<worksheet xmlns="{ns}"><sheetData>{body}</sheetData></worksheet>')
    before = workbook.read_bytes()
    review = module.prepare(mapping, workbook, catalog=TeamCatalog({}))
    assert review['teams'][0]['team_id'] == 123
    assert review['teams'][0]['names'] == [dict(name='United', start='2020-09-01', end='2020-09-01')]
    assert source_id not in json.dumps(review)
    assert workbook.read_bytes() == before
    write_mapping([data, data])
    with pytest.raises(ValueError, match='Duplicate or inexact'):
        module.prepare(mapping, workbook, catalog=TeamCatalog({}))
    changed = data.copy(); changed[0] = str(2**53)
    write_mapping([changed])
    with pytest.raises(ValueError, match='absent from the original workbook'):
        module.prepare(mapping, workbook, catalog=TeamCatalog({}))
    changed = data.copy(); changed[3] = 'Germany'; write_mapping([changed])
    with pytest.raises(ValueError, match='country is absent'):
        module.prepare(mapping, workbook, catalog=TeamCatalog({}))
