"""Dated identity reconciliation must never merge clubs by an unscoped name."""
from copy import deepcopy
import hashlib
import json

import pytest

from xdiyo_analytics.data.catalogue import Registry, apply_team_identity_decisions, load_team_identity_decisions
from xdiyo_analytics.data.catalogue.team_identities import identity_digest
from xdiyo_analytics.reporting import TeamCatalog


def document(rows):
    result = dict(schema=1, policy='dated_reviewed_native_team_aliases_v1', records=rows)
    result['sha256'] = identity_digest(result)
    return result


def setup(tmp_path):
    badge = tmp_path / 'badge.svg'
    badge.write_text('<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0h10v10z"/></svg>')
    digest = hashlib.sha256(badge.read_bytes()).hexdigest()
    catalog = TeamCatalog({team: dict(name=name, badge_path=str(badge), sha256=digest)
                           for team, name in [(1, 'Real Club'), (2, 'University Club'), (2**53+1, 'Other Club')]})
    registry = Registry()
    for comp in [8, 54]:
        registry.seed('competition', comp, name=str(comp), region='Spain', entity_kind='club_league')
    registry.seed('team', 1, name='Real Club', entity_kind='club')
    row = dict(team_id=1, canonical_name='Real Club', name='Murcia', competition_id=54,
               start='2011-01-01', end='2014-12-31', evidence='Reviewed ordered native fixtures')
    return registry, catalog, row


def test_dates_scope_idempotence_and_roundtrip(tmp_path):
    registry, catalog, row = setup(tmp_path)
    previous = deepcopy(registry.payload)
    later = dict(row, team_id=2, canonical_name='University Club', start='2016-01-01', end='2017-12-31')
    decisions = document([row, later])
    result = apply_team_identity_decisions(registry, decisions, catalog=catalog)
    assert registry.payload == previous
    assert result.resolve('team', 'Murcia', 54, '2014-12-31') == 1
    assert result.resolve('team', 'Murcia', 54, '2016-01-01') == 2
    assert result.resolve('team', 'Murcia', 54, '2015-01-01') is None
    assert result.resolve('team', 'Murcia', 8, '2016-01-01') is None
    assert result.squad(2, '2016-01-01') == 'needs_review'
    assert not result.payload['entities'].get('fixture')
    assert apply_team_identity_decisions(result, decisions, catalog=catalog).payload == result.payload
    result.save(tmp_path/'registry.json')
    assert Registry.load(tmp_path/'registry.json').payload == result.payload
    (tmp_path/'decisions.json').write_text(json.dumps(decisions), encoding='utf-8')
    assert load_team_identity_decisions(tmp_path/'decisions.json') == decisions


def test_conflict_rolls_back_whole_batch_and_original_aliases(tmp_path):
    registry, catalog, row = setup(tmp_path)
    registry.alias('team', 1, 'Murcia', scope=54, start='2011-01-01', end='2014-12-31', decision='original')
    before = deepcopy(registry.payload)
    bad = dict(row, team_id=2, canonical_name='University Club', start='2014-12-31')
    with pytest.raises(ValueError, match='Conflicting team alias'):
        apply_team_identity_decisions(registry, document([row, bad]), catalog=catalog)
    assert registry.payload == before


@pytest.mark.parametrize('field,value', [('team_id', 1.0), ('team_id', True), ('team_id', '1'),
    ('competition_id', 999), ('start', 'bad'), ('start', '2015-01-01'), ('name', ''),
    ('evidence', ''), ('canonical_name', 'Different Club')])
def test_invalid_decisions_fail_before_mutation(tmp_path, field, value):
    registry, catalog, row = setup(tmp_path)
    before = deepcopy(registry.payload)
    with pytest.raises(ValueError):
        apply_team_identity_decisions(registry, document([dict(row, **{field:value})]), catalog=catalog)
    assert registry.payload == before


def test_content_binding_and_badge_integrity(tmp_path):
    registry, catalog, row = setup(tmp_path)
    altered = document([row]); altered['records'][0]['end'] = '2016-01-01'
    with pytest.raises(ValueError, match='SHA-256'):
        apply_team_identity_decisions(registry, altered, catalog=catalog)
    catalog.entries['1']['sha256'] = 'bad'
    with pytest.raises(ValueError, match='badge hash'):
        apply_team_identity_decisions(registry, document([row]), catalog=catalog)


def test_exact_large_ids_and_no_identity_collision(tmp_path):
    registry, catalog, row = setup(tmp_path)
    large = dict(row, team_id=2**53+1, canonical_name='Other Club', name='Other')
    result = apply_team_identity_decisions(registry, document([large]), catalog=catalog)
    assert result.resolve('team', 'Other', 54, '2012-01-01') == 2**53+1
    registry.seed('team', 2, name='Another Team', entity_kind='club')
    with pytest.raises(ValueError, match='existing canonical identity'):
        apply_team_identity_decisions(registry, document([dict(row, team_id=2, canonical_name='University Club')]), catalog=catalog)


def test_record_order_invariance(tmp_path):
    registry, catalog, row = setup(tmp_path)
    rows = [row, dict(row, team_id=2, canonical_name='University Club', name='Second')]
    left = apply_team_identity_decisions(registry, document(rows), catalog=catalog)
    right = apply_team_identity_decisions(registry, document(rows[::-1]), catalog=catalog)
    for name in ['Murcia', 'Second']:
        assert left.resolve('team', name, 54, '2012-01-01') == right.resolve('team', name, 54, '2012-01-01')


def test_shipped_bridge_reuses_native_badges_in_portable_bundle(tmp_path):
    decisions = load_team_identity_decisions()
    ids = sorted({row['team_id'] for row in decisions['records']})
    assert len(ids) > 450 and 22 in ids
    catalog = TeamCatalog.builtin()
    path = catalog.export_bundle(tmp_path/'assets', team_ids=ids)
    portable = TeamCatalog.from_json(path)
    for identity in ids:
        assert portable.display(identity, badges=True) == catalog.display(identity, badges=True)
        assert portable.display(identity, badges=True)[1].startswith('data:image/svg+xml;base64,')


def test_shipped_murcia_alias_is_not_a_global_merge():
    rows = [r for r in load_team_identity_decisions()['records'] if r['name'] == 'Murcia' and r['competition_id'] == 54]
    assert {r['team_id'] for r in rows} == {4487, 55785}
    for a in rows:
        for b in rows:
            if a['team_id'] != b['team_id']:
                assert a['end'] < b['start'] or b['end'] < a['start']


def test_discovery_uses_fixtures_not_score_or_fuzzy_names(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    from zipfile import ZipFile
    import pandas as pd

    native = tmp_path/'native'; native.mkdir()
    rows = []
    source = []
    for index, (team, name) in enumerate([(38, 'Chelsea'), (33, 'Tottenham Hotspur'), (17, 'Manchester City')], 1):
        day = f'2020-09-{index:02}'
        rows.append(dict(event_id=index, competition_id=17, home_id=44, away_id=team,
                         home_name='Liverpool FC', away_name=name,
                         kickoff_utc=pd.Timestamp(day, tz='UTC').timestamp()))
        source.append(dict(Sea='2020-2021', Lge='ENG1', Date=f'{index:02}/09/2020',
                           HT='An unrecognizable alias', AT=name, HS=1, AS=0, GD=1, WDL='W'))
    table = native/'matches.parquet'; pd.DataFrame(rows).to_parquet(table, compression='zstd')
    manifest = dict(scope=dict(league_name='Premier_League'), tables=dict(matches=dict(
        file=table.name, sha256=hashlib.sha256(table.read_bytes()).hexdigest())))
    (native/'manifest.json').write_text(json.dumps(manifest))
    (native/'Premier_League_20_21.manifest.json').write_text(json.dumps(dict(manifest='manifest.json')))
    txt = tmp_path/'history.txt'; pd.DataFrame(source).to_csv(txt, sep='\t', index=False)
    workbook = tmp_path/'workbook.xlsx'
    fields = ['country','league','season','home_team','away_team','home_team_id','away_team_id','start_datetime']
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    headers = ''.join(f'<c r="{chr(65+i)}1" t="inlineStr"><is><t>{name}</t></is></c>' for i,name in enumerate(fields))
    # A reused source key is not proof that two countries' clubs are identical.
    workbook_rows = []
    for i, row in enumerate(rows, 1):
        serial = (pd.Timestamp(f'2020-09-{i:02}') - pd.Timestamp('1899-12-30')).days
        workbook_rows.append(['England', 'Premier League', '2020/2021',
                              'An unrecognizable alias', row['away_name'], '777', str(row['away_id']), str(serial)])
    workbook_rows.append(['Germany', 'Bundesliga', '2020/2021', 'Different German club', 'Unknown', '777', '778', '44075'])
    body = ''.join('<row r="'+str(i)+'">'+''.join(
        f'<c r="{chr(65+j)}{i}" t="inlineStr"><is><t>{value}</t></is></c>'
        for j,value in enumerate(row))+'</row>' for i,row in enumerate(workbook_rows, 2))
    with ZipFile(workbook, 'w') as z:
        z.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Matches_Results" sheetId="1" r:id="r1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr('xl/sharedStrings.xml', f'<sst xmlns="{ns}"/>')
        z.writestr('xl/worksheets/sheet1.xml', f'<worksheet xmlns="{ns}"><sheetData><row r="1">{headers}</row>{body}</sheetData></worksheet>')
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, str(root/'examples/audit_international_team_identities.py'),
               '--historical', str(txt), '--workbook', str(workbook), '--native-root', str(native)]
    env = dict(os.environ, PYTHONPATH=str(root/'src'))
    def run(name):
        output = tmp_path/name
        result = subprocess.run([*command, '--output', str(output)], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return json.loads((output/'full-reconciliation.json').read_text(encoding='utf-8'))
    first = run('first')
    found = next(row for row in first if row['source_names'] == ['An unrecognizable alias'])
    assert found['native_team_id'] == 44 and found['supporting_fixtures'] == 3
    conflict = next(row for row in first if 'Different German club' in row['source_names'])
    assert conflict['native_team_id'] is None
    # Scores are irrelevant; row order and independent source IDs cannot drive the match.
    changed = pd.DataFrame(source[::-1]); changed['HS'] = 19
    changed.to_csv(txt, sep='\t', index=False)
    second = run('second')
    assert sorted(first, key=lambda x:x['source_names']) == sorted(second, key=lambda x:x['source_names'])
    # Similar spelling alone with no matching native fixture remains unresolved.
    changed.loc[changed.HT.eq('An unrecognizable alias'), 'HT'] = 'Liverpoool FC'
    changed['Date'] = '01/01/1990'; changed.to_csv(txt, sep='\t', index=False)
    assert all(row['native_team_id'] is None for row in run('no-evidence'))
