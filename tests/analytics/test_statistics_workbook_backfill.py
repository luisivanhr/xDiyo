import copy
import importlib.util
from pathlib import Path

import pytest

from xdiyo_analytics.data.statistics_backfill import (
    fill_missing_statistics, make_candidate, statistic_value, statistics_schema,
)


def candidate(value=0, field='corners', **kwargs):
    return make_candidate(dict(event_id=1, home_id=10, away_id=20), field=field,
        period=kwargs.get('period', 'ft'), side=kwargs.get('side', 'home'), value=value,
        evidence=dict(workbook_sha256='a'*64, sheet='Statistics_FT', row=2,
                      column='home_corners_ft', vendor_match_id='900'), observed_at=123.0)


def native(value=4, **changes):
    row = candidate(value if value is not None else 0)
    row['value'] = value
    row.pop('evidence')
    row.update(source_key='native', raw_hash='original', source_item_json='original evidence', **changes)
    return row


def test_preserve_complete_observations_and_real_zero():
    original = [native(0), native(4, side='away', team_id=20)]
    result, audit = fill_missing_statistics(original, [candidate(8), candidate(9, side='away')])
    assert result == original and audit == []


def test_missing_only_side_period_and_idempotence():
    original = [native(4)]
    donors = [candidate(9), candidate(3, side='away'), candidate(2, period='1h')]
    result, audit = fill_missing_statistics(original, donors)
    assert result[0] == original[0]
    assert len(result) == 3 and len(audit) == 2
    assert {r['action'] for r in audit} == {'append'}
    assert {r['value'] for r in result} == {4, 3, 2}
    again, second = fill_missing_statistics(result, donors)
    assert again == result and second == []


def test_null_fill_preserves_native_metadata_and_display_only_values():
    original = [native(None, display=None)]
    snapshot = copy.deepcopy(original)
    result, audit = fill_missing_statistics(original, [candidate(0)])
    assert original == snapshot
    assert result[0] == {**original[0], 'value': 0, 'display': '0'}
    assert audit[0]['action'] == 'fill_null'
    display_only = [native(None, display='4')]
    assert fill_missing_statistics(display_only, [candidate(8)]) == (display_only, [])


def test_duplicate_groups_protect_existing_values_and_conflicting_donors_fail():
    original = [native(6, key='goalkeeperSaves', group_name='Goalkeeping')]
    result, audit = fill_missing_statistics(original, [candidate(2, field='goalkeeper_saves')])
    assert result == original and not audit
    with pytest.raises(ValueError, match='Conflicting donor'):
        fill_missing_statistics([], [candidate(1), candidate(2)])


def test_units_and_missing_values():
    assert statistic_value('', 'corners') is None
    assert statistic_value(None, 'corners') is None
    assert statistic_value('0', 'yellow_cards') == 0
    assert statistic_value('63', 'ball_possession') == 63
    assert statistic_value('1.25', 'xg') == 1.25
    for field, raw in [('corners', '-1'), ('corners', '0.5'), ('xg', 'nan'),
                       ('ball_possession', '101'), ('fouls', 'inf')]:
        with pytest.raises(ValueError):
            statistic_value(raw, field)


def runner():
    path = Path(__file__).resolve().parents[2]/'examples'/'backfill_statistics.py'
    spec = importlib.util.spec_from_file_location('statistics_backfill_example', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_matching_rejects_wrong_sides_dates_ambiguity():
    r = runner()
    fixture = dict(source_league='League', source_season='24_25', home_id=10, away_id=20,
                   home_name='Home', away_name='Away', event_id=1, stem='League_24_25', kickoff_utc=1735689600)
    donor = dict(source_league='League', source_season='24_25', home_team='Home', away_team='Away',
                 start_datetime='45658')
    mapped, _ = r.match_fixtures({'900': [donor]}, [fixture], {}, [])
    assert mapped['900']['event_id'] == 1
    for changed in ({'home_team': 'Away', 'away_team': 'Home'}, {'start_datetime': '45659'}, {'home_team': 'Unknown'}):
        assert not r.match_fixtures({'900': [{**donor, **changed}]}, [fixture], {}, [])[0]
    assert not r.match_fixtures({'900': [donor]}, [fixture, {**fixture, 'event_id': 2}], {}, [])[0]
    assert not r.match_fixtures({'900': [donor], '901': [donor]}, [fixture], {}, [])[0]


def test_parquet_writer_preserves_schema_metadata_and_zstd(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    r = runner()
    table = pa.table({'value': [0., None, 1.5]}).replace_schema_metadata({'version': '2'})
    path, output = tmp_path/'original.parquet', tmp_path/'new.parquet'
    pq.write_table(table, path, compression='zstd')
    info = r.write_table(table, path, output)
    assert pq.ParquetFile(output).read().equals(table, check_metadata=True)
    assert pq.ParquetFile(output).metadata.row_group(0).column(0).compression == 'ZSTD'
    assert info['sha256'] == r.digest(output)


@pytest.mark.parametrize('inconsistent,absent', [(False, False), (True, False), (False, True)])
def test_publication_keeps_nested_and_native_in_sync(tmp_path, inconsistent, absent):
    import json
    from types import SimpleNamespace
    import pyarrow as pa
    import pyarrow.parquet as pq
    r = runner()
    root = tmp_path/'data'
    root.mkdir()
    statistics_path = root/'statistics.parquet'
    original = pa.Table.from_pylist([] if absent else [native(4)], schema=statistics_schema())
    if not absent:
        pq.write_table(original, statistics_path, compression='zstd')
    nested_rows = [native(8 if inconsistent else 4)]
    nested = pa.table({'event_id': [1], 'home_name': ['Original club'],
                      'statistics': pa.array([nested_rows], type=pa.list_(pa.struct(original.schema)))})
    if absent:
        nested = nested.drop(['statistics'])
    nested_path = root/'season.parquet'
    pq.write_table(nested, nested_path, compression='zstd')
    publication_path = root/'season.manifest.json'
    r.dump(publication_path, dict(season_file='season.parquet', sha256=r.digest(nested_path)))
    manifest_path = root/'manifest.json'
    source = SimpleNamespace(path=manifest_path, manifest={'tables': {}}, table_path=lambda name: statistics_path)
    r.dump(manifest_path, source.manifest)
    merged, audit = fill_missing_statistics(original.to_pylist(), [candidate(3, side='away')])
    table = pa.Table.from_pylist(merged, schema=original.schema)
    audit_path = tmp_path/'audit.json'
    r.dump(audit_path, audit)
    before = {p: r.digest(p) for p in (statistics_path, nested_path, publication_path, manifest_path) if p.exists()}
    if inconsistent:
        with pytest.raises(ValueError, match='Nested/native statistics disagree'):
            r.publish(root, source, publication_path, table, {1: merged}, audit_path, 'abc')
        assert before == {p: r.digest(p) for p in before}
    else:
        r.publish(root, source, publication_path, table, {1: merged}, audit_path, 'abc')
        assert pq.ParquetFile(statistics_path).read().to_pylist() == merged
        updated = pq.ParquetFile(nested_path).read()
        assert updated['statistics'].to_pylist() == [merged]
        assert updated['home_name'].equals(nested['home_name'])
        assert json.loads(publication_path.read_bytes())['sha256'] == r.digest(nested_path)
        assert json.loads(manifest_path.read_bytes())['tables']['statistics']['sha256'] == r.digest(statistics_path)
        assert audit_path.with_suffix('.done.json').exists()


def test_resume_refuses_concurrent_changes_before_replacing_any_file(tmp_path):
    r = runner()
    paths = [tmp_path/'first.json', tmp_path/'second.json']
    entries, before = [], {}
    for p in paths:
        p.write_text('original')
        before[p.name] = r.digest(p)
        staged = p.with_suffix(p.suffix+'.statistics-tmp')
        staged.write_text('enriched')
        entries.append(dict(path=p.name, sha256=r.digest(staged)))
    journal = tmp_path/'update.pending.json'
    r.dump(journal, dict(files=entries, before=before))
    paths[1].write_text('other writer')
    with pytest.raises(ValueError, match='changed since staging'):
        r.finish(tmp_path, journal)
    assert paths[0].read_text() == 'original'
    assert paths[1].read_text() == 'other writer'
