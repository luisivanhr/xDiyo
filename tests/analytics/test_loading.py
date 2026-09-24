"""Everyday table loading, exact data preservation and optional audit checks."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from xdiyo_analytics.data import load_season, load_season_table

STEM = 'Premier_League_24_25'


def test_separate_requested_tables_and_shortcut(export):
    before = export.snapshot()
    season = load_season(export.root, STEM, tables=['pregame', 'statistics', 'matches'])
    assert tuple(season.tables) == ('pregame', 'statistics', 'matches')
    assert season.matches is season['matches']
    assert season.statistics is season['statistics']
    assert season.shots is None
    assert season.provenance['tables'] == tuple(season.tables)
    assert season.matches['event_id'].tolist() == [2**53 + 1, 7]
    assert season.matches['optional_flag'].tolist() == [pd.NA, False]
    single = load_season_table(export.root, STEM, table='statistics')
    pd.testing.assert_frame_equal(single, season.statistics)
    assert export.snapshot() == before


@pytest.mark.parametrize('audit', [False, True])
def test_exact_nullable_unsigned_nested_values_and_row_order(export, audit):
    before = export.snapshot()
    source = export.table('shots')
    frame = load_season(export.root, STEM, tables='shots', verify_hashes=audit).shots
    pd.testing.assert_frame_equal(frame, source.to_pandas(types_mapper=pd.ArrowDtype, ignore_metadata=True))
    assert frame['shot_id'].tolist() == [2**63 + 9, 2**63 + 10, 2**63 + 11]
    assert frame['xg'].array.__arrow_array__().null_count == 1
    assert frame.index.equals(pd.RangeIndex(3))
    assert export.snapshot() == before


def test_selected_only_reads_and_requested_empty_table(export):
    export.change_manifest(lambda m: m['tables']['matches'].update(file='absent.parquet'))
    export.replace('shots', export.table('shots').slice(0, 0))
    season = load_season(export.root, STEM, tables='shots')
    assert season.matches is None and isinstance(season.shots, pd.DataFrame)
    assert season.shots.empty and season.shots.columns.tolist() == ['event_id', 'shot_id', 'xg', 'flag', 'details']


@pytest.mark.parametrize('names', [[], None, {'matches'}, [''], ['matches', 'matches']])
def test_invalid_selection_is_clear_before_source_access(tmp_path, names):
    with pytest.raises(ValueError, match='Choose'):
        load_season(tmp_path / 'absent', STEM, tables=names)


def test_unavailable_late_name_prechecked_without_reading_tables(export, tmp_path, monkeypatch):
    monkeypatch.setattr(pq, 'ParquetFile', lambda *a, **k: pytest.fail('Read before availability precheck'))
    record = tmp_path / 'new/selection.json'
    with pytest.raises(KeyError, match='unavailable; available:'):
        load_season(export.root, STEM, tables=['matches', 'standings'], record_path=record)
    assert not record.parent.exists()


@pytest.mark.parametrize('kind,message', [
    ('missing', ''), ('size', 'File size'), ('rows', 'Row count'), ('parquet', 'Cannot read Parquet'),
])
def test_late_broken_table_creates_no_record_or_parent(export, tmp_path, kind, message):
    path = export.version / 'pregame.parquet'
    if kind == 'missing':
        export.change_manifest(lambda m: m['tables']['pregame'].update(file='absent.parquet'))
    elif kind == 'size':
        path.write_bytes(path.read_bytes() + b'!')
    elif kind == 'rows':
        export.change_manifest(lambda m: m['tables']['pregame'].update(rows=100))
    else:
        path.write_bytes(b'!' * path.stat().st_size)
    before = export.snapshot()
    record = tmp_path / 'unsaved/selection.json'
    error = FileNotFoundError if kind == 'missing' else ValueError
    with pytest.raises(error, match=message or None):
        load_season(export.root, STEM, tables=['matches', 'pregame'], record_path=record)
    assert not record.parent.exists()
    assert export.snapshot() == before


def test_full_table_fingerprint_verification_is_opt_in(export):
    export.change_manifest(lambda m: m['tables']['shots'].update(sha256='0' * 64))
    assert len(load_season(export.root, STEM, tables='shots').shots) == 3
    with pytest.raises(ValueError, match='File fingerprint.*shots'):
        load_season(export.root, STEM, tables='shots', verify_hashes=True)


def test_missing_timing_warns_and_preserves_rows(export):
    table = export.table('matches')
    table = table.set_column(table.schema.get_field_index('kickoff_utc'), 'kickoff_utc', pa.array([None, 1736017200.0]))
    export.replace('matches', table)
    with pytest.warns(UserWarning, match='rows retained'):
        matches = load_season(export.root, STEM).matches
    assert matches['event_id'].tolist() == [2**53 + 1, 7]
    assert pd.isna(matches['kickoff_utc'].iloc[0])


def test_wrong_match_scope_fails_without_saving(export, tmp_path):
    table = export.table('matches')
    export.replace('matches', table.set_column(table.schema.get_field_index('season_id'), 'season_id', pa.array([1, 1])))
    record = tmp_path / 'unsaved/selection.json'
    with pytest.raises(ValueError, match='different competition or season'):
        load_season(export.root, STEM, record_path=record)
    assert not record.parent.exists()
