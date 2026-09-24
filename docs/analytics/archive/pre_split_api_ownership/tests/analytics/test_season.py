"""Metadata-only season discovery, physical schemas and readable descriptions."""

import pyarrow.parquet as pq
import pytest

from xdiyo_analytics.data import inspect_season, load_season

STEM = 'Premier_League_24_25'


def test_catalogue_is_metadata_only_and_preserves_physical_schemas(export, monkeypatch):
    before = export.snapshot()
    monkeypatch.setattr(pq.ParquetFile, 'read', lambda *a, **k: pytest.fail('Inspection decoded rows'))
    monkeypatch.setattr(pq, 'read_table', lambda *a, **k: pytest.fail('Inspection decoded rows'))
    catalogue = inspect_season(export.root, STEM)
    assert catalogue.columns.tolist() == ['rows', 'column_count', 'description', 'columns', 'column_types', 'column_summary']
    assert catalogue.index.tolist() == ['future_context', 'matches', 'pregame', 'shots', 'statistics']
    assert catalogue.loc['shots', 'rows'] == 3
    assert catalogue.loc['shots', 'columns'] == ['event_id', 'shot_id', 'xg', 'flag', 'details']
    assert catalogue.loc['shots', 'column_types']['shot_id'] == 'uint64'
    assert 'list<element: struct<x: int64>>' in catalogue.loc['shots', 'column_summary']
    assert 'Individual shots' in catalogue.loc['shots', 'description']
    assert 'position' in catalogue.loc['pregame', 'description']
    assert 'not a complete league standings table' in catalogue.loc['pregame', 'description']
    assert catalogue.loc['future_context', 'rows'] == 0
    assert 'No description documented' in catalogue.loc['future_context', 'description']
    assert export.snapshot() == before


def test_inspection_replays_existing_record_and_never_creates_one(export, tmp_path):
    missing = tmp_path / 'absent/selection.json'
    with pytest.raises(FileNotFoundError):
        inspect_season(export.root, STEM, record_path=missing)
    assert not missing.parent.exists()
    record = tmp_path / 'selection.json'
    load_season(export.root, STEM, record_path=record)
    payload = record.read_bytes()
    export.publication_path.write_text('invalid current publication', encoding='utf-8')
    assert inspect_season(export.root, STEM, record_path=record).attrs['source']['version'] == 'v1'
    assert record.read_bytes() == payload


@pytest.mark.parametrize('kind,message', [('size', 'File size'), ('rows', 'Row count'), ('footer', 'Cannot inspect Parquet')])
def test_broken_catalogue_metadata_is_clear(export, kind, message):
    path = export.version / 'pregame.parquet'
    if kind == 'size':
        path.write_bytes(path.read_bytes() + b'!')
    elif kind == 'rows':
        export.change_manifest(lambda m: m['tables']['pregame'].update(rows=100))
    else:
        path.write_bytes(b'!' * path.stat().st_size)
    with pytest.raises(ValueError, match=message):
        inspect_season(export.root, STEM)
