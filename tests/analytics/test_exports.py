"""Saved selections and stable publication selection through the public API."""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from xdiyo_analytics.data import inspect_season, load_season, load_season_table

STEM = 'Premier_League_24_25'


def test_legacy_version_one_record_replays_when_current_publication_is_broken(export, tmp_path):
    record = tmp_path / 'legacy.json'
    export.legacy_record(record)
    expected_record = record.read_bytes()
    export.publication_path.write_text('broken current pointer', encoding='utf-8')
    season = load_season(export.root, STEM, tables=['matches', 'statistics'], record_path=record)
    assert season.provenance['version'] == 'v1'
    assert inspect_season(export.root, STEM, record_path=record).loc['matches', 'rows'] == 2
    assert record.read_bytes() == expected_record


@pytest.mark.parametrize('creator', ['bundle', 'single'])
def test_new_records_pin_version_work_with_both_loaders_and_preserve_source(export, tmp_path, creator):
    record = tmp_path / 'experiment/selection.json'
    before = export.snapshot()
    if creator == 'bundle':
        load_season(export.root, STEM, tables=['matches', 'statistics'], record_path=record)
    else:
        load_season_table(export.root, STEM, record_path=record)
    assert export.snapshot() == before
    payload = record.read_bytes()
    assert json.loads(payload)['schema_version'] == 1
    export.advance()
    assert load_season(export.root, STEM).provenance['version'] == 'v2'
    bundle = load_season(export.root, STEM, tables=['matches', 'statistics'], record_path=record)
    single = load_season_table(export.root, STEM, table='statistics', record_path=record)
    pd.testing.assert_frame_equal(single, bundle.statistics)
    assert bundle.provenance['version'] == 'v1'
    assert record.read_bytes() == payload


def test_pointer_change_between_reads_does_not_mix_versions(export, monkeypatch):
    original_read, original_open, original_bytes = pq.ParquetFile.read, Path.open, Path.read_bytes
    opened, publication_reads = [], []

    def tracked_open(path, *args, **kwargs):
        if path.suffix == '.parquet':
            opened.append(path)
        return original_open(path, *args, **kwargs)

    def tracked_bytes(path):
        if path == export.publication_path:
            publication_reads.append(path)
        return original_bytes(path)

    def change_pointer(parquet, *args, **kwargs):
        table = original_read(parquet, *args, **kwargs)
        if len(opened) == 1:
            # Ignore synthetic publisher I/O when measuring the loader's reads.
            with monkeypatch.context() as setup:
                setup.setattr(Path, 'open', original_open)
                setup.setattr(Path, 'read_bytes', original_bytes)
                export.advance()
        return table

    monkeypatch.setattr(Path, 'open', tracked_open)
    monkeypatch.setattr(Path, 'read_bytes', tracked_bytes)
    monkeypatch.setattr(pq.ParquetFile, 'read', change_pointer)
    season = load_season(export.root, STEM, tables=['statistics', 'pregame', 'matches'])
    assert len(publication_reads) == 1 and len(opened) == 3
    assert all(path.parent == export.version for path in opened)
    assert season.provenance['version'] == 'v1'


def test_changed_pinned_manifest_fails_without_refreshing_record(export, tmp_path):
    record = tmp_path / 'selection.json'
    load_season(export.root, STEM, record_path=record)
    payload = record.read_bytes()
    export.advance()
    export.manifest_path.write_bytes(export.manifest_path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='saved season manifest has changed'):
        load_season(export.root, STEM, record_path=record)
    assert record.read_bytes() == payload


def test_record_must_stay_outside_source(export):
    with pytest.raises(ValueError, match='outside the source'):
        load_season(export.root, STEM, record_path=export.root / 'new/selection.json')
    assert not (export.root / 'new').exists()


def test_escaped_table_path_is_rejected(export):
    export.change_manifest(lambda m: m['tables']['shots'].update(file='../shots.parquet'))
    with pytest.raises(ValueError, match='inside the selected version'):
        load_season(export.root, STEM, tables='shots')
