"""Match validation, simple loading and automatic selection replay contracts."""

import base64
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from xdiyo_analytics.data import inspect_season, load_season_table, validate_matches
from xdiyo_analytics.data.exports import resolve_season_export


STEM = "Premier_League_24_25"


@pytest.fixture
def season(tmp_path):
    root = tmp_path / "exports"
    version_dir = root / "_tables/competition=17/season=61627/versions/v1"
    version_dir.mkdir(parents=True)
    tables = {
        "matches": pa.table({
            "event_id": [9007199254740993, 7], "home_id": [50, 42], "away_id": [42, 50],
            "competition_id": [17, 17], "season_id": [61627, 61627],
            "kickoff_utc": [1735752600.0, 1736017200.0],
            "optional_flag": pa.array([None, False], type=pa.bool_()),
        }),
        "shots": pa.table({"event_id": [7, 7, 7], "shot_index": [0, 1, 2]}),
    }
    descriptors = {}
    for name, table in tables.items():
        path = version_dir / f"{name}.parquet"
        pq.write_table(table, path)
        payload = path.read_bytes()
        descriptors[name] = {"file": path.name, "rows": table.num_rows,
                             "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    manifest = {
        "schema_version": "2", "parser_version": "2", "version": "v1",
        "scope": {"league_id": 17, "league_name": "Premier_League", "season_id": 61627,
                  "season_start": 2024, "season_end": 2025}, "tables": descriptors,
    }
    (version_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    publication = {"schema_version": "2", "complete": True, "matches": 2,
                   "season_file": STEM + ".parquet",
                   "manifest": (version_dir / "manifest.json").relative_to(root).as_posix()}
    (root / (STEM + ".manifest.json")).write_text(json.dumps(publication), encoding="utf-8")
    return root


@pytest.fixture
def match_inputs(season):
    resolved = resolve_season_export(season, STEM)
    from xdiyo_analytics.data.exports import read_season_table
    return read_season_table(resolved, "matches"), resolved


def test_validator_preserves_rows_types_and_optional_missingness(match_inputs):
    frame, resolved = match_inputs
    before = frame.copy(deep=True)
    report = validate_matches(frame, resolved)
    assert report.rows == 2
    assert report.missing_kickoff_event_ids == report.invalid_kickoff_event_ids == ()
    pd.testing.assert_frame_equal(frame, before)
    assert frame["event_id"].iloc[0] == 9007199254740993
    assert pd.isna(frame["optional_flag"].iloc[0])
    assert frame["optional_flag"].iloc[1] == False


@pytest.mark.parametrize("column", ["event_id", "home_id", "away_id", "competition_id", "season_id", "kickoff_utc"])
def test_required_columns_fail_explicitly(match_inputs, column):
    frame, resolved = match_inputs
    with pytest.raises(ValueError, match="missing required columns"):
        validate_matches(frame.drop(columns=column), resolved)


@pytest.mark.parametrize("bad_id", [None, pd.NA, True, 0, -1, 7.5, 7.0, "7"])
def test_invalid_ids_are_not_coerced(match_inputs, bad_id):
    frame, resolved = match_inputs
    frame["home_id"] = pd.Series([bad_id, 42], dtype=object)
    with pytest.raises(ValueError, match="home_id.*invalid positive integer IDs"):
        validate_matches(frame, resolved)


@pytest.mark.parametrize("column,values,message", [
    ("event_id", [7, 7], "event_id must be unique"),
    ("home_id", [42, 42], "home_id and away_id must differ"),
    ("competition_id", [17, 8], "competition_id must match"),
    ("season_id", [61627, 1], "season_id must match"),
])
def test_structural_inconsistency_rejected(match_inputs, column, values, message):
    frame, resolved = match_inputs
    frame[column] = values
    with pytest.raises(ValueError, match=message):
        validate_matches(frame, resolved)


def test_duplicate_column_labels_rejected(match_inputs):
    frame, resolved = match_inputs
    with pytest.raises(ValueError, match="repeated column labels"):
        validate_matches(pd.concat([frame, frame[["home_id"]]], axis=1), resolved)


@pytest.mark.parametrize("value", [None, pd.NA, float("nan"), pd.NaT])
def test_missing_timing_reported_with_exact_event_id(match_inputs, value):
    frame, resolved = match_inputs
    frame["kickoff_utc"] = pd.Series([value, 1736017200.0], dtype=object)
    before = frame.copy(deep=True)
    report = validate_matches(frame, resolved)
    assert report.missing_kickoff_event_ids == (9007199254740993,)
    assert report.invalid_kickoff_event_ids == ()
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("value", [True, "1735752600", float("inf"), -1, 0, 10**20, 10**400, [1]])
def test_invalid_timing_reported_without_repair(match_inputs, value):
    frame, resolved = match_inputs
    frame["kickoff_utc"] = pd.Series([value, 1736017200.0], dtype=object)
    assert validate_matches(frame, resolved).invalid_kickoff_event_ids == (9007199254740993,)


def test_empty_and_optional_column_absence_are_valid(match_inputs):
    frame, resolved = match_inputs
    assert validate_matches(frame.drop(columns="optional_flag"), resolved).rows == 2
    assert validate_matches(frame.iloc[:0], resolved).rows == 0


def test_invalid_input_types(match_inputs):
    frame, resolved = match_inputs
    with pytest.raises(TypeError, match="DataFrame"):
        validate_matches([], resolved)
    with pytest.raises(TypeError, match="ResolvedSeason"):
        validate_matches(frame, None)


def test_simple_call_is_read_only_and_attaches_provenance(season):
    before = {p.relative_to(season): p.read_bytes() for p in season.rglob("*") if p.is_file()}
    frame = load_season_table(season, STEM)
    assert frame.shape == (2, 7)
    assert frame.attrs["xdiyo"]["validation"]["status"] == "structure_valid"
    assert frame.attrs["xdiyo"]["source"]["manifest"]["version"] == "v1"
    assert before == {p.relative_to(season): p.read_bytes() for p in season.rglob("*") if p.is_file()}


def test_shots_remain_separate_and_do_not_receive_match_grain_validation(season):
    frame = load_season_table(season, STEM, table="shots")
    assert frame.shape == (3, 2)
    assert frame["event_id"].tolist() == [7, 7, 7]
    assert frame.attrs["xdiyo"]["validation"]["status"] == "not_implemented"


def _replace_match_table(root, change):
    resolved = resolve_season_export(root, STEM)
    path = resolved.manifest_path.parent / "matches.parquet"
    frame = pd.read_parquet(path)
    change(frame)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    payload = path.read_bytes()
    manifest = resolved.manifest
    manifest["tables"]["matches"].update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    resolved.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_loader_warns_and_retains_missing_timing(season):
    _replace_match_table(season, lambda frame: frame.__setitem__("kickoff_utc", [None, float("inf")]))
    with pytest.warns(UserWarning, match="1 missing and 1 invalid"):
        frame = load_season_table(season, STEM)
    assert len(frame) == 2
    assert frame.attrs["xdiyo"]["validation"]["missing_kickoff_event_ids"] == (9007199254740993,)
    assert frame.attrs["xdiyo"]["validation"]["invalid_kickoff_event_ids"] == (7,)


def test_failed_validation_does_not_create_record_or_parent(season, tmp_path):
    _replace_match_table(season, lambda frame: frame.__setitem__("event_id", [7, 7]))
    record = tmp_path / "experiment/selection.json"
    with pytest.raises(ValueError, match="event_id must be unique"):
        load_season_table(season, STEM, record_path=record)
    assert not record.parent.exists()


def test_record_replays_old_version_after_publication_advances(season, tmp_path):
    record = tmp_path / "experiment/selection.json"
    first = load_season_table(season, STEM, record_path=record)
    original_record = record.read_bytes()
    original = resolve_season_export(season, STEM)
    newer = original.manifest_path.parent.parent / "v2"
    shutil.copytree(original.manifest_path.parent, newer)
    manifest = json.loads((newer / "manifest.json").read_bytes())
    manifest["version"] = "v2"
    (newer / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    publication = original.publication
    publication["manifest"] = (newer / "manifest.json").relative_to(season).as_posix()
    original.publication_path.write_text(json.dumps(publication), encoding="utf-8")
    assert load_season_table(season, STEM).attrs["xdiyo"]["source"]["manifest"]["version"] == "v2"
    replay = load_season_table(season, STEM, record_path=record)
    pd.testing.assert_frame_equal(first, replay)
    assert replay.attrs == first.attrs
    original.publication_path.unlink()
    shots = load_season_table(season, STEM, table="shots", record_path=record)
    assert shots.attrs["xdiyo"]["source"]["manifest"]["version"] == "v1"
    assert record.read_bytes() == original_record


def test_record_can_replay_from_relocated_source(season, tmp_path):
    record = tmp_path / "selection.json"
    load_season_table(season, STEM, record_path=record)
    relocated = tmp_path / "relocated"
    shutil.copytree(season, relocated)
    assert len(load_season_table(relocated, STEM, record_path=record)) == 2


@pytest.mark.parametrize("target", ["manifest", "matches"])
def test_record_rejects_changed_pinned_files(season, tmp_path, target):
    record = tmp_path / "selection.json"
    load_season_table(season, STEM, record_path=record)
    resolved = resolve_season_export(season, STEM)
    path = resolved.manifest_path if target == "manifest" else resolved.manifest_path.parent / "matches.parquet"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed|byte count"):
        load_season_table(season, STEM, record_path=record)


def test_record_never_falls_back_when_pinned_manifest_disappears(season, tmp_path):
    record = tmp_path / "selection.json"
    load_season_table(season, STEM, record_path=record)
    resolve_season_export(season, STEM).manifest_path.unlink()
    with pytest.raises(FileNotFoundError):
        load_season_table(season, STEM, record_path=record)


@pytest.mark.parametrize("change", [
    lambda record: record.update(schema_version=True),
    lambda record: record.update(season_stem="La_Liga_24_25"),
    lambda record: record.update(publication_base64="not base64!"),
    lambda record: record.update(publication_sha256="0" * 64),
    lambda record: record.update(manifest_sha256="0" * 64),
])
def test_malformed_or_mismatched_records_fail(season, tmp_path, change):
    record_path = tmp_path / "selection.json"
    load_season_table(season, STEM, record_path=record_path)
    record = json.loads(record_path.read_bytes())
    change(record)
    payload = json.dumps(record)
    record_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_season_table(season, STEM, record_path=record_path)
    assert record_path.read_text(encoding="utf-8") == payload


def test_saved_snapshot_still_receives_resolver_scope_checks(season, tmp_path):
    path = tmp_path / "selection.json"
    load_season_table(season, STEM, record_path=path)
    record = json.loads(path.read_bytes())
    publication = json.loads(base64.b64decode(record["publication_base64"]))
    publication["season_file"] = "La_Liga_24_25.parquet"
    payload = json.dumps(publication).encode()
    record.update(publication_base64=base64.b64encode(payload).decode(),
                  publication_sha256=hashlib.sha256(payload).hexdigest())
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="season_file does not match"):
        load_season_table(season, STEM, record_path=path)


def test_record_cannot_write_into_source_directory(season):
    with pytest.raises(ValueError, match="outside data_root"):
        load_season_table(season, STEM, record_path=season / "selection.json")


def test_inspection_lists_physical_columns_without_reading_observations(season, monkeypatch):
    before = {p: p.read_bytes() for p in season.rglob("*") if p.is_file()}

    def forbid_row_read(*args, **kwargs):
        raise AssertionError("Inspection must not decode observations")

    monkeypatch.setattr(pq.ParquetFile, "read", forbid_row_read)
    monkeypatch.setattr(pq, "read_table", forbid_row_read)
    catalogue = inspect_season(season, STEM)
    assert catalogue.index.tolist() == ["matches", "shots"]
    assert catalogue.loc["shots", "columns"] == ["event_id", "shot_index"]
    assert catalogue.loc["shots", "declared_rows"] == 3
    assert catalogue.loc["shots", "column_types"] == {"event_id": "int64", "shot_index": "int64"}
    assert catalogue.loc["shots", "column_summary"] == "event_id: int64, shot_index: int64"
    assert catalogue.loc["matches", "column_count"] == 7
    assert "Individual shots" in catalogue.loc["shots", "description"]
    assert catalogue.attrs["xdiyo"]["inspection"]["table_hashes_verified"] is False
    assert catalogue.attrs["xdiyo"]["inspection"]["row_values_read"] is False
    assert before == {p: p.read_bytes() for p in season.rglob("*") if p.is_file()}


def test_inspection_reuses_record_without_publication_or_writes(season, tmp_path):
    path = tmp_path / "selection.json"
    frame = load_season_table(season, STEM, record_path=path)
    before = path.read_bytes()
    (season / (STEM + ".manifest.json")).unlink()
    catalogue = inspect_season(season, STEM, record_path=path)
    assert catalogue.attrs["xdiyo"]["source"]["manifest_sha256"] == frame.attrs["xdiyo"]["source"]["manifest_sha256"]
    assert path.read_bytes() == before


def test_inspection_requires_existing_record_and_creates_nothing(season, tmp_path):
    path = tmp_path / "absent/selection.json"
    with pytest.raises(FileNotFoundError):
        inspect_season(season, STEM, record_path=path)
    assert not path.parent.exists()


def test_inspection_rejects_record_inside_source(season):
    with pytest.raises(ValueError, match="outside data_root"):
        inspect_season(season, STEM, record_path=season / "selection.json")


def test_inspection_keeps_unknown_empty_tables_and_nested_schema(season):
    resolved = resolve_season_export(season, STEM)
    table = pa.table({"details": pa.array([], type=pa.list_(pa.struct([("code", pa.int64())]))),
                      "known": pa.array([], type=pa.bool_())})
    path = resolved.manifest_path.parent / "future_context.parquet"
    pq.write_table(table, path)
    payload = path.read_bytes()
    manifest = resolved.manifest
    manifest["tables"]["future_context"] = {"file": path.name, "rows": 0, "bytes": len(payload),
                                             "sha256": hashlib.sha256(payload).hexdigest()}
    resolved.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    catalogue = inspect_season(season, STEM)
    assert catalogue.loc["future_context", "columns"] == ["details", "known"]
    # Parquet stores this list child as "element", rather than Arrow's default "item".
    assert catalogue.loc["future_context", "column_types"]["details"] == "list<element: struct<code: int64>>"
    assert catalogue.loc["future_context", "declared_rows"] == 0
    assert "No description documented" in catalogue.loc["future_context", "description"]


@pytest.mark.parametrize("change,message", [("missing", "missing"), ("size", "byte count"),
                                            ("footer", "Parquet metadata"), ("rows", "footer row count")])
def test_inspection_rejects_unreadable_or_inconsistent_metadata(season, change, message):
    resolved = resolve_season_export(season, STEM)
    path = resolved.manifest_path.parent / "shots.parquet"
    if change == "missing":
        path.unlink()
        with pytest.raises(FileNotFoundError):
            inspect_season(season, STEM)
        return
    if change == "size":
        path.write_bytes(path.read_bytes() + b"!")
    elif change == "footer":
        path.write_bytes(b"!" * path.stat().st_size)
    else:
        manifest = resolved.manifest
        manifest["tables"]["shots"]["rows"] = 100
        resolved.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        inspect_season(season, STEM)


def test_inspection_does_not_claim_data_page_integrity(season):
    resolved = resolve_season_export(season, STEM)
    path = resolved.manifest_path.parent / "shots.parquet"
    with pq.ParquetFile(path) as parquet:
        chunk = parquet.metadata.row_group(0).column(0)
        offset = chunk.dictionary_page_offset or chunk.data_page_offset
    payload = bytearray(path.read_bytes())
    payload[offset] ^= 1  # Same size; metadata footer is untouched.
    path.write_bytes(payload)
    catalogue = inspect_season(season, STEM)
    assert catalogue.loc["shots", "declared_rows"] == 3
    assert catalogue.attrs["xdiyo"]["inspection"]["table_hashes_verified"] is False
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_season_table(season, STEM, table="shots")


def test_inspection_rejects_changed_pinned_manifest(season, tmp_path):
    path = tmp_path / "selection.json"
    load_season_table(season, STEM, record_path=path)
    manifest_path = resolve_season_export(season, STEM).manifest_path
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Pinned canonical manifest changed"):
        inspect_season(season, STEM, record_path=path)
