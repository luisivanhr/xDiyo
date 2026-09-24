"""Resolver and single-table reader contracts using tiny local publications."""

import hashlib
import json
from pathlib import Path

import pytest
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from xdiyo_analytics.data.exports import read_season_table, resolve_season_export


STEM = "Premier_League_24_25"


@pytest.fixture
def publication_files(tmp_path):
    root = tmp_path / "exports"
    manifest_path = (root / "_tables/competition=17/season=61627/versions/v1"
                     / "manifest.json")
    manifest_path.parent.mkdir(parents=True)
    manifest = {
        "schema_version": "2", "parser_version": "2", "version": "v1",
        "scope": {"league_id": 17, "league_name": "Premier_League",
                  "season_id": 61627, "season_start": 2024, "season_end": 2025},
        "tables": {"matches": {"file": "matches.parquet", "rows": 380,
                               "bytes": 123, "sha256": "a" * 64}},
        "provenance": {"collection": "test", "optional_evidence": None},
    }
    publication = {
        "complete": True, "schema_version": "2", "matches": 380,
        "season_file": STEM + ".parquet",
        "manifest": manifest_path.relative_to(root).as_posix(),
    }
    publication_path = root / (STEM + ".manifest.json")
    publication_path.write_text(json.dumps(publication), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, publication_path, manifest_path


def test_resolves_metadata_without_parquet_and_preserves_unknowns(publication_files):
    root, outer_path, inner_path = publication_files
    resolved = resolve_season_export(root, STEM)
    assert resolved.publication_path == outer_path.resolve()
    assert resolved.manifest_path == inner_path.resolve()
    assert resolved.publication_sha256 == hashlib.sha256(outer_path.read_bytes()).hexdigest()
    assert resolved.manifest_sha256 == hashlib.sha256(inner_path.read_bytes()).hexdigest()
    assert resolved.manifest["scope"]["season_id"] == 61627
    assert resolved.manifest["provenance"]["optional_evidence"] is None
    assert "team_seasons" not in resolved.manifest["tables"]
    assert not list(root.rglob("*.parquet"))  # No table files are needed or created.


def test_existing_reference_stays_pinned_when_publication_advances(publication_files):
    root, outer_path, inner_path = publication_files
    first = resolve_season_export(root, STEM)
    second_path = inner_path.parent.parent / "v2/manifest.json"
    second_path.parent.mkdir()
    metadata = json.loads(inner_path.read_text())
    metadata["version"] = "v2"
    second_path.write_text(json.dumps(metadata), encoding="utf-8")
    publication = json.loads(outer_path.read_text())
    publication["manifest"] = second_path.relative_to(root).as_posix()
    outer_path.write_text(json.dumps(publication), encoding="utf-8")
    second = resolve_season_export(root, STEM)
    assert first.manifest_path == inner_path
    assert first.manifest["version"] == "v1"
    assert first.publication["manifest"].endswith("v1/manifest.json")
    assert second.manifest_path == second_path
    assert first.publication_sha256 != second.publication_sha256


def test_ignores_unreferenced_versions_and_current_pointer(publication_files):
    root, _, inner_path = publication_files
    other = inner_path.parent.parent / "newer/manifest.json"
    other.parent.mkdir()
    other.write_text("broken unreferenced JSON", encoding="utf-8")
    (inner_path.parents[2] / "CURRENT.json").write_text("broken CURRENT", encoding="utf-8")
    assert resolve_season_export(root, STEM).manifest_path == inner_path


def test_preserves_later_optional_table_descriptor(publication_files):
    root, _, path = publication_files
    data = json.loads(path.read_text())
    data["tables"]["team_seasons"] = {
        "file": "team_seasons.parquet", "rows": 20, "bytes": 40, "sha256": "b" * 64,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    resolved = resolve_season_export(root, STEM)
    assert resolved.manifest["tables"]["team_seasons"] == data["tables"]["team_seasons"]


@pytest.mark.parametrize("stem", ["", "../Premier_League_24_25", "x/y_24_25",
                                     "Premier_League_24_25.parquet", "Premier_League", None])
def test_invalid_selection_fails_before_file_access(tmp_path, stem):
    with pytest.raises(ValueError, match="filename stem"):
        resolve_season_export(tmp_path, stem)


@pytest.mark.parametrize("which", [1, 2])
def test_missing_json_file_is_explicit(publication_files, which):
    publication_files[which].unlink()
    with pytest.raises(FileNotFoundError):
        resolve_season_export(publication_files[0], STEM)


@pytest.mark.parametrize("which", [1, 2])
@pytest.mark.parametrize("content", [b"{broken", b"[]", b"\xff"])
def test_invalid_json_reports_the_file(publication_files, which, content):
    path = publication_files[which]
    path.write_bytes(content)
    with pytest.raises(ValueError) as error:
        resolve_season_export(publication_files[0], STEM)
    assert str(path) in str(error.value)


@pytest.mark.parametrize("key,value,message", [
    ("complete", False, "not complete"),
    ("complete", 1, "not complete"),
    ("schema_version", "99", "schema_version"),
    ("season_file", "Another_24_25.parquet", "season_file"),
    ("manifest", None, "relative path"),
    ("manifest", "../outside.json", "outside data_root"),
    ("matches", 379, "row declarations disagree"),
])
def test_invalid_publication(publication_files, key, value, message):
    root, path, _ = publication_files
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        resolve_season_export(root, STEM)


@pytest.mark.parametrize("key,value,message", [
    ("schema_version", "99", "schema_version"),
    ("scope", None, "scope object"),
    ("version", "v2", "scope IDs and version"),
    ("parser_version", None, "parser_version"),
    ("tables", {}, "matches descriptor"),
])
def test_invalid_canonical_manifest(publication_files, key, value, message):
    root, _, path = publication_files
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        resolve_season_export(root, STEM)


@pytest.mark.parametrize("key,value,message", [
    ("league_name", "La_Liga", "requested league/season"),
    ("season_start", 2023, "requested league/season"),
    ("season_end", 2125, "requested league/season"),
    ("league_id", True, "positive integer"),
    ("season_id", 61628, "scope IDs and version"),
])
def test_mismatched_scope(publication_files, key, value, message):
    root, _, path = publication_files
    data = json.loads(path.read_text())
    data["scope"][key] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        resolve_season_export(root, STEM)


@pytest.mark.parametrize("key,value,message", [
    ("file", "../matches.parquet", "local Parquet filename"),
    ("rows", -1, "nonnegative integer"),
    ("bytes", True, "nonnegative integer"),
    ("sha256", "not-a-hash", "SHA256 hex digest"),
])
def test_unusable_table_descriptor(publication_files, key, value, message):
    root, _, path = publication_files
    data = json.loads(path.read_text())
    data["tables"]["matches"][key] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        resolve_season_export(root, STEM)


def _store_test_table(publication_files, table, name="matches"):
    """Write a tiny fixture and update its publication declarations."""
    root, publication_path, manifest_path = publication_files
    table_path = manifest_path.parent / f"{name}.parquet"
    pq.write_table(table, table_path)
    data = table_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tables"][name] = {"file": table_path.name, "rows": table.num_rows,
                              "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    if name == "matches":
        publication = json.loads(publication_path.read_text(encoding="utf-8"))
        publication["matches"] = table.num_rows
        publication_path.write_text(json.dumps(publication), encoding="utf-8")
    return resolve_season_export(root, STEM)


@pytest.fixture
def table_reference(publication_files):
    table = pa.table({
        "event_id": pa.array([9007199254740993, None, 7], type=pa.int64()),
        "kickoff_utc": pa.array([1735752600.0, None, 1735752601.0], type=pa.float64()),
        "got_promoted": pa.array([True, None, False], type=pa.bool_()),
        "value": pa.array([0.0, None, float("nan")], type=pa.float64()),
        "label": pa.array(["", None, "sample"], type=pa.string()),
        "details": pa.array([[], None, [{"value": 3}]],
                            type=pa.list_(pa.struct([("value", pa.int64())]))),
    })
    return _store_test_table(publication_files, table), table


def test_table_roundtrip_preserves_order_types_values_and_nulls(table_reference):
    resolved, original = table_reference
    frame = read_season_table(resolved, "matches")
    assert frame.shape == (3, 6)
    assert list(frame.columns) == original.column_names
    assert frame.index.equals(pd.RangeIndex(3))
    for field in original.schema:
        assert frame[field.name].dtype == pd.ArrowDtype(field.type)
        actual = pa.array(frame[field.name])
        expected = original[field.name].combine_chunks()
        assert actual.is_null().to_pylist() == expected.is_null().to_pylist()
        if field.name != "value":
            assert actual.equals(expected)
    # Large IDs survive exactly; valid NaN, null, zero and False stay distinct.
    assert frame.loc[0, "event_id"] == 9007199254740993
    assert frame.loc[0, "value"] == 0.0
    assert frame.loc[1, "value"] is pd.NA
    assert pa.array(frame["value"]).is_nan().to_pylist() == [False, None, True]
    assert frame.loc[1, "got_promoted"] is pd.NA
    assert frame.loc[2, "got_promoted"] == False
    assert "competition" not in frame.columns and "season" not in frame.columns


def test_reading_is_independent_and_does_not_write_sources(table_reference):
    resolved, _ = table_reference
    paths = [resolved.publication_path, resolved.manifest_path,
             resolved.manifest_path.parent / "matches.parquet"]
    before = {p: p.read_bytes() for p in paths}
    first = read_season_table(resolved, "matches")
    first.loc[0, "event_id"] = 1
    assert read_season_table(resolved, "matches").loc[0, "event_id"] == 9007199254740993
    assert {p: p.read_bytes() for p in paths} == before


def test_unknown_optional_table_raises_instead_of_inventing_data(table_reference):
    resolved, _ = table_reference
    with pytest.raises(KeyError, match="team_seasons.*not declared"):
        read_season_table(resolved, "team_seasons")


def test_reads_later_optional_flags_without_filling_unknowns(publication_files, table_reference):
    flags = pa.table({"team_id": pa.array([50, 42], type=pa.int64()),
                      "got_promoted": pa.array([False, None], type=pa.bool_())})
    resolved = _store_test_table(publication_files, flags, "team_seasons")
    frame = read_season_table(resolved, "team_seasons")
    assert frame.shape == (2, 2)
    assert frame.loc[0, "got_promoted"] == False
    assert frame.loc[1, "got_promoted"] is pd.NA


def test_reads_only_selected_table(publication_files, table_reference):
    root, _, manifest_path = publication_files
    metadata = json.loads(manifest_path.read_text())
    metadata["tables"]["shots"] = {"file": "absent.parquet", "rows": 10,
                                    "bytes": 100, "sha256": "a" * 64}
    manifest_path.write_text(json.dumps(metadata), encoding="utf-8")
    resolved = resolve_season_export(root, STEM)
    assert len(read_season_table(resolved, "matches")) == 3


def test_publication_advancing_does_not_reselect_table(table_reference):
    resolved, _ = table_reference
    publication = json.loads(resolved.publication_path.read_text())
    publication["manifest"] = "newer/version/manifest.json"
    resolved.publication_path.write_text(json.dumps(publication), encoding="utf-8")
    assert len(read_season_table(resolved, "matches")) == 3


@pytest.mark.parametrize("target", ["manifest", "table"])
def test_missing_pinned_file_is_explicit(table_reference, target):
    resolved, _ = table_reference
    path = (resolved.manifest_path if target == "manifest"
            else resolved.manifest_path.parent / "matches.parquet")
    path.unlink()
    with pytest.raises(FileNotFoundError):
        read_season_table(resolved, "matches")


def test_modified_canonical_file_fails(table_reference):
    resolved, _ = table_reference
    with resolved.manifest_path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="Pinned canonical manifest changed"):
        read_season_table(resolved, "matches")


def test_modified_metadata_dictionary_fails(table_reference):
    resolved, _ = table_reference
    resolved.manifest["tables"]["matches"]["rows"] = 1
    with pytest.raises(ValueError, match="metadata was modified"):
        read_season_table(resolved, "matches")


@pytest.mark.parametrize("change,message", [("append", "byte count"), ("replace", "SHA256")])
def test_changed_table_fails_before_decoding(table_reference, change, message):
    resolved, _ = table_reference
    path = resolved.manifest_path.parent / "matches.parquet"
    original = path.read_bytes()
    changed = original + b"!" if change == "append" else bytes([original[0] ^ 1]) + original[1:]
    path.write_bytes(changed)
    with pytest.raises(ValueError, match=message):
        read_season_table(resolved, "matches")


def test_matching_hash_does_not_make_invalid_parquet_readable(publication_files):
    root, _, manifest_path = publication_files
    data = b"This is not a Parquet file."
    (manifest_path.parent / "matches.parquet").write_bytes(data)
    metadata = json.loads(manifest_path.read_text())
    metadata["tables"]["matches"].update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    manifest_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid or unreadable Parquet"):
        read_season_table(resolve_season_export(root, STEM), "matches")


def test_declared_row_count_must_match_decoded_rows(publication_files, table_reference):
    root, publication_path, manifest_path = publication_files
    outer = json.loads(publication_path.read_text())
    inner = json.loads(manifest_path.read_text())
    outer["matches"] = 4
    inner["tables"]["matches"]["rows"] = 4
    publication_path.write_text(json.dumps(outer), encoding="utf-8")
    manifest_path.write_text(json.dumps(inner), encoding="utf-8")
    with pytest.raises(ValueError, match="row count mismatch"):
        read_season_table(resolve_season_export(root, STEM), "matches")


def test_empty_typed_table_keeps_its_columns(publication_files):
    empty = pa.table({"event_id": pa.array([], type=pa.int64()),
                      "got_promoted": pa.array([], type=pa.bool_())})
    frame = read_season_table(_store_test_table(publication_files, empty), "matches")
    assert frame.shape == (0, 2)
    assert frame["event_id"].dtype == pd.ArrowDtype(pa.int64())
    assert frame["got_promoted"].dtype == pd.ArrowDtype(pa.bool_())


def test_pandas_index_metadata_does_not_hide_a_physical_column(publication_files):
    source = pd.DataFrame({"event_id": [9, 3]}, index=pd.Index([200, 100], name="stored_index"))
    table = pa.Table.from_pandas(source, preserve_index=True)
    frame = read_season_table(_store_test_table(publication_files, table), "matches")
    assert list(frame.columns) == ["event_id", "stored_index"]
    assert frame["stored_index"].tolist() == [200, 100]
    assert frame.index.equals(pd.RangeIndex(2))


@pytest.mark.parametrize("name", ["", None, ["matches"]])
def test_invalid_table_name(table_reference, name):
    with pytest.raises(ValueError, match="nonempty table name"):
        read_season_table(table_reference[0], name)


def test_requires_resolved_reference():
    with pytest.raises(TypeError, match="ResolvedSeason"):
        read_season_table(Path("somewhere"), "matches")
