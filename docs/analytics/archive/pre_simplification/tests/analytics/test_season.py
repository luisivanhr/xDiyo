"""One-selection bundles, validation boundaries and shared-record compatibility."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from xdiyo_analytics.data import SeasonData, load_season, load_season_table
from xdiyo_analytics.data import season as season_module
from xdiyo_analytics.data.exports import read_season_table, resolve_season_export


STEM = "Premier_League_24_25"
EVENT, HOME, AWAY = 2**53 + 1, 2**53 + 3, 2**53 + 5


@pytest.fixture
def season_root(tmp_path):
    root = tmp_path / "exports"
    version = root / "_tables/competition=17/season=61627/versions/v1"
    version.mkdir(parents=True)
    tables = {
        "matches": pa.table({
            "event_id": [EVENT, 7], "home_id": [HOME, 42], "away_id": [AWAY, 50],
            "competition_id": [17, 17], "season_id": [61627, 61627],
            "kickoff_utc": [1735752600.0, 1736017200.0],
            "optional_flag": pa.array([None, False], type=pa.bool_()),
        }),
        "statistics": pa.table({
            "event_id": [EVENT, EVENT, EVENT, 7], "period": ["ALL", "ALL", "ALL", "1ST"],
            "group_name": ["Match overview", "Shots", "Shots", "Attack"],
            "key": ["totalShotsOnGoal", "totalShotsOnGoal", "totalShotsOnGoal", "goals"],
            "side": ["home", "home", "away", "home"],
            "team_id": pa.array([HOME, HOME, AWAY, None], type=pa.int64()),
            "value": pa.array([3.0, 3.0, 4.0, None], type=pa.float64()),
            "display": ["3", "3", "4", "not supplied"],
        }),
        "pregame": pa.table({"event_id": [EVENT, 7], "side": ["home", "home"], "position": [3, 2]}),
        "shots": pa.table({"event_id": [EVENT, EVENT], "shot_index": [0, 1], "xg": [None, 0.0]}),
        "future_context": pa.table({"details": pa.array([], type=pa.list_(pa.int64()))}),
    }
    descriptors = {}
    for name, table in tables.items():
        path = version / f"{name}.parquet"
        pq.write_table(table, path)
        payload = path.read_bytes()
        descriptors[name] = {"file": path.name, "rows": table.num_rows, "bytes": len(payload),
                             "sha256": hashlib.sha256(payload).hexdigest()}
    manifest = {
        "schema_version": "2", "parser_version": "2", "version": "v1",
        "scope": {"league_id": 17, "league_name": "Premier_League", "season_id": 61627,
                  "season_start": 2024, "season_end": 2025}, "tables": descriptors,
    }
    (version / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    publication = {"schema_version": "2", "complete": True, "matches": 2,
                   "season_file": STEM + ".parquet",
                   "manifest": (version / "manifest.json").relative_to(root).as_posix()}
    (root / (STEM + ".manifest.json")).write_text(json.dumps(publication), encoding="utf-8")
    return root


def _snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _rewrite_table(root, name, transform):
    resolved = resolve_season_export(root, STEM)
    path = resolved.manifest_path.parent / f"{name}.parquet"
    with pq.ParquetFile(path) as parquet:
        table = transform(parquet.read())
    pq.write_table(table, path)
    payload = path.read_bytes()
    manifest = resolved.manifest
    manifest["tables"][name].update(rows=table.num_rows, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    resolved.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _replace_column(table, name, values):
    return table.set_column(table.schema.get_field_index(name), name, pa.array(values))


def _advance_publication(root):
    resolved = resolve_season_export(root, STEM)
    newer = resolved.manifest_path.parent.parent / "v2"
    shutil.copytree(resolved.manifest_path.parent, newer)
    manifest = json.loads((newer / "manifest.json").read_bytes())
    manifest["version"] = "v2"
    (newer / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    publication = resolved.publication
    publication["manifest"] = (newer / "manifest.json").relative_to(root).as_posix()
    resolved.publication_path.write_text(json.dumps(publication), encoding="utf-8")


def test_default_load_contains_only_matches_and_preserves_exact_nullable_data(season_root):
    before = _snapshot(season_root)
    bundle = load_season(season_root, STEM)
    assert isinstance(bundle, SeasonData)
    assert list(bundle.tables) == ["matches"]
    assert bundle.matches is bundle.tables["matches"]
    assert bundle.statistics is bundle.pregame is bundle.shots is None
    assert bundle.matches["event_id"].tolist() == [EVENT, 7]
    assert bundle.matches["home_id"].tolist() == [HOME, 42]
    assert isinstance(bundle.matches["event_id"].dtype, pd.ArrowDtype)
    assert bundle.matches["optional_flag"].tolist() == [pd.NA, False]
    assert bundle.matches.index.equals(pd.RangeIndex(2))
    assert bundle.validation == {"matches": {
        "status": "structure_valid", "rows": 2,
        "missing_kickoff_event_ids": (), "invalid_kickoff_event_ids": (),
    }}
    assert _snapshot(season_root) == before


@pytest.mark.parametrize("requested", [
    ["statistics", "shots", "matches", "pregame"], ("pregame",), ("shots",), ("future_context",),
])
def test_requested_order_exact_table_set_and_convenience_properties(season_root, requested):
    names = tuple(requested)
    bundle = load_season(season_root, STEM, tables=requested)
    assert tuple(bundle.tables) == names
    assert tuple(bundle.validation) == names
    assert bundle.provenance["loaded_tables"] == names
    for name in ("matches", "statistics", "pregame", "shots"):
        assert getattr(bundle, name) is bundle.tables.get(name)
    for name, frame in bundle.tables.items():
        assert frame.attrs["xdiyo"]["source"]["table"] == name
        assert frame.attrs["xdiyo"]["source"]["loaded_tables"] == names
        assert frame.attrs["xdiyo"]["validation"] == bundle.validation[name]
    if isinstance(requested, list):
        requested.clear()
        assert bundle.provenance["loaded_tables"] == names


def test_requested_empty_table_is_a_dataframe_not_none(season_root):
    _rewrite_table(season_root, "shots", lambda table: table.slice(0, 0))
    bundle = load_season(season_root, STEM, tables=["shots"])
    assert isinstance(bundle.shots, pd.DataFrame)
    assert bundle.shots.empty and bundle.shots.columns.tolist() == ["event_id", "shot_index", "xg"]
    assert bundle.matches is None


def test_unrequested_table_files_are_not_read_or_auto_added(season_root):
    resolve_season_export(season_root, STEM).manifest_path.with_name("matches.parquet").unlink()
    bundle = load_season(season_root, STEM, tables=["shots"])
    assert list(bundle.tables) == ["shots"] and bundle.matches is None
    assert len(bundle.shots) == 2


@pytest.mark.parametrize("bad", [None, "matches", b"matches", {"matches"}, {"matches": True}, 7])
def test_wrong_selection_types_fail_before_resolution(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(season_module, "resolve_season_export", lambda *a: pytest.fail("Unexpected resolution"))
    with pytest.raises(TypeError, match="ordered sequence"):
        load_season(tmp_path / "absent", STEM, tables=bad)


def test_generator_selection_is_not_an_ordered_sequence(tmp_path):
    with pytest.raises(TypeError, match="ordered sequence"):
        load_season(tmp_path, STEM, tables=(name for name in ["matches"]))


@pytest.mark.parametrize("bad", [[], (), [""], [" \t"], ["matches", None], [1], [True], ["matches", "matches"]])
def test_invalid_or_duplicate_names_fail_before_resolution(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(season_module, "resolve_season_export", lambda *a: pytest.fail("Unexpected resolution"))
    with pytest.raises(ValueError, match="nonempty|repeated"):
        load_season(tmp_path / "absent", STEM, tables=bad)


@pytest.mark.parametrize("requested", [["statistics"], ["shots", "statistics"]])
def test_statistics_requires_explicit_matches_before_resolution(tmp_path, monkeypatch, requested):
    monkeypatch.setattr(season_module, "resolve_season_export", lambda *a: pytest.fail("Unexpected resolution"))
    with pytest.raises(ValueError, match="statistics requires matches"):
        load_season(tmp_path / "absent", STEM, tables=requested)


@pytest.mark.parametrize("unknown", ["standings", " matches ", "Matches"])
def test_all_names_are_prechecked_before_any_table_read(season_root, tmp_path, monkeypatch, unknown):
    record = tmp_path / "new_directory/selection.json"
    monkeypatch.setattr(season_module, "read_season_table", lambda *a: pytest.fail("Read before availability precheck"))
    with pytest.raises(KeyError, match="Tables not declared:.*available:") as failure:
        load_season(season_root, STEM, tables=["matches", unknown], record_path=record)
    assert unknown in str(failure.value)
    assert not record.parent.exists()


@pytest.mark.parametrize("change", ["advance", "remove"])
def test_resolves_once_and_keeps_same_reference_when_publication_changes(season_root, monkeypatch, change):
    calls, references, sidecar_reads = [], [], []
    original_resolve = season_module.resolve_season_export
    original_read = season_module.read_season_table
    original_bytes = Path.read_bytes
    publication = season_root / (STEM + ".manifest.json")

    def resolve_once(*args):
        calls.append("resolve")
        return original_resolve(*args)

    def tracked_bytes(path):
        if path == publication:
            sidecar_reads.append(path)
        return original_bytes(path)

    def read_and_change_pointer(resolved, name):
        references.append(resolved)
        calls.append(name)
        result = original_read(resolved, name)
        if len(references) == 1:
            # The helper needs its own metadata read; exclude that setup from the spy.
            with monkeypatch.context() as setup:
                setup.setattr(Path, "read_bytes", original_bytes)
                if change == "advance":
                    _advance_publication(season_root)
                else:
                    publication.unlink()
        return result

    monkeypatch.setattr(Path, "read_bytes", tracked_bytes)
    monkeypatch.setattr(season_module, "resolve_season_export", resolve_once)
    monkeypatch.setattr(season_module, "read_season_table", read_and_change_pointer)
    bundle = load_season(season_root, STEM, tables=["statistics", "pregame", "matches"])
    assert calls == ["resolve", "statistics", "pregame", "matches"]
    assert len(sidecar_reads) == 1
    assert all(reference is references[0] for reference in references)
    assert bundle.provenance["version"] == "v1"
    assert all(frame.attrs["xdiyo"]["source"]["version"] == "v1" for frame in bundle.tables.values())
    assert bundle.validation["statistics"]["rows"] == 4


def test_automatic_statistics_validation_preserves_groups_and_reports_gaps(season_root):
    before = _snapshot(season_root)
    bundle = load_season(season_root, STEM, tables=["statistics", "matches"])
    assert bundle.statistics["group_name"].tolist()[:2] == ["Match overview", "Shots"]
    assert bundle.validation["statistics"] == {
        "status": "structure_valid", "rows": 4, "matches_referenced": 2,
        "team_id_column_present": True, "missing_team_id_row_positions": (3,),
        "missing_value_row_positions": (3,), "match_sides_without_statistics": ((7, "away"),),
    }
    assert bundle.statistics["display"].iloc[3] == "not supplied"
    assert _snapshot(season_root) == before


@pytest.mark.parametrize("kind,message", [
    ("duplicate", "Repeated statistics entries"), ("orphan", "unknown match IDs"),
    ("team", "does not match the match's home team"), ("schema", "missing required columns"),
])
def test_late_statistics_structural_failure_saves_no_record(season_root, tmp_path, kind, message):
    def change(table):
        if kind == "duplicate":
            return pa.concat_tables([table, table.slice(0, 1)])
        if kind == "orphan":
            return _replace_column(table, "event_id", [EVENT, EVENT, EVENT, 8])
        if kind == "team":
            return _replace_column(table, "team_id", [AWAY, HOME, AWAY, None])
        return table.drop(["period"])

    _rewrite_table(season_root, "statistics", change)
    record = tmp_path / "unsaved/selection.json"
    before = _snapshot(season_root)
    with pytest.raises(ValueError, match=message):
        load_season(season_root, STEM, tables=["matches", "pregame", "statistics"], record_path=record)
    assert not record.parent.exists()
    assert _snapshot(season_root) == before


@pytest.mark.parametrize("table_name", ["pregame", "shots"])
def test_other_tables_receive_integrity_checks_without_domain_validation(season_root, table_name):
    _rewrite_table(season_root, table_name, lambda table: pa.table({"event_id": [999, 999], "side": ["invalid", "invalid"]}))
    bundle = load_season(season_root, STEM, tables=["matches", table_name])
    assert bundle.validation[table_name] == {"status": "not_implemented", "table": table_name}
    assert bundle.tables[table_name]["event_id"].tolist() == [999, 999]


def test_missing_and_invalid_match_timing_warns_without_dropping_or_filling(season_root):
    _rewrite_table(season_root, "matches", lambda table: _replace_column(table, "kickoff_utc", [None, float("inf")]))
    with pytest.warns(UserWarning, match="1 missing and 1 invalid kickoff values; rows retained") as caught:
        bundle = load_season(season_root, STEM, tables=["matches", "statistics"])
    assert len(caught) == 1
    assert bundle.matches["event_id"].tolist() == [EVENT, 7]
    assert bundle.matches["kickoff_utc"].tolist() == [pd.NA, float("inf")]
    assert bundle.validation["matches"]["missing_kickoff_event_ids"] == (EVENT,)
    assert bundle.validation["matches"]["invalid_kickoff_event_ids"] == (7,)
    assert len(bundle.statistics) == 4


def test_invalid_match_structure_saves_no_record(season_root, tmp_path):
    _rewrite_table(season_root, "matches", lambda table: _replace_column(table, "home_id", [AWAY, 42]))
    record = tmp_path / "unsaved/selection.json"
    with pytest.raises(ValueError, match="home_id and away_id must differ"):
        load_season(season_root, STEM, tables=["statistics", "matches"], record_path=record)
    assert not record.parent.exists()


@pytest.mark.parametrize("kind", ["missing", "size", "hash", "invalid_parquet"])
def test_late_file_failure_creates_neither_record_nor_parent(season_root, tmp_path, kind):
    resolved = resolve_season_export(season_root, STEM)
    path = resolved.manifest_path.parent / "pregame.parquet"
    if kind == "missing":
        path.unlink()
    elif kind == "size":
        path.write_bytes(path.read_bytes() + b"!")
    else:
        payload = b"!" * path.stat().st_size
        path.write_bytes(payload)
        if kind == "invalid_parquet":
            manifest = resolved.manifest
            manifest["tables"]["pregame"]["sha256"] = hashlib.sha256(payload).hexdigest()
            resolved.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    record = tmp_path / "unsaved/selection.json"
    before = _snapshot(season_root)
    expected = FileNotFoundError if kind == "missing" else ValueError
    with pytest.raises(expected):
        load_season(season_root, STEM, tables=["matches", "statistics", "pregame"], record_path=record)
    assert not record.parent.exists()
    assert _snapshot(season_root) == before


def test_record_writer_runs_after_all_reads_and_domain_checks(season_root, tmp_path, monkeypatch):
    record = tmp_path / "new_parent/selection.json"
    steps = []
    original_read = season_module.read_season_table
    original_matches = season_module.validate_matches
    original_stats = season_module.validate_statistics
    original_write = season_module._write_season_record

    def read(resolved, name):
        assert not record.parent.exists()
        steps.append("read:" + name)
        return original_read(resolved, name)

    def match_check(*args):
        assert not record.parent.exists()
        steps.append("validate:matches")
        return original_matches(*args)

    def stats_check(*args):
        assert not record.parent.exists()
        steps.append("validate:statistics")
        return original_stats(*args)

    def write(*args):
        steps.append("write")
        return original_write(*args)

    monkeypatch.setattr(season_module, "read_season_table", read)
    monkeypatch.setattr(season_module, "validate_matches", match_check)
    monkeypatch.setattr(season_module, "validate_statistics", stats_check)
    monkeypatch.setattr(season_module, "_write_season_record", write)
    load_season(season_root, STEM, tables=["statistics", "matches", "pregame"], record_path=record)
    assert steps == ["read:statistics", "read:matches", "read:pregame", "validate:matches", "validate:statistics", "write"]
    assert json.loads(record.read_bytes())["schema_version"] == 1


@pytest.mark.parametrize("creator", ["single", "bundle"])
def test_saved_records_work_in_both_directions_and_never_overwrite(season_root, tmp_path, monkeypatch, creator):
    record = tmp_path / "selection.json"
    if creator == "single":
        load_season_table(season_root, STEM, record_path=record)
    else:
        load_season(season_root, STEM, tables=["matches", "statistics"], record_path=record)
    payload = record.read_bytes()
    _advance_publication(season_root)
    assert resolve_season_export(season_root, STEM).manifest["version"] == "v2"
    (season_root / (STEM + ".manifest.json")).unlink()
    monkeypatch.setattr(season_module, "resolve_season_export", lambda *a: pytest.fail("Replay reread current publication"))
    bundle = load_season(season_root, STEM, tables=["pregame", "statistics", "matches"], record_path=record)
    single = load_season_table(season_root, STEM, table="statistics", record_path=record)
    pd.testing.assert_frame_equal(single, bundle.statistics)
    assert bundle.provenance["version"] == "v1"
    assert single.attrs["xdiyo"]["source"]["manifest"]["version"] == "v1"
    assert bundle.validation["statistics"]["status"] == "structure_valid"
    assert single.attrs["xdiyo"]["validation"]["status"] == "not_implemented"
    assert record.read_bytes() == payload


def test_both_loaders_write_identical_record_format(season_root, tmp_path):
    single_record, bundle_record = tmp_path / "single.json", tmp_path / "bundle.json"
    load_season_table(season_root, STEM, record_path=single_record)
    load_season(season_root, STEM, tables=["statistics", "matches"], record_path=bundle_record)
    assert single_record.read_bytes() == bundle_record.read_bytes()


def test_saved_record_replays_from_relocated_source(season_root, tmp_path):
    record = tmp_path / "selection.json"
    first = load_season(season_root, STEM, tables=["matches", "statistics"], record_path=record)
    relocated = tmp_path / "relocated"
    shutil.copytree(season_root, relocated)
    (relocated / (STEM + ".manifest.json")).unlink()
    replay = load_season(relocated, STEM, tables=["statistics", "matches"], record_path=record)
    pd.testing.assert_frame_equal(replay.statistics, first.statistics)
    assert Path(replay.provenance["manifest_path"]).is_relative_to(relocated)


@pytest.mark.parametrize("kind", ["invalid_record", "changed_manifest", "missing_pinned_table"])
def test_existing_record_failures_do_not_refresh_or_overwrite(season_root, tmp_path, kind):
    record = tmp_path / "selection.json"
    original = load_season(season_root, STEM, record_path=record)
    if kind == "invalid_record":
        record.write_text("invalid JSON", encoding="utf-8")
    elif kind == "changed_manifest":
        path = Path(original.provenance["manifest_path"])
        path.write_bytes(path.read_bytes() + b" ")
    else:
        _advance_publication(season_root)
        Path(original.provenance["manifest_path"]).with_name("statistics.parquet").unlink()
    before = record.read_bytes()
    expected = FileNotFoundError if kind == "missing_pinned_table" else ValueError
    with pytest.raises(expected):
        load_season(season_root, STEM, tables=["matches", "statistics"], record_path=record)
    assert record.read_bytes() == before


def test_record_inside_source_is_rejected_before_resolution(season_root, monkeypatch):
    monkeypatch.setattr(season_module, "resolve_season_export", lambda *a: pytest.fail("Unexpected resolution"))
    with pytest.raises(ValueError, match="outside data_root"):
        load_season(season_root, STEM, record_path=season_root / "new_parent/record.json")
    assert not (season_root / "new_parent").exists()


def test_concurrent_first_creation_preserves_winner_record(season_root, tmp_path, monkeypatch):
    record = tmp_path / "selection.json"
    writer = season_module._write_season_record
    winning_payload = b"record created by another caller"

    def write_after_race(resolved, stem, path):
        path.write_bytes(winning_payload)
        writer(resolved, stem, path)

    monkeypatch.setattr(season_module, "_write_season_record", write_after_race)
    with pytest.raises(FileExistsError):
        load_season(season_root, STEM, record_path=record)
    assert record.read_bytes() == winning_payload


def test_bundle_frame_and_resolved_metadata_are_independent_copies(season_root, monkeypatch):
    resolved = resolve_season_export(season_root, STEM)
    original_manifest = deepcopy(resolved.manifest)
    monkeypatch.setattr(season_module, "resolve_season_export", lambda *a: resolved)
    bundle = load_season(season_root, STEM, tables=["matches", "statistics", "pregame"])
    bundle.provenance["manifest"]["scope"]["league_name"] = "bundle edit"
    bundle.validation["statistics"]["rows"] = -1
    assert resolved.manifest == original_manifest
    assert bundle.statistics.attrs["xdiyo"]["validation"]["rows"] == 4
    assert bundle.statistics.attrs["xdiyo"]["source"]["manifest"] == original_manifest
    bundle.matches.attrs["xdiyo"]["source"]["manifest"]["scope"]["league_name"] = "frame edit"
    bundle.matches.attrs["xdiyo"]["validation"]["rows"] = -2
    assert bundle.validation["matches"]["rows"] == 2
    assert bundle.pregame.attrs["xdiyo"]["source"]["manifest"] == original_manifest
    assert resolved.manifest == original_manifest


def test_frozen_fields_contain_mutable_frames_and_snapshot_reports(season_root):
    bundle = load_season(season_root, STEM, tables=["matches", "statistics"])
    with pytest.raises(FrozenInstanceError):
        bundle.tables = {}
    validation_before = deepcopy(bundle.validation)
    bundle.statistics.loc[0, "value"] = None
    assert pd.isna(bundle.statistics.loc[0, "value"])
    assert bundle.validation == validation_before
    assert bundle.statistics.attrs["xdiyo"]["validation"]["missing_value_row_positions"] == (3,)
    bundle.tables["caller_context"] = pd.DataFrame({"value": [1]})
    assert "caller_context" in bundle.tables
