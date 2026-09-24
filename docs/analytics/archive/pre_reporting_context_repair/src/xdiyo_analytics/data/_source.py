"""Internal publication selection and saved-selection support."""

import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


@dataclass
class _Source:
    path: Path
    manifest: dict
    record: dict
    record_path: Path | None
    save_record: bool

    def table_path(self, name: str) -> Path:
        if name not in self.manifest["tables"]:
            available = ", ".join(sorted(self.manifest["tables"]))
            raise KeyError(f"Table {name!r} is unavailable; available: {available}")
        info = self.manifest["tables"][name]
        if not isinstance(info, dict) or not isinstance(info.get("file"), str):
            raise ValueError(f"Invalid table description: {name}")
        path = (self.path.parent / info["file"]).resolve()
        if path.parent != self.path.parent or path.suffix != ".parquet":
            raise ValueError(f"Table file must stay inside the selected version: {name}")
        if any(type(info.get(key)) is not int or info[key] < 0 for key in ("rows", "bytes")):
            raise ValueError(f"Invalid row or byte count for table {name}")
        return path

    def save(self) -> None:
        if self.save_record:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            with self.record_path.open("x", encoding="utf-8") as stream:
                json.dump(self.record, stream, indent=2)
                stream.write("\n")


def _object(data: bytes, label: str) -> dict:
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSON in {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {label}")
    return value


def _open_source(data_root, season_stem, record_path=None, *, existing_record=False) -> _Source:
    if not isinstance(season_stem, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*_[0-9]{2}_[0-9]{2}", season_stem):
        raise ValueError("Use a season name such as 'Premier_League_24_25'")
    root = Path(data_root).resolve()
    record_path = Path(record_path).resolve() if record_path is not None else None
    if record_path is not None and record_path.is_relative_to(root):
        raise ValueError("Save selection records outside the source data directory")
    replay = record_path is not None and (existing_record or record_path.exists())
    saved = None
    if replay:
        saved = _object(record_path.read_bytes(), str(record_path))
        try:
            if type(saved.get("schema_version")) is not int or saved["schema_version"] != 1 or saved.get("season_stem") != season_stem:
                raise ValueError("Record belongs to a different season or format")
            publication_bytes = base64.b64decode(saved["publication_base64"], validate=True)
            if hashlib.sha256(publication_bytes).hexdigest() != saved["publication_sha256"]:
                raise ValueError("Saved publication has changed")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid selection record: {record_path}") from exc
    else:
        publication_path = (root / f"{season_stem}.manifest.json").resolve()
        if not publication_path.is_relative_to(root):
            raise ValueError("Publication must stay inside the source data directory")
        publication_bytes = publication_path.read_bytes()
    publication = _object(publication_bytes, season_stem)
    if publication.get("complete") is not True or publication.get("schema_version") != "2":
        raise ValueError(f"Season {season_stem} is not a complete schema-2 publication")
    if publication.get("season_file") != f"{season_stem}.parquet":
        raise ValueError("Publication does not match the requested season")
    link = publication.get("manifest")
    if not isinstance(link, str) or not link or Path(link).is_absolute():
        raise ValueError("Publication must link to a relative manifest path")
    path = (root / link).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Manifest must stay inside the source data directory")
    manifest_bytes = path.read_bytes()
    fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
    if saved is not None and fingerprint != saved.get("manifest_sha256"):
        raise ValueError("The saved season manifest has changed")
    manifest = _object(manifest_bytes, str(path))
    scope = manifest.get("scope", {})
    if manifest.get("schema_version") != "2" or not isinstance(scope, dict):
        raise ValueError("Unsupported or invalid season manifest")
    league, start, end = season_stem.rsplit("_", 2)
    if (any(type(scope.get(k)) is not int or scope[k] <= 0 for k in ("league_id", "season_id", "season_start", "season_end"))
            or scope.get("league_name") != league or scope["season_start"] % 100 != int(start)
            or scope["season_end"] % 100 != int(end) or scope["season_end"] != scope["season_start"] + 1):
        raise ValueError("Manifest competition/season does not match the requested season")
    version = manifest.get("version")
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", version):
        raise ValueError("Invalid season version")
    expected = root / "_tables" / f"competition={scope['league_id']}" / f"season={scope['season_id']}" / "versions" / version / "manifest.json"
    if path != expected.resolve():
        raise ValueError("Manifest location does not match its season version")
    tables = manifest.get("tables")
    if not isinstance(tables, dict) or not isinstance(tables.get("matches"), dict):
        raise ValueError("Manifest must list a matches table")
    if type(publication.get("matches")) is not int or publication["matches"] != tables["matches"].get("rows"):
        raise ValueError("Publication and manifest disagree on the match count")
    record = {"schema_version": 1, "season_stem": season_stem,
              "publication_base64": base64.b64encode(publication_bytes).decode("ascii"),
              "publication_sha256": hashlib.sha256(publication_bytes).hexdigest(),
              "manifest_sha256": fingerprint}
    return _Source(path, manifest, record, record_path, record_path is not None and not replay)
