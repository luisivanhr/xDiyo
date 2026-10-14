"""Stable-ID team names and optional local badges for offline reports."""

import base64
from copy import deepcopy
import hashlib
from dataclasses import dataclass
import json
import math
from pathlib import Path
from numbers import Real, Integral
import re
import xml.etree.ElementTree as ET

import pandas as pd


def team_key(value):
    """String form without converting integer identities through floating point."""
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real) and math.isfinite(value) and value == int(value):
        return str(int(value))
    return str(value)


def _local_reference(value):
    if value.startswith("#"):
        return True
    # Some genuine vector crests contain raster tiles. Keep those self-contained
    # PNG/JPEG tiles intact without allowing remote or nested SVG resources.
    for prefix, signature in (("data:image/png;base64,", b"\x89PNG\r\n\x1a\n"),
                              ("data:image/jpeg;base64,", b"\xff\xd8\xff")):
        if value.startswith(prefix):
            try:
                payload = re.sub(r"[ \t\r\n]", "", value[len(prefix):])
                return base64.b64decode(payload, validate=True).startswith(signature)
            except ValueError:
                return False
    return False


def _badge(path):
    path = Path(path)
    if path.stat().st_size > 2_000_000:
        raise ValueError("Badge exceeds the 2 MB embedding limit.")
    data = path.read_bytes()
    # Provider badge filenames are not reliable: SVG crests can have .png names.
    # Sniff bytes, and still apply SVG validation regardless of the suffix.
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        mime = "image/svg+xml"
        source = data.decode("utf-8-sig")
        if re.search(r"<!\s*(?:DOCTYPE|ENTITY)", source, re.I):
            raise ValueError("SVG badge must not contain document/entity declarations.")
        root = ET.fromstring(source)
        if root.tag.split("}")[-1] != "svg":
            raise ValueError("Badge is not an SVG document.")
        for node in root.iter():
            if node.tag.split("}")[-1].lower() in {"script", "foreignobject", "iframe", "object", "embed"}:
                raise ValueError("SVG badge contains active content.")
            for name, value in node.attrib.items():
                name = name.split("}")[-1].lower()
                if name.startswith("on") or (name in {"href", "src"} and not _local_reference(value)):
                    raise ValueError("SVG badge contains active or external references.")
            text = " ".join([node.text or "", *node.attrib.values()])
            urls = re.findall(r"url\(([^)]*)\)", text, re.I)
            if re.search(r"@import", text, re.I) or any(not url.strip().strip("\"'").startswith("#") for url in urls):
                raise ValueError("SVG badge contains external style references.")
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


@dataclass
class TeamCatalog:
    """Map stable team IDs to names and optional local badge paths.

    entries maps IDs to names or dictionaries containing name and badge_path.
    Extra fields (source_url, attribution, license) are retained as provenance.
    No network request is made. Missing badges fall back to the team name.
    """
    entries: dict

    def __post_init__(self):
        self.entries = {team_key(key): ({"name": value} if isinstance(value, str) else dict(value))
                        for key, value in self.entries.items()}

    @classmethod
    def from_json(cls, path):
        """Read a JSON ID-to-record mapping; badge paths resolve beside this file."""
        path = Path(path).resolve()
        entries = json.loads(path.read_text(encoding="utf-8-sig"))
        for value in entries.values():
            if isinstance(value, dict) and value.get("badge_path"):
                value["badge_path"] = str((path.parent / value["badge_path"]).resolve())
        return cls(entries)

    @classmethod
    def builtin(cls):
        """Load the complete shipped catalog, in a checkout or installed wheel."""
        installed = Path(__file__).with_name('team_assets') / 'catalog.json'
        checkout = Path(__file__).resolve().parents[3] / 'docs/analytics/team_assets/catalog.json'
        return cls.from_json(installed if installed.is_file() else checkout)

    def export_bundle(self, directory, *, team_ids=None):
        """Copy selected badges and their catalog into a new portable directory.

        By default export every catalogued team (no league/season filtering).
        Fail before creating the directory if any selected badge is unavailable,
        unsafe or inconsistent with its recorded hash. Files retain their actual
        image format. Source provenance and names are preserved. Alternative
        asset records remain provenance only; the selected badge is packaged.
        The resulting catalog can be passed directly to ``from_json``.
        """
        keys = list(self.entries) if team_ids is None else list(dict.fromkeys(map(team_key, team_ids)))
        records, payloads = {}, {}
        suffixes = {'image/svg+xml': '.svg', 'image/png': '.png',
                    'image/jpeg': '.jpg', 'image/webp': '.webp'}
        for key in keys:
            if key not in self.entries:
                raise ValueError(f'Team {key}: missing catalog entry.')
            entry = deepcopy(self.entries[key])
            if not entry.get('badge_path'):
                raise ValueError(f'Team {key}: missing badge_path.')
            try:
                uri = _badge(entry['badge_path'])
                mime, encoded = uri.removeprefix('data:').split(';base64,', 1)
                data = base64.b64decode(encoded)
            except (OSError, ValueError, ET.ParseError) as error:
                raise ValueError(f'Team {key}: cannot package badge ({error}).') from error
            digest = hashlib.sha256(data).hexdigest()
            if entry.get('sha256') and entry['sha256'] != digest:
                raise ValueError(f'Team {key}: badge SHA-256 mismatch.')
            relative = f'badges/{digest}{suffixes[mime]}'
            entry.update(badge_path=relative, sha256=digest, format=suffixes[mime][1:])
            records[key] = entry
            payloads[relative] = data
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'badges').mkdir()
        for relative, data in payloads.items():
            (directory / relative).write_bytes(data)
        path = directory / 'catalog.json'
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return path

    @classmethod
    def from_matches(cls, matches):
        """Build names from home_id/home_name and away_id/away_name columns.

        Accept a DataFrame or iterable of DataFrames. Later nonmissing names for
        an ID replace earlier spellings; no identity is inferred from a name.
        """
        entries = {}
        for frame in ([matches] if isinstance(matches, pd.DataFrame) else matches):
            for side in ("home", "away"):
                for key, name in frame[[f"{side}_id", f"{side}_name"]].itertuples(index=False, name=None):
                    if pd.notna(key) and pd.notna(name):
                        entries[team_key(key)] = {"name": str(name)}
        return cls(entries)

    def display(self, key, *, badges=False):
        """Return name, embedded image (or None), and optional badge failure note."""
        entry = self.entries.get(team_key(key), {})
        name = entry.get("name") or f"Team {key}"
        if not badges or not entry.get("badge_path"):
            return str(name), None, None
        try:
            return str(name), _badge(entry["badge_path"]), None
        except (OSError, ValueError, ET.ParseError) as error:
            return str(name), None, f"{name}: badge unavailable ({error})."
