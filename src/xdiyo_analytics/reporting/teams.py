"""Stable-ID team names and optional local badges for offline reports."""

import base64
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
    mime = {".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(path.suffix.lower())
    if mime is None:
        raise ValueError("Badges must be local SVG, PNG, JPEG or WebP files.")
    if path.stat().st_size > 2_000_000:
        raise ValueError("Badge exceeds the 2 MB embedding limit.")
    data = path.read_bytes()
    if path.suffix.lower() == ".svg":
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
