"""Presentation inventory, loaded once; runtime forms do not inspect source code."""

from functools import lru_cache
import json
from pathlib import Path


def inventory():
    path = Path(__file__).with_name('inventory.json')
    return _load(path.stat().st_mtime_ns)


@lru_cache(maxsize=1)
def _load(revision):
    return json.loads(Path(__file__).with_name('inventory.json').read_text(encoding='utf-8'))


def component_schema(key):
    return inventory()['components'].get(key)
