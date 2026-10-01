"""Resolve the current locally imported odds database without a version picker."""

import json
from pathlib import Path


def current_database(root):
    """Return the most recently imported database, or an explicit legacy version.

    The selected manifest is pinned by OddsSeries for the duration of a run.
    Import directories are immutable; their manifest modification time records
    local import order. Merely reading an older database does not activate it.
    """
    root = Path(root)
    if (root / 'manifest.json').is_file():
        return root
    candidates = []
    for path in root.glob('*/*/manifest.json'):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('adapter_version') == '1':
            candidates.append(path)
    # Also support a root containing import versions directly.
    for path in root.glob('*/manifest.json'):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('adapter_version') == '1':
            candidates.append(path)
    if not candidates:
        raise ValueError('No imported odds database found. Import the odds workbook first.')
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p))).parent
