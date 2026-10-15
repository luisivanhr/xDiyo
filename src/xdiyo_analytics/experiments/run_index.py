"""Rebuildable run lookup. Authoritative manifest bytes are checked every read.

The disk index is only a publication hint, never a source of result records.
A new process reconstructs the validated in-memory index. This deliberately
retains directory enumeration and content reads to detect unannounced edits.
"""
from copy import deepcopy
import hashlib
import json
import os
from threading import RLock
from uuid import uuid4


class RunIndex:
    def __init__(self, root, experiment_id):
        self.root, self.experiment_id = root, experiment_id
        self._records = {}
        self._by_key = {}
        self._lock = RLock()
        self.decoded = 0

    def refresh(self):
        updated = {}
        for path in sorted((self.root / 'runs').glob('*/run.json')):
            if path.parent.name.startswith('.pending-'):
                continue
            try:
                raw = path.read_bytes()
            except FileNotFoundError:
                continue  # A concurrently removed run cannot be returned.
            prior = self._records.get(path.parent.name)
            if prior is not None and prior[0] == raw:
                record = prior[1]
            else:
                record = json.loads(raw.decode('utf-8'))
                self.decoded += 1
                if (record.get('schema') != 1 or record.get('experiment_id') != self.experiment_id
                        or record.get('run_id') != path.parent.name):
                    raise ValueError(f'Incompatible experiment run record: {path}')
            updated[path.parent.name] = (raw, record)
        changed = updated.keys() != self._records.keys() or any(
            key not in self._records or raw != self._records[key][0] for key, (raw, _) in updated.items())
        self._records = updated
        if changed:
            self._by_key = {}
            for _, record in sorted(updated.values(), key=lambda v: (v[1]['created_at'], v[1]['run_id'])):
                if record['status'] == 'complete' and 'recovery' in record.get('artifacts', {}):
                    self._by_key.setdefault(record.get('recovery_key'), []).append(record)
        document = dict(schema=1, experiment_id=self.experiment_id,
                        manifests={k: hashlib.sha256(raw).hexdigest() for k, (raw, _) in updated.items()})
        target = self.root / 'run-index.json'
        try:
            stale = json.loads(target.read_text(encoding='utf-8')) != document
        except (OSError, ValueError):
            stale = True
        if stale:
            stage = self.root / ('.run-index-' + uuid4().hex + '.tmp')
            try:
                with stage.open('x', encoding='utf-8') as stream:
                    json.dump(document, stream, sort_keys=True)
                    stream.flush(); os.fsync(stream.fileno())
                stage.replace(target)
            except OSError:
                # Read-only stores or racing writers do not change authority.
                pass
            finally:
                try:
                    stage.unlink(missing_ok=True)
                except OSError:
                    pass  # An unpublished hint never participates in recovery.

    def read(self, role=None, group=None, *, key=None, completed=False):
        with self._lock:
            self.refresh()
            records = (self._by_key.get(key, []) if completed else [v[1] for v in self._records.values()])
            records = [r for r in records if (role is None or r.get('role', 'final') == role)
                       and (group is None or r.get('run_group', r['run_id']) == group)]
            records.sort(key=lambda r: (r['created_at'], r['run_id']))
            return deepcopy((records[-1] if records else None) if completed else records)
