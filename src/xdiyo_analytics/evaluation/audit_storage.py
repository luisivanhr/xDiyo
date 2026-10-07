"""Explicit compact traceability; never substitutes for computation/validation."""
from dataclasses import asdict, is_dataclass
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import subprocess

import pandas as pd


def fingerprint(frame):
    """Hash original values, ordered schema/dtypes and row identities."""
    schema = json.dumps([(str(c), str(t)) for c, t in frame.dtypes.items()])
    digest = hashlib.sha256(schema.encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True, categorize=False).to_numpy().tobytes())
    return digest.hexdigest()


def _spec(value):
    if is_dataclass(value):
        return {'type': type(value).__qualname__, 'parameters': asdict(value)}
    return repr(value) if value is not None else None


def summary_manifest(ledger, templates, tickets, summaries, attrs, stake_policy, stake_context, risk_limits):
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    revision = None
    for parent in package.parents:
        if (parent / '.git').exists():
            try:
                revision = subprocess.check_output(['git', '-C', str(parent), 'rev-parse', 'HEAD'],
                                                   stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
            except (OSError, subprocess.SubprocessError):
                pass
            break
    dependencies = {'python': platform.python_version()}
    for name in ('pandas', 'numpy', 'xdiyo-analytics'):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = 'uninstalled'
    models = {}
    for name, template in templates.items():
        models[name] = {}
        for model, source in (getattr(template, 'probability_columns', None) or {}).items():
            fields = list(dict.fromkeys([source, *(c for c in ledger if str(c).startswith(str(model) + '::'))]))
            fields = [c for c in fields if c in ledger]
            models[name][model] = {'source': source, 'stream_sha256': fingerprint(ledger[fields])}
    pins = {str(c): sorted({str(v) for v in ledger[c].dropna()}) for c in ledger
            if str(c).endswith(('snapshot_hash', 'crosswalk_hash'))}
    bounds = {}
    for c in ledger:
        if str(c).split('::')[-1] in ('kickoff_at', 'decision_at', 'quote_at', 'assumed_available_at',
                                      'issued_at', 'trained_through', 'artifact_vintage'):
            # Inventory unconsumed rows too, without adding new validation to
            # rows the native policy intentionally excludes. Guards ran earlier.
            times = pd.to_datetime(ledger[c], utc=True, errors='coerce')
            bounds[str(c)] = {'minimum': None if times.isna().all() else times.min().isoformat(),
                              'maximum': None if times.isna().all() else times.max().isoformat(),
                              'missing': int(ledger[c].isna().sum()),
                              'unparseable': int((times.isna() & ledger[c].notna()).sum())}
    manifest = dict(audit_level='summary', complete_row_audit=False, source_revision=revision,
        runtime_source_sha256=digest.hexdigest(), dependencies=dependencies,
        input_sha256=fingerprint(ledger), models=models, model_fingerprint_kind='retained_stream_values',
        quote_pins=pins, original_time_bounds=bounds,
        strategies={name: _spec(template) for name, template in templates.items()},
        stake_policy=_spec(stake_policy), stake_context=_spec(stake_context), risk_limits=_spec(risk_limits),
        candidate_count=sum(int(s['candidate_tickets']) for s in summaries), selected_count=len(tickets),
        rejected_count=sum(int(s['candidate_tickets']) for s in summaries) - len(tickets),
        expansion_summary=summaries, gate_counts=attrs.get('gate_counts', {}), gate_reasons=attrs.get('gate_summary', []),
        quote_contracts={name: _spec(getattr(template, 'quote_availability', None)) for name, template in templates.items()},
        omitted_details=['per-candidate expansion records', 'rejected per-model ballots',
                         'repeated ticket quote_legs payloads', 'per-ballot quote/timing copies'],
        caveat='Summary traceability is not a complete rejected-candidate audit. Unknown observed quote times remain unknown. Research availability and retrospective simulation-clock declarations do not establish historical tradability or observed model issuance.')
    # Keep attrs / Parquet / JSON round trips free of custom Python objects.
    return json.loads(json.dumps(manifest, default=str))
