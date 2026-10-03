"""Strict input parsing and lossless evidence copying for standalone formats."""
import base64
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import io
import json
from pathlib import Path
import re


def json_read(text):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError(f'Duplicate JSON key: {key}')
            obj[key] = value
        return obj
    def invalid(value):
        raise ValueError(f'Nonfinite JSON number: {value}')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def manifest(path, kind=None):
    path = Path(path).resolve()
    value = json_read(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict) or value.get('schema_version') != '1.0':
        raise ValueError('Manifest schema_version must be "1.0".')
    if kind and value.get('kind') != kind:
        raise ValueError(f'Manifest kind must be {kind!r}.')
    require_text(value.get('title'), 'title')
    return path, value


def require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string.')
    return value


def decimal(value, name):
    # Strings retain the precision of the producer. Refuse binary float inputs.
    if not isinstance(value, str) or not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', value):
        raise ValueError(f'{name} must be a finite decimal string.')
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f'{name} must be a finite decimal string.') from error
    if not number.is_finite() or abs(number.as_tuple().exponent) > 10000 or len(number.as_tuple().digits) > 10000:
        raise ValueError(f'{name} has an unsupported decimal length/exponent.')
    return number


def exact_sum(values):
    values = list(values)
    # Sufficient precision for exact alignment, cancellation and carry.
    precision = max((v.adjusted() for v in values), default=0) - min(
        (v.as_tuple().exponent for v in values), default=0) + len(str(len(values))) + 4
    with localcontext() as ctx:
        ctx.prec = max(28, precision)
        return sum(values, Decimal(0))


def reconcile(actual, expected, tolerance, name):
    difference = exact_sum([actual, expected.copy_negate()]).copy_abs()
    if difference > tolerance:
        raise ValueError(f'{name}: supplied {expected}, observed {actual}, tolerance {tolerance}.')


def timestamp(value, name):
    require_text(value, name)
    try:
        date = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError(f'{name}: invalid ISO timestamp {value!r}.') from error
    if date.tzinfo is None or date.utcoffset().total_seconds() != 0:
        raise ValueError(f'{name}: supply an explicit UTC timestamp (Z or +00:00).')
    return date.astimezone(timezone.utc)


def read_csv(path):
    raw = Path(path).read_bytes()
    try:
        rows = list(csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline=''), strict=True))
    except (UnicodeError, csv.Error) as error:
        raise ValueError(f'{path}: invalid UTF-8 CSV: {error}') from error
    if not rows or not rows[0] or any(not name for name in rows[0]):
        raise ValueError(f'{path}: CSV needs nonempty headers.')
    headers, rows = rows[0], rows[1:]
    if len(set(headers)) != len(headers):
        raise ValueError(f'{path}: duplicate CSV headers.')
    if any(len(row) != len(headers) for row in rows):
        raise ValueError(f'{path}: inconsistent CSV row widths.')
    return raw, headers, rows


def js_json(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def encoded(raw):
    return base64.b64encode(raw).decode('ascii')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def output_directory(path):
    path = Path(path).resolve()
    # Sharing bundles are immutable builds; never overwrite input evidence.
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output directory must be empty: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def attribution_files(m, root):
    result = []
    for index, name in enumerate(m.get('attribution_files', [])):
        source = root / require_text(name, 'attribution_files path')
        result.append((f'attribution/{index:03d}-{source.name}', source.read_bytes()))
    return result


def write_evidence(output, m, inputs, attribution):
    (output / 'manifest.json').write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding='utf-8')
    for name, raw in attribution:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    (output / 'BUILD.json').write_text(json.dumps({
        'format_version': '1.0', 'inputs': inputs,
        'attribution': [{'path': name, 'sha256': digest(raw)} for name, raw in attribution],
        'semantics': 'Presentation of supplied evidence; no fitting, selection or settlement.'
    }, indent=2), encoding='utf-8')
