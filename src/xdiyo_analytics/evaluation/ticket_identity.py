"""Persist original numeric identities before pandas boxes membership rows.

This is versioned consistency evidence, not a signature or source authentication.
Public ticket hashes retain their existing canonical string representation.
"""
import hashlib
import json
import numpy as np
import pandas as pd


FIELD = 'identity_evidence'


def _scalar(value):
    if isinstance(value, (float, np.floating)):
        kind = str(value.dtype) if isinstance(value, np.floating) else 'float'
        if kind not in ('float', 'float16', 'float32', 'float64') or not np.isfinite(value):
            raise ValueError('Floating ticket identities require finite binary16/32/64 values.')
        return [kind, float(value).hex()]
    if isinstance(value, (bool, np.bool_)):
        return ['bool', str(bool(value))]
    if isinstance(value, (int, np.integer)):
        return ['int', str(int(value))]
    if isinstance(value, str):
        return ['str', str(value)]
    return ['other', type(value).__module__ + '.' + type(value).__qualname__, str(value)]


def _token(cell):
    """Reconstruct and validate a canonical token from the original scalar."""
    if not isinstance(cell, list) or len(cell) < 2 or not all(isinstance(v, str) for v in cell):
        raise ValueError('Malformed original identity scalar.')
    kind = cell[0]
    if kind in ('float', 'float16', 'float32', 'float64') and len(cell) == 2:
        value = float.fromhex(cell[1])
        if kind != 'float':
            with np.errstate(over='ignore', under='ignore'):
                value = np.dtype(kind).type(value)
        if _scalar(value) != cell:
            raise ValueError('Original identity scalar is not an exact canonical value.')
        return str(value)
    if kind == 'int' and len(cell) == 2 and str(int(cell[1])) == cell[1]:
        return cell[1]
    if kind == 'bool' and len(cell) == 2 and cell[1] in ('True', 'False'):
        return cell[1]
    if kind == 'str' and len(cell) == 2:
        return cell[1]
    if kind == 'other' and len(cell) == 3:
        return cell[2]
    raise ValueError('Unsupported original identity scalar.')


def _same(cell, value):
    """Accept exact numeric boxing, never rounding or parsing text keys."""
    current = _scalar(value)
    numeric = ('float', 'float16', 'float32', 'float64', 'int')
    if cell[0] in numeric and current[0] in numeric:
        original = int(cell[1]) if cell[0] == 'int' else float.fromhex(cell[1])
        supplied = int(current[1]) if current[0] == 'int' else float.fromhex(current[1])
        return original == supplied
    return cell == current


class OriginalIdentity:
    """One column-wise capture per pool, before iterrows/to_dict conversion."""
    def __init__(self, offered, groups, keys):
        self.groups, self.keys = list(groups), list(keys)
        columns = list(dict.fromkeys([*groups, *keys]))
        values = [list(offered[k]) for k in columns]
        self.cells = None
        if any(isinstance(v, (float, np.floating)) for values_ in values for v in values_):
            self.cells = [[_scalar(v) for v in row] for row in zip(*values)]
            self.dtypes = [str(offered[k].dtype) for k in columns]

    def evidence(self, positions):
        if self.cells is None:
            return {}
        return {FIELD: json.dumps(dict(version=1, groups=self.groups, keys=self.keys,
                                       dtypes=self.dtypes, cells=[self.cells[p] for p in positions]),
                                  ensure_ascii=False, separators=(',', ':'))}


def validate_identity(row, legs, groups, keys):
    """Linear validation with exactly one native hash; no spelling search."""
    from .all_combinations import _identity
    columns = list(dict.fromkeys(groups + keys))
    values = {k: list(legs[k]) for k in columns}
    evidence = row.get(FIELD)
    if isinstance(evidence, str):
        try:
            doc = json.loads(evidence)
            if (not isinstance(doc, dict) or set(doc) != {'version', 'groups', 'keys', 'dtypes', 'cells'}
                    or type(doc['version']) is not int or doc['version'] != 1
                    or doc['groups'] != groups or doc['keys'] != keys
                    or not isinstance(doc['dtypes'], list) or len(doc['dtypes']) != len(columns)
                    or any(not isinstance(v, str) or not v for v in doc['dtypes'])
                    or not isinstance(doc['cells'], list) or len(doc['cells']) != len(legs)):
                raise ValueError('Invalid identity evidence schema.')
            for dtype in doc['dtypes']:
                pd.api.types.pandas_dtype(dtype)
            tokens = {k: [] for k in columns}
            for i, cells in enumerate(doc['cells']):
                if not isinstance(cells, list) or len(cells) != len(columns):
                    raise ValueError('Invalid identity evidence width.')
                for k, cell in zip(columns, cells):
                    tokens[k].append(_token(cell))
                    if not _same(cell, values[k][i]):
                        raise ValueError('Consumed membership changed its original numeric or typed identity.')
            for k in groups:
                if k in row and not _same(doc['cells'][0][columns.index(k)], row[k]):
                    raise ValueError(f'Ticket grouping differs from its consumed membership at {k}.')
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise ValueError(f'Invalid original ticket identity evidence: {exc}') from exc
    else:
        # Legacy string hashes cannot prove whether "1.1" came from binary32
        # or binary64. Reconstruct from the original ledger, never guess.
        if evidence is not None and not (isinstance(evidence, float) and np.isnan(evidence)):
            raise ValueError('Invalid original ticket identity evidence.')
        if any(isinstance(v, (float, np.floating)) for vs in values.values() for v in vs):
            raise ValueError('Floating ticket identities require original identity evidence; rebuild from the original ledger.')
        tokens = {k: [str(v) for v in values[k]] for k in columns}
        for k in groups:
            if k in row and str(row[k]) != tokens[k][0]:
                raise ValueError(f'Ticket grouping differs from its consumed membership at {k}.')
    group_id = _identity(tokens[k][0] for k in groups)
    events = sorted(_identity(v) for v in zip(*(tokens[k] for k in keys)))
    identity = json.dumps([row['bet'], group_id, events], separators=(',', ':'))
    if row['ticket_id'] != row['bet'] + ':' + hashlib.sha256(identity.encode()).hexdigest():
        raise ValueError('Consumed membership differs from the original ticket identity.')
