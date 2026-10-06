"""Small, JSON-native lineage contract for scalar spatial compositions.

Raw coverage is keyed by source selection and map identity. Historical coverage
is keyed by the complete calculation (including its window), never added across
branches. Derived scalar units/frame are deliberately unspecified.
"""
from copy import deepcopy
from hashlib import sha256
from numbers import Real
import json

from .composition import Constant, Ratio, operands


def lineage_id(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def distinct_records(*collections):
    """Preserve first-seen order without sharing mutable cached-child records."""
    found = {}
    for records in collections:
        for record in records:
            found.setdefault(lineage_id(record), record)
    return deepcopy(list(found.values()))


def calculation_for(frame, expression):
    specs = frame.attrs.get('spatial_features', {})
    if specs:
        return deepcopy(next(iter(specs.values()))['calculation'])
    if isinstance(expression, Constant):
        return dict(operator='Constant', value=float(expression.value))
    if isinstance(expression, Real):
        return dict(operator='Constant', value=float(expression))
    return dict(operator='Nonspatial', expression=repr(expression))


def compose_spatial_metadata(result, node, left, right):
    """Called only after ordinary single-column/H2H/numeric validation succeeds."""
    if not any(frame.attrs.get('spatial_features') for frame in (left, right)):
        return
    a, b = operands(node)
    calculation = dict(operator=type(node).__name__, left=calculation_for(left, a), right=calculation_for(right, b))
    if isinstance(node, Ratio):
        calculation['zero_value'] = None if node.zero_value is None else float(node.zero_value)
    sources = []
    for frame in (left, right):
        for spec in frame.attrs.get('spatial_features', {}).values():
            sources.extend(spec.get('sources', [
                {k: v for k, v in spec.items() if k not in ('columns', 'operators', 'feature', 'family')}
            ]))
    result.attrs['spatial_features'] = {'value':dict(
        kind='scalar', columns=['value'], field_label='Derived spatial value',
        source='SpatialArithmetic', derived=True, units='derived expression units',
        orientation='derived', side=None, calculation=calculation,
        sources=distinct_records(sources), operators=[])}
    for attr in ('point_map_coverage', 'point_history_coverage'):
        if any(attr in frame.attrs for frame in (left, right)):
            result.attrs[attr] = distinct_records(*(frame.attrs.get(attr, []) for frame in (left, right)))
