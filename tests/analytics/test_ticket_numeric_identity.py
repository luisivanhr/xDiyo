"""2144ddc audit: original values bind boxed floating identities in linear time."""
from dataclasses import replace
from io import StringIO
import json
import pickle
import base64
import re
import numpy as np
import pandas as pd
import pytest
from test_guarded_bulk import both, sync
from test_guarded_repairs import composite_population, ID
from test_guarded_boundaries import invoke
from test_quote_availability import policy
from xdiyo_analytics.evaluation import compose_bets
from xdiyo_analytics.evaluation import ticket_identity as identity
from xdiyo_analytics.evaluation.stake_policy import FixedStake, StakeContext
from xdiyo_analytics.reporting.tickets import ticket_html
from xdiyo_analytics.reporting.contracts import Artifact, StudyResult, StudyRun, AnalysisReport


FIELDS = ['fold_id', 'competition_id', 'season_id', 'round', 'stage',
          'event_id', 'source_league', 'source_season']


def source(field, origin, level):
    data = composite_population().iloc[:2].copy()
    values = [1.1, 2.2] if field == 'event_id' else [1.1, 1.1]
    data[field] = pd.Series([str(v) for v in values] if origin == 'text' else values,
                            dtype='Float32' if origin == 'float32' else object if origin == 'text' else 'float64')
    data['a::probability'] = data['b::probability'] = 1.
    p = replace(policy(), audit_level=level)
    sync(data, p.quote_availability)
    return data, p


@pytest.mark.parametrize('field', FIELDS)
@pytest.mark.parametrize('origin', ['float64', 'float32', 'text'])
@pytest.mark.parametrize('level', ['full', 'summary'])
def test_directional_numeric_rebinding_rejected(field, origin, level):
    data, p = source(field, origin, level)
    # Optional model key evidence must not be necessary for membership binding.
    for model in ('a', 'b'):
        for key in ID: data[f'{model}::{key}'] = data[key]
    t, m, _ = compose_bets(data, p, match_columns=ID)
    before = m.copy(deep=True)
    for explicit in (False, True):
        members = m if explicit else m.drop(columns=[f'{model}::{key}' for model in ('a', 'b') for key in ID])
        for helper in ('batch', 'finalize'):
            invoke(helper, t.copy(), members.copy(), p)
            values = np.asarray(members[field], dtype=float)
            replacements = [values.astype('float32').astype('float64'),
                            np.array([float(str(np.float32(v))) for v in values]),
                            np.nextafter(values, np.inf),
                            np.nextafter(values.astype('float32'), np.float32(np.inf)).astype(float)]
            for replacement in replacements:
                if origin != 'text' and np.array_equal(values, replacement): continue
                changed = members.copy(); changed[field] = replacement
                with pytest.raises(ValueError): invoke(helper, t.copy(), changed, p)
            changed = members.copy(); changed[field] = [str(v) for v in values]
            if origin != 'text':
                with pytest.raises(ValueError): invoke(helper, t.copy(), changed, p)
    pd.testing.assert_frame_equal(m, before, check_exact=True)


@pytest.mark.parametrize('field', FIELDS)
@pytest.mark.parametrize('level', ['full', 'summary'])
def test_float32_allocated_and_serialized(field, level, monkeypatch):
    data, p = source(field, 'float32', level)
    (t, m, _), _ = both(data, p, monkeypatch, match_columns=ID,
                       stake_policy=FixedStake(1., 'u'),
                       stake_context=StakeContext(100., 100., 'u', data.decision_at.max()))
    t, m = pickle.loads(pickle.dumps((t, m)))
    payload = t[identity.FIELD].iloc[0]
    assert json.loads(payload)['version'] == 1
    csv = pd.read_csv(StringIO(t.to_csv(index=False)))
    assert csv[identity.FIELD].iloc[0] == payload
    study = StudyResult('Identity persistence', artifacts=[Artifact('html', ticket_html(t, m, {}), 'Tickets')],
                        tables={'tickets': t, 'ticket_legs': m})
    report = AnalysisReport([StudyRun('Identity', 'aggregate', 'all', None, 'match', np.arange(2), 2, study)])
    downloads = [pd.read_csv(StringIO(base64.b64decode(raw).decode('utf-8-sig')))
                 for raw in re.findall(r'data:text/csv;base64,([A-Za-z0-9+/=]+)', report.to_html())]
    exported = next(frame for frame in downloads if identity.FIELD in frame)
    assert exported[identity.FIELD].iloc[0] == payload
    for helper in ('batch', 'finalize'):
        invoke(helper, t.copy(), m.copy(), p)
        # Group assertions can be partially or wholly absent, never contradicted.
        groups = json.loads(payload)['groups']
        invoke(helper, t.drop(columns=groups), m.copy(), p)
        for group in groups:
            partial = t.drop(columns=[g for g in groups if g != group]).copy()
            invoke(helper, partial.copy(), m.copy(), p)
            partial[group] = 'contradiction'
            with pytest.raises(ValueError, match='Ticket grouping differs'):
                invoke(helper, partial, m.copy(), p)


@pytest.mark.parametrize('level', ['full', 'summary'])
def test_legacy_float_evidence_requires_explicit_recovery(level):
    for origin in ('float64', 'float32'):
        data, p = source('round', origin, level)
        t, m, _ = compose_bets(data, p, match_columns=ID)
        for helper in ('batch', 'finalize'):
            with pytest.raises(ValueError, match='rebuild from the original ledger'):
                invoke(helper, t.drop(columns=identity.FIELD), m, p)
            for payload in (None, np.nan, '', '{}', '[]', 1, 'null'):
                changed = t.copy(); changed[identity.FIELD] = payload
                with pytest.raises(ValueError): invoke(helper, changed, m, p)


@pytest.mark.parametrize('edit', ['version', 'groups', 'keys', 'dtypes', 'cells', 'value', 'kind', 'token'])
def test_evidence_is_validated_every_call(edit):
    data, p = source('round', 'float32', 'summary')
    t, m, _ = compose_bets(data, p, match_columns=ID)
    doc = json.loads(t[identity.FIELD].iloc[0])
    if edit == 'value': doc['cells'][0][4][1] = float(9.).hex()
    elif edit == 'kind': doc['cells'][0][4][0] = 'float'
    elif edit == 'token': doc['cells'][0][4][1] = '0x1.199999999999ap+0'
    else: doc[edit] = None
    t[identity.FIELD] = json.dumps(doc)
    for helper in ('batch', 'finalize'):
        with pytest.raises(ValueError): invoke(helper, t.copy(), m.copy(), p)


@pytest.mark.parametrize('width', [4, 6, 8, 32])
def test_identity_hash_work_is_bounded(width, monkeypatch):
    data, p = source('round', 'float32', 'summary')
    keys = []
    for i in range(width):
        key = f'key_{i}'; keys.append(key)
        data[key] = pd.Series([1.1, 2.2], dtype='Float32')
    t, m, _ = compose_bets(data, p, match_columns=tuple(keys))
    row = t.iloc[0].to_dict(); groups = json.loads(row[identity.FIELD])['groups']
    original = identity.hashlib.sha256
    calls = []
    def counted(value): calls.append(value); return original(value)
    monkeypatch.setattr(identity.hashlib, 'sha256', counted)
    identity.validate_identity(row, m, groups, keys)
    assert len(calls) == 1
    row['ticket_id'] = 'stale'
    with pytest.raises(ValueError, match='original ticket identity'):
        identity.validate_identity(row, m, groups, keys)
    assert len(calls) == 2


def test_integer_workflow_does_not_gain_evidence_column():
    data = composite_population()
    t, _, _ = compose_bets(data, policy(), match_columns=ID)
    assert identity.FIELD not in t
