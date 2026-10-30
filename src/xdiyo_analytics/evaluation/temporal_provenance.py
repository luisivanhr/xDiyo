"""Mandatory, evidence-derived temporal provenance for shared betting paths.

Certificates are integrity records, not signatures or attestations of truth.
Unknown legacy evidence stays inspectable and unverified. No date is invented.
"""
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from numbers import Number
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import hashlib
import json
import base64
import zlib
import numpy as np
import pandas as pd
from .timestamps import timestamps

VERSION = 'xdiyo.temporal.v1'
CERT = 'temporal_certificate'
POOL = 'temporal_pool'
STATUS = 'temporal_status'
EVIDENCE = 'forecast_evidence'
_ORDER = {'verified': 0, 'conditional_research': 1, 'unverified': 2}
_OUTPUT = {CERT, POOL, STATUS, 'temporal_issues', 'comparison_eligible', 'temporal_offer'}
_MUTABLE = {'take', 'eligible', 'reason', 'stake', 'nominal_stake', 'funded',
            'ticket_id', 'leg_number', 'template', 'accounting_status'}
_STAGES = {'preprocessing', 'ratings', 'calibration', 'tuning', 'fitting'}
_DEPENDENCY_TIMES = ('model_available_as_of', 'features_available_as_of',
                     'preprocessing_available_as_of', 'ratings_available_as_of',
                     'calibration_available_as_of', 'tuning_available_as_of',
                     'training_label_available_as_of', 'training_cutoff')
_CONTENT_CACHE = ContextVar('temporal_content_cache',default=None)


def _validation_session(fn):
    @wraps(fn)
    def run(*args,**kwargs):
        if _CONTENT_CACHE.get() is not None:
            return fn(*args,**kwargs)
        token = _CONTENT_CACHE.set({})
        try:
            return fn(*args,**kwargs)
        finally:
            _CONTENT_CACHE.reset(token)
    return run


def _evidence_series(values, index):
    """Native round-trip string dtype without duplicating repeated pool buffers."""
    dtype = pd.Series(['']).dtype
    if isinstance(dtype, pd.StringDtype) and dtype.storage == 'pyarrow' and len(values):
        import pyarrow as pa
        chunks = {}
        shared = []
        for value in values:
            if value not in chunks:
                chunks[value] = pa.array([value], type=pa.large_string())
            shared.append(chunks[value])
        return pd.Series(pd.array(pa.chunked_array(shared), dtype=dtype), index=index)
    return pd.Series(values, index=index, dtype=dtype)


def _content(dependency, issues, identity):
    """Content-addressed inline evidence or an explicitly referenced local file.

    No network discovery, guessed paths, or persisted approval caches. Missing
    external artifacts downgrade verification; changed content is contradictory.
    """
    expected = dependency.get('hash')
    if expected is None:
        issues.append('missing_dependency_hash')
        return
    cache = _CONTENT_CACHE.get()
    if 'content' in dependency:
        if _policy_status(_json(dependency['content'])) == 'unverified':
            issues.append('unsupported_dependency_content')
        actual = dependency_hash(dependency['content'])
    elif dependency.get('path'):
        path = Path(dependency['path'])
        if not path.is_absolute():
            raise ValueError(f'Temporal provenance {identity}: artifact paths must be absolute.')
        if not path.is_file():
            issues.append('unavailable_dependency_content')
            return
        key = ('file',str(path),expected)
        if cache is not None and key in cache:
            actual = cache[key]
        else:
            with path.open('rb') as source:
                actual = hashlib.file_digest(source,'sha256').hexdigest()
            if cache is not None:
                cache[key] = actual
    else:
        issues.append('unavailable_dependency_content')
        return
    if actual != expected:
        raise ValueError(f'Temporal provenance {identity}: dependency content hash changed.')


def _json(value):
    if callable(value):
        return {'unsupported_callable':getattr(value,'__qualname__',type(value).__qualname__)}
    if is_dataclass(value):
        parameters = {f.name:getattr(value,f.name) for f in fields(value)}
        # Model ballot order is not a decision dependency for native AND/OR.
        if type(value).__name__ == 'DecisionLayer' and type(value).__module__ == 'xdiyo_analytics.evaluation.decision_layer':
            parameters['models'] = sorted(parameters['models'])
        return {'type': type(value).__qualname__, 'parameters': _json(parameters)}
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)) or value is pd.NaT:
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value,float) and not np.isfinite(value):
        return {'nonfinite_float':str(value)}
    if isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, 'dictionaries'):
        return _json(value.dictionaries())
    return {'unsupported_object':type(value).__module__+'.'+type(value).__qualname__}


def _dump(value):
    return json.dumps(_json(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def dependency_hash(value):
    """Stable JSON digest; integer identities never pass through float arrays."""
    return hashlib.sha256(_dump(value).encode()).hexdigest()


def _pool_digest(value, cache):
    key = ('digest', value)
    if key not in cache:
        cache[key] = hashlib.sha256(value.encode()).hexdigest()
    return cache[key]


def _policy_status(policy):
    if isinstance(policy,dict):
        if 'unsupported_callable' in policy or 'unsupported_object' in policy:
            return 'unverified'
        if policy.get('type') in {'LearnedGate','LearnedAllocation','HistoricalRateSource'}:
            return 'unverified' # No invented fit/label availability for legacy fitted policies.
        if policy.get('assume_available_at_decision') is True:
            return 'unverified' # A bare boolean is not a bounded assumption.
        return max((_policy_status(v) for v in policy.values()),default='verified',key=_ORDER.get)
    if isinstance(policy,list):
        return max((_policy_status(v) for v in policy),default='verified',key=_ORDER.get)
    return 'verified'


def _stamp(value, name, identity, *, strict=False):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    if (isinstance(value, Number) or (isinstance(value,str) and not value.strip())
            or not isinstance(value,(str,date,datetime,np.datetime64))):
        raise ValueError(f'Temporal provenance {identity}: {name}: Timestamps require date/time values, not numeric epochs or objects.')
    t = pd.Timestamp(value)
    if pd.isna(t):
        return None
    if strict and pd.Timestamp(value).tzinfo is None:
        raise ValueError(f'Temporal provenance {identity}: {name} requires an explicit timezone: {value!r}.')
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def decision_inputs(frame):
    """Opaque audit evidence is for validation, never an input to a policy.

    A complete audit can contain other independent decision groups. Exposing its
    serialized contents to a callback would expose later groups' forecasts.
    """
    bound=set().union(*(row.keys() for row in _records(frame))) if len(frame) else set(frame)
    columns=[k for k in frame if k in bound and k not in {EVIDENCE,CERT,POOL,'comparison_eligible'}
             and not str(k).startswith('temporal_')]
    result=frame.loc[:,columns].copy()
    result.attrs={}
    return result


def _safe(row):
    from .decision_layer import _outcome_column
    result = {}
    for k, v in row.items():
        if k in _OUTPUT or k in _MUTABLE or str(k).startswith('_'):
            continue
        if _outcome_column(k) and k not in {'expected_profit','training_label_available_as_of'}:
            continue
        if (k == 'odds' or str(k).endswith('::odds')) and isinstance(v,str):
            try:
                v = float(v)
            except ValueError:
                pass # The existing numeric contract reports malformed prices.
        if k.endswith(('_at', '_as_of', '_cutoff')) and v is not None:
            t = _stamp(v, k, 'binding')
            v = ((t if pd.Timestamp(v).tzinfo is not None else t.tz_localize(None)).isoformat()
                 if t is not None else None)
        if k == 'quote_legs' and isinstance(v,str):
            try:
                nested=json.loads(v)
                if isinstance(nested,dict):
                    nested=pd.DataFrame(nested).to_dict('records')
                if isinstance(nested,list):
                    v=[_safe(r) for r in nested]
            except (ValueError,TypeError):
                pass # Native retained-quote validation reports malformed evidence.
        result[k] = _json(v)
    return result


def _later(a, b, names, identity, *, equal=False):
    if a is not None and b is not None and (a >= b if equal else a > b):
        raise ValueError(f'Temporal provenance {identity}: {names[0]}={a.isoformat()} '
                         f'{">=" if equal else ">"} {names[1]}={b.isoformat()}.')


def _available_before(time, cutoff, decision, name, identity):
    # Missing intermediate evidence cannot hide an already known contradiction.
    _later(time, decision, (name, 'decision_at'), identity)
    _later(time, cutoff, (name, 'forecast_as_of'), identity)


def _dependency_times(dep, cutoff, decision, identity, *, training_cutoff=None):
    """Check explicit evidence nodes, never arbitrary artifact content/creation dates."""
    time = _stamp(dep.get('available_as_of'), 'available_as_of', identity, strict=True)
    _available_before(time, cutoff, decision, 'dependency_available_as_of', identity)
    _later(time, training_cutoff, ('training_input_available_as_of', 'training_cutoff'), identity)
    for collection in ('dependencies', 'inputs'):
        for child in dep.get(collection, []):
            _dependency_times(child, cutoff, decision, f'{identity}/{collection} {child.get("id")}', training_cutoff=training_cutoff)
    return time


def _assumptions(evidence):
    assumptions = evidence.get('assumptions', [])
    found = {}
    for a in assumptions:
        if not isinstance(a, dict) or any(not a.get(k) for k in ('id','scope','rationale','reference','available_as_of')):
            raise ValueError('Temporal assumptions require identity, bounded scope, rationale, reference and available_as_of.')
        if a['id'] in found:
            raise ValueError('Duplicate temporal assumption identity.')
        _stamp(a['available_as_of'], 'assumed available_as_of', a['id'], strict=True)
        found[a['id']] = a
    return found


def _forecast(row, keys, *, decision_limit=None):
    cache = _CONTENT_CACHE.get()
    key = ('forecast',tuple(keys),dependency_hash(_safe(row)),decision_limit)
    if cache is not None and key in cache:
        return cache[key]
    result = _forecast_impl(row,keys,decision_limit=decision_limit)
    if cache is not None:
        cache[key] = result
    return result


def _forecast_impl(row, keys, *, decision_limit=None):
    identity = {k: row.get(k) for k in keys}
    issues = []
    fields = ('decision_at','kickoff_at','first_kickoff_at','forecast_as_of','prediction_cutoff',
              'model_available_as_of','features_available_as_of','issued_at','quote_at','assumed_available_at')
    ts = {k:_stamp(row.get(k), k, identity) for k in fields}
    decision = ts['decision_at']
    if decision_limit is not None:
        decision = decision_limit if decision is None else min(decision, decision_limit)
    first = ts['first_kickoff_at'] if ts['first_kickoff_at'] is not None else ts['kickoff_at']
    _later(decision, first, ('decision_at','first_kickoff_at'), identity, equal=True)
    for k in ('forecast_as_of','prediction_cutoff','issued_at','quote_at','assumed_available_at'):
        _later(ts[k], decision, (k,'decision_at'), identity)
    cutoff = ts['forecast_as_of'] if ts['forecast_as_of'] is not None else ts['prediction_cutoff']
    if ts['forecast_as_of'] is not None and ts['prediction_cutoff'] is not None and ts['forecast_as_of'] != ts['prediction_cutoff']:
        raise ValueError(f'Temporal provenance {identity}: forecast_as_of differs from prediction_cutoff.')
    for k in _DEPENDENCY_TIMES:
        _available_before(_stamp(row.get(k), k, identity), cutoff, decision, k, identity)
    train = _stamp(row.get('training_cutoff'), 'training_cutoff', identity)
    _later(_stamp(row.get('training_label_available_as_of'), 'training_label_available_as_of', identity),
           train, ('training_label_available_as_of','training_cutoff'), identity)
    # Legacy model stream fields retain their existing interpretation in gates.
    # Do not reinterpret trained_through/artifact_vintage as simulated availability.
    for k, value in row.items():
        if '::' in str(k) and str(k).split('::')[-1] in ('issued_at','forecast_as_of','prediction_cutoff','quote_at',*_DEPENDENCY_TIMES):
            _later(_stamp(value,k,identity), decision, (k,'decision_at'), identity)
    for model in {k.split('::')[0] for k in row if str(k).endswith('::probability')}:
        issued = _stamp(row.get(model+'::issued_at'),'issued_at',identity)
        trained = _stamp(row.get(model+'::trained_through'),'trained_through',identity)
        vintage = _stamp(row.get(model+'::artifact_vintage'),'artifact_vintage',identity)
        # These comparisons are the existing gate contract, not a new meaning
        # for these ambiguous legacy fields or a claim of verified availability.
        _later(trained,issued,('trained_through','issued_at'),identity,equal=True)
        _later(vintage,issued,('artifact_vintage','issued_at'),identity)
    if any(row.get(k) is None or pd.isna(row.get(k)) for k in keys):
        issues.append('missing_fixture_identity')
    for k in ('decision_at','forecast_as_of','first_kickoff_at'):
        if ts[k] is None:
            issues.append('missing_'+k)
    for k in fields:
        if ts[k] is not None and pd.Timestamp(row[k]).tzinfo is None:
            issues.append('legacy_naive_'+k)
    quote_conditional = _quotes(row,ts,ts['decision_at'],issues)
    raw = row.get(EVIDENCE)
    if not isinstance(raw, str) or not raw:
        return 'unverified', sorted(set(issues+['missing_forecast_lineage']))
    evidence = json.loads(raw)
    if evidence.get('version') != VERSION:
        raise ValueError(f'Temporal provenance {identity}: unsupported forecast evidence version.')
    bound = evidence.get('binding', {})
    current = _safe({k:row.get(k) for k in bound})
    if current != bound or evidence.get('binding_hash') != dependency_hash(bound):
        raise ValueError(f'Temporal provenance {identity}: forecast binding changed; regenerate features/forecast evidence.')
    required = {*keys, 'forecast_as_of','first_kickoff_at','feature_hash','model_hash','preprocessing_hash'}
    probabilities = {k for k in row if k in ('p_win','p_loss','p_push','probability') or str(k).endswith('::probability')}
    required.update(probabilities)
    if not probabilities:
        issues.append('missing_forecast_probabilities')
    if not required <= set(bound) or any(bound.get(k) is None for k in required):
        issues.append('incomplete_forecast_binding')
    assumptions = _assumptions(evidence)
    conditional = quote_conditional
    for k in ('generated_at','forecast_as_of','model_available_as_of'):
        if _stamp(evidence.get(k),k,identity,strict=True) is None:
            issues.append('missing_'+k)
    declared = _stamp(evidence.get('forecast_as_of'),'forecast_as_of',identity,strict=True)
    _later(declared, decision, ('forecast_as_of', 'decision_at'), identity)
    if declared is not None and cutoff is not None and declared != cutoff:
        raise ValueError(f'Temporal provenance {identity}: declared feature cutoff differs from forecast cutoff.')
    # Use a known evidence cutoff for comparisons, without upgrading missing-row
    # provenance or inventing a cutoff in the forecast binding.
    cutoff = cutoff if cutoff is not None else declared
    available = _stamp(evidence.get('model_available_as_of'),'model_available_as_of',identity,strict=True)
    _available_before(available, cutoff, decision, 'model_available_as_of', identity)
    for k in _DEPENDENCY_TIMES:
        _available_before(_stamp(row.get(k), k, identity), cutoff, decision, k, identity)
        _available_before(_stamp(evidence.get(k), k, identity, strict=True), cutoff, decision, k, identity)
    if ts['model_available_as_of'] is not None and available is not None and ts['model_available_as_of'] != available:
        raise ValueError(f'Temporal provenance {identity}: model availability differs from its evidence.')
    generated = _stamp(evidence.get('generated_at'),'generated_at',identity,strict=True)
    if generated is not None and cutoff is not None and generated > cutoff:
        if not any(a['scope']=='retrospective_replay' for a in assumptions.values()):
            issues.append('retrospective_replay_not_declared')
        conditional = True
    dependencies = evidence.get('dependencies', [])
    roles = {d.get('role') for d in dependencies}
    if not {'features','source','schedule','membership'} <= roles:
        issues.append('incomplete_information_dependencies')
    for dep in dependencies:
        ident = f'{identity}/dependency {dep.get("id")}'
        if not all(dep.get(k) for k in ('id','hash','reference')):
            issues.append('unbound_dependency')
        _content(dep,issues,ident)
        if dep.get('role') == 'features' and dep.get('hash') is not None and row.get('feature_hash') is not None and dep.get('hash') != row['feature_hash']:
            raise ValueError(f'Temporal provenance {ident}: feature hash differs from forecast binding.')
        time = _dependency_times(dep, cutoff, decision, ident)
        if time is None:
            issues.append('unknown_dependency_availability')
        kind = dep.get('availability_kind')
        if kind == 'assumed':
            a = assumptions.get(dep.get('assumption_id'))
            if a is None or a['scope'] != dep.get('id') or _stamp(a['available_as_of'],'assumption',ident,strict=True) != time:
                raise ValueError(f'Temporal provenance {ident}: unbounded availability assumption.')
            conditional = True
        elif kind != 'observed':
            issues.append('unsupported_dependency_availability')
    stages = evidence.get('stages', [])
    if {s.get('kind') for s in stages} != _STAGES or len(stages) != len(_STAGES):
        issues.append('incomplete_pipeline_lineage')
    for stage in stages:
        ident = f'{identity}/{stage.get("kind")}'
        artifact_key = {'fitting':'model_hash','preprocessing':'preprocessing_hash'}.get(stage.get('kind'))
        if artifact_key and stage.get('artifact_hash') is not None and row.get(artifact_key) is not None and stage.get('artifact_hash') != row[artifact_key]:
            raise ValueError(f'Temporal provenance {ident}: artifact hash differs from forecast binding.')
        if stage.get('mode') == 'not_used':
            if not stage.get('reason'):
                issues.append('undocumented_unused_stage')
            continue
        if stage.get('mode') != 'fitted' or not stage.get('artifact_hash'):
            issues.append('unbound_fitted_artifact')
        artifact = stage.get('artifact',{})
        _dependency_times(artifact, cutoff, decision, f'{ident}/artifact')
        if artifact.get('hash') != stage.get('artifact_hash'):
            issues.append('unbound_fitted_artifact_content')
        else:
            _content(artifact,issues,ident)
        train = _stamp(stage.get('training_cutoff'),'training_cutoff',ident,strict=True)
        ready = _stamp(stage.get('available_as_of'),'available_as_of',ident,strict=True)
        _available_before(train, cutoff, decision, 'training_cutoff', ident)
        _available_before(ready, cutoff, decision, 'stage_available_as_of', ident)
        if stage.get('kind')=='fitting':
            _later(ready,available,('fitted_artifact_available_as_of','model_available_as_of'),ident)
        created = _stamp(stage.get('created_at'),'created_at',ident,strict=True)
        _later(created,generated,('physical_artifact_created_at','physical_forecast_generated_at'),ident)
        if created is not None and ready is not None and created > ready:
            conditional = True
            if not any(a['scope']=='retrospective_replay' for a in assumptions.values()):
                issues.append('retrospective_replay_not_declared')
        for k in ('created_at',):
            if _stamp(stage.get(k),k,ident,strict=True) is None:
                issues.append('missing_physical_artifact_creation')
        if train is None or ready is None or not stage.get('fit_rows_hash'):
            issues.append('incomplete_training_provenance')
        membership=stage.get('fit_rows',{})
        _dependency_times(membership, cutoff, decision, f'{ident}/fit_rows')
        if membership.get('hash') != stage.get('fit_rows_hash'):
            issues.append('unbound_training_membership')
        else:
            _content(membership,issues,ident)
        _later(train,ready,('training_cutoff','model_available_as_of'),ident)
        if stage.get('fit_partition') != 'train':
            raise ValueError(f'Temporal provenance {ident}: fitted stage is not train-only.')
        for dep in stage.get('inputs', []):
            time = _dependency_times(dep, cutoff, decision, f'{ident}/input {dep.get("id", dep.get("role"))}', training_cutoff=train)
            if time is None or not dep.get('hash'):
                issues.append('missing_training_input_evidence')
            _content(dep,issues,ident)
            if dep.get('role')=='training_labels' and dep.get('rows_hash') != stage.get('fit_rows_hash'):
                issues.append('unbound_training_label_membership')
            _later(time,train,('training_input_available_as_of','training_cutoff'),ident)
        if not stage.get('inputs'):
            issues.append('missing_training_inputs')
        if stage.get('kind') in {'fitting','calibration','tuning'} and not any(d.get('role')=='training_labels' for d in stage.get('inputs', [])):
            issues.append('missing_training_label_availability')
    if not any(s.get('kind')=='fitting' and s.get('mode')=='fitted' for s in stages):
        issues.append('missing_fitted_model')
    streams = evidence.get('model_streams',{})
    for model in {k.split('::')[0] for k in row if str(k).endswith('::probability')}:
        if model not in streams:
            issues.append('missing_model_stream_lineage')
            continue
        own = {k:v for k,v in row.items() if '::' not in k}
        own.update({k.split('::',1)[1]:v for k,v in row.items() if k.startswith(model+'::')})
        own[EVIDENCE] = streams[model]
        status, missing = _forecast(own,keys,decision_limit=decision)
        issues.extend(missing)
        conditional |= status=='conditional_research'
    return ('unverified' if issues else 'conditional_research' if conditional else 'verified'), sorted(set(issues))


def _quotes(row,ts,decision,issues):
    """Reuse the native contract even when forecast lineage is absent."""
    from .quote_availability import QuoteAvailability, QUOTE_IDENTITY, ASSUMPTION_FIELDS
    conditional = False
    if all(row.get(k) is not None and not pd.isna(row.get(k)) for k in QUOTE_IDENTITY if k != 'quote_at'):
        mode = row.get('quote_availability_mode', 'observed')
        if mode == 'research_assumed':
            contract = QuoteAvailability(mode=mode, assumption_id=row.get('quote_assumption_id'),
                rationale=row.get('quote_assumption_rationale'), reference=row.get('quote_assumption_reference'))
            conditional = True
        else:
            contract = QuoteAvailability()
        if ts['quote_at'] is None and mode != 'research_assumed':
            issues.append('unknown_quote_publication')
        elif decision is not None:
            try:
                cache=_CONTENT_CACHE.get()
                fields=(*QUOTE_IDENTITY,*ASSUMPTION_FIELDS,'quote_legs')
                key=('quote',dependency_hash({k:row[k] for k in fields if k in row}),dependency_hash(contract))
                if cache is None or key not in cache:
                    contract.validate(pd.DataFrame([row]), decision, complete=True)
                    if cache is not None:
                        cache[key]=True
            except TypeError as exc:
                raise ValueError(f'Temporal quote evidence {row.get("event_id")}: invalid price or identity.') from exc
    else:
        issues.append('incomplete_quote_identity')
    return conditional


@_validation_session
def bind_forecasts(frame, evidence, *, match_columns=('event_id',)):
    """Producer hook: evidence must align exactly by unique index, never by position."""
    if not frame.index.is_unique or set(evidence) != set(frame.index):
        raise ValueError('Forecast evidence must match exact unique forecast row identities.')
    if EVIDENCE in frame or CERT in frame:
        validate_frame(frame)
    result = frame.copy()
    records, statuses, limitations = [], [], []
    for key, row in zip(frame.index, frame.to_dict('records')):
        item = dict(evidence[key])
        binding = _safe({k:v for k,v in row.items() if k != EVIDENCE})
        item.update(version=VERSION, binding=binding, binding_hash=dependency_hash(binding),match_columns=list(match_columns))
        row[EVIDENCE] = _dump(item)
        status, issues = _forecast(row, match_columns)
        statuses.append(status); limitations.append(_dump(issues))
        records.append(row[EVIDENCE])
    result[EVIDENCE] = records
    result[STATUS] = statuses
    result['comparison_eligible'] = result[STATUS].eq('verified')
    result['temporal_issues'] = limitations
    return result


def _seal(payload):
    document = _dump({'version':VERSION,'payload':payload,'sha256':dependency_hash(payload)})
    if payload['kind'] in {'candidate_pool','comparison'}:
        # Lossless portable evidence, not lossy summary mode. Keep shared pools
        # small when CSV/HTML/data-only bundles repeat a row's dependency link.
        return VERSION+':z:'+base64.b64encode(zlib.compress(document.encode(),level=6)).decode('ascii')
    return document


@_validation_session
def _open(value, cache):
    if value in cache:
        return cache[value]
    shared = _CONTENT_CACHE.get()
    shared_key = ('certificate',value)
    if shared_key in shared:
        cache[value]=shared[shared_key]
        return cache[value]
    document = zlib.decompress(base64.b64decode(value.split(':z:',1)[1],validate=True)).decode() if value.startswith(VERSION+':z:') else value
    packet = json.loads(document)
    if packet.get('version') != VERSION or packet.get('sha256') != dependency_hash(packet.get('payload')):
        raise ValueError('Temporal certificate version/hash mismatch; dependency evidence was modified.')
    p = packet['payload']
    if p.get('policy_hash') != dependency_hash(p.get('policy')):
        raise ValueError('Temporal policy dependency hash mismatch.')
    keys = p['match_columns']
    statuses = [_policy_status(p['policy'])]
    if not p['candidates'] and not p.get('parents'):
        statuses.append('unverified')
    for row in p['candidates']:
        statuses.append(_forecast(row, keys)[0])
    if p.get('selection_groups') and p['candidates']:
        validate_selection_groups(pd.DataFrame(p['candidates']),p['selection_groups'])
    for parent in p.get('parents', []):
        statuses.append(_open(parent,cache)['classification'])
    if p.get('inspected'):
        inspected=pd.DataFrame([r['current'] for r in p['inspected']])
        for column,field in ((CERT,'certificate'),(POOL,'pool')):
            inspected[column]=pd.Series([p['parents'][r[field]] if r[field] is not None else None
                for r in p['inspected']],dtype=object)
        statuses.append(validate_frame(inspected,_cache=cache)[0])
    p['classification'] = max(statuses or ['unverified'], key=_ORDER.get)
    p['_candidate_fields'] = set().union(*(r.keys() for r in p['candidates']))
    cache[value] = p
    shared[shared_key] = p
    return p


def inspect_certificate(value):
    """Decode and revalidate an exported ticket/pool/comparison certificate."""
    return {k:v for k,v in _open(value,{}).items() if not k.startswith('_')}


@_validation_session
def _records(frame):
    if CERT in frame and len(frame) and frame[CERT].map(lambda v:isinstance(v,str)).all():
        cache={}
        bound=set()
        for value in frame[CERT].unique():
            packet=_open(value,cache)
            bound.update(packet.get('_candidate_fields',()))
            bound.update(packet.get('ticket',{}))
        required = bound | _OUTPUT | _MUTABLE | {'bet','kind','time','amount','wealth','reserve','cash','realized_net_pnl','currency',EVIDENCE}
        # Newly supplied temporal assertions are checked too. Unused wide
        # display columns are not decision dependencies of an already bound offer.
        columns=[k for k in frame if k in required or str(k).startswith('temporal_')
                 or str(k).startswith('settlement_evidence')
                 or str(k).endswith(('_at','_as_of','_cutoff','_hash'))]
        frame=frame.loc[:,columns]
    records = frame.to_dict('records')
    if frame.index.name == 'ticket_id' and 'ticket_id' not in frame:
        for key, row in zip(frame.index, records):
            row['ticket_id'] = key
    return records


def _ticket_binding(row):
    # Ticket identity and its nominal stake are immutable decision inputs.
    result = _safe({k:v for k,v in row.items() if k != 'bet'})
    for key in ('ticket_id','nominal_stake'):
        if key in row:
            result[key] = _json(row[key])
    return result


def _member(row, packet, cache):
    """Call-local indexed lookup; certificates are never approval caches."""
    groups = packet.get('_member_index')
    if groups is None:
        groups = {}
        for candidate in packet['candidates']:
            columns = tuple(sorted(candidate))
            groups.setdefault(columns, set()).add(_dump(candidate))
        packet['_member_index'] = groups
    current = _safe(row)
    if any(_dump({k:current.get(k) for k in columns}) in values for columns,values in groups.items()):
        return True
    # Legacy helpers allow optional duplicated model identity assertions to be
    # absent. Contradictions still fail; missing assertions cannot verify.
    for candidate in packet['candidates']:
        missing = set(candidate)-set(current)
        if missing and all('::' in k and k.split('::')[-1] in packet['match_columns'] for k in missing):
            if all(current.get(k)==v for k,v in candidate.items() if k not in missing):
                return True
    return any(_member(row, _open(parent,cache),cache) for parent in packet.get('parents', []))


@_validation_session
def validate_subset(frame, pool):
    """A reusable helper's membership must still belong to the supplied pool."""
    cache = {}
    packets = [_open(value,cache) for value in pool[CERT].dropna().unique()]
    for row in _records(frame):
        if not any(_member(row,packet,cache) for packet in packets):
            raise ValueError('Temporal candidate membership differs from the supplied complete pool.')


@_validation_session
def validate_ticket_members(tickets, members):
    cache = {}
    grouped = members.groupby('ticket_id',sort=False).indices if len(members) else {}
    for row in _records(tickets):
        if not isinstance(row.get(CERT),str):
            continue
        packet = _open(row[CERT],cache)
        if packet.get('kind') != 'ticket':
            continue
        actual = [_safe(r) for r in _records(members.iloc[grouped.get(row['ticket_id'],[])] )]
        expected = packet['legs']
        # Additional reporting fields may be present, but every bound field and
        # the exact multiplicity must survive; no membership rebinding.
        fields = set().union(*(r.keys() for r in expected))
        optional = {k for k in fields if '::' in k and k.split('::')[-1] in packet['match_columns'] and k not in members}
        fields -= optional
        expected = [{k:v for k,v in r.items() if k not in optional} for r in expected]
        projected = [{k:r.get(k) for k in fields} for r in actual]
        if sorted(map(_dump,projected)) != sorted(map(_dump,expected)):
            raise ValueError(f'Temporal ticket {row["ticket_id"]}: exact leg membership changed.')



def ledger_event_hash(row):
    fields=('time','kind','ticket_id','amount','wealth','reserve','cash','realized_net_pnl','currency',
            CERT,POOL,'temporal_offer','temporal_allocation_policy_hash','settlement_evidence_hash')
    bound={k:row.get(k) for k in fields}
    bound['time']=_stamp(bound['time'],'ledger time',bound['ticket_id'])
    return dependency_hash(bound)


@_validation_session
def validate_frame(frame, *, placement_at=None, _cache=None):
    """Revalidate retained certificates and exact bindings on every consumption."""
    cache, states, issues = {} if _cache is None else _cache, [], []
    for row in _records(frame):
        if row.get('kind') in {'place','unfunded','settle'}:
            digest=row.get('temporal_event_hash')
            if not isinstance(digest,str):
                states.append('unverified'); issues.append('missing_ledger_event_binding')
            elif digest != ledger_event_hash(row):
                raise ValueError(f'Temporal ledger event changed for ticket {row.get("ticket_id")}.')
            if all(row.get(k) is not None for k in ('cash','reserve','wealth')):
                values=np.asarray([row[k] for k in ('cash','reserve','wealth')],dtype=float)
                if not np.isfinite(values).all() or min(values[:2]) < -1e-9 or abs(values[0]+values[1]-values[2]) > 1e-9:
                    raise ValueError('Saved bankroll cash/reserve conservation violated.')
        if row.get('kind') in {'place','unfunded','settle'} and not isinstance(row.get('temporal_allocation_policy'),str):
            states.append('unverified')
            issues.append('missing_allocation_policy_evidence')
        if isinstance(row.get('temporal_allocation_policy'),str):
            policy=json.loads(row['temporal_allocation_policy'])
            if dependency_hash(policy) != row.get('temporal_allocation_policy_hash'):
                raise ValueError('Allocation policy evidence hash changed.')
            states.append(_policy_status(policy))
        if row.get('kind')=='settle':
            if not isinstance(row.get('settlement_evidence'),str):
                states.append('unverified')
                issues.append('missing_complete_settlement_evidence')
            else:
                from .bankroll import _settlement_support
                proof=json.loads(row['settlement_evidence'])
                if dependency_hash(proof)!=row.get('settlement_evidence_hash'):
                    raise ValueError('Settlement evidence hash changed.')
                original=json.loads(row.get('temporal_offer','{}'))
                original.update({k:row[k] for k in (CERT,POOL) if k in row})
                if not _settlement_support(proof,original):
                    states.append('unverified')
                latest=timestamps(pd.Series(proof['available_at'])).max()
                _later(latest,_stamp(row.get('time'),'settlement time',row.get('ticket_id')),
                       ('last_leg_available_at','settlement time'),row.get('ticket_id'))
        if isinstance(row.get('temporal_offer'), str):
            original = json.loads(row['temporal_offer'])
            if original.get('ticket_id') != _json(row.get('ticket_id')):
                raise ValueError('Ledger ticket identity differs from its temporal offer.')
            row = {**original, **{k:row[k] for k in (CERT,POOL) if k in row}}
        value = row.get(CERT)
        if isinstance(value,str) and value:
            p = _open(value,cache)
            binding = p.get('ticket')
            if binding is not None:
                missing = set(binding)-set(row)
                if missing:
                    states.append('unverified')
                if _ticket_binding({k:row.get(k) for k in binding if k not in missing}) != {k:v for k,v in binding.items() if k not in missing}:
                    changed = {k for k,v in binding.items() if k not in missing and _ticket_binding({k:row[k]}).get(k)!=v}
                    if changed & {'fold_id','competition_id','season_id','source_league','source_season','stage','round'}:
                        raise ValueError('Ticket grouping differs from certified temporal identity.')
                    raise ValueError(f'Temporal ticket binding changed: {sorted(changed)} differ for ticket {row.get("ticket_id")}.')
                legs = p['legs']
                _check_ticket(legs, p['match_columns'])
                pool = row.get(POOL)
                if not isinstance(pool,str) or _pool_digest(pool,cache) != p.get('pool_hash'):
                    raise ValueError('Temporal ticket candidate pool is missing or changed.')
                pool_packet = _open(pool,cache)
                pool_status = pool_packet['classification']
                if any(not _member(leg,pool_packet,cache) for leg in legs) and pool_packet.get('parents'):
                    raise ValueError('Temporal ticket legs are not members of its certified candidate pool.')
                states.append(pool_status)
                for key,value in binding.items():
                    row.setdefault(key,value)
            else:
                if not _member(row,p,cache):
                    raise ValueError('Temporal candidate binding changed; Retained ticket quote evidence or forecast identity differs from its pool certificate.')
                # Also inspect new timestamp fields, not just those in the certificate.
                _forecast(row, p['match_columns'])
                if p['_candidate_fields']-set(_safe(row)):
                    states.append('unverified')
            states.append(p['classification'])
        else:
            status, reasons = _forecast(row, ('event_id',))
            states.append(status); issues.extend(reasons)
        if placement_at is not None:
            if not isinstance(row.get(CERT),str):
                states.append('unverified')
                issues.append('missing_complete_candidate_pool')
            decision = _stamp(row.get('decision_at'),'decision_at','placement')
            if decision is not None and decision != _stamp(placement_at,'placement_at','placement'):
                raise ValueError('Placement time differs from certified decision_at.')
            first = row.get('first_kickoff_at',row.get('kickoff_at'))
            _later(_stamp(placement_at,'placement_at','placement'),_stamp(first,'first_kickoff_at','placement'),
                   ('placement_at','first_kickoff_at'), row.get('event_id'), equal=True)
    return max(states or ['unverified'],key=_ORDER.get), sorted(set(issues))


def validate_selection_groups(frame, groups, *, at=None):
    """Check information used together, including subsequently discarded rows.

    Callers supply their existing strategy grouping; this defines no new rounds.
    """
    grouped = frame.groupby([frame[k] for k in groups],sort=False,dropna=False,observed=True) if groups else [((),frame)]
    for group, rows in grouped:
        decisions = timestamps(rows.get('decision_at',pd.Series(pd.NaT,index=rows.index)))
        cutoff = _stamp(at,'decision_at',group) if at is not None else decisions.min()
        if pd.isna(cutoff):
            continue
        for row in rows.to_dict('records'):
            for key in row:
                field = str(key).split('::')[-1]
                if field in {'forecast_as_of','prediction_cutoff','issued_at','quote_at','assumed_available_at',
                             *_DEPENDENCY_TIMES}:
                    _later(_stamp(row[key],key,row.get('event_id')),cutoff,(key,'candidate_group_decision_at'),
                           {'event_id':row.get('event_id'),'group':group})
            # Validate nested lineage at the actual selection time too. Keep the
            # original row/binding intact; this is a comparison bound, no relabel.
            _forecast(row, ('event_id',), decision_limit=cutoff)


@_validation_session
def guard_candidates(frame, policy, match_columns=('event_id',), *, selection_groups=None):
    """Complete preselection pool, including take=False rows. No optional bypass."""
    result = frame.copy()
    if CERT in frame:
        validate_frame(frame)
    if selection_groups:
        validate_selection_groups(frame,selection_groups)
    rows = [_safe(row) for row in frame.to_dict('records')]
    states, issues = [], []
    for row in rows:
        status, reasons = _forecast(row,match_columns)
        states.append(status); issues.extend(reasons)
    parents = sorted(set(frame[CERT].dropna())) if CERT in frame else []
    cache = {}
    states.extend(_open(p,cache)['classification'] for p in parents)
    states.append(_policy_status(_json(policy)))
    payload = dict(kind='candidate_pool', match_columns=list(match_columns),
        candidates=sorted(rows,key=_dump), policy=_json(policy), policy_hash=dependency_hash(policy), parents=parents,
        selection_groups=selection_groups)
    certificate = _seal(payload)
    result[CERT] = _evidence_series([certificate]*len(result), result.index)
    result[STATUS] = max(states or ['unverified'],key=_ORDER.get)
    result['temporal_issues'] = _dump(sorted(set(issues)))
    result['comparison_eligible'] = result[STATUS].eq('verified')
    return result


def _check_ticket(legs, keys):
    times = [_stamp(r.get('decision_at'),'decision_at',{k:r.get(k) for k in keys}) for r in legs]
    known = {t for t in times if t is not None}
    if len(known)>1:
        raise ValueError('Every ticket leg must share one decision timestamp.')
    for row in legs:
        _forecast(row,keys)


@_validation_session
def attach_tickets(tickets, members, templates, match_columns, pool=None):
    """Publish leg/pool bindings even for legacy flat, gate-free accounting."""
    tickets, members = tickets.copy(), members.copy()
    if CERT in tickets:
        validate_frame(tickets)
        validate_ticket_members(tickets,members)
    certificate_values, statuses, pool_values, cache = [], [], [], {}
    parent_values = sorted(set(members[CERT].dropna())) if CERT in members else []
    if pool is not None and CERT in pool:
        parent_values = sorted(set(parent_values+pool[CERT].dropna().tolist()))
    pool_value = _seal(dict(kind='candidate_pool',match_columns=list(match_columns),
        candidates=[],parents=parent_values,policy=_json(templates),policy_hash=dependency_hash(templates)))
    pool_status = _open(pool_value,cache)['classification']
    pool_hash = _pool_digest(pool_value,cache)
    grouped = members.groupby('ticket_id',sort=False).indices if len(members) else {}
    decisions = []
    for row in tickets.to_dict('records'):
        legs = members.iloc[grouped[row['ticket_id']]]
        records = [_safe(r) for r in _records(legs)]
        _check_ticket(records,match_columns)
        known = {_stamp(r.get('decision_at'),'decision_at','ticket') for r in records}
        derived = next(iter(known)) if len(known)==1 and None not in known else None
        if row.get('decision_at') is not None and derived is not None and _stamp(row['decision_at'],'decision_at',row['ticket_id']) != derived:
            raise ValueError(f'Temporal ticket {row["ticket_id"]}: decision differs from its exact legs.')
        row.setdefault('decision_at',derived)
        decisions.append(row['decision_at'])
        if CERT in legs:
            validate_frame(legs,_cache=cache)
        retained_pool = row.get(POOL) if pool is None else None
        current_pool = retained_pool if isinstance(retained_pool,str) else pool_value
        current_pool_hash = _pool_digest(current_pool,cache)
        current_pool_status = _open(current_pool,cache)['classification']
        # Absence of a complete original pool can never verify a direct helper.
        payload = dict(kind='ticket', match_columns=list(match_columns),
            candidates=records, legs=records, pool_hash=current_pool_hash,
            ticket=_ticket_binding(row), policy=_json(templates[row['bet']]), policy_hash=dependency_hash(templates[row['bet']]))
        value = _seal(payload)
        status = max([_open(value,cache)['classification'],current_pool_status],key=_ORDER.get)
        certificate_values.append(value); statuses.append(status); pool_values.append(current_pool)
    tickets[CERT] = _evidence_series(certificate_values, tickets.index)
    tickets[POOL] = _evidence_series(pool_values, tickets.index)
    tickets[STATUS] = statuses
    if 'decision_at' not in tickets:
        tickets['decision_at'] = pd.Series(decisions,index=tickets.index,dtype='datetime64[ns, UTC]')
    tickets['comparison_eligible'] = tickets[STATUS].eq('verified')
    return tickets, members


@_validation_session
def annotate_metrics(metrics, tickets):
    result = metrics.copy()
    status, _ = validate_frame(tickets)
    parents = sorted({v for column in (CERT,POOL) if column in tickets for v in tickets[column].dropna()})
    parent_index={value:i for i,value in enumerate(parents)}
    inspected=[]
    for row in _records(tickets):
        current=_safe(row)
        current.update({k:_json(row[k]) for k in ('ticket_id','nominal_stake') if k in row})
        inspected.append(dict(current=current,certificate=parent_index.get(row.get(CERT)),pool=parent_index.get(row.get(POOL))))
    evidence = _seal(dict(kind='comparison',match_columns=['event_id'],candidates=[],parents=parents,
                         policy='comparison',policy_hash=dependency_hash('comparison'),inspected=inspected))
    result[STATUS] = status
    result['comparison_eligible'] = status == 'verified'
    result['temporal_comparison_evidence'] = _evidence_series([evidence]*len(result), result.index)
    fields = ['metric','calculation','target','output','value','parameters','sample_hash']
    result['temporal_metric_binding'] = [_dump({k:r.get(k) for k in fields}) for r in metrics.to_dict('records')]
    return result


def metric_classification(metric):
    """A saved success label alone cannot admit legacy ROI into rankings."""
    evidence = metric.get('temporal_comparison_evidence')
    if not isinstance(evidence,str):
        return 'unverified'
    binding = json.loads(metric['temporal_metric_binding'])
    if _json({k:metric.get(k) for k in binding}) != binding:
        raise ValueError('Temporal comparison metric binding changed.')
    return _open(evidence,{})['classification']


def save_temporal_table(frame, path):
    """Portable table with persisted JSON evidence; validate before writing."""
    validate_frame(frame)
    frame.to_parquet(path, compression='zstd', index=True)


def load_temporal_table(path):
    """Legacy migration never invents timestamps or trusts saved success labels."""
    frame = pd.read_parquet(path)
    status, issues = validate_frame(frame)
    frame[STATUS] = status
    frame['comparison_eligible'] = status == 'verified'
    frame['temporal_issues'] = _dump(issues)
    return frame


@_validation_session
def revalidate_artifact(frame):
    """Recovery/export boundary. Legacy tables remain numerical, never verified."""
    if 'output' in frame and frame.output.eq('bets').any():
        result=frame.copy()
        for idx,row in frame.loc[frame.output.eq('bets')].iterrows():
            status=metric_classification(row)
            result.loc[idx,STATUS]=status
            result.loc[idx,'comparison_eligible']=status=='verified'
        return result
    legacy = {'odds','take','profit'} <= set(frame)
    if CERT not in frame and STATUS not in frame and not legacy:
        return frame
    result = frame.copy()
    status, issues = validate_frame(result)
    result[STATUS] = status
    result['comparison_eligible'] = status == 'verified'
    return result


@_validation_session
def revalidate_report(report):
    """Shallow report copy: keep plots, revalidate table/CSV evidence on export."""
    from dataclasses import replace
    tables={}
    def checked(value):
        if not isinstance(value,pd.DataFrame):
            return value
        if id(value) not in tables:
            tables[id(value)]=revalidate_artifact(value)
        return tables[id(value)]
    studies=[]
    for study in report.studies:
        current_tables={k:checked(v) for k,v in study.result.tables.items()}
        artifacts=[replace(a,data=checked(a.data)) if checked(a.data) is not a.data else a for a in study.result.artifacts]
        changed=any(current_tables[k] is not v for k,v in study.result.tables.items()) or any(a is not b for a,b in zip(artifacts,study.result.artifacts))
        result=replace(study.result,tables=current_tables,artifacts=artifacts) if changed else study.result
        studies.append(replace(study,result=result) if changed else study)
    return replace(report,studies=studies) if any(a is not b for a,b in zip(studies,report.studies)) else report


def require_verified_comparison(*frames):
    """Reject an attempted verified comparison of incomplete/assumed evidence."""
    for frame in frames:
        status, issues = validate_frame(frame)
        if status != 'verified':
            raise ValueError(f'Verified comparison requires verified temporal evidence; got {status}: {issues}.')
