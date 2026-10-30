"""Known contradictions survive missing intermediate temporal evidence."""
from copy import deepcopy
import json

import pandas as pd
import pytest

from xdiyo_analytics.evaluation import compose_bets
from xdiyo_analytics.evaluation.temporal_provenance import (
    EVIDENCE, STATUS, bind_forecasts, guard_candidates,
)
from test_temporal_provenance import population, policy, T, TRAIN

LATE = '2024-12-31T12:00:00+00:00'
FIELDS = ['model_available_as_of', 'features_available_as_of',
          'preprocessing_available_as_of', 'ratings_available_as_of',
          'calibration_available_as_of', 'tuning_available_as_of',
          'training_label_available_as_of']


def run(frame, entry):
    if entry == 'guard':
        return guard_candidates(frame, policy())
    return compose_bets(frame, policy())[0]


def missing(frame, representation):
    frame = frame.drop(columns=['forecast_as_of', 'prediction_cutoff'], errors='ignore')
    if representation != 'absent':
        for name in ['forecast_as_of', 'prediction_cutoff']:
            frame[name] = None if representation == 'null' else pd.NaT
    return frame


@pytest.mark.parametrize('entry', ['guard', 'compose'])
@pytest.mark.parametrize('representation', ['absent', 'null', 'nat'])
@pytest.mark.parametrize('field', FIELDS)
def test_row_dependency_after_decision_without_cutoff(entry, representation, field):
    f = missing(population()[0], representation)
    f.loc[0, 'take'] = False  # Even discarded candidates affect selection.
    f.loc[0, field] = LATE
    with pytest.raises(ValueError, match=field + '.*decision_at') as exc:
        run(f, entry)
    assert str(f.loc[0, 'event_id']) in str(exc.value)
    assert '2024-12-31T12:00:00+00:00' in str(exc.value)


def bound_without_cutoff():
    f, e = population()
    f = missing(f, 'absent')
    for item in e.values():
        item.pop('forecast_as_of')
    return bind_forecasts(f, e)


@pytest.mark.parametrize('entry', ['guard', 'compose'])
@pytest.mark.parametrize('location', ['dependency', 'nested_dependency', 'model', 'declared_cutoff',
                                     'stage', 'training_input', 'artifact', 'fit_rows', 'evidence_field'])
@pytest.mark.parametrize('kind', ['preprocessing', 'ratings', 'calibration', 'tuning'])
def test_nested_evidence_after_decision_without_cutoff(entry, location, kind):
    f = bound_without_cutoff()
    e = json.loads(f.loc[0, EVIDENCE])
    if location in {'dependency', 'nested_dependency'}:
        dep = deepcopy(e['dependencies'][0])
        dep.update(id=kind, role=kind, available_as_of=LATE)
        if location == 'dependency':
            e['dependencies'].append(dep)
        else:
            e['dependencies'][0]['dependencies'] = [dep]
    elif location == 'model':
        e['model_available_as_of'] = LATE
    elif location == 'declared_cutoff':
        e['forecast_as_of'] = LATE
    elif location == 'evidence_field':
        e[kind + '_available_as_of'] = LATE
    else:
        stage = next(s for s in e['stages'] if s['kind'] == kind)
        if stage['mode'] != 'fitted':
            stage.update(deepcopy(next(s for s in e['stages'] if s['kind'] == 'preprocessing')), kind=kind)
        stage['training_cutoff'] = None
        stage['available_as_of'] = None
        if location == 'stage':
            stage['available_as_of'] = LATE
        elif location == 'training_input':
            stage['inputs'][0]['available_as_of'] = LATE
        else:
            stage[location]['available_as_of'] = LATE
    f.loc[0, EVIDENCE] = json.dumps(e)
    with pytest.raises(ValueError, match='decision_at') as exc:
        run(f, entry)
    assert str(f.loc[0, 'event_id']) in str(exc.value)


@pytest.mark.parametrize('entry', ['guard', 'compose'])
@pytest.mark.parametrize('value', [None, TRAIN, T, '2024-12-31T09:00:00+09:00'])
def test_unknown_or_not_later_remains_unverified(entry, value):
    f = missing(population()[0], 'nat')
    for field in FIELDS:
        f[field] = value
    result = run(f, entry)
    assert result[STATUS].eq('unverified').all()


@pytest.mark.parametrize('entry', ['guard', 'compose'])
@pytest.mark.parametrize('field', FIELDS[:-1])
def test_available_intermediate_cutoff_still_enforced(entry, field):
    f = population()[0]
    f['forecast_as_of'] = TRAIN
    f[field] = T  # Permitted at placement, too late for this forecast.
    with pytest.raises(ValueError, match='forecast_as_of'):
        run(f, entry)


@pytest.mark.parametrize('entry', ['guard', 'compose'])
def test_evidence_cutoff_is_enforced_when_row_cutoff_missing(entry):
    f = bound_without_cutoff()
    e = json.loads(f.loc[0, EVIDENCE])
    e['forecast_as_of'] = TRAIN
    e['dependencies'][0]['available_as_of'] = T
    f.loc[0, EVIDENCE] = json.dumps(e)
    with pytest.raises(ValueError, match='forecast_as_of'):
        run(f, entry)


@pytest.mark.parametrize('entry', ['guard', 'compose'])
def test_prefixed_dependency_without_outer_lineage(entry):
    f = missing(population()[0], 'absent')
    f['model_a::ratings_available_as_of'] = LATE
    with pytest.raises(ValueError, match='ratings_available_as_of.*decision_at'):
        run(f, entry)


@pytest.mark.parametrize('entry', ['guard', 'compose'])
def test_nested_training_input_respects_known_training_cutoff(entry):
    f = bound_without_cutoff()
    e = json.loads(f.loc[0, EVIDENCE])
    stage = next(s for s in e['stages'] if s['kind'] == 'preprocessing')
    stage['inputs'][0]['dependencies'] = [dict(id='nested-labels', available_as_of=T)]
    f.loc[0, EVIDENCE] = json.dumps(e)
    with pytest.raises(ValueError, match='nested-labels.*training_cutoff'):
        run(f, entry)


@pytest.mark.parametrize('entry', ['guard', 'compose'])
def test_nested_model_stream_dependency_without_cutoffs(entry):
    raw, evidence = population()
    raw = missing(raw, 'absent')
    raw['probability'] = raw['p_win']
    for item in evidence.values():
        item.pop('forecast_as_of')
    inner = bind_forecasts(raw, evidence)
    outer_raw = raw.drop(columns='probability').assign(**{'model_a::probability': raw.p_win})
    for i, item in evidence.items():
        item['model_streams'] = {'model_a': inner.loc[i, EVIDENCE]}
    f = bind_forecasts(outer_raw, evidence)
    packet = json.loads(f.loc[0, EVIDENCE])
    stream = json.loads(packet['model_streams']['model_a'])
    stream['dependencies'][0]['available_as_of'] = LATE
    packet['model_streams']['model_a'] = json.dumps(stream)
    f.loc[0, EVIDENCE] = json.dumps(packet)
    with pytest.raises(ValueError, match='dependency.*decision_at'):
        run(f, entry)


def test_reload_revalidates_nested_availability(tmp_path):
    from xdiyo_analytics.evaluation.temporal_provenance import save_temporal_table, load_temporal_table
    f = bound_without_cutoff()
    path = tmp_path/'forecasts.parquet'
    save_temporal_table(f, path)
    restored = pd.read_parquet(path)
    e = json.loads(restored.loc[0, EVIDENCE])
    e['dependencies'][0]['available_as_of'] = LATE
    restored.loc[0, EVIDENCE] = json.dumps(e)
    restored.to_parquet(path)
    with pytest.raises(ValueError, match='decision_at'):
        load_temporal_table(path)


@pytest.mark.parametrize('entry', ['guard', 'compose'])
def test_nested_candidate_dependency_respects_earliest_group_decision(entry):
    raw, evidence = population()
    raw = missing(raw, 'absent')
    raw.loc[0, 'decision_at'] = LATE
    raw.loc[0, 'take'] = False
    for item in evidence.values():
        item.pop('forecast_as_of')
    evidence[0]['dependencies'][0]['available_as_of'] = '2024-12-31T06:00:00Z'
    f = bind_forecasts(raw, evidence)  # Eligible for its own later decision only.
    with pytest.raises(ValueError, match='dependency.*decision_at'):
        if entry == 'guard':
            guard_candidates(f, policy(), selection_groups=['competition_id', 'round'])
        else:
            compose_bets(f, policy())
