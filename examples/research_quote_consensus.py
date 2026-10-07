"""Synthetic native quote-consensus example. No model fitting or real data access.

The caller supplies retained per-model quote references and aligned class PMFs.
The availability attestation is explicit and is not an observed quote timestamp.
"""
import numpy as np
import pandas as pd
from xdiyo_analytics.evaluation import (
    AllCombinations, DecisionContext, DecisionLayer, FrozenTable, QuoteAvailability, compose_bets,
)


def consensus(legs, pmfs, contract, *, audit_level='full'):
    """Two native DecisionLayer calls: fixture OR, then complete-ticket EV AND.

    pmfs[model] uses class labels 'draw' and 'non_draw', indexed by event_id.
    Model-prefixed quote/timing columns must come from each retained stream.
    This adapter only aligns data; decision rules remain native DecisionLayer.
    """
    models = tuple(pmfs)
    if len(models) != 2:
        raise ValueError('This example requires exactly two cached model streams.')
    data = contract.annotate(legs).copy()
    if data.event_id.duplicated().any():
        raise ValueError('Supply one original quote and probability record per fixture.')
    data.index = pd.Index(data.event_id, name='fixture')
    if 'decision_at' not in data:
        raise ValueError('Every fixture requires a decision timestamp before grouping.')
    decision_times = pd.to_datetime(data.decision_at, utc=True, errors='raise')
    if decision_times.isna().any():
        raise ValueError('Every fixture requires a nonmissing decision timestamp before grouping.')
    fields = contract.identities(data)
    candidates = data[list(fields)].copy()
    candidates['economic_key'] = data.event_id.map(lambda k: f'fixture:{k}')
    candidates['payoff'] = 'binary'
    valuations = {}
    for model in models:
        pmf = pmfs[model]
        if not pmf.index.is_unique or set(pmf.columns) != {'draw', 'non_draw'}:
            raise ValueError('Use unique fixture keys and explicitly labelled draw/non_draw classes.')
        aligned = pmf.reindex(data.index)[['draw', 'non_draw']]
        values = aligned.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1), 1., atol=1e-12, rtol=0):
            raise ValueError('Both complete valid cached model streams are required before selection.')
        v = candidates.copy()
        for field in fields:
            v[field] = data[f'{model}::{field}']
        for field in ('issued_at', 'trained_through', 'artifact_vintage'):
            v[field] = data[f'{model}::{field}']
        # Declared complement of the ORIGINAL non-draw output for the OR gate.
        # Do not substitute the independently stored draw column here.
        v['probability'] = 1. - aligned['non_draw']
        valuations[model] = v
        # Original draw probabilities remain unchanged for complete-ticket EV.
        data[f'{model}::probability'] = aligned['draw']
    # Validate the entire source and both model streams before groupby can drop
    # rows or any OR rejection can hide unavailable evidence.
    context = DecisionContext(decision_times.max(), FrozenTable(candidates), quote_availability=contract) if len(candidates) else None
    if context is not None:
        from xdiyo_analytics.evaluation.quote_availability import validate_model_quotes
        validate_model_quotes(data, models, contract, context.time)
    gate = DecisionLayer(models, gate='or', metric='probability', threshold=1. - .80,
                         strict=False, missing='error')
    selected = []
    audits = []
    for time, batch in candidates.groupby(decision_times, sort=True, dropna=False):
        result = gate.decide({m: v.loc[batch.index] for m, v in valuations.items()},
            DecisionContext(time, FrozenTable(batch), quote_availability=contract), audit_level=audit_level)
        selected.extend(result.selected.index)
        audits.append(result.audit)
    data['take'] = data.index.isin(selected)
    composition = AllCombinations(legs=2, stake=1., stage_column='stage', audit_level=audit_level,
        probability_mode='independent', payoff='binary', quote_availability=contract,
        probability_columns={m: f'{m}::probability' for m in models},
        ticket_gate=DecisionLayer(models, gate='and', metric='ev', threshold=0., strict=True, missing='error'))
    tickets, members, metrics = compose_bets(data.reset_index(drop=True), composition)
    audit = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame(columns=[
        'candidate_id', 'model', 'probability', 'value', 'take', 'reason', 'comparator',
        *contract.labels, 'assumed_available_at', 'quote_at', 'quote_id', 'decision_at'])
    if audit_level == 'summary':
        from xdiyo_analytics.evaluation.audit_storage import fingerprint
        tickets.attrs['audit_manifest']['original_pmf_fingerprints'] = {str(model): fingerprint(pmf) for model, pmf in pmfs.items()}
        tickets.attrs['audit_manifest']['fixture_gate_reasons'] = audit.groupby(['model', 'reason'], sort=False).size().reset_index(name='count').to_dict('records')
        tickets.attrs['audit_manifest']['fixture_gate'] = dict(models=list(models), gate='or', metric='probability',
                                                            threshold=1.-.80, strict=False, missing='error')
        audit.attrs['audit_level'] = 'summary'
        audit.attrs['omitted_details'] = ['per-ballot quote/timing copies']
    return tickets, members, metrics, audit


def synthetic_example():
    contract = QuoteAvailability('research_assumed', 'opening-round-minus-1h-v1',
        'Unverified opening-quote availability by league/season/stage/round first kickoff minus one hour.',
        'synthetic-example/protocol-v1')
    kickoff = pd.Timestamp('2025-01-02T15:00:00Z')
    decision = kickoff - pd.Timedelta(hours=1)
    data = pd.DataFrame(dict(event_id=[1, 2, 3], fold_id=0, row_position=[0, 1, 2],
        competition_id=1, season_id=1, stage='main', round=1,
        kickoff_at=[kickoff, kickoff+pd.Timedelta(hours=2), kickoff+pd.Timedelta(days=1)],
        decision_at=decision, assumed_available_at=decision, quote_at=pd.NaT,
        quote_id=['synthetic-q1', 'synthetic-q2', 'synthetic-q3'],
        quote_snapshot_hash='synthetic-snapshot', quote_crosswalk_hash='synthetic-crosswalk',
        market='1x2', selection='draw', odds=[5., 4., 6.],
        bet='draw', take=True, stake=1., settlement=['win', 'loss', 'win']))
    data = contract.annotate(data)
    # Synthetic independent retained quote references; real usage must read
    # these from each model stream, never invent them to pass validation.
    for model in ('xgb', 'lgbm'):
        for field in contract.identities(data):
            data[f'{model}::{field}'] = data[field]
        data[f'{model}::issued_at'] = decision
        data[f'{model}::trained_through'] = decision - pd.Timedelta(days=5)
        data[f'{model}::artifact_vintage'] = decision - pd.Timedelta(days=4)
    pmfs = {
        'xgb': pd.DataFrame({'draw':[.20, .30, .19], 'non_draw':[.80, .70, .81]}, index=[1, 2, 3]),
        'lgbm': pd.DataFrame({'non_draw':[.75, .72, .82], 'draw':[.25, .28, .18]}, index=[1, 2, 3]),
    }
    return data, pmfs, contract


if __name__ == '__main__':
    tickets, members, metrics, audit = consensus(*synthetic_example())
    assert len(tickets) == 1 and tickets.odds.iloc[0] == 20.
    assert tickets.quote_at.isna().all()
    print('Synthetic research simulation only; quote availability is unverified.')
    print(tickets[['ticket_id', 'odds', 'stake', 'quote_availability_mode']].to_string(index=False))
