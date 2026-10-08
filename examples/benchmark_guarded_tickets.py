"""Fresh-process synthetic consensus worker for pinned-source/mode comparisons.

Use one process at a time with fixed thread limits; never load real data/models.
--backend reference disables only the private dispatcher for matched timings.
Full native HTML, evidence CSVs and local pickle serialization are measured separately.
--trace is a separate allocation diagnostic, not an unprofiled timing sample.
Output pickles are local evidence: never load untrusted comparison pickles.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import pickle
import runpy
import sys
import time
import tracemalloc


def peak_rss():
    if os.name != 'nt':
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if sys.platform == 'darwin' else value * 1024
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
            'peak', 'current', 'qpp', 'qp', 'qpnp', 'qnp', 'page', 'peakpage')]
    record = Counters()
    record.cb = ctypes.sizeof(record)
    process = ctypes.windll.kernel32.GetCurrentProcess
    process.restype = wintypes.HANDLE
    get = ctypes.windll.psapi.GetProcessMemoryInfo
    get.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not get(process(), ctypes.byref(record), record.cb):
        raise ctypes.WinError()
    return record.peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups', type=int, default=3)
    parser.add_argument('--fixtures', type=int, default=8)
    parser.add_argument('--audit-level', choices=['full', 'summary'], default='full')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--backend', choices=['auto','reference'], default='auto')
    args = parser.parse_args()
    sys.path[:0] = [str(args.source/'src'), str(args.source)]
    started = time.perf_counter()
    import numpy as np
    import pandas as pd
    api = runpy.run_path(str(args.source/'examples/research_quote_consensus.py'))
    imports = time.perf_counter() - started
    if args.backend == 'reference':
        from xdiyo_analytics.evaluation import bulk_tickets
        bulk_tickets.supports_bulk = lambda *a: False  # benchmark-only dispatch override
    started = time.perf_counter()
    _, _, contract = api['synthetic_example']()
    records = []
    for group in range(args.groups):
        kickoff = pd.Timestamp('2025-01-02T15:00:00Z') + pd.Timedelta(days=7*group)
        decision = kickoff - pd.Timedelta(hours=1)
        for leg in range(args.fixtures):
            event = group*args.fixtures + leg + 1
            records.append(dict(event_id=event, fold_id=0, row_position=event-1,
                competition_id=1, season_id=1, stage='main', round=group+1,
                kickoff_at=kickoff+pd.Timedelta(hours=2*leg), decision_at=decision,
                assumed_available_at=decision, quote_at=pd.NaT, quote_id=f'synthetic-q{event}',
                quote_snapshot_hash='synthetic-snapshot', quote_crosswalk_hash='synthetic-crosswalk',
                market='1x2', selection='draw', odds=4., bet='draw', take=True, stake=1., settlement='missing'))
    data = contract.annotate(pd.DataFrame(records))
    for model in ('xgb', 'lgbm'):
        for field in contract.identities(data):
            data[f'{model}::{field}'] = data[field]
        data[f'{model}::issued_at'] = data.decision_at
        data[f'{model}::trained_through'] = data.decision_at - pd.Timedelta(days=5)
        data[f'{model}::artifact_vintage'] = data.decision_at - pd.Timedelta(days=4)
    pmfs = {m: pd.DataFrame({'draw': p, 'non_draw': q}, index=data.event_id)
            for m, p, q in [('xgb', .30, .70), ('lgbm', .32, .68)]}
    original = data.copy(deep=True)
    original_pmfs = {name: pmf.copy(deep=True) for name, pmf in pmfs.items()}
    generation = time.perf_counter() - started
    rss_before = peak_rss()
    if args.trace:
        tracemalloc.start()
    wall, cpu = time.perf_counter(), time.process_time()
    result = api['consensus'](data, pmfs, contract, **({'audit_level': 'summary'} if args.audit_level == 'summary' else {}))
    wall, cpu = time.perf_counter()-wall, time.process_time()-cpu
    rss_call = peak_rss()
    allocation_peak = tracemalloc.get_traced_memory()[1] if args.trace else None
    if args.trace:
        tracemalloc.stop()
    t, m, _, fixture_audit = result
    started = time.perf_counter()
    pd.testing.assert_frame_equal(data, original, check_exact=True)
    for name, pmf in pmfs.items():
        pd.testing.assert_frame_equal(pmf, original_pmfs[name], check_exact=True)
    expected = args.groups * args.fixtures * (args.fixtures-1)//2
    assert len(t) == expected and len(m) == expected*2
    assert t.stake.eq(1.).all() and t.odds.eq(16.).all() and t.quote_at.isna().all()
    assert t.payout.isna().all() and t.profit.isna().all()
    if args.audit_level == 'full':
        ballots = pd.DataFrame(t.attrs['decision_policy_audit'])
    else:
        values = t.attrs['selected_model_values']
        ballots = pd.DataFrame(values['data'], columns=values['columns'])
        assert t.attrs['audit_manifest']['candidate_count'] == expected
    assert len(ballots) == expected*2 and len(fixture_audit) == len(data)*2
    for model, probability in [('xgb', .30), ('lgbm', .32)]:
        subset = ballots.loc[ballots.model.eq(model)]
        np.testing.assert_array_equal(subset.probability, np.full(len(subset), probability*probability))
        np.testing.assert_array_equal(subset.value, np.full(len(subset), probability*probability*16.-1.))
    from itertools import combinations
    # The independent oracle needs row values, not repeated copies of the
    # retained manifest. Preserve the result and strip only a working copy.
    oracle_members = m.copy(deep=False)
    oracle_members.attrs = {}
    actual = {tuple(sorted(group)) for group in oracle_members.groupby('ticket_id', sort=False).event_id.agg(list)}
    oracle = {pair for g in range(args.groups) for pair in combinations(range(g*args.fixtures+1, (g+1)*args.fixtures+1), 2)}
    assert actual == oracle and len(actual) == len(t)
    verification = time.perf_counter() - started
    started = time.perf_counter()
    from xdiyo_analytics.reporting.tickets import ticket_html
    # Explicit consumer pattern: preserve all attrs, then render from an
    # independent frame carrying only the disclosures the renderer consumes.
    attrs = dict(t.attrs)
    display = t.copy(deep=False)
    display.attrs = {k: attrs[k] for k in ('quote_assumptions', 'audit_manifest', 'ticket_ev_enabled') if k in attrs}
    html = ticket_html(display, m, {})
    from xdiyo_analytics.reporting.contracts import Artifact, StudyResult, StudyRun, AnalysisReport
    tables = dict(zip(('tickets','ticket_legs','metrics','fixture_audit'), result))
    for key in ('ticket_candidates','decision_policy_audit','ticket_selection_summary','quote_assumptions'):
        if key in attrs: tables[key] = pd.DataFrame(attrs[key])
    tables['complete_output_attrs'] = pd.DataFrame([dict(output=name, attrs_json=json.dumps(frame.attrs,sort_keys=True,default=str)) for name,frame in zip(('tickets','ticket_legs','metrics','fixture_audit'),result)])
    study = StudyResult('Synthetic consensus', artifacts=[Artifact('html',html,'Tickets')], tables=tables)
    html = AnalysisReport([StudyRun('Consensus','aggregate','all',None,'match',np.arange(len(data)),len(data),study)], title='Synthetic consensus').to_html()
    report = time.perf_counter()-started
    csv_start = time.perf_counter()
    csv_payload = {name: frame.to_csv(index=False) for name,frame in tables.items()}
    csv_seconds = time.perf_counter()-csv_start
    rss_report = peak_rss()
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(result, protocol=5)
    args.output.with_suffix('.pickle').write_bytes(payload)
    serialization = time.perf_counter()-started
    input_hash = hashlib.sha256(pd.util.hash_pandas_object(data, index=True).values.tobytes())
    for name, pmf in sorted(pmfs.items()):
        input_hash.update(name.encode())
        input_hash.update(pd.util.hash_pandas_object(pmf, index=True).values.tobytes())
    evidence = dict(backend=args.backend, csv_seconds=csv_seconds, csv_bytes=sum(len(v.encode()) for v in csv_payload.values()), csv_sha256=hashlib.sha256(json.dumps(csv_payload,sort_keys=True).encode()).hexdigest(), mode=args.audit_level, candidates=expected, wall=wall, cpu=cpu, imports=imports,
        generation=generation, verification=verification, report_html=report, serialization=serialization,
        peak_rss_call=rss_call, peak_rss_pre_call=rss_before, peak_rss_report=rss_report,
        traced_allocation_peak=allocation_peak, trace=args.trace, serialized_bytes=len(payload),
        report_html_bytes=len(html.encode()), audit_rows=len(t.attrs.get('decision_policy_audit', [])),
        report_html_sha256=hashlib.sha256(html.encode()).hexdigest(),
        candidate_rows=len(t.attrs.get('ticket_candidates', [])),
        input_sha256=input_hash.hexdigest(),
        python=sys.version, pandas=pd.__version__, numpy=np.__version__)
    args.output.with_suffix('.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    main()
