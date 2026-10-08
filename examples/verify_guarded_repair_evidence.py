"""Bounded replay of the reviewed 39aba81 audit archive, never a model refit.

Run with the repaired checkout on PYTHONPATH. --evidence is an extracted copy
of GUARDED_BULK_39aba81_AUDIT_EVIDENCE.zip. Original archive members are read
only; generated evidence goes to --output. The simulation clock is hypothetical.
This correctness diagnostic is not a fresh-process performance benchmark.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import runpy
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
from xdiyo_analytics.evaluation import AllCombinations, DecisionLayer, DecisionContext, FrozenTable, compose_bets, bulk_tickets
from xdiyo_analytics.evaluation.decision_layer import _outcome_column
from xdiyo_analytics.reporting.tickets import ticket_html
from xdiyo_analytics.reporting.contracts import Artifact, StudyResult, StudyRun, AnalysisReport


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artifacts(outputs, size):
    t, m, metrics, fixture_audit = outputs
    attrs = dict(t.attrs)
    display = t.copy(deep=False)
    display.attrs = {k: attrs[k] for k in ('quote_assumptions', 'audit_manifest', 'ticket_ev_enabled') if k in attrs}
    tables = dict(zip(('tickets', 'ticket_legs', 'metrics', 'fixture_audit'), outputs))
    for key in ('ticket_candidates', 'decision_policy_audit', 'ticket_selection_summary', 'quote_assumptions'):
        if key in attrs: tables[key] = pd.DataFrame(attrs[key])
    tables['complete_output_attrs'] = pd.DataFrame([
        dict(output=name, attrs_json=json.dumps(frame.attrs, sort_keys=True, default=str))
        for name, frame in zip(('tickets', 'ticket_legs', 'metrics', 'fixture_audit'), outputs)])
    study = StudyResult('Real cached consensus replay', artifacts=[Artifact('html', ticket_html(display, m, {}), 'Tickets')], tables=tables)
    html = AnalysisReport([StudyRun('Consensus', 'aggregate', 'all', None, 'match', np.arange(size), size, study)],
                          title='Guarded bulk audit: real retained streams').to_html()
    csvs = {name: f.to_csv(index=False) for name, f in tables.items()}
    return dict(html_sha256=hashlib.sha256(html.encode()).hexdigest(), html_bytes=len(html.encode()),
                csv_sha256=hashlib.sha256(json.dumps(csvs, sort_keys=True).encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence.resolve()
    manifest = json.loads((root / 'MANIFEST_SHA256.json').read_text())
    for relative, digest in manifest.items():
        path = (root / relative).resolve()
        assert path.is_relative_to(root) and hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative
    folder = root / 'guarded_bulk_audit/real_input'
    inputs = load(folder / 'prepare_real_inputs.py', 'repair_inputs')
    benchmark = runpy.run_path(str(Path(__file__).with_name('benchmark_quote_consensus.py')))
    example = runpy.run_path(str(Path(__file__).with_name('research_quote_consensus.py')))
    # Execute only these reviewed, hash-verified pure adapter functions. The
    # original script's environment changes, main(), and study runner never run.
    source = folder / 'recovered_5785800/scripts/run_native_consensus.py'
    names = {'keyframe', 'leg_gate', 'specification', 'native_select'}
    tree = ast.parse(source.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    rows = []
    for size in (30, 100):
        bundle = inputs.load_study_case(size)  # independently checks original input SHA256 pins
        frames = [bundle[k] for k in ('base', 'canonical', 'ledger')] + list(bundle['views'].values())
        before = [f.copy(deep=True) for f in frames]
        captured = []
        def capture(*a, **kw):
            assert kw['match_columns'] == tuple(inputs.IDENTITY)
            output = compose_bets(*a, **kw)
            captured.append(tuple(f.copy(deep=True) for f in output))
            return output
        namespace = dict(np=np, pd=pd, json=json, ID=inputs.IDENTITY,
                         GROUP=['competition_id', 'season_id', 'tournament_id', 'round'],
                         MODELS=('xgb', 'lgbm'), MAPPING={'xgb': 'xgb::probability', 'lgbm': 'lgbm::probability'},
                         CONTRACT=bundle['contract'], _outcome_column=_outcome_column,
                         AllCombinations=AllCombinations, DecisionLayer=DecisionLayer, DecisionContext=DecisionContext,
                         FrozenTable=FrozenTable, compose_bets=capture)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
        results, timings, dispatch = {}, {}, []
        original_support = bulk_tickets.supports_bulk
        for engine in ('reference', 'auto'):
            def support(*a, **kw):
                capable = original_support(*a, **kw)
                dispatch.append(dict(engine=engine, capable=capable, bulk=capable and engine == 'auto'))
                return capable and engine == 'auto'
            with patch.object(bulk_tickets, 'supports_bulk', support):
                start = time.perf_counter()
                universe, fixture_audit = namespace['leg_gate'](bundle['canonical'], bundle['eligible'], bundle['views'])
                namespace['native_select'](bundle['ledger'], bundle['base'], universe)
                timings[engine] = time.perf_counter() - start
                results[engine] = (*captured[-1], fixture_audit)
        assert dispatch == [dict(engine='reference', capable=True, bulk=False), dict(engine='auto', capable=True, bulk=True)]
        benchmark['assert_outputs_equal'](results['reference'], results['auto'])
        benchmark['assert_outputs_equal'](results['auto'], inputs.invoke(bundle, identity='full'))
        reference_artifacts = artifacts(results['reference'], size)
        assert artifacts(results['auto'], size) == reference_artifacts
        benchmark['assert_outputs_equal'](results['auto'], pickle.loads(pickle.dumps(results['auto'], protocol=5)))
        mutated = inputs.invoke(bundle, identity='full', mutate_outcomes=True)
        original = results['auto']
        pd.testing.assert_frame_equal(original[0][['ticket_id', 'stake', 'odds', 'probability']],
                                      mutated[0][['ticket_id', 'stake', 'odds', 'probability']], check_exact=True)
        assert original[0].attrs == mutated[0].attrs
        pd.testing.assert_frame_equal(original[3], mutated[3], check_exact=True)
        for a, b in zip(before, frames): pd.testing.assert_frame_equal(a, b, check_exact=True)
        oracle = json.loads((folder / f'oracle_full_identity_{size}.json').read_text())
        assert len(original[0]) == oracle['selected_count']
        assert len(original[0].attrs['ticket_candidates']) == oracle['candidate_count']
        expected = pd.DataFrame(oracle['candidates']).set_index('ticket_id')
        ballots = pd.DataFrame(original[0].attrs['decision_policy_audit'])
        for model in ('xgb', 'lgbm'):
            votes = ballots.loc[ballots.model.eq(model)].set_index('candidate_id')
            np.testing.assert_array_equal(votes.probability, expected.reindex(votes.index)[f'{model}_probability'])
        legs, pmfs, contract = inputs.load_inputs(size, allow_research_clock=True)
        pmf_before = {k: f.copy(deep=True) for k, f in pmfs.items()}
        example['consensus'](legs, pmfs, contract)
        for k, f in pmfs.items(): pd.testing.assert_frame_equal(f, pmf_before[k], check_exact=True)
        row = dict(fixtures=size, candidates=oracle['candidate_count'], selected=len(original[0]),
                   dispatch=dispatch, diagnostic_seconds=timings, **reference_artifacts,
                   exact_frames_attrs_pickle_html_csv=True, exact_original_adapter=True,
                   scalar_probability_oracle=True, original_pmfs_unchanged=True, inputs_unchanged=True,
                   outcome_mutation_invariant=True)
        rows.append(row)
        print(json.dumps(row), flush=True)
    # Validate all retained PMFs, including rows outside bounded selection replay.
    pmf_rows = {}
    for model in ('xgb', 'lgbm'):
        frame = pd.read_parquet(inputs.PATHS[model])[['p_draw', 'p_non_draw']]
        frame.columns = ['draw', 'non_draw']
        old = frame.to_numpy().tobytes()
        example['_validate_pmf'](frame)
        assert frame.to_numpy().tobytes() == old
        pmf_rows[model] = len(frame)
    report = dict(status='PASS', archive_members_verified=len(manifest), checks=rows, all_retained_pmf_rows=pmf_rows,
                  pandas=pd.__version__, numpy=np.__version__, historical_availability_verified=False,
                  new_model_fits=0, research_clock='Recovered original hypothetical clock; observed quote_at remains NaT')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
