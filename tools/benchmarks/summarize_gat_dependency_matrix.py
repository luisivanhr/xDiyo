"""Read frozen 24-fit receipts; write a separate, fail-closed acceptance report.

No model imports, fitting, checkpoint loading or outer-test reads. Coordinator
completion means jobs finished; only this report evaluates the frozen gates.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import statistics

SOURCE = Path(__file__).resolve().parents[2]
EXPECTED = {(arm, fold, seed, mode) for arm in ('plain', 'bayes11')
            for fold in (0, 2) for seed in (17, 29, 43)
            for mode in ('native', 'dependency_packed')}
QUALITY = dict(max_pair_ce_increase=.005, max_mean_ce_increase=.001,
               max_accuracy_decrease=.005, max_brier_increase=.003,
               max_auxiliary_relative_increase=.01)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def memory(receipt):
    result = {}
    host = [receipt.get(name, {}) for name in
            ('training_host_peak', 'selected_host_peak')]
    host.append(receipt.get('initialization', {}).get('host_peak', {}))
    os_info = receipt.get('process_os_memory', {})
    for name, sample, operating_system in (('resident', 'rss', 'peak_wset'),
                                           ('private', 'private', 'peak_pagefile')):
        values = [entry.get(sample) for entry in host] + [os_info.get(operating_system)]
        values = [value for value in values if finite(value) and value > 0]
        result[name] = max(values) if values else None
    # End-to-end peaks include selected-model evaluation/export, unlike the
    # training-only snapshot. Absence cannot be silently treated as zero.
    cuda = receipt.get('end_to_end_cuda_peak', {}) or {}
    for field in ('allocated', 'reserved'):
        value = cuda.get(field)
        result['cuda_' + field] = value if finite(value) and value > 0 else None
    return result


def validate_fit(root, fit, manifest, digest, completed):
    directory = root / 'fits' / fit['id']
    errors = []
    if not directory.resolve().is_relative_to((root / 'fits').resolve()):
        return None, ['fit_path_escapes_matrix']
    path = directory / 'receipt.json'
    if not path.exists():
        return None, ['missing_receipt']
    try:
        receipt = read(path)
        if completed.get(fit['id']) != sha(path): errors.append('completed_receipt_hash_mismatch')
        if receipt.get('manifest_sha256') != digest: errors.append('manifest_identity_mismatch')
        if receipt.get('fit') != fit: errors.append('fit_recipe_mismatch')
        if receipt.get('source') != manifest.get('source'): errors.append('source_identity_mismatch')
        if receipt.get('inputs_unchanged') is not True: errors.append('inputs_not_verified')
        if receipt.get('outer_test_untouched') is not True: errors.append('outer_test_not_verified_untouched')
        if receipt.get('device') != manifest.get('device'): errors.append('device_mismatch')
        if receipt.get('executable') != manifest.get('executable'): errors.append('interpreter_mismatch')
        if receipt.get('python') != manifest.get('python'): errors.append('python_mismatch')
        file_hashes = receipt.get('file_hashes', {})
        if 'step_timings.jsonl' not in file_hashes:
            errors.append('missing_sealed_timing_log')
        for relative, expected in file_hashes.items():
            artifact = (directory / relative).resolve()
            if not artifact.is_relative_to(directory.resolve()) or not artifact.is_file() or sha(artifact) != expected:
                errors.append('fit_artifact_hash_mismatch: ' + relative)
        if not manifest.get('runtime') or receipt.get('runtime') != manifest.get('runtime'):
            errors.append('frozen_runtime_mismatch')
        scores = receipt['selected_metrics']
        required = ('outcome_ce', 'auxiliary_quantile_loss', 'combined_objective', 'accuracy', 'multiclass_brier')
        if not all(finite(scores.get(key)) for key in required):
            return None, errors + ['nonfinite_selected_metrics']
        if not (scores['outcome_ce'] >= 0 and scores['auxiliary_quantile_loss'] >= 0 and
                0 <= scores['accuracy'] <= 1 and 0 <= scores['multiclass_brier'] <= 2):
            errors.append('invalid_metric_domains')
        if scores.get('population') != 'frozen_inner_validation': errors.append('wrong_metric_population')
        if not isinstance(scores.get('rows'), int) or scores['rows'] < 1: errors.append('missing_metric_population_count')
        selected = directory / 'selected_metrics.json'
        if not selected.exists() or read(selected) != scores: errors.append('selected_metrics_file_mismatch')
        summary = receipt.get('training_summary', {})
        if summary.get('termination_reason') != 'max_steps': errors.append('incomplete_or_early_termination')
        attempts = summary.get('attempts', [])
        if len(attempts) != 1 or attempts[0].get('steps') != 100: errors.append('not_exactly_100_updates')
        if scores.get('selected_step') != summary.get('retained_step'): errors.append('retained_step_mismatch')
        timings = [json.loads(line) for line in (directory / 'step_timings.jsonl').read_text().splitlines() if line.strip()]
        if len(timings) != 100 or not all(finite(row.get('seconds')) and row['seconds'] > 0 for row in timings):
            return None, errors + ['invalid_update_validation_timings']
        if any(value is None for value in memory(receipt).values()): errors.append('missing_peak_memory_evidence')
        receipt['_median_update_validation_seconds'] = statistics.median(row['seconds'] for row in timings) if timings else None
        receipt['_receipt_sha256'] = sha(path)
        return receipt, errors
    except (KeyError, ValueError, OSError, TypeError) as error:
        return None, ['invalid_receipt: ' + str(error)]


def summarize(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = read(manifest_path)
    digest = sha(manifest_path)
    errors = []
    if (root / 'manifest.sha256').read_text().strip() != digest: errors.append('manifest_hash_mismatch')
    if manifest.get('protocol') != 'full_matrix': errors.append('not_full_matrix_protocol')
    if manifest.get('full_fit_updates') != 100: errors.append('wrong_update_count')
    if Path(manifest.get('output', '')).resolve() != root: errors.append('matrix_output_location_mismatch')
    if manifest.get('predictive_gates') != QUALITY: errors.append('predictive_gates_changed')
    if manifest.get('performance_gate', {}).get('median_warm_update_validation_speedup') != 1.5:
        errors.append('speed_gate_changed')
    expected_memory = {'peak_resident_ram', 'peak_private_commit', 'cuda_allocated', 'cuda_reserved'}
    if set(manifest.get('performance_gate', {}).get('no_increase', ())) != expected_memory:
        errors.append('memory_gates_changed')
    fits = manifest.get('fits', [])
    keys = [(fit.get('arm'), fit.get('fold'), fit.get('seed'), fit.get('mode')) for fit in fits]
    if len(fits) != 24 or len(set(keys)) != 24 or set(keys) != EXPECTED: errors.append('not_exact_24_fit_matrix')
    splits = manifest.get('splits', {})
    frozen = manifest.get('frozen_files', {})
    required_frozen = {'configurations.json'}
    if set(splits) != {'0', '2'} or not all(isinstance(splits.get(key, {}).get('path'), str) for key in ('0', '2')):
        errors.append('missing_frozen_fold_splits')
    else:
        required_frozen.update(splits[key]['path'] for key in ('0', '2'))
    if not required_frozen.issubset(frozen): errors.append('missing_frozen_config_or_split_hashes')
    for relative, expected in manifest.get('frozen_files', {}).items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected:
            errors.append('frozen_file_changed: ' + relative)
    status = read(root / 'status.json') if (root / 'status.json').exists() else {}
    if status.get('state') != 'complete': errors.append('coordinator_not_complete')
    if status.get('error'): errors.append('coordinator_reported_failure')
    completed = {row['id']: row.get('receipt_sha256') for row in status.get('completed', [])}
    if len(completed) != 24: errors.append('incomplete_completed_receipt_list')
    records = {}
    fit_errors = {}
    for fit in fits:
        receipt, problems = validate_fit(root, fit, manifest, digest, completed)
        if problems: fit_errors[fit['id']] = problems
        if receipt is not None: records[(fit['arm'], fit['fold'], fit['seed'], fit['mode'])] = receipt
    pairs = []
    for arm in ('plain', 'bayes11'):
        for fold in (0, 2):
            for seed in (17, 29, 43):
                n = records.get((arm, fold, seed, 'native'))
                c = records.get((arm, fold, seed, 'dependency_packed'))
                pair = dict(arm=arm, fold=fold, seed=seed, complete=n is not None and c is not None)
                if not pair['complete']:
                    pair.update(passed=False, failure='missing_or_invalid_pair')
                    pairs.append(pair)
                    continue
                ns, cs = n['selected_metrics'], c['selected_metrics']
                delta = {name: cs[name] - ns[name] for name in
                         ('outcome_ce', 'accuracy', 'multiclass_brier', 'auxiliary_quantile_loss', 'combined_objective')}
                auxiliary_relative = (delta['auxiliary_quantile_loss'] / ns['auxiliary_quantile_loss']
                    if ns['auxiliary_quantile_loss'] > 0 else 0. if cs['auxiliary_quantile_loss'] == 0 else None)
                nm, cm = memory(n), memory(c)
                memory_pass = all(nm[key] is not None and cm[key] is not None and cm[key] <= nm[key] for key in nm)
                nt, ct = n['_median_update_validation_seconds'], c['_median_update_validation_seconds']
                speed = nt / ct if finite(nt) and finite(ct) and ct > 0 else None
                gates = dict(ce=delta['outcome_ce'] <= .005, accuracy=delta['accuracy'] >= -.005,
                    brier=delta['multiclass_brier'] <= .003,
                    auxiliary=auxiliary_relative is not None and auxiliary_relative <= .01,
                    memory=memory_pass, speed=speed is not None and speed >= 1.5,
                    receipts=n['fit']['id'] not in fit_errors and c['fit']['id'] not in fit_errors,
                    matched_population=ns.get('rows') == cs.get('rows') and ns.get('rows', 0) > 0)
                pair.update(native_metrics=ns, candidate_metrics=cs, differences=delta,
                    auxiliary_relative_increase=auxiliary_relative, native_peak_memory=nm, candidate_peak_memory=cm,
                    memory_differences={key: None if nm[key] is None or cm[key] is None else cm[key]-nm[key] for key in nm},
                    native_median_update_validation_seconds=nt, candidate_median_update_validation_seconds=ct,
                    speedup=speed, gates=gates, passed=all(gates.values()),
                    checkpoint_comparison=dict(native_selected_step=ns['selected_step'], candidate_selected_step=cs['selected_step'],
                        selected_step_difference=cs['selected_step']-ns['selected_step'],
                        native_best_step=n['training_summary'].get('best_step'), candidate_best_step=c['training_summary'].get('best_step'),
                        native_termination=n['training_summary'].get('termination_reason'), candidate_termination=c['training_summary'].get('termination_reason'),
                        native_early_stopping=n['fit']['control'].get('early_stopping'), candidate_early_stopping=c['fit']['control'].get('early_stopping'),
                        interpretation='Selection differences are recorded separately; they do not override predictive, memory or speed gates.'),
                    native_receipt_sha256=n['_receipt_sha256'], candidate_receipt_sha256=c['_receipt_sha256'])
                pairs.append(pair)
    ce_differences = [pair['differences']['outcome_ce'] for pair in pairs if pair.get('complete') and 'differences' in pair]
    mean_ce = statistics.mean(ce_differences) if len(ce_differences) == 12 else None
    mean_gate = mean_ce is not None and mean_ce <= .001
    accepted = not errors and not fit_errors and len(pairs) == 12 and all(pair['passed'] for pair in pairs) and mean_gate
    return dict(schema=1, generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        manifest=str(manifest_path), manifest_sha256=digest, protocol=manifest.get('protocol'),
        source=manifest.get('source'), runtime=manifest.get('runtime'),
        accepted=accepted, result='accepted' if accepted else 'not_accepted',
        integrity_errors=errors, fit_errors=fit_errors, completed_fits=len(records), expected_fits=24,
        mean_ce_increase=mean_ce, mean_ce_gate=mean_gate, predictive_gates=QUALITY,
        performance_gate=manifest.get('performance_gate'), pairs=pairs,
        notes=['No fitting, checkpoint loading or outer-test evaluation performed.',
            'Accuracy uses absolute proportions; Brier is the mean sum of squared errors over three classes.',
            'Timing median uses every recorded update including first update, each including validation; no samples discarded.',
            'Memory uses maximum available host sampling/OS high-water and end-to-end CUDA allocator peaks.',
            'Every pair must pass every gate, plus the global mean CE gate; missing or failed evidence never accepts.',
            'Results apply to the frozen workload/runtime; no universal numerical or memory guarantee is asserted.'])


def markdown(report):
    lines = ['# GAT dependency matrix acceptance', '',
             '**' + report['result'].replace('_', ' ').upper() + '**', '',
             f"Completed receipts: {report['completed_fits']}/24. Mean CE increase: {report['mean_ce_increase']}.", '',
             '| Arm | Fold | Seed | Speedup | CE change | Accuracy change | Brier change | Aux relative | Memory | Result |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for pair in report['pairs']:
        d = pair.get('differences', {})
        def fmt(value): return 'missing' if value is None else f'{value:.6g}'
        lines.append('| ' + ' | '.join((pair['arm'], str(pair['fold']), str(pair['seed']),
            fmt(pair.get('speedup')), fmt(d.get('outcome_ce')), fmt(d.get('accuracy')),
            fmt(d.get('multiclass_brier')), fmt(pair.get('auxiliary_relative_increase')),
            str(pair.get('gates', {}).get('memory', False)), 'pass' if pair['passed'] else 'fail')) + ' |')
    lines += ['', 'Checkpoint selection, stopping and patience comparisons are recorded separately in the JSON report.', '',
              'Integrity errors: ' + json.dumps(report['integrity_errors']), '',
              'Fit errors: ' + json.dumps(report['fit_errors']), '', *report['notes'], '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='NEW external report directory')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(SOURCE) or output.is_relative_to(args.manifest.resolve().parent):
        parser.error('report output must be outside source and frozen matrix directories')
    report = summarize(args.manifest)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'acceptance.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (output / 'acceptance.md').write_text(markdown(report), encoding='utf-8')
    print(json.dumps(dict(accepted=report['accepted'], report=str(output / 'acceptance.json'))))
    return 0 if report['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
