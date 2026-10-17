"""Small JSON-only acceptance fixtures; no torch, models or fitting are used."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[2] / 'tools/benchmarks/summarize_gat_dependency_matrix.py'
SPEC = importlib.util.spec_from_file_location('gat_matrix_summary_tests', PATH)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def fixture(tmp_path):
    root = tmp_path / 'frozen_matrix'
    root.mkdir()
    fits = [dict(id=f'{arm}-{fold}-{seed}-{mode}', arm=arm, fold=fold, seed=seed,
                 mode=mode, config={}, control=dict(early_stopping=None))
            for arm, fold, seed, mode in sorted(summary.EXPECTED)]
    manifest = dict(protocol='full_matrix', full_fit_updates=100, output=str(root.resolve()),
        fits=fits, predictive_gates=summary.QUALITY, source={'sha256': 'frozen_source'},
        runtime={'torch': 'frozen_runtime'}, executable='frozen_python', python='version', device='cuda:0',
        performance_gate=dict(median_warm_update_validation_speedup=1.5,
            no_increase=['peak_resident_ram', 'peak_private_commit', 'cuda_allocated', 'cuda_reserved']),
        frozen_files={})
    write(root / 'configurations.json', {'fits': fits})
    manifest['splits'] = {}
    for fold in (0, 2):
        relative = f'splits/fold{fold:02}.parquet'
        # The aggregator hashes frozen split bytes without loading their tables.
        write(root / relative, {'synthetic_split': fold})
        manifest['splits'][str(fold)] = {'path': relative}
    manifest['frozen_files'] = {relative: summary.sha(root / relative) for relative in
        ('configurations.json', 'splits/fold00.parquet', 'splits/fold02.parquet')}
    write(root / 'manifest.json', manifest)
    digest = summary.sha(root / 'manifest.json')
    (root / 'manifest.sha256').write_text(digest)
    completed = []
    for fit in fits:
        candidate = fit['mode'] == 'dependency_packed'
        scores = dict(outcome_ce=1.0005 if candidate else 1., auxiliary_quantile_loss=.1,
            combined_objective=1.1005 if candidate else 1.1, accuracy=.5,
            multiclass_brier=.6, selected_step=65 if candidate else 50,
            population='frozen_inner_validation', rows=100)
        receipt = dict(fit=fit, manifest_sha256=digest, source=manifest['source'], runtime=manifest['runtime'],
            inputs_unchanged=True, outer_test_untouched=True, executable=manifest['executable'],
            python=manifest['python'], device=manifest['device'], selected_metrics=scores,
            training_summary=dict(retained_step=scores['selected_step'], best_step=scores['selected_step'],
                termination_reason='max_steps', attempts=[dict(steps=100)]),
            training_host_peak=dict(rss=900 if candidate else 1000, private=1800 if candidate else 2000),
            selected_host_peak={}, initialization={'host_peak': {}}, process_os_memory={},
            end_to_end_cuda_peak=dict(allocated=450 if candidate else 500, reserved=700 if candidate else 800))
        directory = root / 'fits' / fit['id']
        write(directory / 'receipt.json', receipt)
        write(directory / 'selected_metrics.json', scores)
        (directory / 'step_timings.jsonl').write_text('\n'.join(
            json.dumps(dict(step=i+1, seconds=1. if candidate else 2.)) for i in range(100)))
        receipt['file_hashes'] = {name: summary.sha(directory / name) for name in
                                 ('step_timings.jsonl', 'selected_metrics.json')}
        write(directory / 'receipt.json', receipt)
        completed.append(dict(id=fit['id'], receipt_sha256=summary.sha(directory / 'receipt.json')))
    write(root / 'status.json', dict(state='complete', completed=completed))
    return root


def mutate_receipt(root, mutate):
    manifest = summary.read(root / 'manifest.json')
    fit = next(fit for fit in manifest['fits'] if fit['mode'] == 'dependency_packed')
    path = root / 'fits' / fit['id'] / 'receipt.json'
    receipt = summary.read(path)
    mutate(receipt)
    write(path, receipt)
    write(path.parent / 'selected_metrics.json', receipt['selected_metrics'])
    receipt['file_hashes']['selected_metrics.json'] = summary.sha(path.parent / 'selected_metrics.json')
    write(path, receipt)
    status = summary.read(root / 'status.json')
    next(row for row in status['completed'] if row['id'] == fit['id'])['receipt_sha256'] = summary.sha(path)
    write(root / 'status.json', status)


def test_exact_complete_matrix_passes_and_checkpoint_selection_difference_is_informational(tmp_path):
    root = fixture(tmp_path)
    report = summary.summarize(root / 'manifest.json')
    assert report['accepted'] is True
    assert len(report['pairs']) == 12
    assert all(pair['checkpoint_comparison']['selected_step_difference'] == 15 for pair in report['pairs'])


def test_missing_receipt_cannot_accept(tmp_path):
    root = fixture(tmp_path)
    next((root / 'fits').glob('*/receipt.json')).unlink()
    report = summary.summarize(root / 'manifest.json')
    assert not report['accepted']
    assert report['fit_errors']


def test_failed_coordinator_cannot_accept_even_with_24_valid_receipts(tmp_path):
    root = fixture(tmp_path)
    status = summary.read(root / 'status.json')
    status.update(state='failed', error='worker failed')
    write(root / 'status.json', status)
    report = summary.summarize(root / 'manifest.json')
    assert not report['accepted']
    assert 'coordinator_reported_failure' in report['integrity_errors']


@pytest.mark.parametrize('metric,value,gate', [
    ('outcome_ce', 1.006, 'ce'), ('accuracy', .494, 'accuracy'),
    ('multiclass_brier', .604, 'brier'), ('auxiliary_quantile_loss', .1011, 'auxiliary')])
def test_each_predictive_pair_gate_fails_closed(tmp_path, metric, value, gate):
    root = fixture(tmp_path)
    mutate_receipt(root, lambda receipt: receipt['selected_metrics'].update({metric: value}))
    report = summary.summarize(root / 'manifest.json')
    assert not report['accepted']
    assert any(not pair['gates'][gate] for pair in report['pairs'])


def test_global_mean_ce_gate_applies_when_every_pair_ce_gate_passes(tmp_path):
    root = fixture(tmp_path)
    manifest = summary.read(root / 'manifest.json')
    status = summary.read(root / 'status.json')
    for fit in manifest['fits']:
        if fit['mode'] != 'dependency_packed':
            continue
        path = root / 'fits' / fit['id'] / 'receipt.json'
        receipt = summary.read(path)
        receipt['selected_metrics']['outcome_ce'] = 1.002
        write(path, receipt)
        write(path.parent / 'selected_metrics.json', receipt['selected_metrics'])
        receipt['file_hashes']['selected_metrics.json'] = summary.sha(path.parent / 'selected_metrics.json')
        write(path, receipt)
        next(row for row in status['completed'] if row['id'] == fit['id'])['receipt_sha256'] = summary.sha(path)
    write(root / 'status.json', status)
    report = summary.summarize(root / 'manifest.json')
    assert all(pair['gates']['ce'] for pair in report['pairs'])
    assert not report['mean_ce_gate'] and not report['accepted']


@pytest.mark.parametrize('location,field,value', [
    ('training_host_peak', 'rss', 1001), ('training_host_peak', 'private', 2001),
    ('end_to_end_cuda_peak', 'allocated', 501), ('end_to_end_cuda_peak', 'reserved', 801)])
def test_any_peak_memory_increase_prevents_acceptance(tmp_path, location, field, value):
    root = fixture(tmp_path)
    mutate_receipt(root, lambda receipt: receipt[location].update({field: value}))
    report = summary.summarize(root / 'manifest.json')
    assert not report['accepted']
    assert any(not pair['gates']['memory'] for pair in report['pairs'])


def test_missing_cuda_evidence_is_not_zero_memory(tmp_path):
    root = fixture(tmp_path)
    mutate_receipt(root, lambda receipt: receipt.pop('end_to_end_cuda_peak'))
    report = summary.summarize(root / 'manifest.json')
    assert not report['accepted']
    assert any('missing_peak_memory_evidence' in issues for issues in report['fit_errors'].values())


def test_worker_smoke_protocol_is_never_full_acceptance(tmp_path):
    root = fixture(tmp_path)
    manifest = summary.read(root / 'manifest.json')
    manifest.update(protocol='worker_smoke', full_fit_updates=1, fits=manifest['fits'][:4])
    write(root / 'manifest.json', manifest)
    (root / 'manifest.sha256').write_text(summary.sha(root / 'manifest.json'))
    report = summary.summarize(root / 'manifest.json')
    assert not report['accepted']
    assert 'not_full_matrix_protocol' in report['integrity_errors']
    assert 'not_exact_24_fit_matrix' in report['integrity_errors']


def test_changed_receipt_hash_prevents_acceptance(tmp_path):
    root = fixture(tmp_path)
    path = next((root / 'fits').glob('*/receipt.json'))
    receipt = summary.read(path)
    receipt['finished_utc'] = 'modified'
    write(path, receipt)
    report = summary.summarize(root / 'manifest.json')
    assert not report['accepted']
    assert any('completed_receipt_hash_mismatch' in issues for issues in report['fit_errors'].values())


def test_insufficient_speedup_prevents_acceptance(tmp_path):
    root = fixture(tmp_path)
    manifest = summary.read(root / 'manifest.json')
    fit = next(fit for fit in manifest['fits'] if fit['mode'] == 'dependency_packed')
    path = root / 'fits' / fit['id'] / 'step_timings.jsonl'
    path.write_text('\n'.join(json.dumps({'step': i+1, 'seconds': 1.5}) for i in range(100)))
    report = summary.summarize(root / 'manifest.json')
    assert not report['accepted']
    assert any(not pair['gates']['speed'] for pair in report['pairs'])


def test_missing_frozen_split_hash_never_accepts(tmp_path):
    root = fixture(tmp_path)
    (root / 'splits/fold00.parquet').unlink()
    report = summary.summarize(root / 'manifest.json')
    assert not report['accepted']
    assert any(error.startswith('frozen_file_changed:') for error in report['integrity_errors'])
