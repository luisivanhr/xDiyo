"""Compare deterministic saved-fold benchmark receipts and full update states."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root/'src'), str(root/'tests/analytics')]
    from test_graph_preparation_memory import assert_exact
    receipt = dict(status='RUNNING', command=[sys.executable, *sys.argv],
                   comparison='bit-exact deterministic CUDA', declared_atol=1e-6,
                   declared_rtol=1e-5, cross_process_resume_tested=False)
    def write():
        args.output.write_text(json.dumps(receipt, indent=2))
    write()
    try:
        a, b = [json.loads(p.read_text()) for p in (args.reference, args.candidate)]
        # Also accept the optimized path with external deterministic flags as
        # a control for native scoped execution-policy overhead.
        assert a['mode'] in ('reference', 'native') and b['mode'] == 'native'
        receipt['implementations'] = dict(reference=a['mode'], candidate=b['mode'])
        for key in ('input_sha256', 'executable', 'harness_sha256', 'runtime',
                    'source_sha256', 'reference_sha256', 'issuances', 'source_edges'):
            assert_exact(a[key], b[key])
        assert a['runtime']['deterministic'] is True
        assert a['inputs_unchanged'] and b['inputs_unchanged']
        for key in ('fixtures', 'directed_edges'):
            assert a['cold_preparation'][key] == b['cold_preparation'][key]
        assert len(a['samples']) == len(b['samples'])
        for left, right in zip(a['samples'], b['samples']):
            for key in ('label', 'index', 'edges', 'width', 'issuance', 'tensor_sha256'):
                assert left[key] == right[key], key
        assert len(a['updates']) == len(b['updates'])
        for left, right in zip(a['updates'], b['updates']):
            for key in ('repeat', 'profiled', 'values', 'metrics', 'rng_unchanged'):
                assert_exact(left[key], right[key])
        # Include all parameter groups, moments and step counters, not only the
        # outputs. Each repetition starts from the saved weights. The benchmark
        # checks RNG does not advance during an update; independent processes
        # have unseeded initial RNG states, so those states are not cross-compared.
        states = [torch.load(p.with_suffix('.pt'), map_location='cpu', weights_only=True)
                  for p in (args.reference, args.candidate)]
        for key in ('model', 'gradients', 'optimizer'):
            assert_exact(states[0][key], states[1][key])
        receipt['files'] = {str(p): sha256(p.read_bytes()).hexdigest()
                            for base in (args.reference, args.candidate)
                            for p in (base, base.with_suffix('.pt'))}
        receipt['checks'] = ['identical runtime and source hashes', 'unchanged inputs',
            'sampled prepared tensor hashes', 'all repeated update losses',
            'first full update gradients', 'parameters', 'full AdamW state', 'RNG unchanged within each process']
        receipt['measurements'] = {}
        for label, field in (('training', 'training_seconds'), ('post_update_loss', 'metric_seconds'),
                             ('combined_update', 'total_seconds')):
            ref, cand = [statistics.median(r[field] for r in x['updates'] if not r['profiled'])
                         for x in (a, b)]
            receipt['measurements'][label] = dict(reference_seconds=ref, candidate_seconds=cand,
                                                   speedup=ref/cand)
        for label, values in (('cold_preparation', [x['cold_preparation']['seconds'] for x in (a,b)]),
                              ('store_setup', [x['setup_seconds'] for x in (a,b)])):
            ref, cand = values
            receipt['measurements'][label] = dict(reference_seconds=ref, candidate_seconds=cand,
                                                   speedup=ref/cand)
        receipt['memory_bytes'] = {label: dict(peak_rss=x['rss']['peak'],
            baseline_rss=x['rss']['baseline'], peak_cuda_allocated=x['peak_device_allocated'],
            peak_cuda_reserved=x['peak_device_reserved']) for label,x in (('reference',a),('candidate',b))}
        receipt['status'] = 'PASS_BIT_EXACT'
        write()
    except BaseException as exc:
        receipt.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
        write()
        raise


if __name__ == '__main__':
    main()
