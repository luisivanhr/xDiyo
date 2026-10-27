"""Audit frozen inputs, stopping, scores, continuation fidelity and saved models."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import compare_gat_temporal as experiment


def verify(run):
    protocol,store,frames = experiment.load(run)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if protocol.get('cuda_allocator_config'):
        os.environ['PYTORCH_ALLOC_CONF'] = protocol['cuda_allocator_config']
    import torch
    from xdiyo_analytics.graphs import TemporalGraphSerializer
    torch.set_num_threads(protocol['inner_threads'])
    torch.set_num_interop_threads(1)
    if protocol['device'].startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(protocol['cuda_memory_fraction'],protocol['device'])
    parent = protocol.get('parent_run')
    if parent:
        parent_path = Path(parent['path'])
        assert experiment.digest(parent_path/'protocol.json') == parent['protocol_sha256']
        original = json.loads((parent_path/'protocol.json').read_text())
        for field in ('data_hashes','populations','store_version','config','seeds','modes','device',
                      'inner_threads','input_cache_bytes','input_cache_headroom_bytes',
                      'cuda_allocator_config','cuda_memory_fraction','worker_rss_limit_gib'):
            assert protocol[field] == original[field],field
        for name,files in parent['fits'].items():
            for relative,expected in files.items():
                assert experiment.digest(parent_path/'fit'/name/relative) == expected
    checks = []
    for mode in protocol['modes']:
        for seed in protocol['seeds']:
            name = f'{mode}-{seed}'
            folder = run/'fit'/name
            record = json.loads((folder/'result.json').read_text())
            progress = json.loads((folder/'progress.json').read_text())
            metadata = json.loads((folder/'model/graph.json').read_text())
            assert record['protocol_sha256'] == experiment.digest(run/'protocol.json')
            assert record['config'] == dict(protocol['config'],temporal_mode=mode)
            assert metadata['control']['seed'] == seed
            assert metadata['control']['max_steps'] == protocol['max_steps']
            assert metadata['control']['early_stopping'] == protocol.get('early_stopping')
            experiment.validate_training_progress(progress,record['training_summary'],protocol)
            predictions = pd.read_parquet(folder/'predictions.parquet')
            assert experiment.frame_hash(predictions) == protocol['populations']['test']['identity_sha256']
            np.testing.assert_array_equal(predictions.truth,experiment.make_context(frames['test']).y.result)
            p = predictions[['p_home','p_draw','p_away']].to_numpy()
            y = np.array([experiment.CLASS_ORDER.index(value) for value in predictions.truth])
            log_loss = -np.log(p[np.arange(len(p)),y]).mean()
            brier = (np.square(p).sum(1)-2*p[np.arange(len(p)),y]+1).mean()
            assert abs(log_loss-record['test']['log_loss']) < 1e-12
            assert abs(brier-record['test']['brier']) < 1e-12
            assert record['checkpoint_sha256'] == experiment.digest(folder/'model/weights.pt')
            assert record['environment']['device'] == protocol['device']
            assert record['input_cache']['transient_bytes'] == 0
            prefix = None
            if parent:
                old = json.loads((parent_path/'fit'/name/'progress.json').read_text())
                assert len(progress) >= len(old)
                metrics = set(old[0])-{'step','seconds_since_previous'}
                for previous,current in zip(old,progress):
                    assert previous['step'] == current['step']
                    for metric in metrics:
                        assert previous[metric] == current[metric],(name,previous['step'],metric)
                prefix = dict(updates=len(old),metrics=sorted(metrics),bit_exact=True)
            checks.append(dict(mode=mode,seed=seed,steps=len(progress),
                retained_step=record['training_summary']['retained_step'],
                termination_reason=record['training_summary']['termination_reason'],
                original_prefix=prefix,identities_and_metrics='passed'))
    # Complete first/last issue groups preserve original fixture-head batch shapes.
    cfg = experiment.TemporalGraphConfig(**protocol['config'])
    issue = store.issue_times(frames['test'],cfg)
    keep = issue.isin([issue.min(),issue.max()])
    selected = frames['test'].loc[keep]
    reloaded = []
    seed = protocol['seeds'][0]
    for mode in protocol['modes']:
        folder = run/'fit'/f'{mode}-{seed}'
        model = TemporalGraphSerializer(event_store=store,device=protocol['device']).load(folder/'model')
        actual = model.predict(experiment.make_context(selected,fitting=False))['predict_proba'].to_numpy()
        saved = pd.read_parquet(folder/'predictions.parquet').loc[keep.to_numpy(),['p_home','p_draw','p_away']].to_numpy()
        np.testing.assert_array_equal(actual,saved)
        reloaded.append(dict(mode=mode,seed=seed,fixtures=len(selected),exact_probabilities=True))
        del model
    for source in protocol['provenance']['sources']:
        assert experiment.digest(source['manifest_path']) == source['manifest_sha256']
    result = dict(run_checks=checks,checkpoint_checks=reloaded,source_manifests_unchanged=True,
                  prepared_inputs_unchanged=True,source_code_unchanged=True,parent_artifacts_unchanged=bool(parent))
    experiment.write_json(run/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    verify(parser.parse_args().run.resolve())
