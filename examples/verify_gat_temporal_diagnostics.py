"""Read-only verification of the completed diagnostic evidence and exports."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import gat_temporal_diagnostics as diagnostic
from gat_temporal_mechanisms import predict_groups


def verify(run):
    protocol,parent,store,frames=diagnostic.load(run)
    base=Path(protocol['base'])
    evidence=json.loads((run/'scores.json').read_text())
    max_error=0.
    recomputed=0
    for mode in parent['modes']:
        for seed,saved in zip(parent['seeds'],evidence['per_seed'][mode]):
            f=pd.read_parquet(base/'fit'/f'{mode}-{seed}'/'predictions.parquet')
            a=f[['p_home','p_draw','p_away']].to_numpy(dtype=float)
            a/=a.sum(1,keepdims=True)
            draw=f.truth.to_numpy()==0
            idx=np.array([diagnostic.experiment.CLASS_ORDER.index(x) for x in f.truth])
            three=-np.log(a[np.arange(len(a)),idx])
            binary=-np.where(draw,np.log(a[:,1]),np.log1p(-a[:,1]))
            ratio=a[:,0]/(a[:,0]+a[:,2])
            winner=-np.where(f.truth.to_numpy()[~draw]==1,np.log(ratio[~draw]),np.log1p(-ratio[~draw]))
            error=abs(three.mean()-binary.mean()-(~draw).mean()*winner.mean())
            max_error=max(max_error,float(error))
            assert error<1e-12
            assert abs(saved['draw_log_loss']-binary.mean())<1e-12
            assert abs(saved['draw_brier']-np.square(a[:,1]-draw).mean())<1e-12
            assert abs(saved['conditional_winner_log_loss']-winner.mean())<1e-12
            recomputed+=1
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['PYTORCH_ALLOC_CONF']=protocol['cuda_allocator_config']
    import torch
    from xdiyo_analytics.graphs import TemporalGraphSerializer
    from xdiyo_analytics.graphs.execution import deterministic_execution
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(protocol['cuda_memory_fraction'],protocol['device'])
    cfg=diagnostic.experiment.TemporalGraphConfig(**parent['config'])
    checks=[]
    with deterministic_execution(cfg,protocol['device']):
        for arm in protocol['matched_arms']:
            folder=run/'matched'/arm
            result=json.loads((folder/'result.json').read_text())
            progress=json.loads((folder/'progress.json').read_text())
            diagnostic.experiment.validate_training_progress(progress,result['training_summary'],protocol)
            model=TemporalGraphSerializer(event_store=store,device=protocol['device']).load(folder/'model')
            backend=model.backend_
            base_dim=backend.preprocessor.edge_dim-len(backend.preprocessor.temporal_columns)
            additions=[projection.weight[:,base_dim:] for layer in backend.model.layers for projection in (layer.e,layer.u)]
            if arm=='zero_time': assert all(torch.count_nonzero(value)==0 for value in additions)
            else: assert all(torch.count_nonzero(value)>0 for value in additions)
            maximum_initial=0.
            for split in ('train','validation'):
                current=pd.read_parquet(folder/f'initial_{split}_predictions.parquet')
                other=pd.read_parquet(run/'matched'/'zero_time'/f'initial_{split}_predictions.parquet')
                np.testing.assert_array_equal(current,other)
                reference=pd.read_parquet(run/'probe/snapshot'/f'initial_{split}_predictions.parquet')
                p=current[['p_home','p_draw','p_away']].to_numpy()
                q=reference[['p_home','p_draw','p_away']].to_numpy()
                np.testing.assert_allclose(p,q,atol=1e-6,rtol=1e-5)
                maximum_initial=max(maximum_initial,float(np.abs(p-q).max()))
            # Explicit diagnostic re-evaluation, retaining the public model's
            # fitted/validation-selection boundary rather than changing it.
            context=diagnostic.experiment.make_context(frames['validation'],fitting=False)
            try:
                model.predict(context)
            except ValueError as error:
                assert 'selection boundary' in str(error)
            else:
                raise AssertionError('Public historical prediction boundary was weakened.')
            actual,_=predict_groups(backend,backend._stream(context))
            saved=pd.read_parquet(folder/'selected_validation_predictions.parquet')[['p_home','p_draw','p_away']].to_numpy()
            np.testing.assert_array_equal(actual,saved)
            selected_control_difference=None
            if arm=='zero_time':
                reference=pd.read_parquet(run/'probe/snapshot/selected_validation_predictions.parquet')[['p_home','p_draw','p_away']].to_numpy()
                np.testing.assert_allclose(actual,reference,atol=1e-6,rtol=1e-5)
                selected_control_difference=float(np.abs(actual-reference).max())
            assert not list(folder.glob('*test*'))
            checks.append(dict(arm=arm,epochs=len(progress),retained=result['training_summary']['retained_step'],
                identical_initial_arm_probabilities=True,max_initial_probability_difference_from_snapshot=maximum_initial,
                selected_control_max_probability_difference_from_snapshot=selected_control_difference,
                validation_reload_bit_exact=True,public_prediction_boundary_preserved=True,test_evaluation=False))
            del model,backend
    for mode in parent['modes']:
        probe=json.loads((run/'probe'/mode/'result.json').read_text())
        record=json.loads((base/'fit'/f"{mode}-{protocol['mechanism_seed']}"/'result.json').read_text())
        assert abs(probe['states']['selected']['splits']['validation']['log_loss']-record['training_summary']['monitored_value'])<2e-6
        assert not list((run/'probe'/mode).glob('*test*'))
    protected=[]
    for path in (base,Path(parent['parent_run']['path'])):
        receipt=json.loads((path/'COMPLETE.json').read_text())
        assert all(diagnostic.experiment.digest(path/name)==value for name,value in receipt['files'].items())
        protected.append(dict(run=str(path),completion_files_verified=len(receipt['files'])))
    diagnostic.load(run)
    result=dict(score_models_recomputed=recomputed,decomposition_max_absolute_error=max_error,matched_checks=checks,
        protected_experiments=protected,diagnostic_sources_unchanged=True,parent_inputs_unchanged=True)
    diagnostic.experiment.write_json(run/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    verify(parser.parse_args().run.resolve())
