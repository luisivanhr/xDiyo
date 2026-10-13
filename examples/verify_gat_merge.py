"""Bounded real-input merge gate; no writes to the completed research run.

Uses all four frozen recipe configurations with three training, two validation
and one prediction issuance sampled from completed fold 02. This verifies recipe
shapes on real graphs; it is not a full-fold training or cross-process resume test.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests/analytics')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ATOL, RTOL = 1e-6, 1e-5


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.resolve().is_relative_to(args.run.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import pandas as pd
    import torch
    from test_graph_speed import reference
    from test_graph_preparation_memory import assert_exact
    from xdiyo_analytics.graphs import adapter, store, preprocessing, streaming
    from xdiyo_analytics.graphs import TemporalGraphConfig, TemporalGraphAdapter, TemporalGraphSerializer
    from xdiyo_analytics.training import TrainingControl
    from xdiyo_analytics.training.control import run_iterations
    from xdiyo_analytics.training.contracts import FitContext, PredictionContext
    sys.path.insert(0, str(args.run/'runner'))
    from data_adapter import attach_edge_stats, attach_fixture_head
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    assert torch.cuda.get_device_name(0) == 'NVIDIA GeForce RTX 4070 SUPER'
    protocol = json.loads((args.run/'STUDY_PROTOCOL.json').read_text())
    manifest = json.loads((args.run/'INPUT_MANIFEST.json').read_text())
    files = [args.run/name for name in manifest['files']]
    files += [args.run/'baseline'/name for name in manifest['baseline_input_hashes']]
    fold = args.run/'gpu_runs/gat_expanded_totals/refit-fold-02'
    files += [fold/'train_metadata.parquet', args.run/'STUDY_PROTOCOL.json',
              args.run/'INPUT_MANIFEST.json', args.run/'runner/data_adapter.py']
    hashes = {str(p.relative_to(args.run)): digest(p) for p in files}
    for name, value in manifest['files'].items():
        assert digest(args.run/name) == value['sha256'], name
    for name, value in manifest['baseline_input_hashes'].items():
        assert digest(args.run/'baseline'/name) == value, name
    source = {str(p.relative_to(ROOT)): digest(p) for folder in ('graphs','training')
              for p in (ROOT/'src/xdiyo_analytics'/folder).glob('*.py')}
    frozen = {p.name: digest(p) for p in (ROOT/'tests/analytics/reference').glob('gat_*_11ffa446.txt')}
    receipt = dict(status='RUNNING', candidate=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        command=[sys.executable,*sys.argv], executable=sys.executable, python=sys.version,
        torch=torch.__version__, pandas=pd.__version__, numpy=np.__version__, gpu=torch.cuda.get_device_name(0),
        declared_atol=ATOL, declared_rtol=RTOL, comparison='bit-exact deterministic CUDA; unsupported operations fail the gate',
        source_sha256=source, frozen_source_sha256=frozen, input_sha256=hashes,
        scope='Three real training issuances, two validation issuances, one prediction issuance per recipe; three updates',
        cross_process_resume_tested=False, arms={})
    def write():
        (args.output/'parity.json').write_text(json.dumps(receipt,indent=2,default=str))
    write()  # Freeze settings before evaluating any candidate.
    history = pd.read_parquet(args.run/'baseline/model_history.parquet')
    members = pd.read_parquet(args.run/'baseline/memberships_schedule_effective.parquet')
    schedule = pd.read_parquet(args.run/'baseline/fixtures.parquet')
    frame = pd.read_parquet(fold/'train_metadata.parquet')
    head = pd.read_parquet(args.run/'fixture_head.parquet')
    modules = [reference(), dict(adapter=adapter,store=store,preprocessing=preprocessing,streaming=streaming)]

    def cpu(value):
        if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
        if isinstance(value, dict): return {k:cpu(v) for k,v in value.items()}
        if isinstance(value, (list,tuple)): return type(value)(cpu(v) for v in value)
        return deepcopy(value)

    try:
        for arm, definition in protocol['arms'].items():
            receipt['active_arm'] = arm; write()
            cfg = TemporalGraphConfig(**definition['native_config_base'])
            family = 'halves' if '_halves' in arm else 'totals'
            assert list(cfg.statistics) == manifest['statistics'][family]
            assert cfg.include_goal_features == manifest['native_goal_predictors'][family]
            h = attach_edge_stats(history, pd.read_parquet(args.run/f'edge_stats_{family}.parquet'),list(cfg.statistics))
            stores = [mod['store'].TemporalEventStore(h,members,schedule=schedule) for mod in modules]
            assert stores[0].version == stores[1].version
            issues = stores[0].issue_times(frame,cfg)
            unique = sorted(issues.unique()); n = len(unique)
            train_times = [unique[i] for i in (0,n//3,2*n//3)]
            val_times = [unique[i] for i in (3*n//4,4*n//5)]
            pred_times = [unique[-1]]
            def context(times, fitting=True):
                meta = frame.loc[issues.isin(times)].copy()
                X = pd.DataFrame(index=meta.index)
                if cfg.fixture_features:
                    meta = attach_fixture_head(meta,head,list(cfg.fixture_features))
                    X = meta[list(cfg.fixture_features)].copy()
                common = dict(X=X, metadata=meta, layout='match', match_columns=('competition_id','season_id','event_id'), fold_id=2)
                return (FitContext(y=pd.DataFrame({'result':np.sign(meta.goals_for-meta.goals_against).astype(int)},index=meta.index),**common)
                        if fitting else PredictionContext(**common))
            fit = context(train_times); fit.validation = context(val_times)
            prediction = context(pred_times,False)
            results = []; initial = []
            control = TrainingControl(max_steps=3,seed=17,monitor='validation_loss',restore_best=True)
            for label, mod, event_store in zip(('reference','candidate'), modules, stores):
                records = []
                class AuditedBackend(mod['adapter']._Backend):
                    def initialize(self, ctx):
                        super().initialize(ctx)
                        initial.append(cpu(self.model.state_dict()))
                    def checkpoint_metrics(self, monitor, values):
                        # Explicitly evaluate post-update training AND validation
                        # at every update while selecting by validation loss.
                        measured = super().checkpoint_metrics('train_loss',values)
                        records.append(dict(losses=cpu(measured),
                            gradients=cpu([p.grad for p in self.model.parameters()]),
                            model=cpu(self.model.state_dict()),optimizer=cpu(self.optimizer.state_dict()),
                            rng=cpu(torch.get_rng_state()),cuda_rng=cpu(torch.cuda.get_rng_state())))
                        return measured
                torch.manual_seed(17); torch.cuda.manual_seed_all(17)
                backend, log, summary = run_iterations(lambda seed:AuditedBackend(cfg,event_store,seed,'cuda:0'),fit,control)
                model = TemporalGraphAdapter(cfg,event_store,control=control,device='cuda:0')
                model.backend_, model.training_history_, model.training_summary_ = backend,log,summary
                pred = model.predict(prediction)
                serializer = TemporalGraphSerializer(embed_event_store=True,device='cuda:0')
                location = args.output/arm/label
                serializer.save(model,location)
                loaded = serializer.load(location)
                assert_exact(backend.model.state_dict(),loaded.backend_.model.state_dict())
                for key, values in pred.items():
                    pd.testing.assert_frame_equal(values,loaded.predict(prediction)[key],check_exact=True)
                chosen = summary['retained_step']-1
                assert chosen == int(np.argmin([x['losses']['validation_loss'] for x in records]))
                assert_exact(cpu(backend.model.state_dict()), records[chosen]['model'])
                results.append((backend,log,summary,records,pred))
                print(json.dumps(dict(arm=arm,implementation=label,updates=len(records),status='FINISHED')),flush=True)
            assert_exact(initial[0],initial[1])
            old,new = results
            assert_exact(old[3],new[3])  # Every update: losses/gradients/weights/full AdamW/RNG.
            pd.testing.assert_frame_equal(old[1],new[1],check_exact=True)
            assert_exact(old[2],new[2])
            assert_exact(old[0].preprocessor.state(),new[0].preprocessor.state())
            for key in old[4]: pd.testing.assert_frame_equal(old[4][key],new[4][key],check_exact=True)
            groups = []
            for name in ('train_groups','validation_groups'):
                left,right = getattr(old[0],name),getattr(new[0],name)
                assert len(left)==len(right)
                for i in range(len(left)):
                    assert_exact(left[i],right[i])
                    a,b = left.plan.raw(i),right.plan.raw(i)
                    pd.testing.assert_frame_equal(a[1].edges,b[1].edges,check_exact=True)
                    for key in ('teams','receiver','sender','log_weight','support','gate','league','entry','cutoff'):
                        assert_exact(getattr(a[1],key),getattr(b[1],key))
                    pd.testing.assert_frame_equal(a[1].edges,stores[0].eligible(a[1].cutoff),check_exact=True)
                    x=right[i][1]['x']; cols=list(new[0].preprocessor.edge_columns)
                    raw=b[1].edges[cols].to_numpy(dtype=float,na_value=np.nan)
                    np.testing.assert_array_equal(x[:,len(cols):2*len(cols)].cpu().numpy(),np.isnan(raw))
                    groups.append(dict(partition=name,issuance=str(a[1].issuance),edges=len(a[1].edges)))
            receipt['arms'][arm]=dict(status='PASS_BIT_EXACT',config=definition['native_config_base'],groups=groups,
                train_rows=len(fit.X),validation_rows=len(fit.validation.X),prediction_rows=len(prediction.X),
                steps=[r['losses'] for r in new[3]],selected_step=new[2]['retained_step'],
                checks=['prepared inputs','causal eligibility','graph order','masks','scaling','per-update losses',
                        'gradients','parameters','full AdamW state','RNG','predictions','checkpoint selection','save/load'])
            write()
            del results, stores, old, new, backend, loaded, model
            import gc
            gc.collect();torch.cuda.empty_cache()
        assert hashes == {str(p.relative_to(args.run)):digest(p) for p in files}
        receipt.update(status='PASS',inputs_unchanged=True);receipt.pop('active_arm',None);write()
    except BaseException as exc:
        receipt.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
        write();raise


if __name__ == '__main__':
    main()
