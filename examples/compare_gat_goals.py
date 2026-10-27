"""Seven predeclared snapshot goal-supervision fits; never score test fixtures.

Commands: prepare, run, initialize, fit, report. Workers run sequentially in fresh
processes. The protocol, source, input files and copied initial states are checked.
"""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd

import compare_gat_temporal as base
from gat_temporal_diagnostics import score_rows, summarize
from temporal_team_gat import make_context
from xdiyo_analytics.graphs.observed_gap import migrate_gap_config
from xdiyo_analytics.graphs import GoalAuxiliary, TemporalGraphConfig, TemporalEventStore, TemporalGraphAdapter, TemporalGraphSerializer
from xdiyo_analytics.graphs.adapter import _Backend, inference_mode
from xdiyo_analytics.graphs.config import CLASS_ORDER, MATCH_KEYS
from xdiyo_analytics.graphs.goals import goal_labels, poisson_goal_loss
from xdiyo_analytics.graphs.execution import configured_execution, prepare_determinism
from xdiyo_analytics.training import TrainingControl, EarlyStopping, run_iterations
from xdiyo_analytics.training.residency import InputCachePolicy

ROOT = base.ROOT
SEEDS = (17, 42, 73)
ARMS = {'control': GoalAuxiliary(beta=0.), 'attached': GoalAuxiliary(beta=.1),
        'detached': GoalAuxiliary(beta=.1, detach_encoder=True)}
DATA = ('history.parquet', 'memberships.parquet', 'schedule.parquet', 'train.parquet', 'validation.parquet')
FILES = ('examples/compare_gat_goals.py', 'examples/render_gat_goals.py',
         'examples/gat_temporal_diagnostics.py', 'tests/analytics/test_graph_goal_auxiliary.py',
         'tests/analytics/test_gat_goal_experiment.py')


def source_hashes():
    return {**base.source_hashes(), **{name:base.digest(ROOT/name) for name in FILES}}


def digest_state(value):
    """Canonical tensor/config digest, also used to verify diagnostics are read-only."""
    import torch
    h = sha256()
    def add(v):
        if isinstance(v, (torch.Tensor, np.ndarray)):
            a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            h.update(str((a.dtype, a.shape)).encode()); h.update(a.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str): h.update(str(k).encode()); add(v[k])
        elif isinstance(v, (tuple, list)):
            for x in v: add(x)
        else: h.update(repr(v).encode())
    add(value)
    return h.hexdigest()


def control(seed, maximum=200):
    return TrainingControl(max_steps=maximum, early_stopping=EarlyStopping(15, 0.), seed=seed,
                           monitor='validation_loss', restore_best=True)


def prepare(parent, run):
    p = json.loads((parent/'protocol.json').read_text())
    if p['device'] != 'cuda:0' or p['config']['temporal_mode'] != 'snapshot' or p['config']['layers'] != 2:
        raise ValueError('Expected the two-layer CUDA snapshot reference.')
    run.mkdir(parents=True, exist_ok=False)
    for name in DATA:
        if base.digest(parent/name) != p['data_hashes'][name]: raise ValueError(f'Reference input changed: {name}')
        shutil.copyfile(parent/name, run/name)
    cfg = TemporalGraphConfig(**migrate_gap_config(p['config']))
    assert cfg.goal_auxiliary is None
    protocol = dict(schema=1, parent=str(parent.resolve()), parent_protocol_sha256=base.digest(parent/'protocol.json'),
        source_hashes=source_hashes(), data_hashes={name:base.digest(run/name) for name in DATA},
        provenance=p['provenance'], store_version=p['store_version'],
        populations={name:p['populations'][name] for name in ('train', 'validation')},
        config=asdict(cfg), arms={k:asdict(v) for k,v in ARMS.items()}, seeds=list(SEEDS),
        matrix=[dict(arm=arm, seed=seed) for seed in SEEDS for arm in ('control', 'attached')]+[dict(arm='detached', seed=17)],
        control=asdict(control(17)), device=p['device'], inner_threads=1,
        input_cache_bytes=p['input_cache_bytes'], input_cache_headroom_bytes=p['input_cache_headroom_bytes'],
        worker_rss_limit_gib=p['worker_rss_limit_gib'], cuda_allocator_config=p['cuda_allocator_config'],
        cuda_memory_fraction=p['cuda_memory_fraction'], primary='validation three-class log loss at each own selected checkpoint',
        diagnostic_population='all fitting fixtures, original weights and issuance groups',
        gradient_parameters='global_prior, league, entry, layers; exclude both heads and regularizer',
        clipping='none; native AdamW has per-parameter states, no global clipping or adaptive shared loss scaling',
        scalar_regularization='prior embedding penalty; decoupled AdamW weight decay remains an optimizer setting',
        no_test_evaluation=True, screen='negative mean attached-control validation log loss and at least 2/3 seeds improve',
        interpretation='exploratory development on already inspected validation; no beta adaptation',
        tolerances=dict(values_rtol=1e-5, values_atol=1e-6, gradients_rtol=2e-5, gradients_atol=2e-6))
    preserved = {}
    for name in ('gat_temporal_cuda_allocator_20261025', 'gat_temporal_cuda_patience15_20261025', 'gat_temporal_diagnostics_20261025'):
        for path in sorted((ROOT/'experiments'/name).rglob('*')):
            if path.is_file(): preserved[str(path.relative_to(ROOT)).replace('\\','/')] = base.digest(path)
    protocol['preserved_artifacts'] = preserved
    base.write_json(run/'protocol.json', protocol)
    with zipfile.ZipFile(run/'frozen_source.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in protocol['source_hashes']: archive.write(ROOT/name, name)
    protocol, store, frames = load(run)
    manifest = {}
    for name, frame in frames.items():
        ctx = make_context(frame)
        labels = goal_labels(store, frame, store.label_availability(frame, cfg).max(), ctx.y.iloc[:, 0])
        issue = store.issue_times(frame, cfg)
        store.assert_targets_unobserved(frame, issue, cfg)
        manifest[name] = dict(fixtures=len(frame), goal_labels=len(labels), target_columns=['home', 'away'],
                              label_sha256=sha256(labels.tobytes()).hexdigest(), means=labels.mean(0).tolist())
    base.write_json(run/'label_verification.json', manifest)
    print(json.dumps(dict(stage='prepared', run=str(run), fits=7, populations=manifest)), flush=True)


def load(run):
    p = json.loads((run/'protocol.json').read_text())
    if source_hashes() != p['source_hashes']: raise ValueError('Source changed after protocol freeze.')
    for name, expected in p['data_hashes'].items():
        if base.digest(run/name) != expected: raise ValueError(f'Frozen data changed: {name}')
    store = TemporalEventStore(pd.read_parquet(run/'history.parquet'), pd.read_parquet(run/'memberships.parquet'),
                              schedule=pd.read_parquet(run/'schedule.parquet'), provenance=p['provenance'])
    if store.version != p['store_version']: raise ValueError('Frozen event-store identity changed.')
    frames = {name:pd.read_parquet(run/f'{name}.parquet') for name in ('train','validation')}
    for name, frame in frames.items():
        if base.frame_hash(frame) != p['populations'][name]['identity_sha256']: raise ValueError('Fixture order changed.')
    return p, store, frames


def shared_parameters(model):
    return [(name,p) for name,p in model.named_parameters() if not name.startswith(('head.', 'goal_head.'))]


@configured_execution
def gradient_diagnostics(backend):
    """Full fitting-population task gradients; no .grad writes or optimizer steps."""
    import torch
    model = backend.model
    before = digest_state((model.state_dict(), backend.optimizer.state_dict(), [p.grad for p in model.parameters()]))
    mode = model.training
    model.eval()
    names, params = zip(*shared_parameters(model))
    outcome_sum = [torch.zeros_like(p) for p in params]
    goal_sum = [torch.zeros_like(p) for p in params]
    total = backend.train_weights.sum()
    try:
        for i in range(len(backend.train_groups)):
            with backend.train_groups.lease(i) as (positions, data, home, away, fixture):
                ix = torch.as_tensor(positions, device=backend.device)
                logits, rates = model(data, home, away, fixture, return_goals=True)
                outcome = (torch.nn.functional.cross_entropy(logits, backend.train_targets[ix], reduction='none')*backend.train_weights[ix]).sum()/total
                goal = (poisson_goal_loss(rates, backend.train_goals[ix])*backend.train_weights[ix]).sum()/total
                a = torch.autograd.grad(outcome, params, retain_graph=True, allow_unused=True)
                b = torch.autograd.grad(goal, params, allow_unused=True)
                for acc, grad in zip(outcome_sum, a):
                    if grad is not None: acc.add_(grad.detach())
                for acc, grad in zip(goal_sum, b):
                    if grad is not None: acc.add_(grad.detach())
        a, b = [torch.cat([g.reshape(-1).double() for g in values]) for values in (outcome_sum, goal_sum)]
        if not torch.isfinite(a).all() or not torch.isfinite(b).all(): raise ValueError('Nonfinite diagnostic gradients.')
        na, nb = float(a.norm()), float(b.norm())
        result = dict(outcome_norm=na, goal_norm=nb,
            weighted_gradient_ratio=backend.config.goal_auxiliary.beta*nb/na if na else None,
            cosine_similarity=float(torch.dot(a,b)/(a.norm()*b.norm())) if na and nb else None,
            zero_norms=[name for name,n in [('outcome',na),('goal',nb)] if not n],
            parameters=list(names), parameter_count=sum(p.numel() for p in params), fixtures=len(backend.train_targets),
            regularizer_included=False, detached=backend.config.goal_auxiliary.detach_encoder)
    finally:
        model.train(mode)
    after = digest_state((model.state_dict(), backend.optimizer.state_dict(), [p.grad for p in model.parameters()]))
    if after != before: raise AssertionError('Gradient diagnostics changed model/optimizer/gradient buffers.')
    result['state_unchanged'] = True
    return result


@configured_execution
def predict_population(backend, groups):
    p, rates = np.empty((len(groups.plan.metadata),3)), np.empty((len(groups.plan.metadata),2))
    with inference_mode(backend.model):
        for i in range(len(groups)):
            with groups.lease(i) as (positions, data, home, away, fixture):
                logits, mu = backend.model(data, home, away, fixture, return_goals=True)
                p[positions], rates[positions] = logits.softmax(-1).cpu().numpy(), mu.cpu().numpy()
    return p, rates


def score_population(frame, probabilities, rates, goals):
    from scipy.special import gammaln, xlogy
    outcome = np.sign(goals[:,0]-goals[:,1]).astype(int)
    rows = score_rows(outcome, probabilities)
    goal_rows = (rates-xlogy(goals,rates)+gammaln(goals+1)).sum(1)
    constant_free = (rates-xlogy(goals,rates)).sum(1)
    if not np.isfinite(rates).all() or (rates <= 0).any() or not np.isfinite(goal_rows).all():
        raise ValueError('Invalid rates or goal scores.')
    result = {name:float(value.mean()) for name,value in rows.items()}
    non_draw = outcome != 0
    result.update(conditional_winner_log_loss=float(rows['winner_contribution'][non_draw].mean()) if non_draw.any() else None,
        non_draw_fraction=float(non_draw.mean()),
        decomposition_error=float(np.max(np.abs(rows['log_loss']-rows['draw_log_loss']-rows['winner_contribution']))),
        full_poisson_nll=float(goal_rows.mean()), goal_loss=float(constant_free.mean()),
        home_mae=float(np.abs(rates[:,0]-goals[:,0]).mean()), away_mae=float(np.abs(rates[:,1]-goals[:,1]).mean()),
        home_rate=summarize(rates[:,0]), away_rate=summarize(rates[:,1]),
        observed_mean_home=float(goals[:,0].mean()), observed_mean_away=float(goals[:,1].mean()))
    output = frame[[*MATCH_KEYS, 'round', 'kickoff_at']].copy()
    output['truth'] = outcome
    for i,name in enumerate(('p_home','p_draw','p_away')): output[name] = probabilities[:,i]
    for i,name in enumerate(('home','away')):
        output[f'lambda_{name}'], output[f'goals_{name}'] = rates[:,i], goals[:,i]
    for name,value in rows.items(): output[name] = value
    output['full_poisson_nll'], output['goal_loss'] = goal_rows, constant_free
    return result, output


def input_digest(backend):
    h = sha256()
    for groups in (backend.train_groups, backend.validation_groups):
        for i in range(len(groups)):
            with groups.lease(i) as group: h.update(digest_state(group).encode())
    return h.hexdigest()


def configure(p):
    os.environ['PYTORCH_ALLOC_CONF']=p['cuda_allocator_config']
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    prepare_determinism(TemporalGraphConfig(**migrate_gap_config(p['config'])),p['device'])
    torch.use_deterministic_algorithms(True)
    if p['device'].startswith('cuda'):
        if not torch.cuda.is_available(): raise RuntimeError('Declared CUDA device unavailable.')
        torch.cuda.set_per_process_memory_fraction(p['cuda_memory_fraction'])
        torch.cuda.reset_peak_memory_stats()
    return InputCachePolicy(p['input_cache_bytes'],p['input_cache_headroom_bytes'])


def initialize_seed(run, seed):
    import torch
    from xdiyo_analytics.graphs.network import TemporalNetwork
    p, store, frames = load(run)
    cache = configure(p)
    cfg = replace(TemporalGraphConfig(**migrate_gap_config(p['config'])), goal_auxiliary=ARMS['control'])
    train = make_context(frames['train']); train.validation = make_context(frames['validation'])
    b = _Backend(cfg, store, seed, p['device'], cache)
    b.initialize(train)
    pre = b.preprocessor
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = TemporalNetwork(replace(cfg, goal_auxiliary=None), len(pre.leagues), pre.edge_dim,
                                    pre.fixture_dim).to(p['device'])
    state = b.model.state_dict()
    for name,value in reference.state_dict().items():
        torch.testing.assert_close(state[name], value, rtol=0, atol=0)
        state[name] = value.detach().clone()  # Explicit copy, never rely on a seed alone.
    b.model.load_state_dict(state)
    folder = run/'initial'/str(seed); folder.mkdir(parents=True, exist_ok=False)
    maximum = 0.
    for name, groups in (('train',b.train_groups),('validation',b.validation_groups)):
        prob, mu = predict_population(b, groups)
        with inference_mode(reference):
            for i in range(len(groups)):
                with groups.lease(i) as (positions,data,home,away,fixture):
                    expected = reference(data,home,away,fixture)
                    actual = b.model(data,home,away,fixture)
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    maximum=max(maximum,float((actual-expected).abs().max()))
        pd.DataFrame(np.column_stack((prob,mu)),columns=['p_home','p_draw','p_away','lambda_home','lambda_away']).to_parquet(folder/f'{name}.parquet',index=False)
    torch.save({k:v.detach().cpu() for k,v in b.model.state_dict().items()}, folder/'weights.pt')
    base.write_json(folder/'manifest.json',dict(weights_sha256=base.digest(folder/'weights.pt'),
        state_digest=digest_state(b.model.state_dict()), common_digest=digest_state(reference.state_dict()),
        inputs_sha256=input_digest(b), preprocessing=pre.state(), maximum_reference_logit_difference=maximum,
        population_hashes={name:base.digest(folder/f'{name}.parquet') for name in ('train','validation')}))
    print(json.dumps(dict(stage='initialized',seed=seed,reference_logit_error=maximum)),flush=True)


class MatchedBackend(_Backend):
    def __init__(self, *args, initial_folder, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_folder=initial_folder

    def initialize(self, context):
        import torch
        super().initialize(context)
        manifest=json.loads((self.initial_folder/'manifest.json').read_text())
        if base.digest(self.initial_folder/'weights.pt') != manifest['weights_sha256']: raise ValueError('Initial weights changed.')
        self.model.load_state_dict(torch.load(self.initial_folder/'weights.pt',map_location=self.device,weights_only=True))
        if digest_state(self.model.state_dict()) != manifest['state_digest']: raise AssertionError('Initial parameter copy failed.')
        if input_digest(self) != manifest['inputs_sha256']: raise AssertionError('Prepared input mismatch across arms.')
        if json.loads(json.dumps(self.preprocessor.state())) != manifest['preprocessing']: raise AssertionError('Preprocessing mismatch.')
        for name,groups in (('train',self.train_groups),('validation',self.validation_groups)):
            if base.digest(self.initial_folder/f'{name}.parquet') != manifest['population_hashes'][name]: raise ValueError('Initial predictions changed.')
            actual=np.column_stack(predict_population(self,groups))
            expected=pd.read_parquet(self.initial_folder/f'{name}.parquet').to_numpy()
            np.testing.assert_array_equal(actual,expected)
        self.initial_gradients=gradient_diagnostics(self)
        self.refresh_version()

    def checkpoint_metrics(self, monitor, values):
        # Post-update curves describe the same checkpoint as validation/selection.
        return super().checkpoint_metrics('train_loss', values)


def fit(run, arm, seed):
    import torch
    p,store,frames=load(run)
    if dict(arm=arm,seed=seed) not in p['matrix']: raise ValueError('Fit outside predeclared matrix.')
    cache=configure(p)
    cfg=replace(TemporalGraphConfig(**migrate_gap_config(p['config'])), goal_auxiliary=GoalAuxiliary(**p['arms'][arm]))
    train=make_context(frames['train']); train.validation=make_context(frames['validation'])
    folder=run/'fit'/f'{arm}-{seed}'; folder.mkdir(parents=True,exist_ok=False)
    progress=[]
    def observer(event):
        if event.final: return
        progress.append(dict(step=event.step,**event.metrics))
        base.write_json(folder/'progress.json',progress)
        if event.step==1 or event.step%10==0:
            print(json.dumps(dict(stage='fit',arm=arm,seed=seed,step=event.step,validation_loss=event.metrics['validation_loss'])),flush=True)
    train.observer=observer
    ctrl=control(seed,p['control']['max_steps'])
    start=time.perf_counter()
    with base.MemoryProbe(p['worker_rss_limit_gib']) as memory:
        (b,history,summary),seconds=base.timed(lambda:run_iterations(
            lambda s:MatchedBackend(cfg,store,s,p['device'],cache,initial_folder=run/'initial'/str(seed)),train,ctrl),p['device'])
        base.validate_training_progress(progress,summary,dict(max_steps=ctrl.max_steps,early_stopping=asdict(ctrl.early_stopping)))
        selected_gradients,diagnostic_seconds=base.timed(lambda:gradient_diagnostics(b),p['device'])
        result=dict(arm=arm,seed=seed,config=asdict(cfg),summary=summary,fit_seconds=seconds,
            environment=dict(executable=sys.executable,python=sys.version,torch=str(torch.__version__),
                             cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),numpy=np.__version__,pandas=pd.__version__),
            selected_gradient_seconds=diagnostic_seconds,parameters=sum(x.numel() for x in b.model.parameters()),
            goal_parameters=sum(x.numel() for x in b.model.goal_head.parameters()),
            shared_parameters=sum(x.numel() for _,x in shared_parameters(b.model)),
            initial= b.initial_gradients,selected=selected_gradients,
            initial_weights_sha256=base.digest(run/'initial'/str(seed)/'weights.pt'),
            protocol_sha256=base.digest(run/'protocol.json'))
        for name,groups,goals in (('train',b.train_groups,b.train_goals),('validation',b.validation_groups,b.validation_goals)):
            prob,mu=predict_population(b,groups)
            result[name],table=score_population(frames[name],prob,mu,goals.cpu().numpy().astype(float))
            table.to_parquet(folder/f'{name}_predictions.parquet',index=False)
        np.testing.assert_allclose(result['validation']['log_loss'],summary['monitored_value'],atol=1e-6,rtol=1e-5)
        adapter=TemporalGraphAdapter(cfg,store,control=ctrl,device=p['device'])
        adapter.backend_,adapter.training_history_,adapter.training_summary_=b,history,summary
        TemporalGraphSerializer().save(adapter,folder/'model')
        restored=TemporalGraphSerializer(store,device=p['device']).load(folder/'model')
        reloaded=restored.backend_
        # In-sample diagnostic replay, without weakening the public trained-through guard.
        group=reloaded._stream(make_context(frames['validation'],fitting=False))
        prob,mu=predict_population(reloaded,group)
        saved=pd.read_parquet(folder/'validation_predictions.parquet')
        np.testing.assert_array_equal(np.column_stack((prob,mu)),saved[['p_home','p_draw','p_away','lambda_home','lambda_away']].to_numpy())
        for method in (restored.predict,restored.predict_goals):
            try: method(make_context(frames['validation'],fitting=False))
            except ValueError as e:
                if 'boundary' not in str(e): raise
            else: raise AssertionError('Public prediction boundary bypassed.')
        history.to_csv(folder/'training.csv',index=False)
        result.update(reload_exact=True,public_boundary_enforced=True,memory=memory.result(),elapsed_seconds=time.perf_counter()-start)
        result['memory'].update(cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(),cuda_reserved_peak_bytes=torch.cuda.max_memory_reserved())
        base.write_json(folder/'result.json',result)
    print(json.dumps(dict(stage='complete',arm=arm,seed=seed,steps=summary['attempts'][0]['steps'],best=summary['best_step'],validation=result['validation']['log_loss'])),flush=True)


def run_all(run):
    p,_,_=load(run)
    environment=os.environ.copy()
    environment['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    environment['PYTORCH_ALLOC_CONF']=p['cuda_allocator_config']
    with (run/'workers.log').open('x',encoding='utf-8') as log:
        commands=[('initialize',seed,None) for seed in SEEDS]+[('fit',x['seed'],x['arm']) for x in p['matrix']]
        for command,seed,arm in commands:
            args=[sys.executable,str(Path(__file__).resolve()),command,'--run',str(run.resolve()),'--seed',str(seed)]
            if arm: args += ['--arm',arm]
            print(json.dumps(dict(worker=command,seed=seed,arm=arm)),flush=True)
            subprocess.run(args,check=True,env=environment,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            log.flush()
    from render_gat_goals import report
    report(run)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','initialize','fit','report'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--parent',type=Path,default=ROOT/'experiments/gat_temporal_cuda_patience15_20261025')
    parser.add_argument('--seed',type=int,choices=SEEDS,default=17)
    parser.add_argument('--arm',choices=ARMS,default='control')
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.parent,args.run)
    elif args.command=='run': run_all(args.run)
    elif args.command=='initialize': initialize_seed(args.run,args.seed)
    elif args.command=='fit': fit(args.run,args.arm,args.seed)
    else:
        from render_gat_goals import report
        report(args.run)


if __name__=='__main__': main()
