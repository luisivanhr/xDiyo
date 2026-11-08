"""Sealed, resumable dataset-expansion GAT selection and expanding refits.

All model fits are fresh. Development selects updates once; outer tests never
select anything. This script does not launch unless explicitly invoked.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'examples')]
OUT = ROOT / 'results'
STAGES = ('original13', 'first_divisions', 'first_and_second_divisions')
FAMILIES = ('time_B/N', 'time_C/E1', 'time_B/CD_DW')
SEEDS = (17, 42, 73)
FOLDS = ('fold_1', 'fold_2', 'fold_3', 'fold_4')
STATISTICS = (
    'team::ALL::Match overview::totalShotsOnGoal::value',
    'team::ALL::Shots::shotsOffGoal::value',
    'team::ALL::Match overview::cornerKicks::value',
    'opponent::ALL::Match overview::cornerKicks::value',
    'opponent::ALL::Match overview::totalShotsOnGoal::value',
    'opponent::ALL::Shots::shotsOffGoal::value',
)
MONITOR = 'development_raw_hda_ce_micro'
SELECTION_JOBS = [dict(stage=s, family=f, seed=n, phase='selection')
                  for s in STAGES for f in FAMILIES for n in SEEDS]
REFIT_JOBS = [dict(stage=s, family=f, seed=n, phase='refit', fold=k)
              for s in STAGES for f in FAMILIES for k in FOLDS for n in SEEDS]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)


def folder(job):
    return OUT / job['phase'] / job['stage'] / job['family'] / str(job['seed']) / job.get('fold', 'development')


def setup():
    import torch
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires authorized local CUDA; no CPU fallback.')
    torch.cuda.reset_peak_memory_stats()


def config(family, store):
    from xdiyo_analytics.graphs.research_profiles import GATActualDateBNV1, GATActualDateCE1V1, GATActualDateBCDDWV1
    if family == 'time_B/N':
        cfg = GATActualDateBNV1()
    elif family == 'time_C/E1':
        cfg = GATActualDateCE1V1(store.phase_context.source_id)
    elif family == 'time_B/CD_DW':
        cfg = GATActualDateBCDDWV1(store.phase_context.source_id, 10)
    else:
        raise ValueError(family)
    return replace(cfg, statistics=STATISTICS)


def prepare(job):
    from experiment_inputs import load_stage
    from temporal_team_gat import make_context
    data = load_stage(job['stage'])
    store = data['store']
    cfg = config(job['family'], store)
    if job['phase'] == 'selection':
        context = make_context(data['dev_fit'].reset_index(drop=True))
        context.validation = make_context(data['dev_validation'].reset_index(drop=True))
        outer = None
    else:
        split = data['folds'][job['fold']]
        context = make_context(split['train'].reset_index(drop=True))
        outer = make_context(split['test'].reset_index(drop=True), fitting=False)
    return cfg, store, context, outer


def backend_class():
    import numpy as np
    import torch
    from xdiyo_analytics.graphs.adapter import _Backend
    from xdiyo_analytics.graphs.execution import configured_execution
    from xdiyo_analytics.graphs.goals import poisson_goal_loss
    from xdiyo_analytics.graphs.research_training import digest_state, rps, summarize

    class ExpansionBackend(_Backend):
        """Native updates; raw equal-fixture validation CE, scope-neutral names.

        Validation reduction matches RetainedGATBackend, rather than the native
        goal-training reduction. Refits have no validation/selection path.
        """
        def __init__(self, *args, preparation_path=None, **kwargs):
            self.preparation_path = preparation_path
            super().__init__(*args, **kwargs)

        def initialize(self, context):
            super().initialize(context)
            self._validation_leagues = (None if context.validation is None else
                context.validation.metadata.competition_id.to_numpy(dtype=np.int64))
            if context.validation is not None:
                w = self.validation_weights
                if not torch.equal(w, w[0].expand_as(w)):
                    raise ValueError('Development requires equal per-fixture weights.')
            expected = ('goals_for', 'goals_against', *STATISTICS)
            if tuple(self.preprocessor.edge_columns) != expected:
                raise ValueError('Raw edge coordinate order differs from prespecified order.')
            encoded = tuple(self.preprocessor.edge_encoded_columns)
            n = len(expected)
            if encoded[:n] != tuple(c + '__standardized' for c in expected) or encoded[n:2*n] != tuple(c + '__missing' for c in expected):
                raise ValueError('Encoded value/missingness order differs.')
            for name, tensor in self.model.state_dict().items():
                if name.startswith('layers.') and name.endswith(('.e.weight', '.u.weight')) and tensor.shape[1] != len(encoded):
                    raise ValueError('E/U input dimension differs from named edge coordinates: ' + name)
            self.initial_digest = digest_state(self.model.state_dict())
            if self.preparation_path:
                write(self.preparation_path, dict(config=asdict(self.config), preprocessor=self.preprocessor.state(),
                    edge_columns=encoded, fixture_columns=self.preprocessor.fixture_encoded_columns,
                    initial_digest=self.initial_digest, parameters=sum(p.numel() for p in self.model.parameters()),
                    event_store_version=self.store.version, fitting_count=len(context.X),
                    validation_count=0 if context.validation is None else len(context.validation.X),
                    initialization='Fresh native paired seed. Expanded league dimensions mean shared tensors need not be bit-identical across stages.',
                    ram_guard=False, input_cache_bytes=4*1024**3, headroom_bytes=1024**3,
                    optional_speedups=False))

        @configured_execution
        def _loss(self, groups, targets, weights, backward=False):
            if backward or groups is not self.validation_groups or self.validation_groups is None:
                return super()._loss(groups, targets, weights, backward)
            count = len(targets)
            losses, goals = np.empty(count), np.empty(count)
            probability = np.empty((count, 3))
            seen = np.zeros(count, dtype=np.int64)
            with torch.no_grad():
                for i in range(len(groups)):
                    with self._group_lease(groups, i) as (group, _):
                        positions, data, home, away, fixture = group
                        ix = torch.as_tensor(positions, device=self.device)
                        logits, rates = self.model(data, home, away, fixture, return_goals=True)
                        losses[positions] = torch.nn.functional.cross_entropy(logits, targets[ix], reduction='none').cpu().numpy()
                        goals[positions] = poisson_goal_loss(rates, self.validation_goals[ix]).cpu().numpy()
                        probability[positions] = logits.softmax(1).cpu().numpy()
                        seen[positions] += 1
            if not (seen == 1).all():
                raise ValueError('Every development fixture must be scored exactly once.')
            self._inner_summary = summarize(losses, self._validation_leagues)
            self._inner_summary['rps'] = float(rps(probability, targets.cpu().numpy()).mean())
            self._goal_loss = float(goals.mean())
            return self._inner_summary['micro']

        def step(self):
            self._update_started = time.perf_counter()
            return super().step()

        def checkpoint_metrics(self, monitor, values):
            if monitor not in (MONITOR, 'train_loss'):
                raise ValueError('Unexpected monitor: ' + str(monitor))
            values = super().checkpoint_metrics('train_loss', values)
            if monitor == MONITOR:
                values[MONITOR] = values['validation_loss']
                if values[MONITOR] != self._inner_summary['micro']:
                    raise ValueError('Raw validation monitor changed during metric replay.')
                values['development_raw_hda_ce_macro'] = self._inner_summary['macro']
                values['development_raw_rps_micro'] = self._inner_summary['rps']
                values.update({f'development_raw_ce_league_{k}': v for k, v in self._inner_summary['per_league'].items()})
            if not all(np.isfinite(v) for v in values.values()):
                raise FloatingPointError('Nonfinite metric; no clipping or silent recovery.')
            if str(self.device).startswith('cuda'):
                torch.cuda.synchronize(self.device)
            values['update_seconds'] = time.perf_counter() - self._update_started
            return values
    return ExpansionBackend


def predict(backend, groups, frame):
    import numpy as np
    from xdiyo_analytics.graphs.adapter import inference_mode
    from xdiyo_analytics.graphs.execution import configured_execution
    @configured_execution
    def calculate(self):
        p, rates, seen = np.empty((len(frame), 3)), np.empty((len(frame), 2)), np.zeros(len(frame), int)
        with inference_mode(self.model):
            for i in range(len(groups)):
                with self._group_lease(groups, i) as (group, _):
                    ix, data, h, a, f = group
                    logits, r = self.model(data, h, a, f, return_goals=True)
                    p[ix], rates[ix] = logits.softmax(1).cpu().numpy(), r.cpu().numpy()
                    seen[ix] += 1
        if not (seen == 1).all():
            raise ValueError('Prediction must score every row once.')
        return p, rates
    return calculate(backend)


def table(frame, probability, rates):
    import numpy as np
    if not np.isfinite(probability).all() or (probability <= 0).any() or not np.allclose(probability.sum(1), 1, atol=3e-7, rtol=0):
        raise FloatingPointError('Invalid un-clipped probability export.')
    p = probability / probability.sum(1, keepdims=True)
    x = frame.copy().reset_index(drop=True)
    y = np.where(x.goals_for > x.goals_against, 0, np.where(x.goals_for == x.goals_against, 1, 2))
    x['y'] = y
    x[['p_home', 'p_draw', 'p_away']] = p
    x[['lambda_home', 'lambda_away']] = rates
    x['ce'] = -np.log(p[np.arange(len(y)), y])
    x['rps'] = .5 * ((np.cumsum(p, axis=1)[:, :2] - (y[:, None] <= np.arange(2)))**2).sum(1)
    x['accuracy'] = (p.argmax(1) == y).astype(float)
    x['brier'] = ((p - np.eye(3)[y])**2).sum(1)
    x['draw_ce'] = -np.where(y == 1, np.log(p[:, 1]), np.log1p(-p[:, 1]))
    return x


def verify():
    seal = json.loads((OUT / 'SOURCE_SEAL.json').read_text())
    for path, digest in seal.items():
        if sha(path) != digest:
            raise ValueError('Sealed input/source changed: ' + path)


def selection_for(job):
    key = f"{job['stage']}|{job['family']}|{job['seed']}"
    return json.loads((OUT / 'SELECTION.json').read_text())['choices'][key]


def fit(job, *, smoke=False):
    import torch
    from xdiyo_analytics.training import TrainingControl, EarlyStopping
    from xdiyo_analytics.training.resume_loop import run_resumable_iterations
    from xdiyo_analytics.training.residency import InputCachePolicy
    from xdiyo_analytics.graphs import TemporalGraphAdapter, TemporalGraphSerializer
    directory = (OUT / 'smoke' / job['stage'] / job['family'] / str(job['seed'])) if smoke else folder(job)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'fit.json').exists():
        return
    if not smoke:
        verify()
    setup()
    cfg, store, context, outer = prepare(job)
    if smoke:
        from temporal_team_gat import make_context
        validation = context.validation.metadata.iloc[:32].reset_index(drop=True)
        context = make_context(context.metadata.iloc[-32:].reset_index(drop=True))
        context.validation = make_context(validation)
        steps = 1
    elif job['phase'] == 'selection':
        steps = 200
    else:
        steps = int(selection_for(job)['updates'])
    selection = job['phase'] == 'selection'
    control = TrainingControl(max_steps=steps, early_stopping=EarlyStopping(15, 0) if selection else None,
        seed=job['seed'], monitor=MONITOR if selection else 'train_loss', restore_best=selection)
    Backend = backend_class()
    policy = InputCachePolicy(4*1024**3, 1024**3)
    def factory(seed):
        return Backend(cfg, store, seed, 'cuda:0', policy, preparation_path=directory / 'preparation.json')
    def observer(event):
        write(directory / 'progress.json', dict(step=event.step, metrics=event.metrics, final=event.final,
            pid=os.getpid(), utc_epoch=time.time(), job=job))
    context.observer = observer
    def save(capture, *, complete):
        # A single atomic envelope avoids a checkpoint/hash sidecar crash window.
        path = directory / 'continuation.pt'
        temporary = directory / 'continuation.tmp'
        torch.save(dict(job=job, control=asdict(control), complete=complete, state=capture()), temporary)
        temporary.replace(path)
    state = None
    if (directory / 'continuation.pt').exists():
        envelope = torch.load(directory / 'continuation.pt', map_location='cpu', weights_only=True)
        if envelope['job'] != job or envelope['control'] != asdict(control):
            raise ValueError('Continuation configuration differs.')
        state = envelope['state']
    started = time.perf_counter()
    backend, history, summary = run_resumable_iterations(factory, context, control, state=state, save=save)
    if not selection and summary['retained_step'] != steps:
        raise ValueError('Fixed refit stopped before the development-selected update budget.')
    history.to_csv(directory / 'history.csv', index=False)
    for label, ctx, groups in [('fit', context, backend.train_groups),
                              ('development', context.validation, backend.validation_groups)]:
        if ctx is not None:
            p, rates = predict(backend, groups, ctx.metadata)
            table(ctx.metadata, p, rates).to_parquet(directory / (label + '.parquet'), index=False)
    adapter = TemporalGraphAdapter(cfg, store, control=control, device='cuda:0')
    adapter.input_cache_policy = policy
    adapter.backend_, adapter.training_history_, adapter.training_summary_ = backend, history, summary
    TemporalGraphSerializer(store, device='cuda:0').save(adapter, directory / 'model')
    write(directory / 'fit.json', dict(job=job, summary=summary, seconds=time.perf_counter()-started,
        source_seal=None if smoke else sha(OUT / 'SOURCE_SEAL.json'),
        cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved(),
        completed_at=time.time(), smoke=smoke))


def evaluate(job):
    from xdiyo_analytics.graphs import TemporalGraphSerializer
    directory = folder(job)
    if (directory / 'outer.parquet').exists() and (directory / 'evaluation.json').exists():
        return
    verify()
    setup()
    _, store, _, outer = prepare(job)
    backend = TemporalGraphSerializer(store, device='cuda:0').load(directory / 'model').backend_
    backend._boundary(backend._validate(outer))
    p, rates = predict(backend, backend._stream(outer), outer.metadata)
    temporary = directory / 'outer.tmp.parquet'
    table(outer.metadata, p, rates).to_parquet(temporary, index=False)
    temporary.replace(directory / 'outer.parquet')
    write(directory / 'evaluation.json', dict(rows=len(outer.X), sha256=sha(directory / 'outer.parquet'),
        selection_sha256=sha(OUT / 'SELECTION.json'), model_version=backend.model_version))


def seal():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'SOURCE_SEAL.json').exists():
        verify()
        return
    check = json.loads((OUT / 'PREFLIGHT.json').read_text())
    if check['status'] != 'PASS':
        raise ValueError('Preflight must pass before sealing.')
    # Analysis/report scripts may be completed while compute runs. Seal only
    # code that can affect fitting, plus all immutable inputs and the plan.
    files = [ROOT / 'run.py', ROOT / 'experiment_inputs.py', ROOT / 'REPORTING_STANDARDS.md',
             ROOT / 'EXPERIMENT_PROTOCOL.md', OUT / 'PREFLIGHT.json']
    files += [p for p in ROOT.glob('prepare*.py')]
    for directory in ('src', 'examples', 'data'):
        files.extend(p for p in (ROOT / directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    write(OUT / 'SOURCE_SEAL.json', {str(p): sha(p) for p in sorted(set(files))})


def choose():
    verify()
    choices = {}
    for job in SELECTION_JOBS:
        path = folder(job) / 'fit.json'
        record = json.loads(path.read_text())
        if record['source_seal'] != sha(OUT / 'SOURCE_SEAL.json'):
            raise ValueError('Selection fit source seal differs.')
        summary = record['summary']
        key = f"{job['stage']}|{job['family']}|{job['seed']}"
        choices[key] = dict(updates=int(summary['retained_step']), development_ce=summary['monitored_value'],
            fit_sha256=sha(path), cd_dw_window=10 if job['family'] == 'time_B/CD_DW' else None)
    destination = OUT / 'SELECTION.json'
    record = dict(choices=choices, source_seal=sha(OUT / 'SOURCE_SEAL.json'))
    if destination.exists() and json.loads(destination.read_text()) != record:
        raise ValueError('Cannot change already-frozen selection decisions.')
    write(destination, record)


def run_child(command, index, job):
    write(OUT / 'STATUS.json', dict(stage=command, job=job, pid=os.getpid(), utc_epoch=time.time(),
        selection_complete=sum((folder(j)/'fit.json').exists() for j in SELECTION_JOBS),
        refit_complete=sum((folder(j)/'fit.json').exists() for j in REFIT_JOBS),
        evaluation_complete=sum((folder(j)/'evaluation.json').exists() for j in REFIT_JOBS)))
    logs = OUT / 'logs'
    logs.mkdir(exist_ok=True)
    with (logs / f'{command}_{index:03}.log').open('a', encoding='utf-8') as stream:
        subprocess.run([sys.executable, str(Path(__file__)), command, str(index)], cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT, check=True)


def run():
    seal()
    for index, job in enumerate(SELECTION_JOBS):
        run_child('select', index, job)
    choose()
    for index, job in enumerate(REFIT_JOBS):
        run_child('refit', index, job)
    # Only score outer outcomes after every fit and selection decision is frozen.
    for index, job in enumerate(REFIT_JOBS):
        run_child('evaluate', index, job)
    verify()
    write(OUT / 'COMPUTE_COMPLETE.json', dict(selection_fits=27, refits=108, outer_exports=108,
        status='complete_pending_analysis', utc_epoch=time.time()))
    write(OUT / 'STATUS.json', dict(stage='compute_complete_pending_analysis'))
    from analyze_results import build_manifest, analyze
    build_manifest(ROOT/'data/financial_inputs.parquet')
    analyze()
    write(OUT / 'STATUS.json', dict(stage='analysis_complete_pending_native_report_and_drive'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'seal', 'choose', 'select', 'refit', 'evaluate', 'smoke'))
    parser.add_argument('index', type=int, nargs='?', default=0)
    args = parser.parse_args()
    try:
        {'run': run, 'seal': seal, 'choose': choose,
         'select': lambda: fit(SELECTION_JOBS[args.index]),
         'refit': lambda: fit(REFIT_JOBS[args.index]),
         'evaluate': lambda: evaluate(REFIT_JOBS[args.index]),
         'smoke': lambda: fit(SELECTION_JOBS[args.index], smoke=True)}[args.command]()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / ('FAILURE_' + args.command + '.txt')).write_text(traceback.format_exc(), encoding='utf-8')
        raise
