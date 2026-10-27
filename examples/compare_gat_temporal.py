"""Frozen, bounded temporal-attention comparison on read-only season publications.

Commands: prepare, profile, fit, report. Profile and fit use fresh sequential
processes for each mode/seed. Run prepare before any scoring; never reuse a run
directory for a different protocol. Generated artifacts belong in experiments/.
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
import threading
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from xdiyo_analytics.graphs.observed_gap import migrate_gap_config
from xdiyo_analytics.graphs import TemporalGraphConfig, TemporalEventStore, TemporalGraphAdapter, TemporalGraphSerializer
from xdiyo_analytics.graphs.config import CLASS_ORDER, MATCH_KEYS
from xdiyo_analytics.training import EarlyStopping, TrainingControl
from temporal_team_gat import make_context, scores

MODES = ('snapshot', 'time_features', 'historical')


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    text = json.dumps(value, indent=2, default=str, allow_nan=False) + '\n'
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def source_hashes():
    paths = sorted((ROOT/'src/xdiyo_analytics').rglob('*.py'))
    paths += [Path(__file__).resolve(), ROOT/'examples/temporal_team_gat.py']
    return {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in paths}


def frame_hash(frame):
    # Identity/order receipt independent of Parquet metadata and dtype storage.
    rows = frame[list(MATCH_KEYS)].itertuples(index=False, name=None)
    return sha256(json.dumps([list(map(int, r)) for r in rows]).encode()).hexdigest()


def training_control(protocol, seed):
    policy = protocol.get('early_stopping')
    return TrainingControl(max_steps=protocol['max_steps'], seed=seed,
        early_stopping=EarlyStopping(**policy) if policy else None,
        monitor='validation_loss', restore_best=True)


def validate_training_progress(progress, summary, protocol):
    """Independently check the first permitted stopping point and best state."""
    control = training_control(protocol, 0)
    if not progress or [r['step'] for r in progress] != list(range(1,len(progress)+1)):
        raise ValueError('Training progress must contain consecutive completed updates.')
    if len(progress) > control.max_steps:
        raise ValueError('Training exceeded its declared update cap.')
    significant_best, stale, stop = np.inf, 0, None
    for row in progress:
        value = row['validation_loss']
        if not np.isfinite(value):
            raise ValueError('Nonfinite validation loss in a scored run.')
        delta = control.early_stopping.min_delta if control.early_stopping else 0.
        if value < significant_best-delta:
            significant_best, stale = value, 0
        else:
            stale += 1
        if control.early_stopping and stale >= control.early_stopping.patience:
            stop = row['step']
            break
    expected_steps = stop if stop is not None else control.max_steps
    reason = 'early_stopping' if stop is not None else 'max_steps'
    best = min(progress,key=lambda row:row['validation_loss'])
    if len(progress) != expected_steps or summary['termination_reason'] != reason:
        raise ValueError('Training did not follow the declared cap/patience rule.')
    if (summary['retained_step'] != best['step'] or
            summary['monitored_value'] != best['validation_loss']):
        raise ValueError('The retained state is not the lowest-validation-loss checkpoint.')
    if len(summary['attempts']) != 1 or summary['attempts'][0]['steps'] != len(progress):
        raise ValueError('Unexpected restart or inconsistent update count.')


def prepare_from_run(args):
    """Reuse byte-identical inputs; only the budget/stopping policy may change."""
    parent, out = args.from_run.resolve(), args.run.resolve()
    protocol = json.loads((parent/'protocol.json').read_text(encoding='utf-8'))
    current = source_hashes()
    runner = 'examples/compare_gat_temporal.py'
    if {k:v for k,v in protocol['source_hashes'].items() if k != runner} != {
            k:v for k,v in current.items() if k != runner}:
        raise ValueError('Model or shared experiment source changed since the parent run.')
    if args.max_steps <= protocol['max_steps']:
        raise ValueError('An extension must increase the maximum update budget.')
    policy = dict(patience=args.patience,min_delta=args.min_delta) if args.patience is not None else None
    training_control(dict(max_steps=args.max_steps,early_stopping=policy),0)
    root = Path(protocol['data_root']).resolve()
    if out.exists() or out.is_relative_to(root) or out.is_relative_to(parent):
        raise ValueError('Use a new output directory outside source data and the parent run.')
    for name, expected in protocol['data_hashes'].items():
        if digest(parent/name) != expected:
            raise ValueError(f'Prepared parent input changed: {name}')
    parent_runs = {}
    for mode in protocol['modes']:
        for seed in protocol['seeds']:
            name = f'{mode}-{seed}'
            folder = parent/'fit'/name
            receipt = json.loads((folder/'result.json').read_text())
            if receipt['protocol_sha256'] != digest(parent/'protocol.json'):
                raise ValueError('Parent fit protocol mismatch.')
            parent_runs[name] = {f:digest(folder/f) for f in ('result.json','progress.json','model/weights.pt')}
    # Disposable resource profiling is inherited, explicitly attributed to its
    # original protocol; the model and execution settings above are unchanged.
    reference = protocol.get('profile_reference') or dict(run=str(parent),
        protocol_sha256=digest(parent/'protocol.json'),
        receipts={m:digest(parent/'profile'/m/'result.json') for m in protocol['modes']})
    protocol.update(schema=2,created_at=pd.Timestamp.now(tz='UTC').isoformat(),
        source_hashes=current,max_steps=args.max_steps,early_stopping=policy,
        parent_run=dict(path=str(parent),protocol_sha256=digest(parent/'protocol.json'),
                        fits=parent_runs,method='Fresh deterministic fits from original seeds; no optimizer state in parent exports.'),
        profile_reference=reference)
    protocol['limitations'] = [x for x in protocol['limitations']
        if x != 'Equal full-population updates, not equal wall-clock budgets; no early stopping.']
    protocol['limitations'] += [
        'Common maximum updates and validation stopping rule; realized update counts and wall times may differ.',
        'Follow-up on an already inspected test season; exploratory evidence, not a fresh confirmatory holdout.']
    out.mkdir(parents=True)
    for name in protocol['data_hashes']:
        shutil.copyfile(parent/name,out/name)
    write_json(out/'protocol.json',protocol)
    (out/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    load(out)
    print(json.dumps(dict(stage='prepared_extension',run=str(out),parent=str(parent),
                         max_steps=protocol['max_steps'],early_stopping=policy)),flush=True)


def prepare(args):
    if args.from_run is not None:
        return prepare_from_run(args)
    from xdiyo_analytics.data import load_season
    from xdiyo_analytics.histories import build_team_history
    out, root = args.run.resolve(), args.data_root.resolve()
    if out.exists() or out.is_relative_to(root):
        raise ValueError('Use a new output directory outside the source data root.')
    out.mkdir(parents=True)
    histories, memberships, sources = [], [], []
    for year in range(19, 25):
        for league in ('Bundesliga', 'Bundesliga_2'):
            stem = f'{league}_{year:02}_{year+1:02}'
            data = load_season(root, stem, tables=('matches', 'team_seasons'), verify_hashes=True)
            h = build_team_history(data)
            h['source_season'], h['source_league'] = f'{year:02}_{year+1:02}', league
            histories.append(h)
            m = data['team_seasons'].copy()
            m['entry_at'] = pd.Timestamp(f'20{year:02}-07-01', tz='Europe/Berlin').tz_convert('UTC')
            memberships.append(m)
            source = dict(data.provenance)
            source['manifest_sha256'] = digest(source['manifest_path'])
            sources.append(source)
    provenance = dict(sources=sources, season_entry_assumption='Reviewed membership known July 1 local midnight')
    store = TemporalEventStore(pd.concat(histories, ignore_index=True), pd.concat(memberships, ignore_index=True), provenance=provenance)
    cfg = TemporalGraphConfig('Europe/Berlin', deterministic=True)
    policy = dict(patience=args.patience,min_delta=args.min_delta) if args.patience is not None else None
    training_control(dict(max_steps=args.max_steps,early_stopping=policy),0)
    schedule = store.schedule
    target = schedule[schedule.source_league.eq('Bundesliga') & schedule.status.eq('finished') &
                      schedule.goals_for.notna() & schedule.goals_against.notna()].copy()
    splits = {name: target[target.source_season.eq(season)].copy() for name, season in
              (('train', '22_23'), ('validation', '23_24'), ('test', '24_25'))}
    populations = {}
    previous = None
    for name, frame in splits.items():
        if not len(frame) or frame.duplicated(list(MATCH_KEYS)).any():
            raise ValueError('Missing or duplicate target population.')
        issue = store.issue_times(frame, cfg)
        store.assert_targets_unobserved(frame, issue, cfg)
        first_cutoff = min(cfg.history_cutoff(t) for t in issue)
        known_through = store.label_availability(frame, cfg).max()
        if previous is not None and first_cutoff < previous:
            raise ValueError('Label availability crosses the next split boundary.')
        previous = known_through
        populations[name] = dict(n=len(frame), identity_sha256=frame_hash(frame),
            first_issue=issue.min().isoformat(), last_issue=issue.max().isoformat(),
            first_cutoff=first_cutoff.isoformat(), labels_known_through=known_through.isoformat(),
            issuance_groups=issue.nunique())
        frame.to_parquet(out/f'{name}.parquet', index=True)
    for name, frame in (('history', store.history), ('memberships', store.team_seasons), ('schedule', store.schedule)):
        frame.to_parquet(out/f'{name}.parquet', index=False)
    protocol = dict(schema=1, created_at=pd.Timestamp.now(tz='UTC').isoformat(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        source_hashes=source_hashes(), data_root=str(root), provenance=provenance,
        data_hashes={p.name:digest(p) for p in out.glob('*.parquet')}, store_version=store.version,
        history_matches=len(schedule), history_start=str(schedule.kickoff_at.min()),
        history_end=str(schedule.kickoff_at.max()), populations=populations,
        modes=MODES, seeds=[17,42,73], config=asdict(cfg), max_steps=args.max_steps,
        device=args.device, inner_threads=1, primary_metric='validation_log_loss',
        early_stopping=policy, scheduler=None, restore_best=True,
        profile_steps=2, worker_rss_limit_gib=args.rss_limit_gib,
        input_cache_bytes=int(args.input_cache_gib*2**30),input_cache_headroom_bytes=2**30,
        cuda_allocator_config=args.cuda_allocator_config or None,cuda_memory_fraction=args.cuda_memory_fraction,
        bootstrap=dict(replicates=2000, seed=20261025, consecutive_rounds=4),
        limitations=[
            'Fixed archive starts in 2019/20 for all modes; not the older 15-season pilot.',
            'Retrospective schedule and July 1 membership assumptions, not historical vintages.',
            'Previously explored benchmark seasons; this is not a pristine confirmatory holdout.',
            'Common update cap and validation stopping policy; not an equal wall-clock budget.',
            'Same seeds and initialization algorithm; differing layer shapes consume RNG differently.',
            'No hyperparameter search or test-guided tuning; no recurrent/retrospective extension.',
            'The rolling logistic comparator is contextual; no claim of graph value without a no-opponent encoder.'])
    write_json(out/'protocol.json', protocol)
    (out/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(stage='prepared', run=str(out), history_matches=len(schedule), populations=populations)), flush=True)


def load(run):
    protocol = json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    if protocol['source_hashes'] != source_hashes():
        raise ValueError('Source changed after the protocol was frozen.')
    for name, expected in protocol['data_hashes'].items():
        if digest(run/name) != expected:
            raise ValueError(f'Prepared input changed: {name}')
    store = TemporalEventStore(pd.read_parquet(run/'history.parquet'), pd.read_parquet(run/'memberships.parquet'),
                              schedule=pd.read_parquet(run/'schedule.parquet'), provenance=protocol['provenance'])
    if store.version != protocol['store_version']:
        raise ValueError('Event store changed across prepared-input serialization.')
    frames = {name:pd.read_parquet(run/f'{name}.parquet') for name in protocol['populations']}
    for name, frame in frames.items():
        if frame_hash(frame) != protocol['populations'][name]['identity_sha256']:
            raise ValueError('Fixture identities or ordering changed.')
    return protocol, store, frames


class MemoryProbe:
    """Process RSS sampled every 20 ms, plus the Windows lifetime peak counter."""
    def __init__(self, limit_gib):
        import psutil
        self.process = psutil.Process()
        self.limit = limit_gib * 2**30
        self.baseline = self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.wait(.02):
            info = self.process.memory_info()
            self.peak = max(self.peak, info.rss)
            if info.rss > self.limit:
                print(json.dumps(dict(resource_stop='RSS limit', bytes=info.rss)), flush=True)
                os._exit(77)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()

    def result(self):
        info = self.process.memory_info()
        return dict(baseline_rss_bytes=self.baseline, sampled_peak_rss_bytes=max(self.peak, info.rss),
                    lifetime_peak_rss_bytes=getattr(info, 'peak_wset', None), final_rss_bytes=info.rss,
                    private_bytes=getattr(info, 'private', None), sampling_seconds=.02)


def synchronize(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)


def timed(function, device='cpu'):
    synchronize(device)
    start = time.perf_counter()
    value = function()
    synchronize(device)
    return value, time.perf_counter()-start


def tensor_bytes(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(x) for x in value)
    return 0


def group_receipt(groups, index):
    group, elapsed = timed(lambda:groups[index], groups.device)
    data = group[1]
    return dict(index=index, issuance=str(groups.plan.descriptors[index][0]),
        root_edges=len(data['receiver']), prepared_tensor_bytes=tensor_bytes(group),
        layer_queries=[len(x['self_index']) for x in data.get('temporal_layers', ())],
        layer_edges=[len(x['receiver']) for x in data.get('temporal_layers', ())],
        preparation_seconds=elapsed)


def worker(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    frozen = json.loads((args.run/'protocol.json').read_text(encoding='utf-8'))
    if frozen.get('cuda_allocator_config'):
        os.environ['PYTORCH_ALLOC_CONF'] = frozen['cuda_allocator_config']
    import torch
    from xdiyo_analytics.graphs.adapter import _Backend
    from xdiyo_analytics.graphs.streaming import PreparedGroups
    from xdiyo_analytics.training.residency import InputCachePolicy
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    protocol, store, frames = load(args.run)
    device = protocol.get('device', 'cpu')
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValueError('The frozen protocol requires CUDA.')
        torch.cuda.set_per_process_memory_fraction(protocol.get('cuda_memory_fraction',1.),device)
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    cfg = replace(TemporalGraphConfig(**migrate_gap_config(protocol['config'])), temporal_mode=args.mode)
    cache = InputCachePolicy(protocol.get('input_cache_bytes',0),protocol.get('input_cache_headroom_bytes',2**30))
    train = make_context(frames['train'])
    train.validation = make_context(frames['validation'])
    folder = args.run/args.command/(args.mode if args.command=='profile' else f'{args.mode}-{args.seed}')
    folder.mkdir(parents=True, exist_ok=False)
    receipt = dict(mode=args.mode, seed=args.seed, protocol_sha256=digest(args.run/'protocol.json'),
        store_version=store.version, populations=protocol['populations'], config=asdict(cfg),
        environment=dict(executable=sys.executable,python=sys.version,torch=torch.__version__,pandas=pd.__version__,
                         numpy=np.__version__,device=device,cuda_runtime=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
                         cuda_allocator_config=protocol.get('cuda_allocator_config'),
                         cuda_memory_fraction=protocol.get('cuda_memory_fraction',1.),
                         threads=torch.get_num_threads()))
    started = time.perf_counter()
    with MemoryProbe(protocol['worker_rss_limit_gib']) as memory:
        if args.command == 'profile':
            backend = _Backend(cfg, store, args.seed, device, cache)
            _, receipt['initialize_seconds'] = timed(lambda:backend.initialize(train), device)
            receipt['parameters'] = sum(p.numel() for p in backend.model.parameters())
            receipt['samples'] = {f'train_{label}':group_receipt(backend.train_groups, index)
                for label, index in [('early',0), ('middle',len(backend.train_groups)//2), ('late',len(backend.train_groups)-1)]}
            receipt['samples']['validation_late'] = group_receipt(backend.validation_groups, len(backend.validation_groups)-1)
            test_plan = backend._plan(make_context(frames['test'], fitting=False))
            test_groups = PreparedGroups(test_plan, backend.preprocessor, device)
            receipt['samples']['test_late'] = group_receipt(test_groups, len(test_groups)-1)
            receipt['update_with_validation_seconds'] = []
            for step in range(protocol['profile_steps']):
                _, elapsed = timed(backend.step, device)
                receipt['update_with_validation_seconds'].append(elapsed)
                print(json.dumps(dict(stage='profile',mode=args.mode,step=step+1,seconds=elapsed)), flush=True)
        else:
            control = training_control(protocol,args.seed)
            model = TemporalGraphAdapter(cfg, store, control=control, device=device, inner_threads=1)
            model.configure_input_cache(cache.max_bytes,cache.headroom_bytes)
            step_start = time.perf_counter()
            progress = []
            def observer(event):
                nonlocal step_start
                if event.final:
                    return
                synchronize(device)
                now = time.perf_counter()
                row = dict(step=event.step, seconds_since_previous=now-step_start, **event.metrics)
                progress.append(row)
                step_start = now
                write_json(folder/'progress.json', progress)
                print(json.dumps(dict(stage='fit',mode=args.mode,seed=args.seed,**row)), flush=True)
            train.observer = observer
            _, receipt['fit_seconds'] = timed(lambda:model.fit(train), device)
            receipt['training_summary'] = model.training_summary_
            receipt['parameters'] = sum(p.numel() for p in model.backend_.model.parameters())
            validate_training_progress(progress,model.training_summary_,protocol)
            model.training_history_.to_csv(folder/'training.csv', index=False)
            TemporalGraphSerializer().save(model, folder/'model')
            prediction, receipt['predict_seconds'] = timed(lambda:model.predict(make_context(frames['test'], fitting=False))['predict_proba'], device)
            p = prediction.to_numpy(dtype=float)
            y = make_context(frames['test']).y.result.to_numpy()
            result = frames['test'][list(MATCH_KEYS)+['round','kickoff_at']].copy()
            result['issuance_at'] = store.issue_times(frames['test'], cfg)
            result['truth'] = y
            for j, name in enumerate(('p_home','p_draw','p_away')):
                result[name] = p[:,j]
            result.to_parquet(folder/'predictions.parquet', index=False)
            receipt['test'] = metrics(y, p)
            receipt['model_version'] = model.backend_.model_version
            receipt['checkpoint_sha256'] = digest(folder/'model/weights.pt')
        receipt['elapsed_seconds'] = time.perf_counter()-started
        fitted = backend if args.command=='profile' else model.backend_
        pool = fitted.train_groups._resident.pool
        receipt['input_cache'] = dict(policy=asdict(cache),
            hits=pool.hits if pool else 0,misses=pool.misses if pool else 0,
            resident_bytes=pool.resident if pool else 0,transient_bytes=pool.transient if pool else 0)
        receipt['memory'] = memory.result()
        if device.startswith('cuda'):
            receipt['memory'].update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(folder/'result.json', receipt)
    print(json.dumps(dict(stage=args.command+'_complete',mode=args.mode,seed=args.seed,
                         seconds=receipt['elapsed_seconds'],memory=receipt['memory'])), flush=True)


def metrics(y, probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(y),3) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=2e-6):
        raise ValueError('Invalid ordered H/D/A probabilities.')
    result = scores(y, p)
    class_calibration = {}
    for j, label in enumerate(('home','draw','away')):
        truth = np.asarray(y) == CLASS_ORDER[j]
        bins = np.minimum((p[:,j]*10).astype(int),9)
        rows, ece = [], 0.
        for b in sorted(set(bins)):
            keep = bins == b
            predicted, observed = p[keep,j].mean(), truth[keep].mean()
            ece += keep.mean()*abs(predicted-observed)
            rows.append(dict(bin=int(b),count=int(keep.sum()),predicted=float(predicted),observed=float(observed)))
        class_calibration[label] = dict(ece_10=float(ece),mean_probability=float(p[:,j].mean()),
                                       frequency=float(truth.mean()),bins=rows)
    result['class_calibration'] = class_calibration
    return result


def paired_blocks(loss_difference, rounds, *, replicates, seed, consecutive_rounds):
    """Paired circular moving-round bootstrap, conditional on the fitted seeds."""
    values = np.asarray(loss_difference)
    groups = list(dict.fromkeys(rounds))
    indices = [np.flatnonzero(np.asarray(rounds)==g) for g in groups]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        starts = rng.integers(len(groups), size=int(np.ceil(len(groups)/consecutive_rounds)))
        sampled = ((starts[:,None]+np.arange(consecutive_rounds)) % len(groups)).ravel()[:len(groups)]
        rows = np.concatenate([indices[i] for i in sampled])
        draws.append(values[rows].mean())
    return dict(mean=float(values.mean()),lower_95=float(np.quantile(draws,.025)),
                upper_95=float(np.quantile(draws,.975)),round_blocks=len(groups),
                interpretation='Exploratory paired time-block interval, conditional on these fitted seeds and season')


def rolling_features(store, cfg, frame):
    issue = store.issue_times(frame, cfg)
    output = np.empty((len(frame),6))
    for time, positions in issue.groupby(issue, sort=True).indices.items():
        eligible = store.eligible(cfg.history_cutoff(time))
        tail = eligible.groupby('team_id', sort=False).tail(10)
        means = tail.groupby('team_id')[['goals_for','goals_against']].mean()
        meta = frame.iloc[positions]
        a = means.reindex(meta.home_id.to_numpy()).to_numpy(dtype=float)
        b = means.reindex(meta.away_id.to_numpy()).to_numpy(dtype=float)
        output[positions] = np.column_stack((a,b,a-b))
    return output


def report(args):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    protocol, store, frames = load(args.run)
    cfg = TemporalGraphConfig(**migrate_gap_config(protocol['config']))
    labels = {name:make_context(frame).y.result.to_numpy() for name,frame in frames.items()}
    receipts, predictions, losses = {}, {}, {}
    for mode in MODES:
        receipts[mode], predictions[mode], losses[mode] = [], [], []
        for seed in protocol['seeds']:
            folder = args.run/'fit'/f'{mode}-{seed}'
            receipt = json.loads((folder/'result.json').read_text())
            if receipt['protocol_sha256'] != digest(args.run/'protocol.json'):
                raise ValueError('Mixed experiment protocols.')
            pred = pd.read_parquet(folder/'predictions.parquet')
            if frame_hash(pred) != protocol['populations']['test']['identity_sha256'] or not np.array_equal(pred.truth,labels['test']):
                raise ValueError('Prediction population mismatch.')
            p = pred[['p_home','p_draw','p_away']].to_numpy()
            y = np.array([CLASS_ORDER.index(x) for x in labels['test']])
            receipts[mode].append(receipt)
            predictions[mode].append(p)
            losses[mode].append(-np.log(np.maximum(p[np.arange(len(p)),y],1e-15)))
    baseline = make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(),
                             LogisticRegression(C=1.,max_iter=500,random_state=17))
    tick = time.perf_counter()
    baseline.fit(rolling_features(store,cfg,frames['train']), labels['train'])
    p = baseline.predict_proba(rolling_features(store,cfg,frames['test']))
    p = p[:,[list(baseline.classes_).index(c) for c in CLASS_ORDER]]
    baseline_result = dict(seconds=time.perf_counter()-tick,**metrics(labels['test'],p))
    pd.DataFrame(p,columns=['p_home','p_draw','p_away']).to_parquet(args.run/'rolling_logistic_predictions.parquet',index=False)
    summaries = {}
    for mode, runs in receipts.items():
        summaries[mode] = dict(seeds=protocol['seeds'],per_seed_test=[r['test'] for r in runs],
            validation_losses=[r['training_summary']['monitored_value'] for r in runs],
            training_summaries=[r['training_summary'] for r in runs],
            mean_test_log_loss=float(np.mean([r['test']['log_loss'] for r in runs])),
            sd_test_log_loss=float(np.std([r['test']['log_loss'] for r in runs],ddof=1)),
            mean_test_brier=float(np.mean([r['test']['brier'] for r in runs])),
            mean_test_ece=float(np.mean([r['test']['ece_10'] for r in runs])),
            fit_seconds=[r['fit_seconds'] for r in runs],predict_seconds=[r['predict_seconds'] for r in runs],
            memory=[r['memory'] for r in runs])
    contrasts = {}
    for candidate, reference in [('time_features','snapshot'),('historical','snapshot'),('historical','time_features')]:
        difference = np.asarray(losses[candidate])-np.asarray(losses[reference])
        contrast = paired_blocks(difference.mean(0),frames['test']['round'].tolist(),**protocol['bootstrap'])
        contrast['per_seed_difference'] = difference.mean(1).tolist()
        contrasts[f'{candidate}_minus_{reference}'] = contrast
    reference = protocol.get('profile_reference')
    profile_run = Path(reference['run']) if reference else args.run
    if reference and digest(profile_run/'protocol.json') != reference['protocol_sha256']:
        raise ValueError('Referenced profile protocol changed.')
    profiles = {}
    for mode in MODES:
        path = profile_run/'profile'/mode/'result.json'
        if reference and digest(path) != reference['receipts'][mode]:
            raise ValueError('Referenced profile receipt changed.')
        profiles[mode] = json.loads(path.read_text())
    result = dict(protocol_sha256=digest(args.run/'protocol.json'),protocol=protocol,
                  profiles=profiles,models=summaries,paired_log_loss=contrasts,rolling_logistic=baseline_result)
    write_json(args.run/'results.json',result)
    print(json.dumps(dict(models={m:{k:v[k] for k in ('mean_test_log_loss','sd_test_log_loss','mean_test_brier','mean_test_ece')} for m,v in summaries.items()},
                         paired_log_loss=contrasts,rolling_logistic_log_loss=baseline_result['log_loss'])),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','profile','fit','report'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/xDiyo_data')
    parser.add_argument('--from-run',type=Path,help='Prepare an extension with the parent data, models, seeds and execution settings.')
    parser.add_argument('--max-steps',type=int,default=20,help='Maximum complete training-population updates, frozen by prepare.')
    parser.add_argument('--patience',type=int,help='Stop after this many epochs without validation improvement; omitted means no early stopping.')
    parser.add_argument('--min-delta',type=float,default=0.,help='Required validation-loss improvement to reset patience.')
    parser.add_argument('--device',choices=['cpu','cuda:0'],default='cpu',help='Frozen by prepare; later commands read the protocol.')
    parser.add_argument('--rss-limit-gib',type=float,default=2.5,help='Worker host-memory stop, frozen by prepare.')
    parser.add_argument('--input-cache-gib',type=float,default=0.,help='Common immutable GPU-input budget, frozen by prepare.')
    parser.add_argument('--cuda-allocator-config',default='',help='Worker-only PYTORCH_ALLOC_CONF, frozen by prepare.')
    parser.add_argument('--cuda-memory-fraction',type=float,default=1.,help='Common CUDA allocator limit, frozen by prepare.')
    parser.add_argument('--mode',choices=MODES)
    parser.add_argument('--seed',type=int,default=17)
    args = parser.parse_args()
    args.run = args.run.resolve()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'report':
        report(args)
    elif args.mode:
        worker(args)
    else:
        protocol = json.loads((args.run/'protocol.json').read_text())
        # Keep all workers sequential so RSS and wall-clock observations are comparable.
        for seed in ([17] if args.command=='profile' else protocol['seeds']):
            for mode in MODES:
                command = [sys.executable,str(Path(__file__).resolve()),args.command,'--run',str(args.run),'--mode',mode,'--seed',str(seed)]
                subprocess.run(command,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))


if __name__ == '__main__':
    main()
