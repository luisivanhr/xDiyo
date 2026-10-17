"""Source-pinned, serial 24-fit acceptance matrix; outer test files are never read.

First use --preflight to freeze inputs/splits/configuration in a NEW external
directory. Inspect manifest.json, then explicitly --run --manifest that file.
An interrupted/failed/started matrix is never resumed implicitly. Its immutable
native checkpoints remain available for an explicitly designed recovery run.
Normal coordinator failures clean up only validated owned worker descendants.
Abrupt OS termination or an already-dead launcher cannot guarantee cleanup;
stale leases and per-fit launch/ancestry receipts require explicit PID-verified
manual recovery, never a process-name-wide termination or automatic resume.
"""
from pathlib import Path
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback

sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE/'src'))
import gat_dependency_execution as bounded

ARMS = ('plain', 'bayes11')
FOLDS = {0: (9516, 'ab316813426fc4ea45d48624888af16d8decd7b2743d7e55383fae00cfd45642'),
         2: (18888, '169765fe04ba5cab1d90fb0a46571a43df44e5f9cc68d208aa69db9caee40156')}
SEEDS = (17, 29, 43)
COMMON = tuple(k for k in bounded.HASHES if not k.startswith('baseline/selection/'))
DESKTOP_GPU_NAMES = frozenset(name.casefold() for name in (
    'dwm.exe', 'explorer.exe', 'msedge.exe', 'msedgewebview2.exe', 'chrome.exe',
    'Discord.exe', 'ChatGPT.exe', 'Code.exe', 'PSExpressCore.exe',
    'GCC.exe', 'GBT_DL_LIB.exe', 'ShellExperienceHost.exe', 'StartMenuExperienceHost.exe',
    'SearchHost.exe', 'ApplicationFrameHost.exe', 'TextInputHost.exe',
    'SystemSettings.exe', 'LockApp.exe', 'Widgets.exe', 'WidgetService.exe'))
DESKTOP_GPU_NAMES = DESKTOP_GPU_NAMES | frozenset(name.casefold() for name in (
    'CrossDeviceResume.exe', 'WhatsApp.Root.exe', 'PhoneExperienceHost.exe',
    'system_tray.exe', 'lghub_system_tray.exe', 'ProtonVPN.Client.exe', 'AMDRSSrcExt.exe', 'RadeonSoftware.exe',
    'Dropbox.exe', 'texmaker.exe', 'LINE.exe', 'LineCall.exe', 'ShellHost.exe',
    'codex-computer-use-swift.exe', 'Notepad.exe', 'mscopilot.exe', 'SnippingTool.exe'))
GLOBAL_GPU_LOCK = Path(tempfile.gettempdir())/'xdiyo-gat-dependency-matrix-gpu.lock'


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+'\n'


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.pending')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())


def source_identity():
    files = sorted((SOURCE/'src').rglob('*.py')) + [Path(__file__).resolve(),
        SOURCE/'tools/benchmarks/gat_dependency_execution.py', SOURCE/'pyproject.toml',
        SOURCE/'tools/benchmarks/summarize_gat_dependency_matrix.py',
        SOURCE/'docs/analytics/gat_dependency_packed_execution.md']
    hashes = {p.relative_to(SOURCE).as_posix(): bounded.sha(p) for p in files}
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    return dict(root=str(SOURCE), head=head, files=hashes,
                sha256=hashlib.sha256(encoded(hashes).encode()).hexdigest())


def runtime_identity(device):
    import numpy as np
    import pandas as pd
    import torch
    import psutil
    gpu = None
    if device.startswith('cuda'):
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version',
            '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    return dict(executable=sys.executable, python=sys.version, platform=platform.platform(),
        actual_executable=psutil.Process().exe(),
        numpy=np.__version__, pandas=pd.__version__, torch=str(torch.__version__), psutil=psutil.__version__,
        torch_build=hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=gpu,
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        precision=torch.get_float32_matmul_precision(), matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        caller_deterministic=torch.are_deterministic_algorithms_enabled(),
        intended_inner_threads=1, intended_interop_threads=1)


def runtime_probe(device):
    """Import numerical runtimes only in a disposable subprocess, never parent."""
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--runtime-probe',
        '--device', device], cwd=SOURCE, stdin=subprocess.DEVNULL, capture_output=True,
        text=True, check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    return json.loads(result.stdout)


def checked_relative(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Manifest path escapes its root: '+relative)
    return path


def load_training_population(path):
    """Read the canonical *persisted* unnamed cohort index, never row numbers.

    Canonical fold Parquet stores its original cohort positions in physical
    ``__index_level_0__``; pandas restores that field as an unnamed int64 Index.
    Assigning its descriptive name is lossless. A missing physical index or a
    virtual RangeIndex is rejected, rather than synthesizing cohort positions.
    """
    import pandas as pd
    import pyarrow.parquet as pq
    schema = pq.read_schema(path)
    metadata = json.loads((schema.metadata or {}).get(b'pandas', b'{}'))
    if metadata.get('index_columns') != ['__index_level_0__'] or '__index_level_0__' not in schema.names:
        raise ValueError('Canonical training Parquet requires its physical unnamed cohort index.')
    index_field = next((item for item in metadata.get('columns', [])
                        if item.get('field_name') == '__index_level_0__'), None)
    if index_field is None or index_field.get('name') is not None:
        raise ValueError('Canonical cohort index schema differs from the frozen unnamed index.')
    frame = pd.read_parquet(path)
    if (frame.index.name is not None or isinstance(frame.index, (pd.MultiIndex, pd.RangeIndex))
            or not pd.api.types.is_integer_dtype(frame.index.dtype)
            or pd.api.types.is_bool_dtype(frame.index.dtype) or not frame.index.is_unique
            or frame.index.hasnans or 'cohort_position' in frame.columns):
        raise ValueError('Canonical cohort positions must be a unique persisted unnamed integer index.')
    frame.index = frame.index.rename('cohort_position')
    return frame


def verify_manifest(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    expected = path.with_suffix('.sha256').read_text(encoding='utf-8').strip()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Frozen matrix manifest changed.')
    manifest = json.loads(raw)
    if manifest['source'] != source_identity():
        raise ValueError('Frozen source/commit changed; refuse fitting or implicit migration.')
    root = Path(manifest['input_root'])
    for relative, digest in manifest['input_hashes'].items():
        if bounded.sha(checked_relative(root, relative)) != digest:
            raise ValueError('Canonical input changed: '+relative)
    for relative, digest in manifest['frozen_files'].items():
        if bounded.sha(checked_relative(path.parent, relative)) != digest:
            raise ValueError('Frozen split/configuration changed: '+relative)
    protocol = manifest.get('protocol')
    if protocol == 'full_matrix':
        if manifest['full_fit_updates'] != 100 or len(manifest['fits']) != 24:
            raise ValueError('Full matrix must retain 24 fits and 100 updates.')
    elif protocol != 'worker_smoke' or manifest['full_fit_updates'] != 1 or len(manifest['fits']) != 4:
        raise ValueError('Unknown matrix protocol or modified smoke/full-fit contract.')
    if any(fit['control']['max_steps'] != manifest['full_fit_updates'] for fit in manifest['fits']):
        raise ValueError('Fit update control disagrees with frozen protocol.')
    return manifest, expected


def split_population(frame, config):
    """Latest 20% distinct kickoff batches, then purge at graph-safe issuance.

    Purging at the earliest validation history cutoff is at least as strict as
    its issuance and satisfies the adapter's label-selection boundary contract.
    Equal kickoff batches stay together and original cohort ordering survives.
    """
    import numpy as np
    import pandas as pd
    from xdiyo_analytics.data.availability import respect_result_availability
    times = pd.to_datetime(frame.kickoff_at, utc=True, errors='raise')
    if times.isna().any() or not frame.index.is_unique:
        raise ValueError('Training population needs unique rows and finite kickoff times.')
    batches = times.drop_duplicates().sort_values()
    count = int(math.ceil(len(batches)*.2))
    if count < 1 or count >= len(batches):
        raise ValueError('Inner validation must leave fitting kickoff batches.')
    validation = times.ge(batches.iloc[-count]).to_numpy()
    issue = pd.to_datetime(frame.round_issue_at, utc=True, errors='raise')
    if issue.isna().any() or (issue > times).any():
        raise ValueError('Invalid canonical frozen round issuance.')
    boundary = min(config.history_cutoff(t) for t in issue.loc[validation])
    available = respect_result_availability(frame, config.labels_known_at(times))
    fit = ~validation & available.notna().to_numpy() & available.le(boundary).to_numpy()
    purged = ~validation & ~fit
    if not fit.any():
        raise ValueError('Purging removed every fitting observation.')
    memberships = np.full(len(frame), 'purged', dtype=object)
    memberships[fit] = 'fit'; memberships[validation] = 'validation'
    rows = frame[['competition_id', 'season_id', 'event_id', 'kickoff_at', 'round_issue_at']].copy()
    rows['role'] = memberships
    rows['label_known_at'] = available
    rows.index.name = 'cohort_position'
    return rows, dict(population=len(frame), fit=int(fit.sum()), validation=int(validation.sum()),
        purged=int(purged.sum()), kickoff_batches=len(batches), validation_batches=count,
        validation_first_kickoff=times.loc[validation].min().isoformat(),
        validation_first_issuance=issue.loc[validation].min().isoformat(),
        fitting_label_boundary=boundary.isoformat(),
        purge_policy='label_known_at <= earliest_validation_history_cutoff; unknown labels purged')


def planned_fits(configs):
    fits = []
    for arm in ARMS:
        for fold in FOLDS:
            for i, seed in enumerate(SEEDS):
                # Freeze a reverse-order pair for the middle seed; never choose
                # order after inspecting timing or validation trajectories.
                modes = ('native', 'dependency_packed') if i % 2 == 0 else ('dependency_packed', 'native')
                for mode in modes:
                    fits.append(dict(id=f'{arm}-fold{fold:02}-seed{seed}-{mode}', arm=arm,
                        fold=fold, seed=seed, mode=mode, config=configs[arm][mode],
                        control=dict(max_steps=100, early_stopping=None, scheduler=None,
                            restarts=0, seed=seed, monitor='validation_loss', direction='minimize', restore_best=True)))
    assert len(fits) == 24
    return fits


def preflight(args):
    import pandas as pd
    from xdiyo_analytics.graphs import GraphExecution
    input_root, output = args.input_root.resolve(), args.output.resolve()
    if output.is_relative_to(input_root) or output.is_relative_to(SOURCE):
        raise ValueError('Output must be external to source and canonical inputs.')
    if output.exists():
        raise FileExistsError('Preflight requires a NEW output directory; preserve existing runs.')
    bounded.OLD = input_root
    input_hashes = {}
    for relative in COMMON:
        digest = bounded.sha(checked_relative(input_root, relative))
        if digest != bounded.HASHES[relative]:
            raise ValueError('Canonical input hash mismatch: '+relative)
        input_hashes[relative] = digest
    input_hashes['FOLD_INPUTS.json'] = bounded.sha(input_root/'FOLD_INPUTS.json')
    fold_inputs = json.loads((input_root/'FOLD_INPUTS.json').read_text(encoding='utf-8'))
    fixtures = pd.read_parquet(input_root/'baseline/fixtures.parquet')
    configs = {}
    arm_policies = []
    for arm in ARMS:
        cfg, _ = bounded.recipe(arm)
        arm_policies.append((cfg.timezone, cfg.forecast_policy, cfg.round_keys))
        configs[arm] = {mode: asdict(replace(cfg, execution=GraphExecution(mode=mode,
            host_memory_bytes=args.host_memory_bytes, device_memory_bytes=args.device_memory_bytes,
            max_graphs=args.max_graphs, headroom_bytes=args.headroom_bytes)))
            for mode in ('native', 'dependency_packed')}
    if len(set(arm_policies)) != 1:
        raise ValueError('Paired arms must share timezone and frozen round issuance policy.')
    splits = {}
    frozen_rows = {}
    keys = ['competition_id', 'season_id', 'event_id']
    for fold, (count, expected) in FOLDS.items():
        part = next(f for f in fold_inputs['folds'] if f['fold'] == fold)['parts']['train']
        relative = f'baseline/folds/fold-{fold:02}/train.parquet'
        if part['path'] != relative or part['rows'] != count or part['sha256'] != expected:
            raise ValueError('Canonical fold training manifest differs.')
        digest = bounded.sha(input_root/relative)
        if digest != expected:
            raise ValueError('Canonical fold training file differs.')
        input_hashes[relative] = digest
        frame = load_training_population(input_root/relative)
        if len(frame) != count or frame.duplicated(keys).any():
            raise ValueError('Canonical training row count/identity differs.')
        authoritative = fixtures.set_index(keys).reindex(pd.MultiIndex.from_frame(frame[keys]))
        for field in ('home_id', 'away_id', 'round_issue_at'):
            if not authoritative[field].reset_index(drop=True).equals(frame[field].reset_index(drop=True)):
                raise ValueError('Canonical training rows disagree with authoritative fixtures: '+field)
        rows, info = split_population(frame, bounded.recipe('plain')[0])
        frozen_rows[fold] = rows
        splits[str(fold)] = dict(path=f'splits/fold-{fold:02}.parquet', **info)
    output.mkdir(parents=True, exist_ok=False)
    (output/'splits').mkdir()
    for fold, rows in frozen_rows.items():
        rows.to_parquet(output/splits[str(fold)]['path'], index=True, compression='zstd')
    fits = planned_fits(configs)
    if args.smoke:
        fits = [f for f in fits if f['fold'] == 0 and f['seed'] == 17]
        for fit in fits:
            fit['control']['max_steps'] = 1
    atomic_json(output/'configurations.json', dict(configs=configs, fits=fits))
    frozen_files = {relative: bounded.sha(output/relative) for relative in
        ['configurations.json', *(s['path'] for s in splits.values())]}
    manifest = dict(schema=1, protocol='worker_smoke' if args.smoke else 'full_matrix',
        created_utc=bounded.utc(), source=source_identity(), runtime=runtime_probe(args.device),
        input_root=str(input_root), input_hashes=input_hashes, frozen_files=frozen_files,
        output=str(output), device=args.device, executable=sys.executable, python=sys.version,
        checkpoint_every=args.checkpoint_every, splits=splits, fits=fits,
        windows_desktop_gpu_allowlist=sorted(DESKTOP_GPU_NAMES),
        gpu_gate_policy='Only this worker may run non-desktop GPU computation. On Windows, explicitly verified desktop GPU names are excluded and recorded; unknown/unresolved or Python processes block. No process is stopped.',
        full_fit_updates=1 if args.smoke else 100, outer_test_policy='Outer test files are neither read nor evaluated.',
        split_policy='Latest ceil(20 percent) distinct kickoff batches; availability purge before earliest validation graph cutoff.',
        predictive_gates=dict(max_pair_ce_increase=.005, max_mean_ce_increase=.001,
            max_accuracy_decrease=.005, max_brier_increase=.003, max_auxiliary_relative_increase=.01),
        performance_gate=dict(median_warm_update_validation_speedup=1.5,
            no_increase=['peak_resident_ram','peak_private_commit','cuda_allocated','cuda_reserved']),
        recovery_policy='Stop on any failure; no automatic resume or regrouping. Preserve native checkpoints.')
    raw = encoded(manifest).encode('utf-8')
    (output/'manifest.json').write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    (output/'manifest.sha256').write_text(digest+'\n', encoding='utf-8')
    atomic_json(output/'status.json', dict(state='preflight', utc=bounded.utc(), manifest_sha256=digest,
        completed=[], active=None, total=len(fits)))
    print(encoded(dict(manifest=str(output/'manifest.json'), sha256=digest, splits=splits, fits=len(fits))), flush=True)


def gpu_processes(ignore_pids=()):
    import psutil
    command = ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits']
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    report = dict(utc=bounded.utc(), blocking=[], excluded_desktop=[], ignored_own=[], raw=lines)
    for line in lines:
        pid_text, _, reported_name = line.partition(',')
        try:
            pid = int(pid_text.strip())
            if pid in ignore_pids:
                report['ignored_own'].append(dict(pid=pid, reported_name=reported_name.strip()))
                continue
            process = psutil.Process(pid)
            row = dict(pid=pid, name=process.name(), reported_name=reported_name.strip())
            try:
                row['executable'] = process.exe()
            except (psutil.Error, OSError) as error:
                # Explicitly verified desktop names are sufficient on WDDM,
                # where protected shell executable paths can be inaccessible.
                row['executable'] = None
                row['path_resolution_limitation'] = repr(error)
            if os.name == 'nt' and row['name'].casefold() in DESKTOP_GPU_NAMES:
                report['excluded_desktop'].append(row)
            else:
                report['blocking'].append(row)
        except (ValueError, psutil.Error, OSError) as error:
            report['blocking'].append(dict(raw=line, resolution_error=repr(error)))
    return report


def _same_executable(left, right):
    return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(str(Path(right).resolve()))


def _same_command(actual, expected):
    return bool(actual and expected and _same_executable(actual[0], expected[0])
                and actual[1:] == expected[1:])


def validate_launcher_ancestry(coordinator_pid, *, executable, actual_executable,
                               worker_command, coordinator_command, coordinator_created):
    """Allow a direct parent or ONE exact Windows venv redirector, never peers.

    Windows venv Python is a redirector: its child executes the base interpreter
    while preserving sys.executable and argv. Every intermediate must be that
    exact venv executable with the identical worker command. The live final
    coordinator must match its frozen process creation time, executable and argv.
    """
    import psutil
    current = psutil.Process()
    if not _same_executable(sys.executable, executable) or not _same_executable(current.exe(), actual_executable):
        raise RuntimeError('Worker interpreter differs from frozen runtime.')
    if not _same_command(current.cmdline(), worker_command):
        raise RuntimeError('Worker command differs from its owned matrix invocation.')
    chain = []
    parent = current.parent()
    for intermediates in range(2):
        if parent is None:
            break
        row = dict(pid=parent.pid, created=parent.create_time(), executable=parent.exe(), command=parent.cmdline())
        chain.append(row)
        if parent.pid == coordinator_pid:
            if (row['created'] != coordinator_created
                    or not _same_executable(row['executable'], actual_executable)
                    or not _same_command(row['command'], coordinator_command)):
                raise RuntimeError('Live coordinator identity differs from the matrix lease.')
            return chain
        if (os.name != 'nt' or intermediates != 0
                or not _same_executable(row['executable'], executable)
                or not _same_command(row['command'], worker_command)):
            break
        parent = parent.parent()
    raise RuntimeError('Worker has no validated direct/venv-launcher coordinator ancestry.')


def coordinator_lease(path, manifest, digest):
    import psutil
    process = psutil.Process()
    return dict(pid=os.getpid(), created=process.create_time(), actual_executable=process.exe(),
        executable=sys.executable, command=process.cmdline(), cwd=str(Path.cwd().resolve()), manifest=str(Path(path).resolve()),
        manifest_sha256=digest, source_sha256=manifest['source']['sha256'],
        output=manifest['output'], started_utc=bounded.utc())


def validate_worker_lease(lease, path, manifest, digest, fit_id):
    if (lease.get('manifest_sha256') != digest or lease.get('source_sha256') != manifest['source']['sha256']
            or lease.get('manifest') != str(Path(path).resolve()) or lease.get('output') != manifest['output']
            or not _same_executable(lease.get('executable', ''), manifest['executable'])
            or not _same_executable(lease.get('actual_executable', ''), manifest['runtime']['actual_executable'])):
        raise RuntimeError('Worker matrix lease/source/manifest/interpreter identity differs.')
    command = lease.get('command', [])
    script_position = 1
    while script_position < len(command) and command[script_position] in ('-B', '-u'):
        script_position += 1
    cwd = Path(lease.get('cwd', ''))
    def resolve_argument(value):
        return (cwd/Path(value)).resolve()
    if (len(command) < script_position+4 or not _same_executable(resolve_argument(command[script_position]), Path(__file__).resolve())
            or '--run' not in command or '--manifest' not in command
            or resolve_argument(command[command.index('--manifest')+1]) != Path(path).resolve()):
        raise RuntimeError('Lease coordinator command is not this source-pinned matrix run.')
    worker_command = [manifest['executable'], str(Path(__file__).resolve()), '--worker',
                      '--manifest', str(Path(path).resolve()), '--fit-id', fit_id]
    return validate_launcher_ancestry(lease['pid'], executable=manifest['executable'],
        actual_executable=manifest['runtime']['actual_executable'], worker_command=worker_command,
        coordinator_command=command, coordinator_created=lease['created'])


def validate_spawn_registration(spawn, ancestry, lease, manifest, digest, fit_id):
    import psutil
    expected = [manifest['executable'], str(Path(__file__).resolve()), '--worker',
                '--manifest', lease['manifest'], '--fit-id', fit_id]
    if (spawn.get('coordinator_pid') != lease['pid'] or spawn.get('manifest_sha256') != digest
            or spawn.get('source_sha256') != manifest['source']['sha256'] or spawn.get('fit_id') != fit_id
            or not _same_command(spawn.get('command', []), expected)):
        raise RuntimeError('Worker spawn registration differs from the source-pinned matrix invocation.')
    own = psutil.Process()
    candidates = [dict(pid=own.pid, created=own.create_time()), *ancestry[:-1]]
    if not any(row['pid'] == spawn.get('pid') and row['created'] == spawn.get('created') for row in candidates):
        raise RuntimeError('Worker/venv launcher is not the coordinator registered spawn.')


def terminate_owned_worker(child, *, command, executable, actual_executable, created):
    """Stop only our exact worker/venv redirector, including its owned child.

    A PID/executable/argv/creation-time mismatch refuses termination. Research
    processes outside the Popen subtree are never inspected or signaled.
    """
    import psutil
    try:
        root = psutil.Process(child.pid)
    except psutil.NoSuchProcess:
        return dict(terminated=[], already_exited=True)
    allowed = (executable, actual_executable)
    def owned(process):
        return (any(_same_executable(process.exe(), path) for path in allowed)
                and _same_command(process.cmdline(), command))
    if root.create_time() != created or not owned(root):
        raise RuntimeError('Refusing cleanup of an unverified/reused worker launcher PID.')
    processes = []
    for process in root.children(recursive=True):
        try:
            if process.create_time() >= created and owned(process):
                processes.append(process)
        except psutil.NoSuchProcess:
            continue
    # Terminate the actual training interpreter before its waiting redirector.
    processes = list(reversed(processes))+[root]
    signaled = []
    for process in processes:
        try:
            process.terminate(); signaled.append(process.pid)
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=10)
    for process in alive:
        if owned(process):
            process.kill()
    _, alive = psutil.wait_procs(alive, timeout=10)
    if alive:
        raise RuntimeError('Owned worker processes did not exit after targeted cleanup.')
    child.wait(timeout=10)
    return dict(terminated=signaled, already_exited=False)


def initialize_worker_device(torch, device):
    """Establish the same requested CUDA baseline before allocator statistics.

    is_available() and the disposable runtime probe do not initialize a worker's
    CUDA allocator. Peak reset/synchronization must follow explicit selection
    and initialization in this process, identically for every execution mode.
    """
    if not device.startswith('cuda'):
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)


def run_matrix(path):
    manifest, digest = verify_manifest(path)
    output = Path(manifest['output'])
    status = json.loads((output/'status.json').read_text(encoding='utf-8'))
    if status['state'] != 'preflight' or status['completed']:
        raise RuntimeError('Matrix already started/completed/failed; implicit resume is forbidden.')
    if sys.executable != manifest['executable'] or sys.version != manifest['python']:
        raise RuntimeError('Use the exact preflight Python interpreter/runtime.')
    if runtime_probe(manifest['device']) != manifest['runtime']:
        raise RuntimeError('Frozen numerical runtime/GPU/driver changed.')
    lock = output/'matrix.lock'
    lease = coordinator_lease(path, manifest, digest)
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(lease, stream)
        stream.flush(); os.fsync(stream.fileno())
    child = None
    global_lease = False
    child_created = None
    command = None
    status.update(state='running', started_utc=bounded.utc(), coordinator_pid=os.getpid())
    try:
        if manifest['device'].startswith('cuda'):
            with GLOBAL_GPU_LOCK.open('x', encoding='utf-8') as stream:
                stream.write(encoded(lease))
                stream.flush(); os.fsync(stream.fileno())
            global_lease = True
        (output/'fits').mkdir(exist_ok=False)
        for fit in manifest['fits']:
            verify_manifest(path)
            if runtime_probe(manifest['device']) != manifest['runtime']:
                raise RuntimeError('Frozen numerical runtime/GPU/driver changed before next fit.')
            active = gpu_processes() if manifest['device'].startswith('cuda') else None
            if active is not None:
                append_json(output/'gpu_gates.jsonl', dict(fit_id=fit['id'], **active))
                if active['blocking']:
                    raise RuntimeError('Exclusive GPU gate failed; active/unknown GPU processes: '+repr(active['blocking']))
            directory = output/'fits'/fit['id']
            directory.mkdir(exist_ok=False)
            status.update(active=fit['id'], utc=bounded.utc())
            atomic_json(output/'status.json', status)
            command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                       '--manifest', str(Path(path).resolve()), '--fit-id', fit['id']]
            with (directory/'stdout.log').open('x', encoding='utf-8') as stdout, (directory/'stderr.log').open('x', encoding='utf-8') as stderr:
                child = subprocess.Popen(command, cwd=SOURCE, stdout=stdout, stderr=stderr,
                    stdin=subprocess.DEVNULL,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8'))
                import psutil
                child_created = psutil.Process(child.pid).create_time()
                atomic_json(directory/'launch.json', dict(pid=child.pid, created=child_created,
                    coordinator_pid=os.getpid(), command=command, manifest_sha256=digest,
                    source_sha256=manifest['source']['sha256'], fit_id=fit['id']))
                status['worker_pid'] = child.pid
                atomic_json(output/'status.json', status)
                while child.poll() is None:
                    status['heartbeat_utc'] = bounded.utc()
                    progress = directory/'progress.json'
                    if progress.exists():
                        status['worker_progress'] = json.loads(progress.read_text(encoding='utf-8'))
                    atomic_json(output/'status.json', status)
                    time.sleep(5)
                if child.returncode:
                    raise RuntimeError(f"Fit {fit['id']} failed with exit {child.returncode}; see its stderr.log. No further fits launched.")
            receipt = directory/'receipt.json'
            if not receipt.exists():
                raise RuntimeError('Worker exited without a completed receipt.')
            status['completed'].append(dict(id=fit['id'], receipt_sha256=bounded.sha(receipt)))
            status.update(active=None, worker_pid=None, utc=bounded.utc())
            atomic_json(output/'status.json', status)
            child = None
        status.update(state='complete', finished_utc=bounded.utc(),
            acceptance='Jobs completed; predictive/performance acceptance requires the separate frozen-gate analysis.')
        atomic_json(output/'status.json', status)
    except BaseException as error:
        if child is not None and child.poll() is None:
            try:
                status['worker_cleanup'] = terminate_owned_worker(child, command=command,
                    executable=manifest['executable'], actual_executable=manifest['runtime']['actual_executable'],
                    created=child_created)
            except Exception as cleanup_error:
                status['worker_cleanup'] = dict(error=repr(cleanup_error), unverified_process_not_signaled=True)
        status.update(state='failed', utc=bounded.utc(), error=repr(error))
        atomic_json(output/'status.json', status)
        raise
    finally:
        if global_lease: GLOBAL_GPU_LOCK.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
    if manifest['protocol'] == 'full_matrix':
        report_output = output.with_name(output.name+'-analysis')
        command = [sys.executable, str(SOURCE/'tools/benchmarks/summarize_gat_dependency_matrix.py'),
                   '--manifest', str(Path(path).resolve()), '--output', str(report_output)]
        try:
            with (output/'analysis_stdout.log').open('x', encoding='utf-8') as stdout, (output/'analysis_stderr.log').open('x', encoding='utf-8') as stderr:
                result = subprocess.run(command, cwd=SOURCE, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
            status['analysis'] = dict(output=str(report_output), returncode=result.returncode,
                state='accepted' if result.returncode == 0 else 'rejected' if result.returncode == 1 else 'failed')
        except (OSError, subprocess.SubprocessError) as error:
            status['analysis'] = dict(state='failed', error=repr(error), output=str(report_output))
        atomic_json(output/'status.json', status)


class ProgressObserver:
    def __init__(self, directory, fit_id, digest):
        self.directory, self.fit_id, self.digest = directory, fit_id, digest

    def cache_key(self):
        return dict(fit_id=self.fit_id, manifest_sha256=self.digest)

    def __call__(self, event):
        value = dict(utc=bounded.utc(), fit_id=self.fit_id, attempt=event.attempt,
            step=event.step, metrics=event.metrics, monitor=event.monitor, final=event.final)
        append_json(self.directory/'trajectory.jsonl', value)
        atomic_json(self.directory/'progress.json', dict(state='fitting', **value))
        print(json.dumps(value, allow_nan=False), flush=True)


def construct_fit(manifest, fit, output):
    import runpy
    import numpy as np
    import pandas as pd
    from xdiyo_analytics.graphs import TemporalGraphAdapter, TemporalEventStore, GraphExecution, StatisticsForecast, TemporalGraphConfig
    from xdiyo_analytics.graphs.config import MATCH_KEYS
    from xdiyo_analytics.training import TrainingControl
    from xdiyo_analytics.training.contracts import FitContext
    root = Path(manifest['input_root'])
    config = dict(fit['config'])
    config['statistics_forecast'] = StatisticsForecast.from_dict(config['statistics_forecast'])
    config['execution'] = GraphExecution(**config['execution'])
    config = TemporalGraphConfig(**config)
    control = TrainingControl(**fit['control'])
    helpers = runpy.run_path(str(root/'runtime_d1219063_long200_aux2/runner/data_adapter.py'))
    history = helpers['attach_edge_stats'](pd.read_parquet(root/'baseline/model_history.parquet'),
        pd.read_parquet(root/'edge_stats_totals.parquet'), list(config.statistics))
    inputs = json.loads((root/'INPUT_MANIFEST.json').read_text(encoding='utf-8'))
    store = TemporalEventStore(history, pd.read_parquet(root/'baseline/memberships_schedule_effective.parquet'),
        schedule=pd.read_parquet(root/'baseline/fixtures.parquet'), provenance=dict(
            input_manifest_sha256=bounded.sha(root/'INPUT_MANIFEST.json'), source_commit=bounded.COMMIT,
            arm_family='totals', availability_policy=inputs['edge_availability_policy']))
    del history
    frame = load_training_population(root/f"baseline/folds/fold-{fit['fold']:02}/train.parquet")
    split = pd.read_parquet(output/manifest['splits'][str(fit['fold'])]['path'])
    if not frame.index.equals(split.index):
        raise ValueError('Frozen row order differs from canonical training population.')
    if config.fixture_features:
        frame = helpers['attach_fixture_head'](frame, pd.read_parquet(root/'fixture_head.parquet'), list(config.fixture_features))
    def context(role):
        meta = frame.loc[split.role.eq(role)].copy()
        return FitContext(X=meta[list(config.fixture_features)].copy(),
            y=pd.DataFrame({'result': np.sign(meta.goals_for-meta.goals_against).astype(int)}, index=meta.index),
            metadata=meta, layout='match', match_columns=MATCH_KEYS, fold_id=fit['fold'])
    train = context('fit'); train.validation = context('validation')
    expected = manifest['splits'][str(fit['fold'])]
    if len(train.X) != expected['fit'] or len(train.validation.X) != expected['validation']:
        raise ValueError('Frozen split row counts changed.')
    return TemporalGraphAdapter(config, store, control=control, device=manifest['device'], inner_threads=1), train


def worker(path, fit_id):
    worker_started_utc = bounded.utc()
    worker_started = time.perf_counter()
    manifest, digest = verify_manifest(path)
    fit = next(f for f in manifest['fits'] if f['id'] == fit_id)
    output = Path(manifest['output']); directory = output/'fits'/fit_id
    lock = json.loads((output/'matrix.lock').read_text(encoding='utf-8'))
    ancestry = validate_worker_lease(lock, path, manifest, digest, fit_id)
    spawn_path = directory/'launch.json'
    deadline = time.monotonic()+5
    while not spawn_path.exists() and time.monotonic() < deadline:
        time.sleep(.02)
    if not spawn_path.exists():
        raise RuntimeError('Coordinator worker spawn registration was not published.')
    validate_spawn_registration(json.loads(spawn_path.read_text(encoding='utf-8')), ancestry,
                                lock, manifest, digest, fit_id)
    atomic_json(directory/'launcher_ancestry.json', dict(worker_pid=os.getpid(), chain=ancestry,
        manifest_sha256=digest, source_sha256=manifest['source']['sha256']))
    if runtime_identity(manifest['device']) != manifest['runtime']:
        raise RuntimeError('Frozen numerical runtime/GPU/driver changed in worker.')
    if (directory/'receipt.json').exists() or (directory/'checkpoints').exists():
        raise RuntimeError('Fit evidence/checkpoints already exist; refuse silent resume.')
    import numpy as np
    import torch
    from xdiyo_analytics.graphs.adapter import _Backend, inference_mode
    from xdiyo_analytics.graphs.execution import deterministic_execution
    from xdiyo_analytics.graphs import TemporalGraphSerializer
    from xdiyo_analytics.training import CheckpointPolicy
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    device = manifest['device']
    def sync():
        if device.startswith('cuda'): torch.cuda.synchronize(device)
    if device.startswith('cuda'):
        global_lock = json.loads(GLOBAL_GPU_LOCK.read_text(encoding='utf-8'))
        if global_lock != lock:
            raise RuntimeError('Worker requires exclusive global matrix GPU lease.')
        gate = gpu_processes(ignore_pids=(os.getpid(),))
        atomic_json(directory/'gpu_gate.json', gate)
        if gate['blocking']:
            raise RuntimeError('GPU ceased to be exclusive before worker initialization.')
        initialize_worker_device(torch, device)
    steps_completed = 0
    original_init, original_loss, original_step = _Backend.initialize, _Backend._loss, _Backend.step
    with bounded.Memory() as memory:
        adapter, train = construct_fit(manifest, fit, output)
        initialization = {}
        def measured_init(self, context):
            sync(); start = time.perf_counter()
            value = original_init(self, context); sync()
            initialization.update(seconds=time.perf_counter()-start, host_peak=memory.snapshot())
            initialization['counts'] = bounded.model_counts(self)
            memory.reset()
            return value
        def measured_loss(self, groups, targets, weights, backward=False):
            sync(); start = time.perf_counter()
            value = original_loss(self, groups, targets, weights, backward); sync()
            record = dict(population='train' if groups is self.train_groups else 'validation', backward=backward,
                seconds=time.perf_counter()-start, trace=deepcopy(getattr(self, '_execution_trace', None)))
            append_json(directory/'loss_phases.jsonl', record)
            return value
        def measured_step(self):
            nonlocal steps_completed
            if device.startswith('cuda'):
                gate = gpu_processes(ignore_pids=(os.getpid(),))
                append_json(directory/'gpu_gates.jsonl', gate)
                if gate['blocking']:
                    raise RuntimeError('Another GPU compute process appeared; stopping without regrouping.')
            sync(); start = time.perf_counter()
            value = original_step(self); sync()
            steps_completed += 1
            append_json(directory/'step_timings.jsonl', dict(step=steps_completed,
                seconds=time.perf_counter()-start, metrics=value))
            return value
        _Backend.initialize, _Backend._loss, _Backend.step = measured_init, measured_loss, measured_step
        train.observer = ProgressObserver(directory, fit_id, digest)
        atomic_json(directory/'progress.json', dict(state='initializing', utc=bounded.utc(), fit_id=fit_id))
        fit_started = time.perf_counter()
        wrapped = CheckpointPolicy(every=manifest['checkpoint_every'], unsupported='raise').wrap(
            lambda: adapter, directory/'checkpoints', fit_id)()
        wrapped.fit_controlled(train, adapter.control)
        sync(); fit_seconds = time.perf_counter()-fit_started
        training_host_peak = memory.snapshot()
        training_cuda_peak = None if not device.startswith('cuda') else dict(
            allocated=torch.cuda.max_memory_allocated(device), reserved=torch.cuda.max_memory_reserved(device))
        if adapter.training_summary_['termination_reason'] != 'max_steps' or adapter.training_summary_['attempts'][0]['steps'] != manifest['full_fit_updates']:
            raise RuntimeError('Fit did not complete all predeclared updates.')
        adapter.training_history_.to_parquet(directory/'training_history.parquet', index=False, compression='zstd')
        atomic_json(directory/'training_summary.json', adapter.training_summary_)
        memory.reset()
        backend = adapter.backend_
        selected_started = time.perf_counter()
        with deterministic_execution(adapter.config, device), inference_mode(backend.model):
            ce = original_loss(backend, backend.validation_groups, backend.validation_targets, backend.validation_weights)
            auxiliary = backend._statistics_loss
            # This is the frozen INNER validation population. Public predict
            # correctly rejects historical issuance after validation selection.
            from xdiyo_analytics.graphs.packing import stream_batches
            probabilities = np.empty((len(train.validation.X), 3), dtype=np.float64)
            for group, _, _ in stream_batches(backend, backend.validation_groups):
                positions, data, home, away, fixture = group
                probabilities[positions] = backend.model(data, home, away, fixture).softmax(-1).cpu().numpy()
                del group, data, home, away, fixture
        truth = backend.validation_targets.cpu().numpy()
        onehot = np.eye(3)[truth]
        scores = dict(outcome_ce=ce, auxiliary_quantile_loss=auxiliary,
            combined_objective=ce+adapter.config.statistics_forecast.weight*auxiliary,
            accuracy=float((probabilities.argmax(1) == truth).mean()),
            multiclass_brier=float(np.square(probabilities-onehot).sum(1).mean()),
            brier_definition='Mean sum of squared errors across all three outcome classes.',
            population='frozen_inner_validation', rows=len(truth),
            selected_step=adapter.training_summary_['retained_step'])
        if not all(math.isfinite(scores[k]) for k in ('outcome_ce','auxiliary_quantile_loss','combined_objective','accuracy','multiclass_brier')):
            raise RuntimeError('Selected validation metrics are not finite.')
        atomic_json(directory/'selected_metrics.json', scores)
        TemporalGraphSerializer(adapter.event_store).save(adapter, directory/'selected_model')
        sync(); selected_seconds = time.perf_counter()-selected_started
        selected_host_peak = memory.snapshot()
        end_to_end_cuda_peak = None if not device.startswith('cuda') else dict(
            allocated=torch.cuda.max_memory_allocated(device), reserved=torch.cuda.max_memory_reserved(device))
    verify_manifest(path)
    receipt = dict(fit=fit, manifest_sha256=digest, source=manifest['source'], inputs_unchanged=True,
        started_utc=worker_started_utc, finished_utc=bounded.utc(), executable=sys.executable,
        python=sys.version, platform=platform.platform(), torch=str(torch.__version__),
        cuda_version=torch.version.cuda, device=device,
        gpu_name=torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
        initialization=initialization, fit_seconds=fit_seconds, selected_evaluation_export_seconds=selected_seconds,
        end_to_end_seconds=time.perf_counter()-worker_started, training_host_peak=training_host_peak,
        training_cuda_peak=training_cuda_peak, selected_host_peak=selected_host_peak,
        end_to_end_cuda_peak=end_to_end_cuda_peak, runtime=manifest['runtime'],
        process_os_memory=memory.os_info, selected_metrics=scores,
        training_summary=adapter.training_summary_, checkpoint=str(adapter.training_checkpoint_),
        timing_definition='Per-update timings include native validation and boundary synchronization; fit time additionally includes initialization/checkpoint I/O/observers. First update retained; no timing samples discarded.',
        memory_definition='2ms sampled RSS/private/VMS plus process lifetime OS high-water; CUDA allocated/reserved peaks include initialization through training. No universal memory cap asserted.',
        outer_test_untouched=True)
    receipt['file_hashes'] = {relative: bounded.sha(directory/relative) for relative in (
        'step_timings.jsonl', 'loss_phases.jsonl', 'trajectory.jsonl', 'training_history.parquet',
        'training_summary.json', 'selected_metrics.json', 'selected_model/graph.json', 'selected_model/weights.pt')}
    atomic_json(directory/'receipt.json', receipt)
    atomic_json(directory/'progress.json', dict(state='complete', utc=bounded.utc(), fit_id=fit_id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preflight', action='store_true')
    action.add_argument('--run', action='store_true')
    action.add_argument('--worker', action='store_true')
    action.add_argument('--runtime-probe', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--fit-id')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--host-memory-bytes', type=int, default=128*1024**2)
    parser.add_argument('--device-memory-bytes', type=int, default=256*1024**2)
    parser.add_argument('--max-graphs', type=int, default=8)
    parser.add_argument('--headroom-bytes', type=int, default=256*1024**2)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--smoke', action='store_true',
        help='Preflight a SEPARATE four-fit, one-update worker smoke manifest; never modifies the 24x100 manifest.')
    args = parser.parse_args()
    if args.smoke and not args.preflight: parser.error('--smoke is only valid with --preflight and a new output directory.')
    if args.runtime_probe:
        print(encoded(runtime_identity(args.device)), flush=True)
    elif args.preflight:
        if args.input_root is None or args.output is None: parser.error('Preflight requires --input-root and --output.')
        if args.checkpoint_every < 1: parser.error('--checkpoint-every must be positive.')
        preflight(args)
    elif args.run:
        if args.manifest is None: parser.error('--run requires --manifest.')
        run_matrix(args.manifest)
    else:
        if args.manifest is None or args.fit_id is None: parser.error('Worker requires manifest and fit ID.')
        try:
            worker(args.manifest, args.fit_id)
        except BaseException as error:
            if args.manifest.exists():
                output = args.manifest.parent/'fits'/args.fit_id
                if output.is_dir():
                    atomic_json(output/'failure.json', dict(utc=bounded.utc(), error=repr(error), traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
