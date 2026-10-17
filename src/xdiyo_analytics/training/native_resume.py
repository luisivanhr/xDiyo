"""Versioned native neural training checkpoints, distinct from inference exports.

Only data/tensors are loaded (torch weights_only); callers still trust local
checkpoint files. Publication is owned by CheckpointPolicy's directory transaction.
The execution/data fingerprint is checked before native backend reconstruction.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np


def _cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu').clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    return deepcopy(value)


def rng_state(device):
    import torch
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(state[0], state[1].tolist(), *state[2:]),
                torch=torch.get_rng_state().clone(),
                cuda=[item.cpu().clone() for item in torch.cuda.get_rng_state_all()]
                     if str(device).startswith('cuda') else None)


def restore_rng(state):
    import torch
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Training checkpoint CUDA RNG device count is incompatible.')
        torch.cuda.set_rng_state_all(state['cuda'])


def execution_profile(adapter):
    import torch
    from dataclasses import asdict
    from .residency import InputCachePolicy
    device = str(adapter.device)
    profile = dict(input_cache=asdict(getattr(adapter, "input_cache_policy", InputCachePolicy())), python=sys.version, numpy=np.__version__, torch=str(torch.__version__),
                torch_build=hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
                platform=platform.platform(), device=device, threads=torch.get_num_threads(),
                interop_threads=torch.get_num_interop_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                precision=torch.get_float32_matmul_precision(),
                cublas=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                cuda_build=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
                capability=list(torch.cuda.get_device_capability(device)) if device.startswith('cuda') else None)
    from ..graphs.config import TemporalGraphConfig
    config = getattr(adapter, 'config', None)
    if isinstance(config, TemporalGraphConfig):
        # None is the legacy native policy, not an implicit opt-in to new modes.
        profile['graph_execution'] = None if config.execution is None else asdict(config.execution)
    return profile


def _preprocessing(backend):
    from ..experiments.recovery import pack
    obj = backend.preprocessor if hasattr(backend, 'preprocessor') else backend.inputs
    return json.dumps(pack(obj.state()), ensure_ascii=False, allow_nan=False)


def capture_backend(backend):
    state = dict(model=_cpu(backend.model.state_dict()), optimizer=_cpu(backend.optimizer.state_dict()),
                gradients=[None if p.grad is None else _cpu(p.grad) for p in backend.model.parameters()],
                preprocessing=_preprocessing(backend), training=backend.model.training)
    if hasattr(backend, 'statistics_scale'):
        state['statistics_scale'] = deepcopy(backend.statistics_scale.state())
    if getattr(getattr(backend, 'config', None), 'execution', None) is not None:
        reports = getattr(backend, '_execution_reports', {})
        state['graph_execution_reports'] = {name: deepcopy(reports[name])
            for name in ('train', 'validation', 'train_replay') if name in reports}
    return state


def restore_backend(backend, state):
    if _preprocessing(backend) != state['preprocessing']:
        raise ValueError('Training checkpoint preprocessing is incompatible with rebuilt inputs.')
    if hasattr(backend, 'statistics_scale'):
        if backend.statistics_scale.state() != state.get('statistics_scale'):
            raise ValueError('Training checkpoint statistics scaling is incompatible with rebuilt labels.')
    backend.restore(state['model'])
    backend.optimizer.load_state_dict(state['optimizer'])
    parameters = list(backend.model.parameters())
    if len(parameters) != len(state['gradients']):
        raise ValueError('Training checkpoint parameter order is incompatible.')
    for parameter, gradient in zip(parameters, state['gradients']):
        parameter.grad = None if gradient is None else gradient.to(device=parameter.device, dtype=parameter.dtype).clone()
    backend.model.train(state['training'])
    if getattr(getattr(backend, 'config', None), 'execution', None) is not None:
        reports = state.get('graph_execution_reports', {})
        backend._execution_reports = {name: deepcopy(reports[name])
            for name in ('train', 'validation', 'train_replay') if name in reports}


def _checkpoint_primitives(value):
    """Normalize scalar metadata without copying model/optimizer tensor buffers."""
    if isinstance(value, (np.bool_, np.integer)):
        return value.item()
    if isinstance(value, (np.float16, np.float32, np.float64)):
        return float(value)
    if isinstance(value, dict):
        return {_checkpoint_primitives(k): _checkpoint_primitives(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_checkpoint_primitives(v) for v in value)
    if isinstance(value, list):
        return [_checkpoint_primitives(v) for v in value]
    return value


def write_checkpoint(directory, state, identity, profile):
    import torch
    directory = Path(directory)
    # The outer writer supplies a new staging directory. Never overwrite a
    # checkpoint or publish its commit marker before payload flush/verification.
    state = _checkpoint_primitives(state)
    payload = directory/'training.pt'
    with payload.open('xb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = dict(schema=1, kind='native-training', identity=identity, profile=profile,
                    payload='training.pt', sha256=digest, complete=bool(state['attempt'] > state['restarts']))
    with (directory/'training.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    # Verify staging payload before the directory/pointer can become public.
    read_checkpoint(directory, identity, profile)


def read_checkpoint(directory, identity=None, profile=None):
    import torch
    directory = Path(directory)
    try:
        manifest = json.loads((directory/'training.json').read_text(encoding='utf-8'))
        if manifest.get('schema') != 1 or manifest.get('kind') != 'native-training' or manifest.get('payload') != 'training.pt':
            raise ValueError('Invalid native training checkpoint manifest.')
        if identity is not None and manifest['identity'] != identity:
            raise ValueError('Training checkpoint data/config/source fingerprint is incompatible.')
        if profile is not None and manifest['profile'] != profile:
            raise ValueError('Training checkpoint execution profile is incompatible.')
        payload = directory/'training.pt'
        if hashlib.sha256(payload.read_bytes()).hexdigest() != manifest['sha256']:
            raise ValueError('Incomplete or corrupt native training checkpoint payload.')
        state = torch.load(payload, map_location='cpu', weights_only=True)
        required = {'attempt', 'restarts', 'rows', 'attempts', 'winner', 'winning_value', 'winner_info', 'rng', 'active'}
        if not isinstance(state, dict) or not required <= state.keys():
            raise ValueError('Incomplete native training checkpoint state.')
        if bool(manifest['complete']) != (state['attempt'] > state['restarts']):
            raise ValueError('Training checkpoint completion marker is inconsistent.')
        return state, manifest
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError('Incomplete or unreadable native training checkpoint.') from exc


@contextmanager
def resume_session(adapter, checkpoint, save_checkpoint):
    if hasattr(adapter, '_native_resume_session'):
        raise RuntimeError('A resumable fit is already active on this adapter.')
    adapter._native_resume_session = (checkpoint, save_checkpoint)
    try:
        yield
    finally:
        del adapter._native_resume_session


def run_adapter_iterations(adapter, factory, context, control):
    from .control import run_iterations
    if not hasattr(adapter, '_native_resume_session'):
        if getattr(adapter, '_training_finalized', False):
            raise RuntimeError('Prediction-only finalized adapter requires fit_resumable with a valid training checkpoint.')
        adapter.__dict__.pop('training_checkpoint_', None)
        return run_iterations(factory, context, control)
    from ..experiments.recovery import execution_key
    from .resume_loop import run_resumable_iterations
    checkpoint, publish = adapter._native_resume_session
    if getattr(adapter, '_training_finalized', False) and checkpoint is None:
        raise RuntimeError('A finalized adapter requires a valid training checkpoint.')
    profile = execution_profile(adapter)
    config = getattr(adapter, 'config', None) or adapter.spec
    store = getattr(adapter, 'event_store', None)
    # The native store's public frames are copied; read current content instead
    # of assuming a cached version alone proves private state has not changed.
    evidence = None if store is None else (store.version, store.history, store.team_seasons, store.schedule,
                                           getattr(store, 'heatmaps', None))
    identity = execution_key(type(adapter), context, control, config, evidence, profile)
    state = None if checkpoint is None else read_checkpoint(checkpoint, identity, profile)[0]
    if checkpoint is not None:
        adapter.training_checkpoint_ = str(checkpoint)
    adapter._training_finalized = False
    def save(capture, *, complete):
        def writer(directory):
            # Capture only after the publication policy accepts this offer.
            # Thinned offers must not clone optimizer/model buffers to CPU.
            write_checkpoint(directory, capture(), identity, profile)
        # A final checkpoint is mandatory even when periodic offers are thinned.
        writer._xdiyo_completed_checkpoint = complete
        path = publish(writer)
        if path is not None:
            read_checkpoint(path, identity, profile)
            adapter.training_checkpoint_ = str(path)
    return run_resumable_iterations(factory, context, control, state=state, save=save)


def finalize_training(adapter, *, mode='retain', checkpoint=None):
    if mode == 'retain':
        return
    if mode != 'prediction_only':
        raise ValueError('Finalization mode must be retain or prediction_only.')
    path = checkpoint or getattr(adapter, 'training_checkpoint_', None)
    if path is None:
        raise RuntimeError('Prediction-only finalization requires a verified completed training checkpoint.')
    state, manifest = read_checkpoint(path)
    if not manifest['complete'] or state['winner'] is None:
        raise RuntimeError('Cannot finalize before completed checkpoint publication.')
    # Bind cleanup to this fitted object, not any unrelated completed artifact.
    own_path = getattr(adapter, 'training_checkpoint_', None)
    if own_path is None or Path(own_path).resolve() != Path(path).resolve():
        raise ValueError('Finalization checkpoint does not belong to this fitted adapter.')
    backend = adapter.backend_
    if getattr(adapter, '_training_finalized', False):
        return
    import torch
    current = backend.model.state_dict()
    saved = state['winner']['model']
    if list(current) != list(saved) or any(current[k].dtype != saved[k].dtype or not torch.equal(current[k].detach().cpu(), saved[k]) for k in current):
        raise ValueError('Finalization checkpoint does not match the current fitted parameters.')
    if _preprocessing(backend) != state['winner']['preprocessing']:
        raise ValueError('Finalization checkpoint does not match the current preprocessing.')
    if hasattr(backend, "_input_resident"):
        backend._input_resident.close()
    backend.optimizer.zero_grad(set_to_none=True)
    for name in ('optimizer', 'train_groups', 'validation_groups', 'train_targets', 'validation_targets',
                 'train_weights', 'validation_weights', 'train', 'validation',
                 'train_statistics', 'validation_statistics', '_statistics_observations'):
        backend.__dict__.pop(name, None)
    if hasattr(backend, 'preprocessor'):
        backend.preprocessor._edge_cache = None
        backend.preprocessor._map_cache = None
    adapter._training_finalized = True
