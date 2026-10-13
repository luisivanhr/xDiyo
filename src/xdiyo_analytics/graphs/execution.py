"""Scoped, opt-in deterministic execution; optional torch stays lazily imported."""
from contextlib import contextmanager
from functools import wraps
import os
from threading import RLock


_execution_lock = RLock()
_WORKSPACES = (':4096:8', ':16:8')


def prepare_determinism(config, device):
    """Configure cuBLAS before CUDA initialization, never silently fall back."""
    if not config.deterministic or not str(device).startswith('cuda'):
        return
    from .adapter import _torch
    torch = _torch()
    with _execution_lock:
        workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        if workspace in _WORKSPACES:
            return
        if workspace is not None or torch.cuda.is_initialized():
            raise RuntimeError(
                'Deterministic GAT execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8 '
                '(or :16:8) before CUDA initialization. Restart the notebook kernel/process '
                'with this environment setting, or enable deterministic mode before using CUDA.')
        # CUDA caches its workspace configuration. Keep this process-level setting
        # after execution; restoring the environment would misrepresent that cache.
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


@contextmanager
def deterministic_execution(config, device):
    """Restore caller flags, including warn-only, on success and on failure.

    These flags are process-global. The lock serializes our enabled scopes, but
    callers must not run unrelated torch work concurrently in the same process.
    Disabled configurations inherit caller settings without changing them.
    """
    if not config.deterministic:
        yield
        return
    from .adapter import _torch
    torch = _torch()
    with _execution_lock:
        prepare_determinism(config, device)
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        cudnn_deterministic = torch.backends.cudnn.deterministic
        benchmark = torch.backends.cudnn.benchmark
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            yield
        finally:
            torch.backends.cudnn.benchmark = benchmark
            torch.backends.cudnn.deterministic = cudnn_deterministic
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def configured_execution(function):
    """One scope per backend operation, never one scope per fixture/edge."""
    @wraps(function)
    def wrapped(owner, *args, **kwargs):
        with deterministic_execution(owner.config, owner.device):
            return function(owner, *args, **kwargs)
    return wrapped
