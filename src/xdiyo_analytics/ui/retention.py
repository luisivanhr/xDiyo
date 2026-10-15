"""Optional completed-result retention with verified native recovery."""
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
import sys


def payload_bytes(value, seen=None):
    """Conservative retained payload estimate, not process RSS or allocator RAM."""
    import numpy as np
    import pandas as pd
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(index=True, deep=True).sum()) + payload_bytes(value.attrs, seen)
    if isinstance(value, pd.Series):
        return int(value.memory_usage(index=True, deep=True)) + payload_bytes(value.attrs, seen)
    if isinstance(value, np.ndarray):
        return value.nbytes + (sum(payload_bytes(v, seen) for v in value.flat) if value.dtype.hasobject else 0)
    if type(value).__module__.startswith('torch') and hasattr(value, 'numel'):
        return value.numel() * value.element_size()
    torch = sys.modules.get('torch')
    if torch is not None and isinstance(value, torch.nn.Module):
        return sum(payload_bytes(t, seen) for t in (*value.parameters(), *value.buffers()))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(payload_bytes(k, seen) + payload_bytes(v, seen) for k, v in value.items())
    if isinstance(value, (tuple, list, set)):
        return size + sum(payload_bytes(v, seen) for v in value)
    if hasattr(value, '__dict__') and not isinstance(value, type) and not callable(value):
        return size + payload_bytes(vars(value), seen)
    return size


def digest(value):
    from ..experiments.recovery import pack
    return hashlib.sha256(json.dumps(pack(value), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def model_digest(adapter):
    from ..training.checkpoints import _CheckpointAdapter
    if isinstance(adapter, _CheckpointAdapter):
        adapter = adapter.estimator
    if adapter is None:
        return None
    from ..graphs import TemporalGraphAdapter
    from ..neural import HeatmapCNNAdapter
    if type(adapter) in (TemporalGraphAdapter, HeatmapCNNAdapter):
        from ..training.native_resume import _preprocessing
        backend = adapter.backend_
        return digest((getattr(adapter, 'config', None), getattr(adapter, 'spec', None),
                       getattr(backend, 'fitting_boundary', None), getattr(backend, 'trained_through', None),
                       getattr(backend, 'target_name', None), getattr(backend, 'target_names', None),
                       getattr(backend, 'output_classes', None), getattr(backend, 'classes', None),
                       getattr(getattr(adapter, 'event_store', None), 'version', None), _preprocessing(backend),
                       {k: v.detach().cpu().numpy() for k, v in backend.model.state_dict().items()}))
    # Native sklearn serializers already retain these adapters for prediction.
    from ..training.estimators import EstimatorAdapter
    if type(adapter) is EstimatorAdapter:
        from joblib import hash
        return hash(adapter)
    raise TypeError('Custom model lifecycle is pinned; no verified retention capability.')


def result_digest(result):
    return digest((result.prepared, result.training, result.refit, result.report)), tuple(
        model_digest(f.model) for f in result.training.folds), model_digest(result.refit.model) if result.refit else None


def files_digest(folder):
    folder = Path(folder)
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file() and not p.name.startswith('.')}


class RetainedResult:
    """Lightweight loader. Construction proves report/data/model round trips."""
    def __init__(self, result, recipe):
        from ..experiments.football import ExperimentResult, FootballExperiment
        if type(result) is not ExperimentResult or result._renderers or result.path is None:
            raise ValueError('Unsaved or custom-rendered result stays pinned.')
        self.path = Path(result.path)
        self.name, self.output_dir = recipe['name'], recipe['output_dir']
        self.run_id = result.record['run_id']
        self.files = files_digest(self.path)
        self.signature = result_digest(result)
        loaded = FootballExperiment(self.name, output_dir=self.output_dir).load(self.run_id)
        if result_digest(loaded) != self.signature or files_digest(self.path) != self.files:
            raise ValueError('Persistence/reload does not preserve the complete result; keeping it pinned.')
        # Render once after reload too: unsupported restored artifacts pin the job.
        loaded.to_html()

    def load(self):
        from ..experiments.football import FootballExperiment
        if files_digest(self.path) != self.files:
            raise ValueError('Retained result files changed; open the saved run again explicitly.')
        result = FootballExperiment(self.name, output_dir=self.output_dir).load(self.run_id)
        if result_digest(result) != self.signature:
            raise ValueError('Retained result no longer matches verified data and model state.')
        return result


class RetainedDocument:
    def __init__(self, path, document):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = document.encode('utf-8')
        self.path.write_bytes(raw)
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if self.read() != document:
            raise ValueError('Retained document verification failed.')

    def read(self):
        raw = self.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise ValueError('Retained report document changed.')
        return raw.decode('utf-8')
