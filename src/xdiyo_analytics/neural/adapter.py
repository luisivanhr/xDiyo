"""Native iterative CNN training, independent of graph construction."""
from copy import deepcopy
import numpy as np
import pandas as pd

from .inputs import neural_maps
from ..features.spatial import heatmap_columns
from ..training.control import run_iterations


class HeatmapCNNAdapter:
    supports_sample_weight = True

    def __init__(self, spec, *, device='cpu', inner_threads=1):
        self.spec, self.device, self.inner_threads = spec, device, inner_threads

    def configure_input_cache(self, max_bytes=0, headroom_bytes=1073741824):
        from ..training.residency import InputCachePolicy
        self.input_cache_policy = InputCachePolicy(max_bytes, headroom_bytes)

    def configure_device(self, requested):
        from ..graphs.adapter import TemporalGraphAdapter
        return TemporalGraphAdapter.configure_device(self, requested)

    def configure_threads(self, count):
        from ..graphs.adapter import TemporalGraphAdapter
        return TemporalGraphAdapter.configure_threads(self, count)

    def fit_resumable(self, context, *, control, checkpoint, save_checkpoint):
        from ..training.native_resume import resume_session
        with resume_session(self, checkpoint, save_checkpoint):
            self.fit_controlled(context, control)

    def finalize_training(self, *, mode='retain', checkpoint=None):
        from ..training.native_resume import finalize_training
        finalize_training(self, mode=mode, checkpoint=checkpoint)

    def fit(self, context):
        self.fit_controlled(context, self.spec.control)

    def fit_controlled(self, context, control):
        from ..training.native_resume import run_adapter_iterations
        from ..graphs.adapter import torch_threads
        self.configure_device(self.device)
        self.configure_threads(self.inner_threads)
        effective_control = control or self.spec.control
        with torch_threads(self.inner_threads):
            self.backend_, self.training_history_, self.training_summary_ = run_adapter_iterations(self,
                lambda seed: CNNBackend(self.spec, seed, self.device, getattr(self, "input_cache_policy", None)), context, effective_control)
        self.effective_control_ = effective_control

    def predict(self, context):
        from ..graphs.adapter import torch_threads
        with torch_threads(self.inner_threads):
            return self.backend_.predict(context)

    def encode(self, context):
        """Read-only frozen representations in original fixture-row order."""
        from ..graphs.adapter import torch_threads
        with torch_threads(self.inner_threads):
            return self.backend_.predict(context, embeddings=True)


class CNNInputs:
    def __init__(self, spec):
        self.spec = spec

    def fit(self, X):
        from ..graphs.preprocessing import MaskedScale
        if not X.index.is_unique or not X.columns.is_unique:
            raise ValueError('CNN inputs require unique row and feature identities.')
        groups = heatmap_columns(X.columns)
        self.group_names = list(self.spec.grid_features or groups)
        if not self.group_names or not set(self.group_names) <= set(groups):
            raise ValueError('Select native complete heatmap grid groups for the standalone CNN.')
        selected = [groups[name] for name in self.group_names]
        if len({g['size'] for g in selected}) != 1:
            raise ValueError('CNN input maps must share one grid resolution.')
        self.size = selected[0]['size']
        self.columns = []
        for g in selected:
            if set(g['cells']) != {(y, x) for y in range(self.size) for x in range(self.size)}:
                raise ValueError('CNN input must contain every grid cell exactly once.')
            self.columns.extend(g['cells'][y, x] for y in range(self.size) for x in range(self.size))
        if set(self.columns) & set(self.spec.numeric_features):
            raise ValueError('Grid cells cannot also be selected as numeric features.')
        self.numeric_scale = MaskedScale().fit(X[list(self.spec.numeric_features)].to_numpy(dtype=float, na_value=np.nan))
        self.transform(X)
        return self

    def transform(self, X):
        if not X.index.is_unique or not X.columns.is_unique:
            raise ValueError('CNN inputs require unique row and feature identities.')
        if not set([*self.columns, *self.spec.numeric_features]) <= set(X):
            raise ValueError('CNN prediction inputs are missing fitted feature columns.')
        maps, masks = neural_maps(X[self.columns].to_numpy(dtype=float, na_value=np.nan).reshape(
            len(X), len(self.group_names), self.size, self.size))
        numeric = self.numeric_scale.transform(X[list(self.spec.numeric_features)].to_numpy(dtype=float, na_value=np.nan)).astype(np.float32)
        if not np.isfinite(numeric).all():
            raise ValueError('Scaled numeric features exceed the finite neural tensor range.')
        return maps, masks, numeric

    def state(self):
        return dict(group_names=self.group_names, size=self.size, columns=self.columns, numeric_scale=self.numeric_scale.state())

    @classmethod
    def from_state(cls, spec, state):
        from ..graphs.preprocessing import MaskedScale
        obj = cls(spec)
        obj.group_names, obj.size, obj.columns = state['group_names'], state['size'], state['columns']
        obj.numeric_scale = MaskedScale.from_state(state['numeric_scale'])
        return obj


def make_network(spec, inputs, outputs):
    from .network import HeatmapNetwork
    return HeatmapNetwork(spec, len(inputs.group_names), inputs.size, outputs)


class CNNBackend:
    supports_sample_weight = True

    def __init__(self, spec, seed, device, input_cache_policy=None):
        from ..training.residency import InputResident
        self.input_cache_policy = input_cache_policy
        self._input_resident = InputResident(device, input_cache_policy)
        self.spec, self.seed, self.device = spec, seed, device

    def initialize(self, context):
        import torch
        if not len(context.X) or not len(context.y.columns) or not context.y.index.equals(context.X.index) or context.y.isna().any().any():
            raise ValueError('CNN fitting needs nonempty aligned inputs and observed targets.')
        self.target_names = list(context.y.columns)
        self.classes = None
        if self.spec.task == 'classification':
            if len(self.target_names) != 1:
                raise ValueError('CNN classification requires one target.')
            self.classes = sorted(context.y.iloc[:, 0].unique().tolist())
            if len(self.classes) < 2:
                raise ValueError('CNN classification requires at least two fitting classes.')
        self.inputs = CNNInputs(self.spec).fit(context.X)
        self.train = self.population(context)
        self.validation = self.population(context.validation) if context.validation is not None else None
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.model = make_network(self.spec, self.inputs, len(self.classes) if self.classes is not None else len(self.target_names)).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.spec.learning_rate, weight_decay=self.spec.weight_decay)

    def population(self, context):
        if not context.X.index.equals(context.y.index) or list(context.y.columns) != self.target_names:
            raise ValueError('CNN target columns and row identities must match fitting.')
        if self.classes is not None and not context.y.iloc[:, 0].isin(self.classes).all():
            raise ValueError('CNN validation contains classes absent from the fitting population.')
        y = context.y.to_numpy(dtype=float, copy=True) if self.classes is None else np.array([
            self.classes.index(v) for v in context.y.iloc[:, 0]])
        if not np.isfinite(y).all():
            raise ValueError('CNN targets must be finite.')
        w = np.ones(len(y))
        if context.sample_weight is not None:
            if hasattr(context.sample_weight, 'index') and not context.sample_weight.index.equals(context.X.index):
                raise ValueError('Sample weights must align with CNN rows.')
            w = np.asarray(context.sample_weight, dtype=float).reshape(-1)
        if len(w) != len(y) or not np.isfinite(w).all() or (w < 0).any() or not (w > 0).any():
            raise ValueError('CNN weights must be finite, nonnegative and have positive total.')
        if self.classes is None and (np.abs(y) > np.finfo(np.float32).max).any():
            raise ValueError('CNN targets must fit the finite neural tensor range.')
        maps, masks, numeric = self.inputs.transform(context.X)
        observed = masks.any(axis=1)
        # Normalize once in float64, independently for each loss population.
        # Max scaling avoids overflow, and no float32 cast erases small weights.
        def normalized(selected):
            values = np.where(selected, w, 0.)
            maximum = values.max()
            if maximum == 0:
                return values  # No observed positive-weight reconstruction target.
            values = values / maximum
            return values / values.sum()
        weights = np.stack((normalized(np.ones(len(y), dtype=bool)), normalized(observed)))
        from .prepared import OwnedPopulation, population_binding
        return OwnedPopulation((maps, masks, numeric, y, weights),
            binding=population_binding(self.spec, self.inputs, self.device),
            validate_kl=self.spec.encoder.reconstruction is not None and self.spec.encoder.reconstruction.loss == 'kl')

    def loss(self, population, backward=False):
        import torch
        from .prepared import OwnedPopulation, population_binding
        owned = type(population) is OwnedPopulation
        if owned:
            if population.binding != population_binding(self.spec, self.inputs, self.device):
                raise ValueError('Prepared CNN inputs no longer match configuration, preprocessing or device; initialize again.')
            count = population.rows
        else:
            maps, masks, numeric, y, weights = population
            count = len(y)
        # Accumulate detached float64 scalars on device in batch order. Reading
        # every batch as a Python float would synchronize CUDA repeatedly.
        primary = torch.zeros((), dtype=torch.float64, device=self.device)
        reconstruction = torch.zeros_like(primary)
        for start in range(0, count, self.spec.batch_size):
            sl = slice(start, start+self.spec.batch_size)
            def tensors(device):
                batch = population.batch(sl) if owned else (maps[sl], masks[sl], numeric[sl], y[sl], weights[:, sl])
                return (*[torch.as_tensor(v, device=device) for v in batch[:3]],
                        torch.as_tensor(batch[3], dtype=torch.long if self.classes is not None else torch.float32, device=device),
                        *[torch.as_tensor(v, device=device) for v in batch[4]])
            from contextlib import nullcontext
            lease = (self._input_resident.lease((population.generation, start, self.spec.batch_size),
                     lambda: tensors('cpu'), lambda: tensors(self.device)) if owned else nullcontext(tensors(self.device)))
            with lease as values:
                m, a, n, target, w, rw = values
                logits, z = self.model(m, a, n)
                error = (logits-target).square().mean(dim=1) if self.classes is None else torch.nn.functional.cross_entropy(logits, target, reduction='none')
                loss = (error*w).sum()
                primary.add_(loss.detach())
                if self.spec.encoder.reconstruction is not None:
                    errors = (self.model.encoder._reconstruction_errors(z, m, a, validate=False)
                              if owned and population.kl_validated else self.model.encoder.reconstruction_errors(z, m, a))
                    aux = (errors*rw).sum()
                    reconstruction.add_(aux.detach())
                    loss = loss + self.spec.encoder.reconstruction.weight*aux
                if backward:
                    loss.backward()
                del values, m, a, n, target, w, rw, logits, z, loss, error
                if self.spec.encoder.reconstruction is not None:
                    del errors, aux
        return float(primary), float(reconstruction)

    def step(self):
        import torch
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, aux = self.loss(self.train, True)
        self.optimizer.step()
        values = {'train_loss': loss}
        if self.spec.encoder.reconstruction is not None:
            values.update(train_reconstruction_loss=aux, train_objective=loss+self.spec.encoder.reconstruction.weight*aux)
        if self.validation is not None:
            self.model.eval()
            with torch.no_grad():
                loss, aux = self.loss(self.validation)
            values['validation_loss'] = loss
            if self.spec.encoder.reconstruction is not None:
                values.update(validation_reconstruction_loss=aux, validation_objective=loss+self.spec.encoder.reconstruction.weight*aux)
        return values

    def checkpoint_metrics(self, monitor, values):
        """Training monitors describe the same post-update state as snapshot()."""
        if monitor not in ('train_loss', 'train_objective', 'train_reconstruction_loss'):
            return values
        from ..graphs.adapter import inference_mode
        with inference_mode(self.model):
            loss, aux = self.loss(self.train)
        values = dict(values, train_loss=loss)
        if self.spec.encoder.reconstruction is not None:
            values.update(train_reconstruction_loss=aux,
                          train_objective=loss+self.spec.encoder.reconstruction.weight*aux)
        return values

    def predict(self, context, embeddings=False):
        import torch
        from ..graphs.adapter import inference_mode
        maps, masks, numeric = self.inputs.transform(context.X)
        width = self.spec.encoder.embedding_dim if embeddings else (len(self.classes) if self.classes is not None else len(self.target_names))
        output = np.empty((len(maps), width))
        with inference_mode(self.model):
            for start in range(0, len(maps), self.spec.batch_size):
                sl = slice(start, start+self.spec.batch_size)
                logits, z = self.model(*[torch.as_tensor(v[sl], device=self.device) for v in (maps, masks, numeric)])
                output[sl] = (z if embeddings else (logits.softmax(-1) if self.classes is not None else logits)).cpu().numpy()
        if embeddings:
            return pd.DataFrame(output, index=context.X.index, columns=[f'z_{i}' for i in range(width)])
        if self.classes is None:
            return {'predict': pd.DataFrame(output, index=context.X.index, columns=self.target_names)}
        columns = pd.MultiIndex.from_product([self.target_names, self.classes], names=['target', 'class'])
        return {'predict_proba': pd.DataFrame(output, index=context.X.index, columns=columns),
                'predict': pd.DataFrame({self.target_names[0]: [self.classes[i] for i in output.argmax(1)]}, index=context.X.index)}

    def snapshot(self):
        return deepcopy(self.model.state_dict())

    def restore(self, state):
        self.model.load_state_dict(state)

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def set_learning_rate(self, rate):
        for group in self.optimizer.param_groups:
            group['lr'] = rate
