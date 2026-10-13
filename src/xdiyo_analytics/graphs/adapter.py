"""Native training adapter, bounded iteration backend and exact-cutoff exports."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import numpy as np
import pandas as pd
from .config import CLASS_ORDER, MATCH_KEYS
from .preprocessing import GraphPreprocessor
from .store import _ids, _times
from .streaming import GroupPlan, PreparedGroups, release_temporary_frames
from .spatial import validate_grid_contract
from .profiling import phase, timed
from .execution import configured_execution, prepare_determinism
from ..training.control import TrainingControl, run_iterations


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError("TemporalGraphAdapter requires the optional xdiyo-analytics[graphs] extra.") from exc


@contextmanager
def torch_threads(count):
    torch = _torch()
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(count)
        yield
    finally:
        torch.set_num_threads(old)


@contextmanager
def inference_mode(model):
    """Inspection preserves the caller's module mode, even after an error."""
    was_training = model.training
    try:
        model.eval()
        with _torch().no_grad():
            yield
    finally:
        model.train(was_training)


class TemporalGraphAdapter:
    supports_sample_weight = True

    def __init__(self, config, event_store, *, control=None, device="cpu", inner_threads=1):
        validate_grid_contract(event_store, config.heatmaps)
        self.config, self.event_store = config, event_store
        self.control = control or TrainingControl()
        self.device, self.inner_threads = device, inner_threads

    def configure_device(self, requested):
        torch = _torch()
        if requested == "auto":
            requested = "cuda:0" if torch.cuda.is_available() else "cpu"
        if requested == "cuda":
            requested = "cuda:0"
        device = torch.device(requested)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU and CUDA devices are supported.")
        if device.type == "cuda" and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
            raise ValueError("Requested CUDA device is unavailable.")
        self.device = str(device)
        return self.device

    def configure_threads(self, count):
        if type(count) is not int or count < 1:
            raise ValueError("inner_threads must be a positive integer.")
        self.inner_threads = count

    def fit(self, context):
        self.fit_controlled(context, self.control)

    def fit_controlled(self, context, control):
        validate_grid_contract(self.event_store, self.config.heatmaps)
        self.configure_device(self.device)
        prepare_determinism(self.config, self.device)
        self.configure_threads(self.inner_threads)
        self.control = control or self.control
        with torch_threads(self.inner_threads):
            self.backend_, self.training_history_, self.training_summary_ = run_iterations(
                lambda seed: _Backend(self.config, self.event_store, seed, self.device),
                context, self.control)
        self.training_summary_.update(
            fitting_boundary=self.backend_.fitting_boundary.isoformat(),
            trained_through=self.backend_.trained_through.isoformat(),
            model_version=self.backend_.model_version,
            event_store_version=self.event_store.version,
            forecast_policy=self.config.forecast_policy,
            timezone=self.config.timezone)
        if self.config.deterministic:
            self.training_summary_['deterministic'] = True

    def predict(self, context):
        with torch_threads(self.inner_threads):
            return self.backend_.predict(context)

    def encode_teams(self, queries):
        """queries: team_id and timezone-aware issuance_at; preserve index/order."""
        with torch_threads(self.inner_threads):
            return self.backend_.encode_teams(queries)

    def predict_statistics(self, context):
        with torch_threads(self.inner_threads):
            return self.backend_.predict_statistics(context)

    def attention_weights(self, queries):
        from .inspection import attention_weights
        return attention_weights(self, queries)

    def attach_event_store(self, event_store):
        """Explicit replacement for later observations; trained parameters stay frozen.

        Existing source events/memberships must be preserved. New fixtures and
        new historical evidence may be added, but revised source data requires
        a new fitting run instead of silently changing artifact provenance.
        """
        validate_grid_contract(event_store, self.config.heatmaps)
        old = self.event_store
        for before, after, keys in (
            (old.history, event_store.history, [*MATCH_KEYS, "side"]),
            (old.team_seasons, event_store.team_seasons, ["team_id", "competition_id", "season_id"]),
        ):
            left = before.set_index(keys)
            right = after.set_index(keys)
            try:
                pd.testing.assert_frame_equal(left.reset_index(), right.loc[left.index, left.columns].reset_index(),
                                              check_dtype=False, check_like=True, check_exact=True)
            except (KeyError, AssertionError) as exc:
                raise ValueError("Replacement store must preserve existing evidence.") from exc
        if self.config.forecast_policy == "by_round":
            before = old.issue_times(old.schedule, self.config)
            after = event_store.issue_times(old.schedule, self.config)
            if not before.equals(after):
                raise ValueError("Replacement store changes frozen round anchors.")
        self.event_store = event_store
        self.backend_.store = event_store
        self.backend_.preprocessor._edge_cache = None


class _Backend:
    supports_sample_weight = True

    def __init__(self, config, store, seed, device):
        self.config, self.store, self.seed, self.device = config, store, seed, device

    def _validate(self, context, fitting=False):
        if context.layout != "match" or not context.X.index.equals(context.metadata.index) or not context.X.index.is_unique:
            raise ValueError("Graph contexts require match layout and unique aligned row indices.")
        if context.metadata.duplicated(list(MATCH_KEYS)).any():
            raise ValueError("Context contains repeated fixture identities.")
        _ids(context.metadata, (*MATCH_KEYS, "home_id", "away_id"))
        if context.metadata.home_id.eq(context.metadata.away_id).any():
            raise ValueError("A fixture needs different teams.")
        issue = self.store.issue_times(context.metadata, self.config)
        if fitting:
            if context.y.shape[1] != 1 or not context.y.index.equals(context.X.index):
                raise ValueError("Fitting requires one aligned home/draw/away target.")
            labels = context.y.iloc[:, 0]
            classes = CLASS_ORDER
            if labels.map(lambda x: isinstance(x, (bool, np.bool_))).any() or not labels.isin(classes).all():
                raise ValueError("Target labels must use native home-perspective 1/0/-1.")
            if hasattr(self, 'output_classes') and classes != self.output_classes:
                raise ValueError('Fitting and validation must use the same outcome label encoding.')
        if self.config.fixture_features:
            if not set(self.config.fixture_features) <= set(context.X):
                raise ValueError("Selected fixture features are absent from X.")
            # This is a cutoff declaration for externally constructed features,
            # not publication/release infrastructure.
            if "feature_cutoff_at" not in context.metadata:
                raise ValueError("Explicit fixture features require feature_cutoff_at declarations.")
            supplied = _times(context.metadata.feature_cutoff_at)
            expected = pd.Series([self.config.history_cutoff(t) for t in issue], index=issue.index)
            if not np.array_equal(supplied, expected):
                raise ValueError("Fixture features must use the same previous-day cutoff as graph inputs.")
        return issue

    def _plan(self, context):
        return GroupPlan(self.store, self.config, context, self._validate(context))

    def _stream(self, context):
        return PreparedGroups(self._plan(context), self.preprocessor, self.device)

    def _replay(self, groups):
        if self.config.boundary_memory is None or not len(groups):
            return None
        from .memory import BoundaryReplay
        return BoundaryReplay(self, groups.plan.descriptors[-1][0])

    @configured_execution
    @release_temporary_frames
    def initialize(self, context):
        torch = _torch()
        from .network import TemporalNetwork
        from ..data.availability import respect_result_availability
        self._validate(context, fitting=True)
        if len(context.X) == 0:
            raise ValueError("Fitting population cannot be empty.")
        self.target_name = context.y.columns[0]
        self.output_classes = CLASS_ORDER
        self.fitting_boundary = respect_result_availability(context.metadata,
            self.config.labels_known_at(context.metadata.kickoff_at)).max()
        self.trained_through = self.fitting_boundary
        if context.validation is not None:
            validation_issue = self._validate(context.validation, fitting=True)
            if len(validation_issue) == 0 or min(self.config.history_cutoff(t) for t in validation_issue) < self.fitting_boundary:
                raise ValueError("Fitting labels must be known before validation issuance.")
            self.trained_through = max(self.trained_through,
                respect_result_availability(context.validation.metadata,
                    self.config.labels_known_at(context.validation.metadata.kickoff_at)).max())
        plan = self._plan(context)
        numeric, leagues = plan.fitting_inputs()
        largest = max(self.config.history_cutoff(time) for time, _ in plan.descriptors)
        fitting_history = self.store.eligible(largest)
        self.preprocessor = GraphPreprocessor(self.config).fit(
            fitting_history, (), context.metadata, numeric, extra_leagues=leagues)
        del fitting_history
        # fork_rng avoids altering the caller's CPU random stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.model = TemporalNetwork(self.config, len(self.preprocessor.leagues),
                2*len(self.preprocessor.edge_columns)+1+len(self.preprocessor.leagues)+1,
                2*len(self.preprocessor.fixture_columns)+len(self.preprocessor.leagues)+1).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate,
                                           weight_decay=self.config.weight_decay)
        del numeric
        self.train_groups = PreparedGroups(plan, self.preprocessor, self.device)
        self.train_targets, self.train_weights = self._targets(context)
        self.validation_groups = None
        if context.validation is not None:
            self.validation_groups = self._stream(context.validation)
            # Preserve eager input validation without retaining its tensors.
            for i in range(len(self.validation_groups)):
                self.validation_groups[i]
            self.validation_targets, self.validation_weights = self._targets(context.validation)
        if self.config.statistics_forecast is not None:
            from .supervision import initialize_statistics
            initialize_statistics(self, context)
        self.refresh_version()

    def _targets(self, context):
        torch = _torch()
        y = torch.tensor([self.output_classes.index(x) for x in context.y.iloc[:, 0]], device=self.device)
        w = np.ones(len(y))
        if context.sample_weight is not None:
            if isinstance(context.sample_weight, (pd.Series, pd.DataFrame)) and not context.sample_weight.index.equals(context.X.index):
                raise ValueError("Observation weights must align with X.")
            w = np.asarray(context.sample_weight, dtype=float).reshape(-1)
        if len(w) != len(y) or not np.isfinite(w).all() or (w < 0).any() or not (w > 0).any():
            raise ValueError("Observation weights must be finite, nonnegative and have positive total.")
        # Binary scaling is exact in float64 and leaves relative weights intact.
        # ldexp also handles subnormal maxima and avoids constructing 2**1024.
        # Bound the largest stored weight to [0.5, 1], before the float32 cast:
        # neither an overall tiny scale nor an overflowing raw sum can corrupt
        # the fitting/validation objective. Ordinary binary rescalings retain
        # the existing float32 reduction arithmetic.
        exponent = np.frexp(w.max())[1]
        scaled = np.ldexp(w, -int(exponent)).astype(np.float32)
        if ((w > 0) & (scaled == 0)).any():
            raise ValueError('Relative observation weights exceed the float32 dynamic range; positive weights would disappear.')
        return y, torch.tensor(scaled, device=self.device)

    def _group_loss(self, group, targets, weights, backward, total_weight=None):
        torch = _torch()
        positions, data, home, away, fixture = group
        reconstruction = self.config.heatmaps.encoder.reconstruction if self.config.heatmaps is not None else None
        with phase('forward'):
            if reconstruction is None:
                logits = self.model(data, home, away, fixture)
            else:
                logits, auxiliary = self.model(data, home, away, fixture, return_auxiliary=True)
        ix = torch.as_tensor(positions, device=self.device)
        if total_weight is None:
            total_weight = weights.sum()
        group_weights = weights[ix]
        loss = (torch.nn.functional.cross_entropy(logits, targets[ix], reduction="none") * group_weights).sum()/total_weight
        if reconstruction is not None:
            part = auxiliary * (group_weights.sum()/total_weight)
            objective = loss + reconstruction.weight * part
        else:
            objective = loss
        if backward:
            with phase('backward'):
                objective.backward()
        # Detached scalar storage only; no graph/logits survive this helper.
        return (loss.detach(), part.detach()) if reconstruction is not None else (loss.detach(),)

    @release_temporary_frames
    @configured_execution
    def _loss(self, groups, targets, weights, backward=False):
        self._reconstruction_loss = 0.
        if self.config.statistics_forecast is not None:
            from .supervision import statistics_loss
            return statistics_loss(self, groups, targets, weights, backward)
        replay = self._replay(groups)
        total_weight = weights.sum()
        scalars = []
        for i in range(len(groups)):
            scalars.extend(self._group_loss(replay.group(groups, i) if replay is not None else groups[i], targets, weights, backward, total_weight))
        if not scalars:
            return 0.
        # One device-to-host transfer, then the *original* sequential Python
        # float accumulation. A device sum would change the reduction order.
        width = 2 if self.config.heatmaps is not None and self.config.heatmaps.encoder.reconstruction is not None else 1
        with phase('loss_transfer'):
            values = _torch().stack(scalars).cpu().tolist()
        value = 0.
        for i in range(0, len(values), width):
            value += values[i]
            if width == 2:
                self._reconstruction_loss += values[i+1]
        return value

    @configured_execution
    def step(self):
        torch = _torch()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        train_loss = self._loss(self.train_groups, self.train_targets, self.train_weights, True)
        penalty = self.config.prior_penalty * (self.model.league.weight.square().sum() + self.model.entry.weight.square().sum())
        penalty.backward()
        with phase('optimizer'):
            self.optimizer.step()
        values = {"train_loss": train_loss}
        if self.config.heatmaps is not None and self.config.heatmaps.encoder.reconstruction is not None:
            values['train_reconstruction_loss'] = self._reconstruction_loss
            values['train_objective'] = train_loss + self.config.heatmaps.encoder.reconstruction.weight*self._reconstruction_loss + float(penalty.detach())
        if self.config.statistics_forecast is not None:
            values['train_statistics_loss'] = self._statistics_loss
            values['train_objective'] = values.get('train_objective', train_loss + float(penalty.detach())) + self.config.statistics_forecast.weight*self._statistics_loss
        if self.validation_groups is not None:
            self.model.eval()
            with torch.no_grad():
                values["validation_loss"] = self._loss(self.validation_groups, self.validation_targets, self.validation_weights)
                if self.config.heatmaps is not None and self.config.heatmaps.encoder.reconstruction is not None:
                    values['validation_reconstruction_loss'] = self._reconstruction_loss
                    values['validation_objective'] = values['validation_loss'] + self.config.heatmaps.encoder.reconstruction.weight*self._reconstruction_loss
                if self.config.statistics_forecast is not None:
                    values['validation_statistics_loss'] = self._statistics_loss
                    values['validation_objective'] = values.get('validation_objective', values['validation_loss']) + self.config.statistics_forecast.weight*self._statistics_loss
        self.refresh_version()
        return values

    @configured_execution
    @timed('metric_replay')
    def checkpoint_metrics(self, monitor, values):
        if monitor not in ('train_loss', 'train_objective', 'train_reconstruction_loss', 'train_statistics_loss'):
            return values
        with inference_mode(self.model):
            loss = self._loss(self.train_groups, self.train_targets, self.train_weights)
            penalty = self.config.prior_penalty * (self.model.league.weight.square().sum() + self.model.entry.weight.square().sum())
        values = dict(values, train_loss=loss)
        if self.config.heatmaps is not None and self.config.heatmaps.encoder.reconstruction is not None:
            values.update(train_reconstruction_loss=self._reconstruction_loss,
                          train_objective=loss+self.config.heatmaps.encoder.reconstruction.weight*self._reconstruction_loss+float(penalty))
        if self.config.statistics_forecast is not None:
            values['train_statistics_loss'] = self._statistics_loss
            values['train_objective'] = loss + float(penalty) + self.config.statistics_forecast.weight*self._statistics_loss
            if self.config.heatmaps is not None and self.config.heatmaps.encoder.reconstruction is not None:
                values['train_objective'] += self.config.heatmaps.encoder.reconstruction.weight*self._reconstruction_loss
        return values

    @timed('fingerprint')
    def refresh_version(self):
        digest = sha256()
        for name, value in self.model.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        digest.update(json.dumps(self.preprocessor.state(), sort_keys=True).encode())
        # A representation version includes decay, policy and architectural
        # settings as well as learned coordinates and preprocessing.
        if self.config.statistics_forecast is not None:
            digest.update(json.dumps(self.statistics_scale.state(), sort_keys=True).encode())
        config_state = asdict(self.config)
        for name in ('boundary_memory', 'statistics_forecast'):
            if config_state[name] is None:
                del config_state[name]
        if not config_state['deterministic']:
            del config_state['deterministic']  # Preserve pre-option artifact fingerprints.
        if config_state['include_goal_features']:
            del config_state['include_goal_features']  # Preserve pre-option artifact fingerprints.
        if config_state['heatmaps'] is None:
            del config_state['heatmaps']  # Preserve existing scalar artifact fingerprints.
        else:
            reconstruction = config_state['heatmaps']['encoder']['reconstruction']
            if reconstruction is not None and reconstruction.get('loss') == 'mse':
                del reconstruction['loss']  # MSE was the original auxiliary default.
        self.model_version = sha256((digest.hexdigest() + json.dumps(
            config_state, sort_keys=True)).encode()).hexdigest()

    def snapshot(self):
        return deepcopy(self.model.state_dict())

    def restore(self, state):
        self.model.load_state_dict(state)
        self.refresh_version()

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    def set_learning_rate(self, rate):
        for group in self.optimizer.param_groups:
            group["lr"] = rate

    def _boundary(self, issue):
        if len(issue) and min(self.config.history_cutoff(t) for t in issue) < self.trained_through:
            raise ValueError("Requested issuance precedes the fitted/validation-selection boundary.")

    def _statistic_arrays(self, size):
        arrays = {}
        for t in self.config.statistics_forecast.targets:
            fields = ['mean']
            if t.family != 'point': fields.append('std')
            if t.family == 'negative_binomial': fields.append('dispersion')
            if t.family == 'inflated_beta': fields += ['a', 'b', 'p_zero', 'p_one', 'p_interior']
            for field in fields: arrays[t.name+'::'+field] = np.full(size, np.nan)
        return arrays

    def _predict_group(self, group, output, statistics=None):
        positions, data, home, away, fixture = group
        if statistics is None:
            logits = self.model(data, home, away, fixture)
        else:
            logits, _, forecasts = self.model(data, home, away, fixture, return_statistics=True)
            for target, values in zip(self.config.statistics_forecast.targets, forecasts):
                values = dict(values)
                if 'log_mixture' in values:
                    mixture = values.pop('log_mixture').exp()
                    values.update(zip(('p_zero','p_one','p_interior'), mixture.unbind(1)))
                for field, value in values.items():
                    if value is not None:
                        factor = 100 if target.proportion and field in ('mean','std') else 1
                        statistics[target.name+'::'+field][positions] = value.cpu().numpy()*factor
        output[positions] = logits.softmax(-1).cpu().numpy()

    @configured_execution
    @release_temporary_frames
    def predict(self, context):
        torch = _torch()
        issue = self._validate(context)
        self._boundary(issue)
        output = np.empty((len(context.X), 3))
        statistics = self._statistic_arrays(len(context.X)) if self.config.statistics_forecast is not None else None
        with inference_mode(self.model):
            groups = self._stream(context)
            replay = self._replay(groups)
            for i in range(len(groups)):
                self._predict_group(replay.group(groups, i) if replay is not None else groups[i], output, statistics)
        columns = pd.MultiIndex.from_product([[self.target_name], self.output_classes], names=['target', 'class'])
        probabilities = pd.DataFrame(output, index=context.X.index, columns=columns)
        predicted = pd.DataFrame({self.target_name: [self.output_classes[i] for i in output.argmax(1)]}, index=context.X.index)
        result = {"predict_proba": probabilities, "predict": predicted}
        if statistics is not None:
            result['statistics'] = pd.DataFrame(statistics, index=context.X.index)
        return result

    def _encode_group(self, time, positions, queries, records, replay=None):
        teams = queries.iloc[positions].team_id.tolist()
        if replay is None:
            snap = self.store.snapshot(time, self.config, teams)
            data = self.preprocessor.tensors(snap, self.device)
        else:
            snap, data = replay.snapshot(time, teams)
        z = self.model.encode(data).cpu().numpy()
        priors = data.get("initial_state", self.model.prior(data["league"], data["entry"])).cpu().numpy()
        lookup = {t: n for n, t in enumerate(snap.teams)}
        for pos, team in zip(positions, teams):
            i = lookup[team]
            records[pos] = dict(team_id=team, issuance_at=time, history_cutoff=snap.cutoff,
                support=snap.support[i], gate=snap.gate[i], league=snap.league[i], entry=snap.entry[i],
                evidence=snap.evidence(team, self.config.layers), model_version=self.model_version,
                event_store_version=self.store.version, forecast_policy=self.config.forecast_policy,
                trained_through=self.trained_through,
                **{f"z_{j}": float(v) for j, v in enumerate(z[i])},
                **{f"prior_{j}": float(v) for j, v in enumerate(priors[i])})
            if replay is not None:
                records[pos].update(boundary_at=replay.start, evidence_scope='explicit_segment_edges',
                                    state_mode='detached_boundary_memory')

    @configured_execution
    @release_temporary_frames
    def encode_teams(self, queries):
        torch = _torch()
        if not queries.index.is_unique:
            raise ValueError("Queries require a unique index.")
        _ids(queries, ("team_id",))
        issue = _times(queries.issuance_at)
        self._boundary(issue)
        records = [None] * len(queries)
        with inference_mode(self.model):
            from .memory import BoundaryReplay
            replay = BoundaryReplay(self, issue.max()) if self.config.boundary_memory is not None and len(issue) else None
            for time, positions in issue.groupby(issue, sort=True).indices.items():
                self._encode_group(time, positions, queries, records, replay)
        frame = pd.DataFrame(records, index=queries.index)
        if len(frame):
            # Preserve IDs even when mixed with nullable context fields.
            frame["team_id"] = pd.array(queries.team_id.tolist(), dtype="UInt64")
            frame["league"] = pd.array([r["league"] for r in records], dtype="UInt64")
        return frame

    @configured_execution
    def predict_statistics(self, context):
        if self.config.statistics_forecast is None:
            raise ValueError('This model has no statistics forecast head.')
        return self.predict(context)['statistics']
