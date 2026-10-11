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


class TemporalGraphAdapter:
    supports_sample_weight = True

    def __init__(self, config, event_store, *, control=None, device="cpu", inner_threads=1):
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
        self.configure_device(self.device)
        self.configure_threads(self.inner_threads)
        with torch_threads(self.inner_threads):
            self.backend_, self.training_history_, self.training_summary_ = run_iterations(
                lambda seed: _Backend(self.config, self.event_store, seed, self.device),
                context, control or self.control)
        self.training_summary_.update(
            fitting_boundary=self.backend_.fitting_boundary.isoformat(),
            trained_through=self.backend_.trained_through.isoformat(),
            model_version=self.backend_.model_version,
            event_store_version=self.event_store.version,
            forecast_policy=self.config.forecast_policy,
            timezone=self.config.timezone)

    def predict(self, context):
        with torch_threads(self.inner_threads):
            return self.backend_.predict(context)

    def encode_teams(self, queries):
        """queries: team_id and timezone-aware issuance_at; preserve index/order."""
        with torch_threads(self.inner_threads):
            return self.backend_.encode_teams(queries)

    def attach_event_store(self, event_store):
        """Explicit replacement for later observations; trained parameters stay frozen.

        Existing source events/memberships must be preserved. New fixtures and
        new historical evidence may be added, but revised source data requires
        a new fitting run instead of silently changing artifact provenance.
        """
        old = self.event_store
        for before, after, keys in (
            (old.history, event_store.history, [*MATCH_KEYS, "side"]),
            (old.team_seasons, event_store.team_seasons, ["team_id", "competition_id", "season_id"]),
        ):
            left = before.set_index(keys)
            right = after.set_index(keys)
            try:
                pd.testing.assert_frame_equal(left, right.loc[left.index, left.columns],
                                              check_dtype=False, check_like=True)
            except (KeyError, AssertionError) as exc:
                raise ValueError("Replacement store must preserve existing evidence.") from exc
        if self.config.forecast_policy == "by_round":
            before = old.issue_times(old.schedule, self.config)
            after = event_store.issue_times(old.schedule, self.config)
            if not before.equals(after):
                raise ValueError("Replacement store changes frozen round anchors.")
        self.event_store = event_store
        self.backend_.store = event_store


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
            if not context.y.iloc[:, 0].isin(CLASS_ORDER).all():
                raise ValueError("Target labels must be home, draw or away.")
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

    def _raw_groups(self, context):
        issue = self._validate(context)
        groups, numeric_all = [], np.empty((len(context.X), 5+len(self.config.fixture_features)))
        for time, positions in issue.groupby(issue, sort=True).indices.items():
            meta = context.metadata.iloc[positions]
            teams = set(meta.home_id.tolist()) | set(meta.away_id.tolist())
            snap = self.store.snapshot(time, self.config, teams)
            indices = {t: i for i, t in enumerate(snap.teams)}
            home = np.array([indices[t] for t in meta.home_id], dtype=np.int64)
            away = np.array([indices[t] for t in meta.away_id], dtype=np.int64)
            latest = snap.edges.groupby("team_id").kickoff_at.max()
            def rest(t):
                return (time-latest[t]).total_seconds()/86400 if t in latest else np.nan
            neutral = meta["neutral_venue"].to_numpy(dtype=float, na_value=np.nan) if "neutral_venue" in meta else np.full(len(meta), np.nan)
            if np.any(np.isfinite(neutral) & ~np.isin(neutral, [0, 1])):
                raise ValueError("neutral_venue must be 0, 1 or missing.")
            numeric = np.column_stack((neutral, [rest(t) for t in meta.home_id],
                                       [rest(t) for t in meta.away_id],
                                       snap.support[home], snap.support[away]))
            if self.config.fixture_features:
                numeric = np.column_stack((numeric, context.X.iloc[positions][list(self.config.fixture_features)].to_numpy(dtype=float, na_value=np.nan)))
            numeric_all[positions] = numeric
            groups.append((positions, snap, home, away, meta, numeric))
        return groups, numeric_all

    def _prepare(self, raw):
        torch = _torch()
        groups = []
        for positions, snap, home, away, meta, numeric in raw:
            groups.append((positions, self.preprocessor.tensors(snap, self.device),
                           torch.as_tensor(home, device=self.device),
                           torch.as_tensor(away, device=self.device),
                           torch.as_tensor(self.preprocessor.fixture(meta, numeric), dtype=torch.float32, device=self.device)))
        return groups

    def initialize(self, context):
        torch = _torch()
        from .network import TemporalNetwork
        from ..data.availability import respect_result_availability
        self._validate(context, fitting=True)
        if len(context.X) == 0:
            raise ValueError("Fitting population cannot be empty.")
        self.target_name = context.y.columns[0]
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
        raw, numeric = self._raw_groups(context)
        largest = max(snap.cutoff for _, snap, *_ in raw)
        fitting_history = self.store.eligible(largest)
        self.preprocessor = GraphPreprocessor(self.config).fit(
            fitting_history, [g[1] for g in raw], context.metadata, numeric)
        # fork_rng avoids altering the caller's CPU random stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.model = TemporalNetwork(self.config, len(self.preprocessor.leagues),
                self.preprocessor.edges(raw[0][1]).shape[1],
                self.preprocessor.fixture(context.metadata, numeric).shape[1]).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate,
                                           weight_decay=self.config.weight_decay)
        self.train_groups = self._prepare(raw)
        self.train_targets, self.train_weights = self._targets(context)
        self.validation_groups = None
        if context.validation is not None:
            self.validation_groups = self._prepare(self._raw_groups(context.validation)[0])
            self.validation_targets, self.validation_weights = self._targets(context.validation)
        self.refresh_version()

    def _targets(self, context):
        torch = _torch()
        y = torch.tensor([CLASS_ORDER.index(x) for x in context.y.iloc[:, 0]], device=self.device)
        w = np.ones(len(y))
        if context.sample_weight is not None:
            if isinstance(context.sample_weight, (pd.Series, pd.DataFrame)) and not context.sample_weight.index.equals(context.X.index):
                raise ValueError("Observation weights must align with X.")
            w = np.asarray(context.sample_weight, dtype=float).reshape(-1)
        if len(w) != len(y) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("Observation weights must be finite, nonnegative and have positive total.")
        return y, torch.tensor(w.copy(), dtype=torch.float32, device=self.device)

    def _loss(self, groups, targets, weights, backward=False):
        torch = _torch()
        value = 0.
        for positions, data, home, away, fixture in groups:
            logits = self.model(data, home, away, fixture)
            ix = torch.as_tensor(positions, device=self.device)
            loss = (torch.nn.functional.cross_entropy(logits, targets[ix], reduction="none") * weights[ix]).sum()/weights.sum()
            if backward:
                loss.backward()
            value += float(loss.detach())
        return value

    def step(self):
        torch = _torch()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        train_loss = self._loss(self.train_groups, self.train_targets, self.train_weights, True)
        penalty = self.config.prior_penalty * (self.model.league.weight.square().sum() + self.model.entry.weight.square().sum())
        penalty.backward()
        self.optimizer.step()
        values = {"train_loss": train_loss}
        if self.validation_groups is not None:
            self.model.eval()
            with torch.no_grad():
                values["validation_loss"] = self._loss(self.validation_groups, self.validation_targets, self.validation_weights)
        self.refresh_version()
        return values

    def refresh_version(self):
        digest = sha256()
        for name, value in self.model.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        digest.update(json.dumps(self.preprocessor.state(), sort_keys=True).encode())
        # A representation version includes decay, policy and architectural
        # settings as well as learned coordinates and preprocessing.
        self.model_version = sha256((digest.hexdigest() + json.dumps(
            asdict(self.config), sort_keys=True)).encode()).hexdigest()

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

    def predict(self, context):
        torch = _torch()
        issue = self._validate(context)
        self._boundary(issue)
        output = np.empty((len(context.X), 3))
        self.model.eval()
        with torch.no_grad():
            for positions, data, home, away, fixture in self._prepare(self._raw_groups(context)[0]):
                output[positions] = self.model(data, home, away, fixture).softmax(-1).cpu().numpy()
        probabilities = pd.DataFrame(output, index=context.X.index, columns=CLASS_ORDER)
        predicted = pd.DataFrame({self.target_name: [CLASS_ORDER[i] for i in output.argmax(1)]}, index=context.X.index)
        return {"predict_proba": probabilities, "predict": predicted}

    def encode_teams(self, queries):
        torch = _torch()
        if not queries.index.is_unique:
            raise ValueError("Queries require a unique index.")
        _ids(queries, ("team_id",))
        issue = _times(queries.issuance_at)
        self._boundary(issue)
        records = [None] * len(queries)
        self.model.eval()
        with torch.no_grad():
            for time, positions in issue.groupby(issue, sort=True).indices.items():
                teams = queries.iloc[positions].team_id.tolist()
                snap = self.store.snapshot(time, self.config, teams)
                data = self.preprocessor.tensors(snap, self.device)
                z = self.model.encode(data).cpu().numpy()
                priors = self.model.prior(data["league"], data["entry"]).cpu().numpy()
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
        frame = pd.DataFrame(records, index=queries.index)
        if len(frame):
            # Preserve IDs even when mixed with nullable context fields.
            frame["team_id"] = pd.array(queries.team_id.tolist(), dtype="UInt64")
            frame["league"] = pd.array([r["league"] for r in records], dtype="UInt64")
        return frame
