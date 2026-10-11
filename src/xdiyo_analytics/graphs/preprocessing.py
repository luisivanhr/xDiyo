"""Fit-only numeric transforms and categorical maps; missing values get masks."""
import numpy as np
from .config import ENTRY_ORDER


class MaskedScale:
    def fit(self, values):
        a = np.asarray(values, dtype=float)
        if np.isinf(a).any():
            raise ValueError("Infinite features are not supported.")
        count = np.isfinite(a).sum(0)
        self.mean = np.nansum(a, axis=0) / np.maximum(count, 1)
        variance = np.nansum((a-self.mean)**2, axis=0) / np.maximum(count, 1)
        self.scale = np.where(variance > 0, np.sqrt(variance), 1.)
        return self

    def transform(self, values):
        a = np.asarray(values, dtype=float)
        if np.isinf(a).any():
            raise ValueError("Infinite features are not supported.")
        missing = np.isnan(a)
        filled = np.where(missing, self.mean, a)
        return np.concatenate(((filled-self.mean)/self.scale, missing.astype(float)), axis=1)

    def state(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state(cls, state):
        obj = cls()
        obj.mean, obj.scale = np.array(state["mean"]), np.array(state["scale"])
        return obj


class GraphPreprocessor:
    def __init__(self, config):
        self.config = config
        self.edge_columns = ("goals_for", "goals_against", *config.statistics)
        self.fixture_columns = ("neutral_venue", "home_rest_days", "away_rest_days",
                                "home_support", "away_support", *config.fixture_features)

    def numeric_edges(self, frame):
        if not set(self.edge_columns) <= set(frame):
            raise ValueError("Selected match-level statistics are absent from history.")
        # Permit only paired native match-stat columns, never rolling histories.
        for col in self.config.statistics:
            if not col.startswith(("team::", "opponent::")):
                raise ValueError("statistics must select build_team_history match-stat columns.")
            role, rest = col.split("::", 1)
            paired = ("opponent" if role == "team" else "team") + "::" + rest
            if paired not in frame:
                raise ValueError("Selected statistics require both paired match perspectives.")
        return frame[list(self.edge_columns)].to_numpy(dtype=float, na_value=np.nan)

    def fit(self, history, snapshots, fixtures, numeric, *, extra_leagues=()):
        categories = set(history.competition_id.tolist()) | set(fixtures.competition_id.tolist()) | set(extra_leagues)
        for snap in snapshots:
            categories.update(x for x in snap.league if x is not None)
        self.leagues = {int(key): n+1 for n, key in enumerate(sorted(categories))}
        self.edge_scale = MaskedScale().fit(self.numeric_edges(history))
        self.fixture_scale = MaskedScale().fit(numeric)
        return self

    def one_hot(self, leagues):
        return np.eye(len(self.leagues)+1)[[self.leagues.get(x, 0) for x in leagues]]

    def edges(self, snapshot):
        h = snapshot.edges
        return np.concatenate((self.edge_scale.transform(self.numeric_edges(h)),
                               h.side.eq("home").to_numpy(dtype=float)[:, None],
                               self.one_hot(h.competition_id)), axis=1)

    def fixture(self, fixtures, numeric):
        return np.concatenate((self.fixture_scale.transform(numeric),
                               self.one_hot(fixtures.competition_id)), axis=1)

    def tensors(self, snapshot, device):
        import torch
        def tensor(a, dtype=torch.float32):
            return torch.as_tensor(a, dtype=dtype, device=device)
        return dict(x=tensor(self.edges(snapshot)),
                    receiver=tensor(snapshot.receiver, torch.long),
                    sender=tensor(snapshot.sender, torch.long),
                    log_weight=tensor(snapshot.log_weight, torch.float64),
                    gate=tensor(snapshot.gate),
                    league=tensor([self.leagues.get(x, 0) for x in snapshot.league], torch.long),
                    entry=tensor([ENTRY_ORDER.index(x) for x in snapshot.entry], torch.long))

    def state(self):
        return dict(leagues=list(self.leagues.items()), edge_scale=self.edge_scale.state(),
                    fixture_scale=self.fixture_scale.state())

    @classmethod
    def from_state(cls, config, state):
        obj = cls(config)
        obj.leagues = dict(state["leagues"])
        obj.edge_scale = MaskedScale.from_state(state["edge_scale"])
        obj.fixture_scale = MaskedScale.from_state(state["fixture_scale"])
        return obj
