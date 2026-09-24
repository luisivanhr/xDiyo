"""Reusable numeric rating snapshots, storage and prediction-time lookup."""

from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class RatingRun:
    """State history independent of a particular model's feature selection.

    Other engines (including future embedding producers) can supply the same
    table: scope columns, stream, team_id, recorded_at, latest_kickoff_at, and
    numeric state fields. State producers own their update/training semantics.
    Every snapshot must summarize only events known by recorded_at; its latest
    contributing kickoff is latest_kickoff_at. This is not a pretrained GAT.
    """

    snapshots: object
    initial_state: dict
    scope: tuple = ("competition_id",)
    streams: tuple = ("result",)
    metadata: dict = field(default_factory=dict)

    def save(self, directory):
        """Save reusable snapshots and their definition; refuse existing artifacts."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        table, meta = path / "snapshots.parquet", path / "ratings.json"
        if table.exists() or meta.exists():
            raise FileExistsError("Rating artifacts already exist in this directory.")
        description = {"initial_state": self.initial_state, "scope": self.scope,
                       "streams": self.streams, "metadata": self.metadata}
        serialized = json.dumps(description, indent=2, allow_nan=False)
        self.snapshots.to_parquet(table, index=False)
        meta.write_text(serialized + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory):
        """Read a saved run without recomputing ratings."""
        import pandas as pd
        path = Path(directory)
        description = json.loads((path / "ratings.json").read_text(encoding="utf-8"))
        return cls(pd.read_parquet(path / "snapshots.parquet"),
                   description["initial_state"], tuple(description["scope"]),
                   tuple(description["streams"]), description["metadata"])

    def features(self, history, *, cutoffs=None, side="both", fields=None):
        """Lookup earlier states and align them with every training/prediction row.

        Unrated teams receive the configured initial state. Missing query dates
        return missing features. Optional cutoffs default to the row's kickoff.
        No additional inactivity inflation is applied by lookup.
        """
        import numpy as np
        import pandas as pd
        from ..features.history import aligned_times

        roles = {"for": ("team",), "against": ("opponent",), "both": ("team", "opponent")}
        if side not in roles:
            raise ValueError("side must be for, against or both.")
        fields = tuple(self.initial_state) if fields is None else tuple(fields)
        if not fields or len(set(fields)) != len(fields) or set(fields) - set(self.initial_state):
            raise ValueError("Select distinct fields from the saved rating state.")
        cutoff = aligned_times(history, cutoffs, default="kickoff_at")
        kickoff = aligned_times(history, None, default="kickoff_at")
        if (cutoff > kickoff).any():
            raise ValueError("Prediction cutoffs must not follow target kickoff.")
        if history[list(self.scope)].isna().any().any():
            raise ValueError("Rating scope identifiers must be nonmissing.")
        keys = ["stream", *self.scope, "team_id"]
        groups = {}
        for key, frame in self.snapshots.groupby(keys, sort=False, dropna=False):
            order = ["recorded_at"] + (["snapshot_order"] if "snapshot_order" in frame else [])
            frame = frame.sort_values(order, kind="stable")
            if frame.duplicated(order).any():
                raise ValueError("Duplicate snapshots for a team at one time.")
            groups[key] = (
                pd.to_datetime(frame["recorded_at"], utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy(),
                pd.to_datetime(frame["latest_kickoff_at"], utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy(),
                frame[list(fields)].to_numpy(dtype=float),
            )
        scopes = list(zip(*(history[name].tolist() for name in self.scope))) if self.scope else [()] * len(history)
        cutoff_ns = cutoff.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        valid = cutoff.notna().to_numpy() & kickoff.notna().to_numpy()
        initial = np.array([self.initial_state[name] for name in fields], dtype=float)
        output = {}
        for stream in self.streams:
            for role in roles[side]:
                values = np.full((len(history), len(fields)), np.nan)
                teams = history[f"{role}_id"].tolist()
                for row, (scope, team) in enumerate(zip(scopes, teams)):
                    if not valid[row]:
                        continue
                    values[row] = initial
                    group = groups.get((stream, *scope, team))
                    if group is None:
                        continue
                    times, latest, states = group
                    at = np.searchsorted(times, cutoff_ns[row], side="right") - 1
                    # Equal-time result releases are usable only if their games
                    # already kicked off strictly before the query boundary.
                    while at >= 0 and latest[at] >= cutoff_ns[row]:
                        at -= 1
                    if at >= 0:
                        values[row] = states[at]
                for col, name in enumerate(fields):
                    output[f"{stream}::{role}::{name}"] = values[:, col]
        return pd.DataFrame(output, index=history.index)
