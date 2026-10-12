"""Declarative construction from the ordinary prepared history and movements."""
from dataclasses import dataclass, field
import pandas as pd

from .config import TemporalGraphConfig, MATCH_KEYS
from .store import TemporalEventStore
from .adapter import TemporalGraphAdapter
from ..training.control import TrainingControl


@dataclass(frozen=True)
class TemporalGraphModel:
    """UI model specification; preparation supplies immutable graph inputs.

    first_fixture declares each competition-season's first scheduled kickoff as
    its effective membership boundary. This is retrospective schedule context,
    not evidence of when a promotion announcement was published. explicit uses
    the selected membership file's timezone-aware entry_at values instead.
    """
    config: TemporalGraphConfig = field(default_factory=lambda: TemporalGraphConfig('UTC'))
    control: TrainingControl = field(default_factory=TrainingControl)
    membership_policy: str = 'first_fixture'
    memberships_path: str | None = None

    def event_store(self, history, memberships, heatmaps=None):
        from .spatial import prepare_heatmaps, grid_identity
        if self.config.heatmaps is not None:
            history = prepare_heatmaps(history, heatmaps, self.config.heatmaps)
        if self.memberships_path:
            from pathlib import Path
            path = Path(self.memberships_path)
            memberships = pd.read_parquet(path) if path.suffix.lower() == '.parquet' else pd.read_csv(path)
        if memberships is None:
            raise ValueError('Temporal graph needs the team_seasons table or a membership file.')
        memberships = memberships.copy()
        if self.membership_policy == 'first_fixture':
            anchors = history.groupby(['competition_id', 'season_id']).kickoff_at.min()
            keys = pd.MultiIndex.from_frame(memberships[['competition_id', 'season_id']])
            memberships['entry_at'] = anchors.reindex(keys).to_numpy()
            # Reindexing timezone-aware values must preserve their timezone.
            memberships['entry_at'] = pd.to_datetime(memberships.entry_at, utc=True)
        elif self.membership_policy != 'explicit':
            raise ValueError('membership_policy must be first_fixture or explicit.')
        return TemporalEventStore(history, memberships, provenance={
            'membership_policy': self.membership_policy,
            'memberships_path': self.memberships_path,
            'cutoff_config': {name: getattr(self.config, name) for name in ('timezone', 'forecast_policy', 'round_keys')},
            **({'heatmap_grid': grid_identity(self.config.heatmaps)} if self.config.heatmaps is not None else {}),
        })

    def build(self, store):
        from .spatial import validate_grid_contract
        validate_grid_contract(store, self.config.heatmaps)
        for key in ('membership_policy', 'memberships_path'):
            if key in store.provenance and store.provenance[key] != getattr(self, key):
                raise ValueError('Graph membership settings change preparation; use a separate recipe.')
        expected = store.provenance.get('cutoff_config')
        if expected is not None:
            actual = {name: getattr(self.config, name) for name in expected}
            actual['round_keys'] = list(actual['round_keys'])
            if actual != expected:
                raise ValueError('Graph cutoff settings change preparation; use a separate recipe.')
        return TemporalGraphAdapter(self.config, store, control=self.control)


def fixture_cutoffs(store, config, metadata):
    issue = store.issue_times(metadata, config)
    return pd.Series([config.history_cutoff(t) for t in issue], index=metadata.index)


def history_cutoffs(store, config, history):
    schedule = store.schedule
    lookup = dict(zip(schedule[list(MATCH_KEYS)].itertuples(index=False, name=None),
                      fixture_cutoffs(store, config, schedule)))
    return pd.Series([lookup[k] for k in history[list(MATCH_KEYS)].itertuples(index=False, name=None)],
                     index=history.index)
