"""Declarative construction from the ordinary prepared history and movements."""
from dataclasses import dataclass, field
import pandas as pd

from .config import TemporalGraphConfig, MATCH_KEYS
from .store import TemporalEventStore
from .adapter import TemporalGraphAdapter
from .research_training import RetainedGATTraining
from .research_execution import RetainedGATExecution
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
    phase_context_path: str | None = None
    round_stage_column: str | None = None
    research_training: RetainedGATTraining | None = None
    research_execution: RetainedGATExecution | None = None

    def prepare_history(self, history):
        """Copy an explicitly selected native stage column; never infer stages."""
        name = self.round_stage_column
        if name is None:
            return history
        if not isinstance(name, str) or name not in history or history[name].isna().any():
            raise ValueError('round_stage_column must name a complete native history column.')
        if 'round_stage' in history and not history['round_stage'].equals(history[name]):
            raise ValueError('Explicit round stage differs from the supplied history.')
        result = history.copy()
        result['round_stage'] = history[name]
        return result

    def event_store(self, history, memberships, heatmaps=None):
        from .spatial import prepare_heatmaps, grid_identity
        history = self.prepare_history(history)
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
        source = None
        if self.config.phase_rank_features is not None:
            if self.phase_context_path is None:
                raise ValueError('Enabled phase features require phase_context_path.')
            from .phase_context import PhaseRankLookup
            source = PhaseRankLookup.load(self.phase_context_path)
            spec = self.config.phase_rank_features
            if source.source_id != spec.source_id or source.policy != spec.policy:
                raise ValueError('Phase context source or policy disagrees with configuration.')
        elif self.phase_context_path is not None:
            raise ValueError('phase_context_path requires enabled phase features.')
        return TemporalEventStore(history, memberships, phase_context=source, provenance={
            'membership_policy': self.membership_policy,
            'memberships_path': self.memberships_path,
            **({'round_stage_column': self.round_stage_column} if self.round_stage_column is not None else {}),
            'cutoff_config': {name: getattr(self.config, name) for name in ('timezone', 'forecast_policy', 'round_keys')},
            **({'heatmap_grid': grid_identity(self.config.heatmaps)} if self.config.heatmaps is not None else {}),
        })

    def build(self, store):
        from .spatial import validate_grid_contract
        validate_grid_contract(store, self.config.heatmaps)
        for key in ('membership_policy', 'memberships_path', 'round_stage_column'):
            if key in store.provenance and store.provenance[key] != getattr(self, key):
                raise ValueError('Graph membership settings change preparation; use a separate recipe.')
        expected = store.provenance.get('cutoff_config')
        if expected is not None:
            actual = {name: getattr(self.config, name) for name in expected}
            actual['round_keys'] = list(actual['round_keys'])
            if actual != expected:
                raise ValueError('Graph cutoff settings change preparation; use a separate recipe.')
        from .preprocessing import GraphPreprocessor
        GraphPreprocessor(self.config).bind_context(store)
        if self.research_training is not None:
            if not isinstance(self.research_training, RetainedGATTraining):
                raise TypeError('research_training must be RetainedGATTraining.')
            self.research_training.validate_control(self.control)
            GraphPreprocessor(self.config).numeric_edges(store._history.iloc[:0])
        elif self.research_execution is not None:
            raise ValueError('Research execution requires an explicit retained training contract.')
        if self.research_execution is not None:
            if not isinstance(self.research_execution, RetainedGATExecution):
                raise TypeError('research_execution must be RetainedGATExecution.')
            self.research_execution.validate_config(self.config)
        return TemporalGraphAdapter(self.config, store, control=self.control,
            research_training=self.research_training, research_execution=self.research_execution)


def fixture_cutoffs(store, config, metadata):
    issue = store.issue_times(metadata, config)
    return pd.Series([config.history_cutoff(t) for t in issue], index=metadata.index)


def history_cutoffs(store, config, history):
    schedule = store.schedule
    lookup = dict(zip(schedule[list(MATCH_KEYS)].itertuples(index=False, name=None),
                      fixture_cutoffs(store, config, schedule)))
    return pd.Series([lookup[k] for k in history[list(MATCH_KEYS)].itertuples(index=False, name=None)],
                     index=history.index)
