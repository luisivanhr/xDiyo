"""Local experiment artifacts and reusable comparison rules."""

from .store import ExperimentStore, configuration_hash
from .ranking import rank_runs

__all__ = ["ExperimentStore", "configuration_hash", "rank_runs", "FootballExperiment", "PreparedExperiment",
           "ExperimentResult", "RefitPolicy", "SelectionSummary"]


def __getattr__(name):
    if name in {"FootballExperiment", "PreparedExperiment", "ExperimentResult", "RefitPolicy", "SelectionSummary"}:
        from . import football
        return getattr(football, name)
    raise AttributeError(name)
