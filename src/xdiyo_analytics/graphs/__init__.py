"""Cutoff-safe temporal team graphs. Importing this package does not import torch."""
from .config import TemporalGraphConfig, CLASS_ORDER
from .store import TemporalEventStore, GraphSnapshot
from .adapter import TemporalGraphAdapter
from .persistence import TemporalGraphSerializer

__all__ = ["TemporalGraphConfig", "TemporalEventStore", "GraphSnapshot",
           "TemporalGraphAdapter", "TemporalGraphSerializer", "CLASS_ORDER"]

from .integration import TemporalGraphModel
from .spatial import GraphHeatmaps
__all__ += ["TemporalGraphModel", "GraphHeatmaps"]
from .profiling import GraphProfile
__all__ += ["GraphProfile"]

from .extensions import BoundaryMemory, StatisticTarget, NumericStatisticTarget, StatisticsForecast, GoalAuxiliary
__all__ += ["BoundaryMemory", "StatisticTarget", "NumericStatisticTarget", "StatisticsForecast", "GoalAuxiliary"]
from .phase_context import PhaseRankFeatures, PhaseRankLookup
__all__ += ["PhaseRankFeatures", "PhaseRankLookup"]

from .research_profiles import GATResearchBNV1, GATResearchCE1V1, GATResearchBCDDWV1
__all__ += ["GATResearchBNV1", "GATResearchCE1V1", "GATResearchBCDDWV1"]

from .research_training import RetainedGATTraining
__all__ += ["RetainedGATTraining"]

from .phase_preparation import build_phase_context
__all__ += ["build_phase_context"]

from .research_execution import RetainedGATExecution
__all__ += ["RetainedGATExecution"]
