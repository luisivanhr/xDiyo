"""Small shared contracts for analysis studies and their display artifacts."""

from dataclasses import dataclass, field
from typing import ClassVar, Protocol

import numpy as np
import pandas as pd


@dataclass
class Artifact:
    """One display item, independent of study layout.

    Built-in kinds are text, table, plotly, html and leaderboard. Custom kinds
    use a renderer passed to AnalysisReport.to_html(renderers={kind: callable}).
    A renderer receives this artifact and returns an HTML fragment. html/custom
    renderers are trusted local code. Ordinary text/table labels are escaped.
    options contains kind-specific presentation settings, never execution scope.
    """

    kind: str
    data: object
    title: str = ""
    options: dict = field(default_factory=dict)


@dataclass
class FeatureSelection:
    """Explicit selected feature names; no mutation of the input dataset.

    ranking retains the calculation that led to the selection. The orchestrator
    keeps this result with its study/fold/row scope for later training consumers.
    """

    columns: tuple[str, ...]
    ranking: pd.DataFrame


@dataclass
class StudyResult:
    """A reporter's reusable numerical outputs and independently chosen visuals."""

    title: str
    artifacts: list[Artifact] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    selection: object = None


@dataclass
class AnalysisContext:
    """Selected rows supplied to one reporter invocation, not the entire dataset.

    X, y and metadata are isolated copies with ORIGINAL dataset-position indexes.
    They retain the declared row layout. row_positions is ordered chronologically
    for type=timeline and otherwise follows original input positions. fold_id is
    zero-based or None. Partition unions never duplicate repeated CV occurrences.
    No fitted preprocessing or implicit team-to-match aggregation takes place.
    """

    X: pd.DataFrame
    y: pd.DataFrame
    metadata: pd.DataFrame
    layout: str
    match_columns: tuple[str, ...]
    row_positions: np.ndarray
    type: str
    partition: str
    fold_id: object = None
    fold_metadata: dict = field(default_factory=dict)
    definitions: dict = field(default_factory=dict)
    previous_results: dict = field(default_factory=dict)
    fold_rows: dict = field(default_factory=dict)
    fold_results: dict = field(default_factory=dict)

    @property
    def n_matches(self):
        return len(self.metadata[list(self.match_columns)].drop_duplicates())


class Reporter(Protocol):
    """Structural interface: subclassing or registration is not required.

    Both type and partition are explicit choices, with no orchestrator defaults.
    supported_types lets reporters declare supported execution arrangements.
    run may return any arrangement of typed/custom artifacts and result tables.
    """

    type: str
    partition: str
    supported_types: ClassVar[tuple[str, ...]]

    def run(self, context: AnalysisContext) -> StudyResult: ...


@dataclass
class StudyRun:
    """One named reporter result and the exact scope used to calculate it."""

    name: str
    type: str
    partition: str
    fold_id: object
    layout: str
    row_positions: np.ndarray
    n_matches: int
    result: StudyResult


@dataclass
class AnalysisReport:
    """Computed results, reusable without rerunning studies when rendering.

    An empty run collection is valid. This result/viewer layer is also suitable
    for future post-training analysis; it has no model execution dependency.
    """

    studies: list[StudyRun] = field(default_factory=list)
    title: str = "Pre-training analysis"

    @property
    def selections(self):
        """name -> {fold_id (or None): FeatureSelection}, retaining scope identities."""
        result = {}
        for study in self.studies:
            if study.result.selection is not None:
                result.setdefault(study.name, {})[study.fold_id] = study.result.selection
        return result

    def to_html(self, path=None, *, renderers=None):
        """Return a standalone HTML document and optionally write it locally."""
        from .viewer import render_report
        return render_report(self, path=path, renderers=renderers)
