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
    weights: object = None


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
    fold_metadata_by_id: dict = field(default_factory=dict)

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
    scope: object = None
    scope_label: str | None = None


@dataclass
class PostTrainingContext:
    """One prediction population, indexed by (fold_id, row_position).

    Models are references for read-only inspection, not refitting. Frame inputs
    and definitions are copied per study. pooling declares how repeated held-out
    predictions were retained/combined. experiment is an optional ExperimentStore.
    """
    y: pd.DataFrame
    predictions: dict[str, pd.DataFrame]
    metadata: pd.DataFrame
    layout: str
    identity_columns: tuple[str, ...]
    match_columns: tuple[str, ...]
    type: str
    partition: str
    fold_id: object = None
    pooling: str = "occurrences"
    models: dict = field(default_factory=dict)
    definitions: dict = field(default_factory=dict)
    experiment: object = None
    fold_results: dict = field(default_factory=dict)
    previous_results: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)

    @property
    def row_positions(self):
        return self.y.index.get_level_values("row_position").to_numpy()

    @property
    def n_matches(self):
        return len(self.metadata[list(self.match_columns)].drop_duplicates()) if len(self.metadata) else 0


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

    def to_html(self, path=None, *, renderers=None, spatial_transport=None, spatial_limit=None):
        """Return a standalone HTML document and optionally write it locally."""
        from .viewer import render_report
        if spatial_limit is not None and (isinstance(spatial_limit, bool) or not isinstance(spatial_limit, int) or spatial_limit < 1):
            raise ValueError('spatial_limit must be a positive integer.')
        return render_report(self, path=path, renderers=renderers,
                             spatial_transport=spatial_transport, spatial_limit=spatial_limit)

    def live(self):
        """Serve computed spatial values on demand; close the returned handle when done."""
        from .spatial_live import live_report
        return live_report(self)

    def close_live(self):
        from .spatial_live import close_report
        close_report(self)

    def _notebook_html(self, *, height=800, renderers=None):
        return self.to_notebook(height=height, renderers=renderers)._repr_html_()

    def to_notebook(self, *, height=800, renderers=None, live=None):
        """Return an IPython IFrame display object containing the interactive report.

        Spatial reports use a token-protected loopback server by default, fetching
        only the selected fixture's stored values. close_live releases it. This
        expects a local notebook/browser; live=False embeds all data for remote
        or offline notebooks. Ordinary reports remain self-contained. Use a
        trusted frontend allowing JavaScript. No analysis or fitting is rerun.
        """
        from html import escape
        from IPython.display import IFrame
        if isinstance(height, bool) or not isinstance(height, (int, np.integer)) or height < 1:
            raise ValueError("Notebook report height must be a positive integer in pixels.")
        if live is None:
            live = any(a.kind == 'spatial' for s in self.studies for a in s.result.artifacts)
        if live:
            if renderers:
                raise ValueError('Custom notebook renderers currently require live=False.')
            return IFrame(self.live().url, width='100%', height=int(height))
        document = escape(self.to_html(renderers=renderers), quote=True)
        return IFrame("about:blank", width="100%", height=int(height),
                      extras=[f'title="{escape(self.title, quote=True)}"',
                              'style="border:0;border-radius:12px"', f'srcdoc="{document}"'])

    def show(self, *, height=800, renderers=None, live=None):
        """Display inline in Jupyter/IPython; returns None to avoid duplicate output."""
        from IPython.display import display
        display(self.to_notebook(height=height, renderers=renderers, live=live))

    def _repr_html_(self):
        """Automatic rich display when the report is the notebook cell's last value."""
        return self._notebook_html()
