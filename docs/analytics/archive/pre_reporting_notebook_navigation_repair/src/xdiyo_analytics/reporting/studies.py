"""Initial reporter library: distributions, association rankings and timelines."""

from dataclasses import dataclass, field
from numbers import Real
from typing import ClassVar

import numpy as np
import pandas as pd

from .contracts import Artifact, StudyResult


def _plotting():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError as error:
        raise ImportError("Plots require the xdiyo-collection[reporting] optional dependencies.") from error


def _scipy():
    try:
        from scipy import stats
        return stats
    except ImportError as error:
        raise ImportError("KDE/Kendall require the xdiyo-collection[reporting] optional dependencies.") from error


def _columns(frame, selected, name):
    result = list(frame.columns) if selected is None else ([selected] if isinstance(selected, str) else list(selected))
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must select at least one distinct column.")
    missing = set(result) - set(frame.columns)
    if missing:
        raise KeyError(f"Unknown {name}: {sorted(missing)}")
    return result


def _numeric(series):
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _style(figure, title, x_title, y_title):
    figure.update_layout(template="plotly_dark", title=title, height=400,
                         paper_bgcolor="#17202b", plot_bgcolor="#17202b",
                         font=dict(family="system-ui, sans-serif", color="#e8eef7"),
                         margin=dict(l=65, r=25, t=75, b=65),
                         xaxis_title=x_title, yaxis_title=y_title,
                         legend=dict(orientation="h", y=-0.25))
    return figure


@dataclass(kw_only=True)
class FeatureDistributionReporter:
    """Density histograms and Gaussian KDE for selected feature columns.

    type and partition are required. supports overall/per_fold. Values are
    equally weighted observations in the declared dataset layout; paired team
    rows are not silently averaged. Nonfinite/non-numeric observations are
    excluded with counts. KDE is omitted for fewer than two finite observations
    or zero spread. bins follows numpy.histogram; bandwidth is scipy gaussian_kde's
    Scott/Silverman rule, None (Scott), or a positive finite scalar covariance
    factor; booleans and callables are not accepted. A KDE of counts
    is a smooth visual guide, not a discrete probability mass function.
    """

    type: str
    partition: str
    features: object = None
    features_from: object = None
    bins: object = "auto"
    bandwidth: object = "scott"
    grid_size: int = 256
    supported_types: ClassVar[tuple] = ("per_fold", "overall")

    def run(self, context):
        if self.bandwidth is not None:
            if isinstance(self.bandwidth, str):
                valid_bandwidth = self.bandwidth in {"scott", "silverman"}
            else:
                valid_bandwidth = (isinstance(self.bandwidth, Real) and
                                   not isinstance(self.bandwidth, (bool, np.bool_)) and
                                   np.isfinite(self.bandwidth) and self.bandwidth > 0)
            if not valid_bandwidth:
                raise ValueError("bandwidth must be None, scott, silverman or a positive finite scalar factor.")
        if not len(context.X.columns) and self.features is None:
            return StudyResult("Feature distributions", notes=["No input features selected."])
        if isinstance(self.grid_size, bool) or not isinstance(self.grid_size, (int, np.integer)) or self.grid_size < 2:
            raise ValueError("grid_size must be an integer >= 2.")
        go, stats = _plotting(), _scipy()
        result = StudyResult("Feature distributions", notes=[
            "Density per supplied observation; missing/nonfinite values are excluded.",
            "KDE is a smooth density guide, including for discrete count features."])
        summary, histogram, curves = [], [], []
        for feature in _columns(context.X, self.features, "features"):
            raw = _numeric(context.X[feature])
            values = raw[np.isfinite(raw)]
            figure = go.Figure()
            status = "no_finite_values"
            if len(values):
                counts, edges = np.histogram(values, bins=self.bins)
                widths = np.diff(edges)
                if (widths <= 0).any() or not np.isfinite(edges).all():
                    raise ValueError("Histogram bin edges must be finite and strictly increasing.")
                if counts.sum() != len(values):
                    raise ValueError("Explicit histogram edges must cover every finite observation.")
                density = counts / (len(values) * widths)
                figure.add_trace(go.Bar(x=((edges[:-1] + edges[1:]) / 2).tolist(),
                                        y=density.tolist(), width=widths.tolist(),
                                        marker_color="#58c7b2", opacity=.65, name="Histogram density"))
                histogram.extend(dict(feature=feature, left=left, right=right, count=int(count), density=float(d))
                                 for left, right, count, d in zip(edges[:-1], edges[1:], counts, density))
                status = "insufficient_spread" if len(values) < 2 or np.ptp(values) == 0 else "ok"
                if status == "ok":
                    try:
                        kernel = stats.gaussian_kde(values, bw_method=self.bandwidth)
                        grid = np.linspace(edges[0], edges[-1], self.grid_size)
                        kde = kernel(grid)
                        figure.add_trace(go.Scatter(x=grid.tolist(), y=kde.tolist(), mode="lines",
                                                   line=dict(color="#b9a3ff", width=3), name="Gaussian KDE"))
                        curves.extend(dict(feature=feature, x=float(x), density=float(d)) for x, d in zip(grid, kde))
                    except np.linalg.LinAlgError:
                        status = "singular_kde"
                if status != "ok":
                    figure.add_annotation(text=f"KDE unavailable: {status}", xref="paper", yref="paper",
                                          x=.5, y=.95, showarrow=False)
            else:
                figure.add_annotation(text="No finite observations in this scope", showarrow=False)
            summary.append(dict(feature=feature, rows=len(raw), finite=len(values),
                                excluded=len(raw) - len(values), kde_status=status))
            result.artifacts.append(Artifact("plotly", _style(figure, str(feature), str(feature), "Density"), str(feature)))
        result.tables = {"summary": pd.DataFrame(summary),
                         "histogram": pd.DataFrame(histogram, columns=["feature", "left", "right", "count", "density"]),
                         "kde": pd.DataFrame(curves, columns=["feature", "x", "density"])}
        return result


def _categorical(series, column, declarations, thresholds):
    if column in thresholds:
        threshold = thresholds[column]
        if not np.isfinite(threshold):
            raise ValueError("MCC thresholds must be finite numbers.")
        values = _numeric(series)
        return pd.Series(np.where(np.isfinite(values), (values > threshold).astype(float), np.nan),
                         index=series.index)
    if column in declarations:
        return series.replace([np.inf, -np.inf], np.nan)
    return None


def _mcc(x, y):
    """Multiclass MCC using a shared, explicitly supplied class vocabulary.

    Returns missing for a zero denominator rather than implying an association
    estimate of zero. Binary threshold comparisons use the shared classes 0/1.
    """
    joined = pd.concat([x, y], ignore_index=True)
    codes, classes = pd.factorize(joined, sort=False)
    a, b = codes[:len(x)], codes[len(x):]
    count = float(len(x))
    px, py = np.bincount(a, minlength=len(classes)), np.bincount(b, minlength=len(classes))
    denom = np.sqrt((count * count - np.dot(px.astype(float), px)) *
                    (count * count - np.dot(py.astype(float), py)))
    return np.nan if denom == 0 else (count * float(np.sum(a == b)) - np.dot(px.astype(float), py)) / denom


@dataclass(kw_only=True)
class CorrelationAnalysis:
    """Feature/target association leaderboard, sorted by sum of absolute values.

    Explicit type/partition; overall/per_fold supported. methods selects Pearson,
    Spearman (average tied ranks), Kendall tau-b ('kendall') and/or MCC ('mcc').
    Pairwise finite observations are used for numerical coefficients; n/status
    accompany every cell. No p-values or independence assumptions are reported.

    MCC requires BOTH columns declared categorical or explicitly thresholded.
    feature_thresholds/target_thresholds map names to strict 'value > threshold'
    binary encodings. No automatic bins or threshold fitting. For native
    multiclass values, both columns must use a meaningful shared label vocabulary;
    MCC is sensitive to class correspondence. Undeclared MCC cells stay missing.

    With percentile_ranks=True, each metric/target cell receives its average-tie
    rank among features by absolute coefficient, divided by the number of finite
    coefficients, times 100. The pooled score SUMS absolute selected coefficients
    across all requested metric/target columns; missing cells do not contribute.
    n_coefficients shows this coverage; all-missing scores stay missing. Percentile
    ranks are descriptive orderings, not additional association estimators.
    """

    type: str
    partition: str
    features: object = None
    features_from: object = None
    targets: object = None
    methods: object = ("pearson", "spearman")
    percentile_ranks: bool = True
    categorical_features: object = ()
    categorical_targets: object = ()
    feature_thresholds: dict = field(default_factory=dict)
    target_thresholds: dict = field(default_factory=dict)
    supported_types: ClassVar[tuple] = ("per_fold", "overall")

    def run(self, context):
        if not len(context.X.columns) and self.features is None:
            return StudyResult("Feature–target associations", notes=["No input features selected."])
        features = _columns(context.X, self.features, "features")
        targets = _columns(context.y, self.targets, "targets")
        methods = [self.methods] if isinstance(self.methods, str) else list(self.methods)
        if not methods or len(set(methods)) != len(methods) or set(methods) - {"pearson", "spearman", "kendall", "mcc"}:
            raise ValueError("methods must select distinct pearson, spearman, kendall or mcc values.")
        stats = _scipy() if "kendall" in methods else None
        records = []
        for feature in features:
            for target in targets:
                x, y = _numeric(context.X[feature]), _numeric(context.y[target])
                keep = np.isfinite(x) & np.isfinite(y)
                a, b = x[keep], y[keep]
                for method in methods:
                    n, value, status = len(a), np.nan, "ok"
                    if method == "mcc":
                        cx = _categorical(context.X[feature], feature, self.categorical_features, self.feature_thresholds)
                        cy = _categorical(context.y[target], target, self.categorical_targets, self.target_thresholds)
                        if cx is None or cy is None:
                            n, status = 0, "requires_categorical_declaration_or_threshold"
                        else:
                            valid = cx.notna() & cy.notna()
                            n = int(valid.sum())
                            if n < 2:
                                status = "insufficient_pairs"
                            else:
                                value = _mcc(cx[valid], cy[valid])
                                if not np.isfinite(value):
                                    status = "zero_spread"
                    elif n < 2:
                        status = "insufficient_pairs"
                    elif np.ptp(a) == 0 or np.ptp(b) == 0:
                        status = "zero_spread"
                    elif method == "kendall":
                        value = float(stats.kendalltau(a, b, variant="b").statistic)
                    else:
                        if method == "spearman":
                            a_rank, b_rank = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
                        else:
                            a_rank, b_rank = a, b
                        value = float(np.corrcoef(a_rank, b_rank)[0, 1])
                    if np.isfinite(value):
                        value = float(np.clip(value, -1, 1))
                    records.append(dict(feature=feature, target=target, metric=method, value=value, n=n, status=status))
        cells = pd.DataFrame(records)
        if self.percentile_ranks:
            cells["percentile"] = cells.groupby(["target", "metric"], sort=False)["value"].transform(
                lambda values: values.abs().rank(method="average", pct=True) * 100)
        leaderboard = pd.DataFrame(index=pd.Index(features, name="feature"))
        metric_columns, percentile_columns = [], []
        for target in targets:
            for method in methods:
                column = f"{target} · {method}"
                selected = cells[(cells.target == target) & (cells.metric == method)].set_index("feature")
                leaderboard[column] = selected["value"]
                metric_columns.append(column)
                if self.percentile_ranks:
                    rank_column = f"{column} · percentile"
                    leaderboard[rank_column] = selected["percentile"]
                    percentile_columns.append(rank_column)
        values = leaderboard[metric_columns]
        leaderboard.insert(0, "n_coefficients", values.notna().sum(axis=1))
        leaderboard.insert(0, "pooled_magnitude", values.abs().sum(axis=1, min_count=1))
        leaderboard = leaderboard.sort_values("pooled_magnitude", ascending=False, kind="stable", na_position="last").reset_index()
        result = StudyResult("Feature–target associations", tables={"leaderboard": leaderboard, "coefficients": cells},
                             notes=["Ordered by the sum of absolute selected coefficients across targets and methods.",
                                    "These are descriptive associations, not model performance or causal effects.",
                                    "Missing cells remain missing; n_coefficients shows pooled-score coverage."])
        if "mcc" in methods:
            result.notes.append("MCC requires declared categories or explicit thresholds; zero spread is undefined.")
        result.artifacts.append(Artifact("leaderboard", leaderboard, "Association leaderboard",
                                        {"coefficient_columns": metric_columns, "percentile_columns": percentile_columns}))
        return result


@dataclass(kw_only=True)
class FeatureTimeline:
    """Feature histories by team or competition with explicit execution scope.

    type=per_fold plots the requested fold partition; overall/timeline plot the
    selected full population. Plots always sort points chronologically. entity
    is team or league. team-match layout uses team_id. For match layout a home::
    or away:: feature maps to home_id/away_id; team_column can override explicitly.
    Series use competition_id plus team identity, retaining continuity across
    seasons. teams optionally filters IDs without recoding them.

    League mode emits one value per competition/kickoff. By default finite values
    at that coordinate must agree; different LOO/team values require explicit
    league_aggregation='mean'/'median', or entity='team'. Aggregation is over the
    selected dataset rows at that timestamp, not season/round averages. Missing
    points are retained as gaps. No forward-fill, smoothing or value inference.
    max_points=None displays all points; an explicit limit evenly samples each
    plotted series including endpoints, leaving full downloadable tables intact.
    Additional missing-value break points are retained to avoid drawing across
    gaps, so the number of displayed points can exceed this sampling limit.
    """

    type: str
    partition: str
    features: object = None
    features_from: object = None
    entity: str = "team"
    teams: object = None
    team_column: object = None
    league_aggregation: object = None
    max_points: object = None
    supported_types: ClassVar[tuple] = ("per_fold", "overall", "timeline")

    def run(self, context):
        if not len(context.X.columns) and self.features is None:
            return StudyResult("Feature timelines", notes=["No input features selected."])
        if self.entity not in {"team", "league"} or self.league_aggregation not in {None, "mean", "median"}:
            raise ValueError("entity must be team/league and league_aggregation None/mean/median.")
        if self.entity == "league" and self.teams is not None:
            raise ValueError("teams is a team-timeline filter; omit it for league timelines.")
        if self.max_points is not None and (isinstance(self.max_points, bool) or
                not isinstance(self.max_points, (int, np.integer)) or self.max_points < 2):
            raise ValueError("max_points must be None or an integer >= 2.")
        go = _plotting()
        meta = context.metadata
        times = pd.to_datetime(meta["kickoff_at"], utc=True)
        if times.isna().any():
            raise ValueError("FeatureTimeline requires nonmissing kickoff timestamps.")
        if meta["competition_id"].isna().any():
            raise ValueError("FeatureTimeline requires competition identities.")
        result = StudyResult("Feature timelines", notes=[
            "UTC kickoff order. Missing values remain gaps; lines do not reconstruct unseen states."])
        point_tables, plotted_tables = [], []
        for feature in _columns(context.X, self.features, "features"):
            points = pd.DataFrame({"row_position": context.row_positions,
                                   "kickoff_at": times.to_numpy(),
                                   "competition_id": meta["competition_id"].to_numpy(),
                                   "value": _numeric(context.X[feature])})
            points.loc[~np.isfinite(points.value), "value"] = np.nan
            if self.entity == "team":
                column = self.team_column
                if column is None:
                    if context.layout == "team_match":
                        column = "team_id"
                    elif str(feature).startswith("home::"):
                        column = "home_id"
                    elif str(feature).startswith("away::"):
                        column = "away_id"
                    else:
                        raise ValueError("Match-layout team timelines need home::/away:: features or team_column.")
                if meta[column].isna().any():
                    raise ValueError("Team timelines require nonmissing team identities.")
                points["team_id"] = meta[column].to_numpy()
                if self.teams is not None:
                    points = points[points.team_id.isin(list(self.teams))].copy()
                group_by = ["competition_id", "team_id"]
            else:
                group_by = ["competition_id"]
                grouped = points.groupby(["competition_id", "kickoff_at"], sort=False, dropna=False)["value"]
                if self.league_aggregation is None and (grouped.nunique(dropna=True) > 1).any():
                    raise ValueError("League timeline values differ at one kickoff; select team mode or explicit league_aggregation.")
                aggregation = "first" if self.league_aggregation is None else self.league_aggregation
                points = grouped.agg(aggregation).reset_index()
            points = points.sort_values("kickoff_at", kind="stable")
            points.insert(0, "feature", feature)
            point_tables.append(points)
            figure = go.Figure()
            for key, group in points.groupby(group_by, sort=False, dropna=False):
                key = key if isinstance(key, tuple) else (key,)
                label = " · ".join(str(value) for value in key)
                plotted = group
                if self.max_points is not None and len(group) > self.max_points:
                    chosen = np.unique(np.linspace(0, len(group) - 1, self.max_points).astype(int))
                    missing = np.flatnonzero(group.value.isna().to_numpy())
                    breaks = []
                    for left, right in zip(chosen[:-1], chosen[1:]):
                        between = missing[(missing > left) & (missing < right)]
                        if len(between):
                            breaks.append(between[0])
                    plotted = group.iloc[np.unique(np.concatenate([chosen, np.asarray(breaks, dtype=int)]))]
                plotted_tables.append(plotted.copy())
                figure.add_trace(go.Scatter(x=plotted.kickoff_at.tolist(), y=plotted.value.tolist(),
                                           mode="lines+markers", name=label, connectgaps=False,
                                           marker=dict(size=4), line=dict(width=2)))
            if not len(points):
                figure.add_annotation(text="No observations for this scope/team selection", showarrow=False)
            if len(figure.data) > 1:
                buttons = [dict(label="All entities", method="update", args=[{"visible": [True] * len(figure.data)}])]
                for i, trace in enumerate(figure.data):
                    buttons.append(dict(label=trace.name, method="update",
                                        args=[{"visible": [j == i for j in range(len(figure.data))]}]))
                figure.update_layout(updatemenus=[dict(buttons=buttons, x=1, y=1.2, xanchor="right")])
            result.artifacts.append(Artifact("plotly", _style(figure, str(feature), "Kickoff (UTC)", str(feature)), str(feature)))
        result.tables["points"] = pd.concat(point_tables, ignore_index=True)
        result.tables["plotted_points"] = (pd.concat(plotted_tables, ignore_index=True)
                                            if plotted_tables else result.tables["points"].iloc[:0].copy())
        if self.max_points is not None:
            result.notes.append(f"Display limit: {self.max_points} points per entity; full points remain downloadable.")
        return result
