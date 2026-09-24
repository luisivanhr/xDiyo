"""Prediction diagnostics sharing the existing study/artifact viewer."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from ..evaluation.metrics import Metric, evaluate_metrics, metric_inputs, _probabilities
from .contracts import Artifact, StudyResult
from .studies import _columns, _numeric, _plotting, _scipy, _style


@dataclass(kw_only=True)
class PredictionReporter:
    """Shared scope configuration; subclasses implement run(context)."""
    type: str
    partition: str
    pooling: str | None = None
    supported_types: ClassVar[tuple] = ("per_fold", "overall", "timeline")


@dataclass(kw_only=True)
class PerformanceReporter(PredictionReporter):
    """Selected scalar metrics with coverage, definition and direction metadata.

    metrics is a sequence of Metric requests or registered names. Numerical
    results are reusable through evaluate_metrics without rendering any report.
    """
    metrics: object

    def run(self, context):
        table = evaluate_metrics(context.y, context.predictions, self.metrics, metadata=context.metadata)
        shown = table[['target', 'metric', 'value', 'n', 'n_missing', 'status']]
        return StudyResult("Predictive performance", [Artifact("table", shown, "Metrics")],
                           {"metrics": table}, ["Metrics use paired eligible observations; n_missing records excluded rows."])


def _pairs(context, target, output):
    observed, predicted, mask, _ = metric_inputs(context.y, context.predictions,
                                                Metric("mse", target=target, output=output))
    return pd.DataFrame({"observed": observed[mask], "predicted": predicted[mask],
                         "residual": observed[mask] - predicted[mask]}), mask


@dataclass(kw_only=True)
class ResidualAnalysisReporter(PredictionReporter):
    """Observed-vs-predicted, residual-vs-predicted and residual histogram.

    Residual = observed - predicted. Numeric point predictions only; no automatic
    conversion of class probabilities into point estimates or class labels.
    """
    targets: object = None
    output: str = "predict"
    bins: object = "auto"

    def run(self, context):
        go = _plotting()
        result = StudyResult("Residual analysis")
        summaries = []
        for target in _columns(context.y, self.targets, "targets"):
            pairs, mask = _pairs(context, target, self.output)
            summaries.append(dict(target=target, n=len(pairs), n_missing=int((~mask).sum())))
            result.tables[f"pairs::{target}"] = pairs.reset_index()
            observed = go.Figure(go.Scatter(x=pairs.predicted, y=pairs.observed, mode="markers", name=target))
            if len(pairs):
                lo, hi = pairs[["observed", "predicted"]].min().min(), pairs[["observed", "predicted"]].max().max()
                observed.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction"))
            residual = go.Figure(go.Scatter(x=pairs.predicted, y=pairs.residual, mode="markers", name=target))
            residual.add_hline(y=0)
            counts, edges = np.histogram(pairs.residual, bins=self.bins)
            histogram = pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": counts})
            result.tables[f"histogram::{target}"] = histogram
            distribution = go.Figure(go.Bar(x=(edges[:-1]+edges[1:])/2, y=counts, width=np.diff(edges)))
            for figure, title, x, y in ((observed, "Observed vs predicted", "Predicted", "Observed"),
                                        (residual, "Residual vs predicted", "Predicted", "Observed − predicted"),
                                        (distribution, "Residual distribution", "Residual", "Count")):
                result.artifacts.append(Artifact("plotly", _style(figure, f"{target}: {title}", x, y), f"{target}: {title}"))
        result.tables["coverage"] = pd.DataFrame(summaries)
        return result


@dataclass(kw_only=True)
class PredictionDistributionReporter(PredictionReporter):
    """Overlay observed/predicted distributions for the same eligible rows.

    mode='kde' draws two independently normalized KDE curves on a shared grid,
    with no histogram bars. bandwidth is Scott/Silverman or a positive scalar
    covariance factor. Constants/single observations use a vertical marker.
    mode='ecdf' uses step curves, mode='frequency' compares categorical relative
    frequencies. KDE of counts is descriptive smoothing, not probability mass.
    """
    targets: object = None
    output: str = "predict"
    mode: str = "kde"
    bandwidth: object = "scott"
    grid_size: int = 256

    def run(self, context):
        if self.mode not in {"kde", "ecdf", "frequency"}:
            raise ValueError("mode must be kde, ecdf or frequency.")
        if isinstance(self.grid_size, bool) or not isinstance(self.grid_size, (int, np.integer)) or self.grid_size < 2:
            raise ValueError("grid_size must be an integer >= 2.")
        if self.mode == "kde":
            from numbers import Real
            valid = (self.bandwidth in {"scott", "silverman"} if isinstance(self.bandwidth, str) else
                     isinstance(self.bandwidth, Real) and not isinstance(self.bandwidth, (bool, np.bool_))
                     and np.isfinite(self.bandwidth) and self.bandwidth > 0)
            if not valid:
                raise ValueError("bandwidth must be scott, silverman or a positive finite scalar.")
        go = _plotting()
        result = StudyResult("Observed and predicted distributions", notes=[
            "Each distribution uses the same eligible observations and its own unit normalization.",
            "Agreement of marginal distributions does not imply accurate individual predictions."])
        summaries = []
        for target in _columns(context.y, self.targets, "targets"):
            figure, records = go.Figure(), []
            if self.mode == "frequency":
                y, p = context.y[target], context.predictions[self.output][target]
                mask = y.notna() & p.notna()
                labels = pd.Index(pd.unique(pd.concat([y[mask], p[mask]], ignore_index=True)))
                for name, values in (("Observed", y[mask]), ("Predicted", p[mask])):
                    freq = values.value_counts(normalize=True).reindex(labels, fill_value=0)
                    figure.add_trace(go.Bar(x=[str(label) for label in labels], y=freq, name=name))
                    records.extend(dict(distribution=name, x=label, value=value) for label, value in freq.items())
                    summaries.append(dict(target=target, distribution=name, n=int(mask.sum()),
                                          n_missing=int((~mask).sum()), status="ok" if mask.any() else "empty"))
                figure.update_layout(barmode="group")
            else:
                pairs, mask = _pairs(context, target, self.output)
                combined = pairs[["observed", "predicted"]].to_numpy().ravel()
                if len(combined):
                    pad = max(float(np.std(combined)) * 3, 1e-6)
                    grid = np.linspace(combined.min()-pad, combined.max()+pad, self.grid_size)
                for name, values in (("Observed", pairs.observed), ("Predicted", pairs.predicted)):
                    status = "ok"
                    if not len(values):
                        status = "empty"
                    elif self.mode == "ecdf":
                        x = np.sort(values.to_numpy())
                        density = np.arange(1, len(x)+1)/len(x)
                        figure.add_trace(go.Scatter(x=x, y=density, mode="lines", line_shape="hv", name=name))
                        records.extend(dict(distribution=name, x=a, value=b) for a, b in zip(x, density))
                    elif len(values) < 2 or values.nunique() < 2:
                        status = "constant_marker"
                        figure.add_vline(x=float(values.iloc[0]), annotation_text=name)
                        records.append(dict(distribution=name, x=float(values.iloc[0]), value=np.nan))
                    else:
                        density = _scipy().gaussian_kde(values, bw_method=self.bandwidth)(grid)
                        figure.add_trace(go.Scatter(x=grid, y=density, mode="lines", name=name))
                        records.extend(dict(distribution=name, x=a, value=b) for a, b in zip(grid, density))
                    summaries.append(dict(target=target, distribution=name, n=len(values),
                                          n_missing=int((~mask).sum()), status=status))
            result.tables[f"distribution::{target}"] = pd.DataFrame(records, columns=["distribution", "x", "value"])
            axis = {"kde": "Density", "ecdf": "Cumulative probability", "frequency": "Relative frequency"}[self.mode]
            result.artifacts.append(Artifact("plotly", _style(figure, str(target), "Value", axis), str(target)))
        result.tables["coverage"] = pd.DataFrame(summaries)
        return result


LabelPredictionDistributionReporter = PredictionDistributionReporter


@dataclass(kw_only=True)
class CalibrationReporter(PredictionReporter):
    """One-vs-rest reliability curves for explicit class-probability outputs.

    Uniform or quantile bins; empty bins are omitted. Each plotted point is the
    mean probability and observed class frequency in that bin. No recalibration
    is fitted. Probabilities with incomplete class vectors are excluded/countable.
    """
    targets: object = None
    output: str = "predict_proba"
    n_bins: int = 10
    strategy: str = "uniform"
    classes: object = None

    def run(self, context):
        if isinstance(self.n_bins, bool) or not isinstance(self.n_bins, (int, np.integer)) or self.n_bins < 1:
            raise ValueError("n_bins must be a positive integer.")
        if self.strategy not in {"uniform", "quantile"}:
            raise ValueError("strategy must be uniform or quantile.")
        go, rows, coverage = _plotting(), [], []
        result = StudyResult("Probability calibration")
        for target in _columns(context.y, self.targets, "targets"):
            truth, p, mask, _ = metric_inputs(context.y, context.predictions, Metric("log_loss", target, self.output))
            truth, p = truth[mask], p.loc[mask]
            _probabilities(p)
            if not truth.isin(p.columns).all():
                raise ValueError("Observed classes are absent from probability outputs.")
            labels = list(p.columns) if self.classes is None else list(self.classes)
            if set(labels) - set(p.columns):
                raise ValueError("Requested calibration classes are absent from output.")
            figure = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration"))
            coverage.append(dict(target=target, n=len(truth), n_missing=int((~mask).sum())))
            for label in labels:
                values = p[label].to_numpy(dtype=float)
                if not len(values):
                    continue
                edges = (np.linspace(0, 1, self.n_bins+1) if self.strategy == "uniform" else
                         np.unique(np.quantile(values, np.linspace(0, 1, self.n_bins+1))))
                bins = np.searchsorted(edges[1:-1], values, side="right")
                points = []
                for bin_id in np.unique(bins):
                    keep = bins == bin_id
                    row = dict(target=target, label=label, bin=int(bin_id), n=int(keep.sum()),
                               mean_probability=float(values[keep].mean()),
                               observed_frequency=float(truth.eq(label).to_numpy()[keep].mean()))
                    rows.append(row)
                    points.append(row)
                figure.add_trace(go.Scatter(x=[r["mean_probability"] for r in points],
                                            y=[r["observed_frequency"] for r in points], mode="lines+markers", name=str(label)))
            result.artifacts.append(Artifact("plotly", _style(figure, str(target), "Mean predicted probability",
                                                               "Observed frequency"), str(target)))
        result.tables["calibration"] = pd.DataFrame(rows)
        result.tables["coverage"] = pd.DataFrame(coverage)
        return result


@dataclass(kw_only=True)
class PredictionTimelineReporter(PredictionReporter):
    """Observed, predicted and residual series ordered by kickoff within groups.

    group_by selects metadata columns (e.g. team_id or competition_id). None
    draws one sequence per fold. In match layout choose home_id/away_id explicitly
    for team grouping. No league/team averaging or interpolation is implicit.
    """
    targets: object = None
    output: str = "predict"
    group_by: object = None

    def run(self, context):
        go = _plotting()
        result = StudyResult("Prediction timelines")
        group_columns = [] if self.group_by is None else ([self.group_by] if isinstance(self.group_by, str) else list(self.group_by))
        times = pd.to_datetime(context.metadata["kickoff_at"], utc=True, errors="raise")
        if times.isna().any():
            raise ValueError("Prediction timelines require nonmissing kickoff timestamps.")
        for target in _columns(context.y, self.targets, "targets"):
            # Retain gaps when either numeric value is unavailable.
            frame = context.metadata[group_columns].copy()
            frame["kickoff_at"] = times
            frame["observed"] = _numeric(context.y[target])
            frame["predicted"] = _numeric(context.predictions[self.output][target])
            frame[["observed", "predicted"]] = frame[["observed", "predicted"]].replace([np.inf, -np.inf], np.nan)
            frame["residual"] = frame.observed-frame.predicted
            frame = frame.reset_index().sort_values("kickoff_at", kind="stable")
            result.tables[f"timeline::{target}"] = frame
            figure = go.Figure()
            groups = ["fold_id", *group_columns]
            for identity, values in frame.groupby(groups, sort=False, dropna=False):
                for column in ("observed", "predicted", "residual"):
                    figure.add_trace(go.Scatter(x=values.kickoff_at, y=values[column], mode="lines+markers",
                                                connectgaps=False, name=f"{identity}: {column}"))
            result.artifacts.append(Artifact("plotly", _style(figure, str(target), "Kickoff (UTC)", "Value"), str(target)))
        return result
