"""Weighted comparison of recorded metrics, independent of report rendering."""

import numpy as np
import pandas as pd

from .store import configuration_hash


def rank_runs(records, weights, *, selectors=None, scaling="percentile", reference_scales=None, directions=None):
    """Rank compatible runs by an arbitrary nonnegative weighted metric sum.

    weights maps display keys to weights, normalized internally. selectors maps
    those keys to metric-record filters, e.g. {'mse': {'metric':'mse',
    'target':'corners', 'study':'performance'}}. Without a filter, select an
    overall metric of the same name. Each key must resolve to exactly one record;
    specify type/fold_id explicitly for per-fold metrics. No implicit fold mean.

    Higher utility is better. percentile scales within each comparable group:
    (average rank - 1)/(n - 1); one run/all ties receive 0.5. fixed uses declared
    (low,high) bounds, clips to [0,1], and reverses minimizing metrics. Adding runs
    may change percentile scores. directions can override descriptive preferences.

    Comparability includes sample hashes, target, output, calculation, settings,
    execution scope and direction for every weighted metric. Separate comparison
    groups are ranked independently; scores across groups are not comparable.
    Missing/partial/undefined required metrics leave a run visible but unranked.
    Failed runs stay visible. Zero-weight metrics do not determine eligibility.
    """
    if not weights or any(not isinstance(k, str) or not k for k in weights):
        raise ValueError("weights must map nonempty metric keys to nonnegative weights.")
    values = np.asarray(list(weights.values()), dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not (values > 0).any():
        raise ValueError("Weights must be finite, nonnegative and have positive sum.")
    values = values/values.max()
    weights = {key: float(weight/values.sum()) for key, weight in zip(weights, values) if weight > 0}
    if scaling not in {"percentile", "fixed"}:
        raise ValueError("scaling must be percentile or fixed.")
    selectors, directions = selectors or {}, directions or {}
    reference_scales = reference_scales or {}
    if scaling == "fixed":
        for key in weights:
            limits = np.asarray(reference_scales.get(key, ()), dtype=float)
            if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError(f"Fixed scaling needs finite increasing (low, high) bounds for {key!r}.")
    rows, orientations = [], {}
    for record in records:
        row = dict(run_id=record["run_id"], name=record["name"], config_hash=record["config_hash"],
                   status="unranked_missing_or_undefined", comparison_group=None, score=np.nan)
        identities, valid = [], record.get("status") == "complete"
        if not valid:
            row["status"] = record.get("status", "unknown")
        for key in weights:
            query = dict(selectors.get(key, {}))
            query.setdefault("metric", key)
            if "type" not in query:
                query["type"] = "per_fold" if "fold_id" in query and query["fold_id"] is not None else "overall"
            matches = [metric for metric in record.get("metrics", [])
                       if all(metric.get(field) == value for field, value in query.items())]
            if len(matches) > 1:
                raise ValueError(f"Metric key {key!r} is ambiguous in run {record['name']!r}; specify target/study/output.")
            metric = matches[0] if matches else None
            value = metric.get("value") if metric else None
            row[f"raw::{key}"] = np.nan if value is None else value
            if metric is None or metric.get("status") != "ok" or value is None or not np.isfinite(value):
                valid = False
                continue
            direction = directions.get(key, metric.get("direction"))
            if direction not in {"minimize", "maximize"}:
                raise ValueError(f"Metric {key!r} has no universal ranking direction; supply directions explicitly.")
            orientations[(record["run_id"], key)] = direction
            if not metric.get("sample_hash"):
                valid = False
            identities.append({field: metric.get(field) for field in
                               ("metric", "calculation", "target", "output", "parameters", "sample_hash",
                                "type", "partition", "fold_id", "layout", "scope_label")})
            identities[-1]["direction"] = direction
        if valid:
            row["comparison_group"] = configuration_hash(identities)
            row["status"] = "ranked"
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["run_id", "name", "config_hash", "status", "comparison_group", "score", "rank"])
    frame["rank"] = np.nan
    for key in weights:
        frame[f"utility::{key}"] = np.nan
        frame[f"contribution::{key}"] = np.nan
    for _, group in frame.loc[frame.status.eq("ranked")].groupby("comparison_group", sort=False):
        index = group.index
        total = pd.Series(0.0, index=index)
        for key, weight in weights.items():
            values = group[f"raw::{key}"].astype(float)
            minimize = orientations[(group.iloc[0].run_id, key)] == "minimize"
            if scaling == "percentile":
                utility = ((values.rank(method="average", ascending=not minimize)-1)/(len(group)-1)
                           if len(group) > 1 else pd.Series(0.5, index=index))
            else:
                lo, hi = reference_scales[key]
                utility = ((values-lo)/(hi-lo)).clip(0, 1)
                if minimize:
                    utility = 1-utility
            frame.loc[index, f"utility::{key}"] = utility
            frame.loc[index, f"contribution::{key}"] = weight*utility
            total += weight*utility
        frame.loc[index, "score"] = total
        frame.loc[index, "rank"] = total.rank(method="min", ascending=False)
    frame.attrs["weights"] = weights
    return frame.sort_values(["comparison_group", "score", "run_id"], ascending=[True, False, True], na_position="last").reset_index(drop=True)
