"""Availability-aware chronological calibration populations; no model fitting."""

import numpy as np
import pandas as pd

from ..splits.core import _Matches, _times, _duration


def timing(policy, dataset):
    matches = _Matches(dataset)
    kicks = matches.times(policy.time_column, policy.time_column)
    keys = matches.columns(policy.prediction_group_by)
    groups = (pd.factorize(pd.MultiIndex.from_frame(matches.meta[list(keys)]), sort=False)[0]
              if keys else np.arange(len(matches.positions)))
    if policy.cutoff_column is not None:
        cutoffs = matches.times(policy.cutoff_column, 'calibration prediction cutoff', 'min')
    else:
        lead = _duration(policy.prediction_lead, 'prediction_lead')
        cutoffs = pd.DatetimeIndex(pd.Series(kicks).groupby(groups, sort=False).transform('min')) - lead
    if (cutoffs > kicks).any():
        raise ValueError('Calibration prediction cutoff cannot be later than kickoff.')
    return matches, kicks, groups, cutoffs


def fold_boundary(policy, dataset, fold):
    """Use the earliest actual outer prediction boundary, never a later issue time."""
    metadata = dict(fold.metadata)
    if getattr(policy, 'response_method', 'predict_proba') != 'decision_function':
        return metadata
    matches, _, _, cutoffs = timing(policy, dataset)
    boundary = cutoffs[np.unique(matches.codes[fold.test])].min()
    for value in (metadata.get('fit_at'), policy.issue_at):
        if value is not None:
            timestamp = pd.to_datetime(value, utc=True, errors='raise')
            if pd.isna(timestamp):
                raise ValueError('Calibration issue time must be nonmissing.')
            boundary = min(boundary, timestamp)
    metadata['fit_at'] = boundary
    return metadata


def temporal_partition(policy, dataset, train_positions, fold_metadata=None):
    from .control import ValidationTail
    matches, kicks, groups, cutoffs = timing(policy, dataset)
    train = np.asarray(train_positions, dtype=int)
    codes = np.unique(matches.codes[train])
    group_members = {}
    for code, group in enumerate(groups):
        group_members.setdefault(group, set()).add(code)
    if not np.array_equal(np.sort(train), matches.rows(codes)):
        raise ValueError('Temporal calibration training population must retain whole matches.')
    issue = (fold_metadata or {}).get('fit_at', policy.issue_at)
    if issue is None:
        raise ValueError('Margin calibration needs an outer fit_at or explicit issue_at for refit.')
    issue = pd.to_datetime(issue, utc=True, errors='raise')
    if pd.isna(issue):
        raise ValueError('Calibration issue_at must be nonmissing.')
    if policy.availability_column is not None:
        raw = _times(dataset, policy.availability_column, 'label availability')
        available = pd.DatetimeIndex([raw.iloc[r].max() if raw.iloc[r].notna().all() else pd.NaT
                                      for r in matches.positions])
        availability = {'kind': 'explicit', 'column': policy.availability_column}
    elif policy.availability_delay is not None:
        available = kicks + _duration(policy.availability_delay, 'availability_delay')
        availability = {'kind': 'kickoff_plus_delay_proxy', 'delay': str(policy.availability_delay)}
    else:
        raise ValueError('Margin calibration requires availability_column or an explicit availability_delay proxy.')
    reserved = ValidationTail(policy.fraction, policy.time_column).select(dataset, train)
    tail = set(matches.codes[reserved])
    base = set(codes) - tail
    removed = []

    def purge(ids, reason):
        for code in sorted(ids):
            for row in matches.positions[code]:
                record = {name: dataset.metadata.iloc[row][name] for name in dataset.match_columns}
                removed.append(dict(row_position=int(row), identity=record, reason=reason))
        base.difference_update(ids)
        tail.difference_update(ids)

    # Do not expand a tail across a prediction group or silently split one.
    population = set(codes)
    for group in np.unique(groups[codes]):
        all_ids = group_members[group]
        selected = all_ids & population
        if selected != all_ids or (selected & base and selected & tail):
            purge(selected, 'incomplete_or_boundary_prediction_group')
    observed_rows = dataset.y.notna().all(axis=1).to_numpy()
    observed = np.array([observed_rows[r].all() for r in matches.positions])
    for group in np.unique(groups[list(tail)]):
        ids = group_members[group] & tail
        idx = list(ids)
        if not (available[idx].notna() & (available[idx] <= issue) & (kicks[idx] < issue) & observed[idx]).all():
            purge(ids, 'calibration_label_unavailable_at_issue')
    if not tail:
        raise ValueError(f'Temporal calibration has no eligible calibration tail; purged={removed!r}')
    first = cutoffs[list(tail)].min()
    for group in np.unique(groups[list(base)]):
        ids = group_members[group] & base
        idx = list(ids)
        if not (available[idx].notna() & (available[idx] <= first) & (kicks[idx] < first) & observed[idx]).all():
            purge(ids, 'base_label_unavailable_at_calibration_cutoff')
    if not base:
        raise ValueError(f'Temporal calibration has no eligible earlier fitting rows; purged={removed!r}')
    # Stable chronology/identity order also makes numerical fitting invariant to input order.
    def rows(ids):
        ordered = sorted(ids, key=lambda i: (kicks[i], repr(matches.keys[i])))
        result = []
        for i in ordered:
            result.extend(sorted(matches.positions[i], key=lambda r: repr(tuple(
                dataset.metadata.iloc[r][c] for c in dataset.identity_columns))))
        return np.asarray(result, dtype=int)
    fitting, calibration = rows(base), rows(tail)
    audit = dict(version=1, availability=availability, prediction_group_by=list(policy.prediction_group_by),
                 cutoff_column=policy.cutoff_column, prediction_lead=str(policy.prediction_lead),
                 first_calibration_cutoff=first.isoformat(), fit_at=issue.isoformat(),
                 latest_base_label_available_at=available[list(base)].max().isoformat(),
                 latest_calibration_label_available_at=available[list(tail)].max().isoformat(),
                 purged=removed)
    return fitting, calibration, audit
