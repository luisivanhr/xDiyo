"""Exact per-match point geometry; no grids, model fitting or temporal inference."""
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from copy import deepcopy

import numpy as np
import pandas as pd

from .expressions import Expr


@dataclass(frozen=True)
class SpatialPointSummary(Expr):
    """Observed activity descriptor in the focal team's 0..100 coordinate frame.

    Requires Lag/rolling/EMA before use as a predictor. Unit-weight raw points
    are summarized per match; repeated coordinates are retained. Against uses
    the opponent's points rotated 180 degrees. Target venue never rotates these
    scalars. min_points is map quality, distinct from historical min_periods.
    """
    field: str = 'mean_x'
    kinds: tuple = ('player',)
    side: str = 'for'
    min_points: int = 1

    def __post_init__(self):
        if self.field not in ('mean_x', 'sd_x', 'sd_y'):
            raise ValueError('SpatialPointSummary field must be mean_x, sd_x or sd_y.')
        kinds = (self.kinds,) if isinstance(self.kinds, str) else tuple(self.kinds or ())
        if not kinds or len(set(kinds)) != len(kinds) or set(kinds) - {'player', 'goalkeeper'}:
            raise ValueError('Choose distinct point kinds: player and/or goalkeeper.')
        object.__setattr__(self, 'kinds', kinds)
        if self.side not in ('for', 'against', 'both'):
            raise ValueError('SpatialPointSummary side must be for, against or both.')
        if isinstance(self.min_points, bool) or not isinstance(self.min_points, Integral) or self.min_points < 1:
            raise ValueError('min_points must be a positive integer.')
        object.__setattr__(self, 'min_points', int(self.min_points))


def point_summary_source(history, points, kinds, min_points):
    """Build a shared summary/audit once per point selection in one evaluation."""
    if points is None:
        raise ValueError('SpatialPointSummary requires the heatmap_points table; pass heatmaps=points.')
    scope = []
    for name in ('source_league', 'source_season'):
        if (name in history) != (name in points):
            raise ValueError(f'Spatial point identity requires {name} on both history and points.')
        if name in history:
            scope.append(name)
    keys = [*scope, 'event_id', 'team_id']
    required = [*keys, 'kind', 'x', 'y']
    missing = set(required) - set(points)
    if missing:
        raise ValueError(f'Spatial points are missing columns: {sorted(missing)}')
    if history[keys].isna().any().any() or points[keys].isna().any().any():
        raise ValueError('Spatial map identities must be nonmissing.')
    if history.duplicated(keys).any():
        raise ValueError('Duplicate fixture/team identity in spatial history; retain source league/season keys.')
    if points.kind.isna().any() or not set(points.kind.unique()) <= {'player', 'goalkeeper'}:
        raise ValueError('Unknown heatmap point kind; expected player or goalkeeper.')
    if 'point_order' in points and (points.point_order.isna().any() or
                                   points.duplicated([*keys, 'kind', 'point_order']).any()):
        raise ValueError('Duplicate/missing spatial point_order identity; possible duplicate table or fixture version.')
    # Compute source identity once; do not cache globally across mutable inputs.
    hashed = [*required, *[c for c in ('point_order', 'source_key', 'raw_hash', 'observed_at') if c in points]]
    digest = sha256(pd.util.hash_pandas_object(points[hashed], index=False).to_numpy().tobytes()).hexdigest()
    groups = {}
    kind_mask = points.kind.isin(kinds).to_numpy()
    for key, positions in points.groupby(keys, sort=False, dropna=False).indices.items():
        key = key if isinstance(key, tuple) else (key,)
        frame = points.iloc[positions]
        for name in ('raw_hash', 'source_key'):
            if name in frame and frame[name].nunique(dropna=True) > 1:
                raise ValueError(f'Conflicting spatial {name} for map {key}; select one source version.')
        selected = points.iloc[positions[kind_mask[positions]]]
        try:
            xy = selected[['x', 'y']].to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid spatial coordinates for map {key}: expected numeric 0..100 values.') from exc
        if not np.isfinite(xy).all() or ((xy < 0) | (xy > 100)).any():
            raise ValueError(f'Invalid spatial coordinates for map {key}: expected finite 0..100 values.')
        n = len(xy)
        status = 'empty_selected_kind' if n == 0 else 'low_point_count' if n < min_points else 'ok'
        mean, sd = np.full(2, np.nan), np.full(2, np.nan)
        if status == 'ok':
            mean = xy.mean(axis=0)
            # Two-pass centered sums: nonnegative by construction, no raw-square
            # subtraction or negative-variance clamping tolerance is needed.
            sd = np.sqrt(np.mean((xy-mean)**2, axis=0))
        record = dict(zip(keys, (v.item() if isinstance(v, np.generic) else v for v in key)))
        record.update(point_count=n, total_point_count=len(frame), status=status,
                      valid_map=status == 'ok', mean_x=float(mean[0]), sd_x=float(sd[0]), sd_y=float(sd[1]),
                      x_min=float(xy[:,0].min()) if n else None, x_max=float(xy[:,0].max()) if n else None,
                      y_min=float(xy[:,1].min()) if n else None, y_max=float(xy[:,1].max()) if n else None)
        for name in ('raw_hash', 'source_key'):
            if name in frame:
                values = frame[name].dropna()
                record[name] = str(values.iloc[0]) if len(values) else None
        groups[key] = record
    return dict(keys=keys, groups=groups, source_sha256=digest, formula_version='point_geometry_v1',
                kinds=list(kinds), min_points=int(min_points), weighting='unit', ddof=0,
                observed_kinds=sorted(points.kind.unique()), provenance=deepcopy(points.attrs.get('source', {})),
                duplicate_detection='point_order' if 'point_order' in points else 'source_version_only')


def point_summary_values(history, source, spec):
    outputs, descriptors, audits = {}, {}, []
    keys = source['keys']
    identities = history[keys].to_dict('records')
    for side in (('for','against') if spec.side == 'both' else (spec.side,)):
        lookup = [*keys[:-1], 'team_id' if side == 'for' else 'opponent_id']
        if history[lookup].isna().any().any():
            raise ValueError('Spatial point lookup identities must be nonmissing.')
        values = []
        for position, key in enumerate(history[lookup].itertuples(index=False, name=None)):
            record = source['groups'].get(key)
            value = record[spec.field] if record is not None else np.nan
            if side == 'against' and spec.field == 'mean_x':
                value = 100-value
            values.append(value)
            diagnostic = identities[position].copy()
            diagnostic.update(side=side, source_team_id=key[-1], point_count=0 if record is None else record['point_count'],
                              status='missing_team_map' if record is None else record['status'],
                              valid_map=False if record is None else record['valid_map'])
            if record is not None:
                diagnostic.update({k:record[k] for k in ('total_point_count','x_min','x_max','y_min','y_max','raw_hash','source_key') if k in record})
            audits.append(diagnostic)
        column = f'activity_{side}_{spec.field}'
        outputs[column] = values
        descriptors[column] = dict(kind='scalar', columns=[column], field=spec.field, side=side,
            field_label={'mean_x':'Mean longitudinal position', 'sd_x':'Longitudinal standard deviation',
                         'sd_y':'Lateral standard deviation'}[spec.field],
            orientation='team', units='percent of pitch '+('width' if spec.field == 'sd_y' else 'length'),
            kinds=list(spec.kinds), weighting='unit', ddof=0, min_points=spec.min_points,
            source='SpatialPointSummary', formula_version=source['formula_version'],
            source_sha256=source['source_sha256'], operators=[])
    result = pd.DataFrame(outputs, index=history.index)
    result.attrs['spatial_features'] = descriptors
    result.attrs['point_map_coverage'] = audits
    return result
