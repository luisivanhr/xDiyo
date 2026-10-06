"""Causal map deviations and explicitly fixture-scoped spatial comparisons."""
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
import pandas as pd

from .spatial import Heatmap
from .spatial_distribution import _tolerance, validated_grid


def outfield_grid():
    return Heatmap(grid_size=6, kinds=('player',), orientation='team')


def validate_options(node):
    if not isinstance(node.source, Heatmap):
        raise TypeError('Spatial distances need a Heatmap source; history is configured by window/min_periods.')
    if node.source.normalization not in ('mass', 'density'):
        raise ValueError('Spatial distances require mass or density grids.')
    for name in ('window', 'min_periods'):
        value = getattr(node, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if node.min_periods > node.window:
        raise ValueError('min_periods cannot exceed window.')
    if node.venue not in ('all', 'same'):
        raise ValueError('Historical venue must be all or same.')
    _tolerance(node.mass_tolerance)


@dataclass(frozen=True)
class SpatialHistoricalDeviation:
    """Observed JS distance to the mean map before this match's own cutoff.

    Requires an outer Lag/rolling/EMA to become a predictor. Inner window counts
    eligible matches, including missing maps. Defaults to 6x6 player-only maps.
    """
    source: Heatmap = field(default_factory=outfield_grid)
    window: int = 20
    min_periods: int = 1
    venue: str = 'all'
    mass_tolerance: float = 1e-10

    def __post_init__(self):
        validate_options(self)


@dataclass(frozen=True)
class SpatialFixtureDistance:
    """One fixture JS distance from both teams' causal mean maps.

    style compares own-frame for maps. home_context compares home-for with
    rotated away-against; away_context compares away-for with rotated home-against.
    Source side must be for: the comparison chooses both required perspectives.
    """
    source: Heatmap = field(default_factory=outfield_grid)
    comparison: str = 'style'
    window: int = 20
    min_periods: int = 1
    venue: str = 'all'
    mass_tolerance: float = 1e-10

    def __post_init__(self):
        validate_options(self)
        if self.source.side != 'for':
            raise ValueError('Fixture distance source side must be for; comparison selects the against map when needed.')
        if self.comparison not in ('style', 'home_context', 'away_context'):
            raise ValueError('comparison must be style, home_context or away_context.')


def unit_maps(frame, spec, tolerance):
    """Strict complete-map validation; only normalize roundoff-sized mass drift."""
    values, missing = validated_grid(frame, spec, 'distance source')
    values = values.copy()
    if spec['normalization'] == 'density':
        values = values * (100/spec['grid_size'])**2
    totals = values[~missing].sum(axis=1)
    if ((~np.isfinite(totals)) | (totals <= 0) | (np.abs(totals-1) > tolerance)).any():
        raise ValueError('Spatial distance source must have unit mass within mass_tolerance.')
    values[~missing] /= totals[:, None]
    return values


def js_distance(left, right):
    """Base-2 square-root JS for already validated unit maps; missing propagates."""
    if left.shape != right.shape:
        raise ValueError('JS inputs must have identical grid shapes.')
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    p, q = left[valid], right[valid]
    middle = (p+q)/2
    divergence = np.zeros(len(p))
    for value in (p, q):
        log_ratio = np.zeros_like(value)
        positive = value > 0
        np.divide(value, middle, out=log_ratio, where=positive)
        np.log2(log_ratio, out=log_ratio, where=positive)
        divergence += .5*np.sum(value*log_ratio, axis=1)
    result = np.full(len(left), np.nan)
    result[valid] = np.sqrt(np.clip(divergence, 0, 1))
    return result


def prior_maps(values, candidates, node):
    output = np.full_like(values, np.nan)
    coverage = []
    for row, positions in enumerate(candidates):
        sample = values[positions[-node.window:]]
        valid = np.isfinite(sample).all(axis=1)
        coverage.append(dict(eligible_matches=len(positions), window_matches=len(sample), usable_maps=int(valid.sum())))
        if valid.sum() >= node.min_periods:
            output[row] = sample[valid].mean(axis=0)
    return output, coverage


def descriptor(spec, node, column):
    calculation = dict(operator=type(node).__name__, window=node.window,
        min_periods=node.min_periods, venue=node.venue, mass_tolerance=node.mass_tolerance,
        base=2, source=deepcopy(spec['calculation']))
    if isinstance(node, SpatialFixtureDistance):
        calculation['comparison'] = node.comparison
    return dict(kind='scalar', columns=[column], side=spec.get('side'),
        orientation='team', operators=[], derived=True, grid_size=spec['grid_size'],
        field=column, field_label=column.replace('_', ' ').capitalize(), units='JS distance (base 2, 0–1)',
        formula_version='spatial_js_v1', source=type(node).__name__,
        sources=[deepcopy(spec)], calculation=calculation)


def deviation_values(history, frame, candidates, node):
    outputs, specs, audit = {}, {}, {}
    for name, spec in frame.attrs['spatial_features'].items():
        values = unit_maps(frame, spec, node.mass_tolerance)
        baseline, coverage = prior_maps(values, candidates, node)
        column = name+'::deviation'
        outputs[column] = js_distance(values, baseline)
        specs[column] = descriptor(spec, node, column)
        audit[column] = coverage
    result = pd.DataFrame(outputs, index=history.index)
    result.attrs.update(spatial_features=specs, spatial_distance_coverage=audit)
    return result


def fixture_values(history, for_frame, against_frame, candidates, node, cutoffs):
    from .history import aligned_times
    spec = next(iter(for_frame.attrs['spatial_features'].values()))
    own, own_coverage = prior_maps(unit_maps(for_frame, spec, node.mass_tolerance), candidates, node)
    against, against_coverage = None, None
    if against_frame is not None:
        other_spec = next(iter(against_frame.attrs['spatial_features'].values()))
        against, against_coverage = prior_maps(unit_maps(against_frame, other_spec, node.mass_tolerance), candidates, node)
    times = aligned_times(history, cutoffs, default='kickoff_at')
    keys = [k for k in ('source_league','source_season','competition_id','season_id','event_id') if k in history]
    groups = {}
    for row, key in enumerate(history[keys].itertuples(index=False, name=None)):
        groups.setdefault(key, []).append(row)
    output = np.full(len(history), np.nan)
    ids = history[['team_id','opponent_id','side']].to_dict('records')
    for key, rows in groups.items():
        sides = {ids[r]['side']:r for r in rows}
        if len(rows) > 2 or len(sides) != len(rows) or set(sides)-{'home','away'}:
            raise ValueError(f'Spatial fixture distance needs unique home/away rows for {key}.')
        if len(rows) < 2:
            continue
        home, away = sides['home'], sides['away']
        if ids[home]['team_id'] != ids[away]['opponent_id'] or ids[away]['team_id'] != ids[home]['opponent_id']:
            raise ValueError(f'Spatial fixture distance has inconsistent team identities for {key}.')
        if pd.isna(times.iloc[home]) or pd.isna(times.iloc[away]) or times.iloc[home] != times.iloc[away]:
            raise ValueError(f'Spatial fixture distance needs identical nonmissing prediction cutoffs for {key}.')
        if node.comparison == 'style':
            a, b = own[home], own[away]
        elif node.comparison == 'home_context':
            a, b = own[home], against[away, ::-1]
        else:
            a, b = own[away], against[home, ::-1]
        output[rows] = js_distance(a[None,:], b[None,:])[0]
    column = node.comparison+'_distance'
    metadata = descriptor(spec, node, column)
    metadata.update(scope='fixture', side=None)
    if against_frame is not None:
        metadata['sources'].append(deepcopy(other_spec))
    result = pd.DataFrame({column:output}, index=history.index)
    result.attrs.update(spatial_features={column:metadata},
        spatial_distance_coverage=dict(for_maps=own_coverage, against_maps=against_coverage))
    return result
