"""Fixture-level spatial feature inspection; the prepared values are authoritative."""
from dataclasses import dataclass
from typing import ClassVar
from copy import deepcopy
import numpy as np
import pandas as pd
from .contracts import Artifact, StudyResult
from .teams import TeamCatalog, team_key


def spatial_descriptors(context):
    """Use declared output metadata, not feature-name parsing."""
    specs = deepcopy(context.definitions.get('spatial_features', {}))
    if not specs:
        specs = deepcopy(context.X.attrs.get('spatial_features', {}))
    return specs


def point_coverage_tables(context, positions):
    """Scope identity-keyed diagnostics to exactly the displayed fixtures."""
    audit = context.definitions.get('spatial_point_audit', {})
    if not audit:
        return {}
    metadata = context.metadata.iloc[positions]
    keys = [c for c in ('source_league', 'source_season', 'event_id') if c in metadata]
    if context.layout != 'match':
        keys.append('team_id')
    allowed = set(metadata[keys].itertuples(index=False, name=None))
    tables = {}
    for kind, name in (('maps', 'point_map_coverage'), ('history', 'point_history_coverage')):
        records = []
        for feature, entries in audit.get('features', {}).items():
            for record in entries.get(kind, []):
                if tuple(record[k] for k in keys) in allowed:
                    records.append({'feature':feature, **record})
        tables[name] = pd.DataFrame(records)
    maps = tables['point_map_coverage']
    if not maps.empty:
        scope = [c for c in ('feature', 'source_league', 'source_season', 'side') if c in maps]
        fixture = maps.groupby([*scope, 'event_id'], dropna=False).agg(
            teams=('team_id', 'nunique'), usable_teams=('valid_map', 'sum')).reset_index()
        fixture['both_teams_usable'] = (fixture.teams == 2) & (fixture.usable_teams == 2)
        tables['point_fixture_coverage'] = fixture.groupby(scope, dropna=False).agg(
            fixtures=('event_id', 'size'), both_teams_usable=('both_teams_usable', 'sum')).reset_index()
    return tables


@dataclass
class SpatialFixtureData:
    """Compact numerical store. Only a requested fixture/map becomes JSON live.

    New operators declare kind grid/region/scalar plus columns and optional region,
    grid_size, orientation, side and operators. Embeddings require their own view.
    """
    values: pd.DataFrame
    fixtures: list
    maps: dict
    teams: dict
    cache_size: int = 8

    def manifest(self, limit=None):
        fixtures = self.fixtures if limit is None else self.fixtures[:limit]
        return dict(fixtures=[{k:v for k,v in f.items() if k != 'rows'} for f in fixtures],
                    maps=[dict(id=k, label=v['label'], kind=v['spec']['kind']) for k,v in self.maps.items()],
                    teams=self.teams, cache_size=self.cache_size,
                    total_fixtures=len(self.fixtures), included_fixtures=len(fixtures))

    def panel_pair(self, fixture_id, map_id):
        position = int(fixture_id)
        if str(position) != str(fixture_id) or position < 0 or position >= len(self.fixtures):
            raise KeyError('Unknown spatial fixture.')
        fixture, feature = self.fixtures[position], self.maps[map_id]
        panels = []
        for side in ('home', 'away'):
            spec = feature['sides'].get(side) or feature['sides'].get('team')
            row = fixture['rows'].get(side)
            panel = dict(side=side, team_id=fixture[side])
            if row is None or spec is None:
                panels.append({**panel, 'missing':True}); continue
            values = self.values.iloc[row][spec['columns']].to_numpy(dtype=float, na_value=np.nan)
            spec = {k:v for k,v in spec.items() if k != 'columns'}
            panels.append({**panel, 'spec':spec, 'values':[float(v) if np.isfinite(v) else None for v in values],
                           'missing':not np.isfinite(values).any()})
        return dict(panels=panels, scale=feature['scale'])

    def offline(self, limit=None):
        manifest = self.manifest(limit)
        return dict(manifest=manifest, pairs={f['id']:{m:self.panel_pair(f['id'], m) for m in self.maps}
                                             for f in manifest['fixtures']})


@dataclass
class HeatmapReporter:
    """Select one fixture and show its prepared home/away spatial features.

    No averaging, feature recalculation or fitting occurs in the viewer. Declared
    grids use pitch heatmaps; region outputs shade their region and show the value;
    scalar outputs show a value. Unsupported representations require a renderer.
    """
    type: str
    partition: str
    maps: object = None
    teams: object = None
    catalog: TeamCatalog | None = None
    show_badges: bool = True
    max_fixtures: int | None = None
    cache_size: int = 8
    supported_types: ClassVar[tuple[str, ...]] = ('overall', 'per_fold')

    def run(self, context):
        for name, value in [('cache_size', self.cache_size), ('max_fixtures', self.max_fixtures)]:
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1):
                raise ValueError(f'{name} must be a positive integer.')
        specs = spatial_descriptors(context)
        requested = None if self.maps is None else ([self.maps] if isinstance(self.maps, str) else list(self.maps))
        if requested == []:
            raise ValueError('Select at least one spatial feature, or disable Maps for all.')
        families, columns = {}, []
        for name, spec in specs.items():
            family = spec.get('family', name)
            if requested is not None and family not in requested and name not in requested:
                continue
            if spec['kind'] not in ('grid', 'region', 'scalar'):
                continue
            if not set(spec['columns']) <= set(context.X.columns):
                raise ValueError(f'Spatial feature {name!r} is incomplete. Inspect it before selecting individual cells.')
            if spec['kind'] == 'grid' and len(spec['columns']) != spec['grid_size']**2:
                raise ValueError(f'Spatial feature {name!r} has an incomplete grid.')
            operations = ', '.join(('Rolling mean' if o['operator'] == 'RollingMean' else o['operator']) + (f" {o['window']} matches" if o.get('window') else f" lag {o['periods']}" if o.get('periods') else f" span {o['span']}" if o.get('span') else '') + (' · same venue' if o.get('venue') == 'same' else ' · all venues') for o in spec.get('operators', []))
            label = ' · '.join(filter(None, [spec.get('feature', family), spec.get('side'), spec.get('region'), operations]))
            entry = families.setdefault(family, dict(label=label, spec=spec, sides={}, scale=[0., 0.]))
            entry['sides'][spec.get('fixture_side') or 'team'] = spec
            columns.extend(spec['columns'])
            values = context.X[spec['columns']].to_numpy(dtype=float, na_value=np.nan)
            finite = values[np.isfinite(values)]
            if len(finite):
                entry['scale'][0] = min(entry['scale'][0], float(finite.min()))
                entry['scale'][1] = max(entry['scale'][1], float(finite.max()))
        if not families:
            raise ValueError('HeatmapReporter needs declared spatial feature metadata. Prepare Heatmap, SpatialPointSummary or RegionMass features first.')
        if requested is not None and any(k not in families and k not in specs for k in requested):
            raise KeyError('Unknown spatial feature selection.')
        catalog = self.catalog or TeamCatalog({})
        teams, grouped = {}, {}
        meta = context.metadata
        keys = list(context.match_columns)
        records = list(meta.itertuples(index=False, name=None))
        indexes = {c:i for i,c in enumerate(meta.columns)}
        allowed = None if self.teams is None else {team_key(x) for x in self.teams}
        for pos, row in enumerate(records):
            identity = tuple(team_key(row[indexes[k]]) for k in keys)
            if context.layout == 'match':
                home, away = (team_key(row[indexes[k]]) for k in ('home_id', 'away_id'))
                row_roles = {'home':pos, 'away':pos}
            else:
                side = row[indexes['side']]
                focal, opponent = (team_key(row[indexes[k]]) for k in ('team_id', 'opponent_id'))
                home, away = (focal, opponent) if side == 'home' else (opponent, focal)
                row_roles = {side:pos}
            if allowed is not None and not {home, away}.intersection(allowed):
                continue
            for team in (home, away):
                if team not in teams:
                    name, badge, _ = catalog.display(team, badges=self.show_badges)
                    teams[team] = dict(name=name, badge=badge)
            detail = ' · '.join(str(row[indexes[c]]) for c in ('source_league','source_season','round') if c in indexes)
            label = ' · '.join(filter(None, [f"{teams[home]['name']} vs {teams[away]['name']}", detail]))
            item = grouped.setdefault(identity, dict(id=str(len(grouped)), home=home, away=away,
                         label=label, rows={}))
            item['rows'].update(row_roles)
        fixtures = list(grouped.values())
        if self.max_fixtures is not None:
            fixtures = fixtures[:self.max_fixtures]
        # Retain each scoped row once, rather than duplicating a table per plot.
        kept = sorted({r for f in fixtures for r in f['rows'].values()})
        positions = {r:i for i,r in enumerate(kept)}
        for fixture in fixtures:
            fixture['rows'] = {k:positions[v] for k,v in fixture['rows'].items()}
        values = context.X.iloc[kept][list(dict.fromkeys(columns))].reset_index(drop=True)
        data = SpatialFixtureData(values, fixtures, families, teams, self.cache_size)
        summary = pd.DataFrame([{k:v for k,v in f.items() if k != 'rows'} for f in fixtures])
        return StudyResult('Fixture spatial features', artifacts=[Artifact('spatial', data, title='Select a fixture and spatial feature')],
                           tables={'fixtures':summary, **point_coverage_tables(context, kept)}, notes=[
            'Panels show the exact prepared feature values for this fixture; no additional averaging is applied.',
            'Home-oriented grids share one pitch frame: Home goal left, Away goal right. Against describes opponents historically faced by each focal team.',
            'The displayed feature declares all-venue or same-venue history. Missing values remain unavailable.',
            'Point summaries use unit-weight activity observations, population spatial spread and the team-relative frame. Coverage tables are diagnostics, not additional predictors.',
            f'{len(fixtures)} fixtures available. Live views cache at most {self.cache_size} fixture/feature pairs. Offline exports embed the chosen population.'])
