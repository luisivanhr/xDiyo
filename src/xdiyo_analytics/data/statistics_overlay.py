"""Apply pinned observations to missing native statistic cells in memory."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from .loading import SeasonData

STAT_KEYS = {'cornerKicks': ('Corner kicks', 'Match overview', 'positive'),
             'totalShotsOnGoal': ('Total shots', 'Match overview', 'positive'),
             'shotsOnGoal': ('Shots on target', 'Shots', 'positive'),
             'shotsOffGoal': ('Shots off target', 'Shots', 'negative'),
             'ballPossession': ('Ball possession', 'Match overview', 'positive'),
             'expectedGoals': ('Expected goals', 'Match overview', 'positive')}
SHOT_KEYS = frozenset({'totalShotsOnGoal', 'shotsOnGoal', 'shotsOffGoal'})
def has_observation(row):
    value = row.get('value')
    if value is not None and not pd.isna(value):
        try:
            if math.isfinite(float(value)):
                return True
        except (TypeError, ValueError):
            return True  # Protect malformed native observations too.
    display = row.get('display')
    return display is not None and not pd.isna(display) and str(display).strip() not in ('', '-', '\u2014')


def normalize_overlay_selection(config):
    """Read legacy missing-only recipes without replaying a time-filtered release by mistake."""
    config = dict(config)
    if config.pop('mode', 'retrospective') != 'retrospective' or config.pop('as_of', None) is not None:
        raise ValueError('This statistics overlay used a time filter. Remove mode and as_of '
                         'and rebuild the prepared data to use fixture matching.')
    return config


def _observed_at(cell):
    """Populate optional capture metadata; fixture matching does not depend on it."""
    value = cell.get('evidence_at')
    if isinstance(value, str):
        stamp = pd.to_datetime(value, format='ISO8601', errors='coerce')
        if pd.notna(stamp) and stamp.tzinfo is not None:
            return stamp.timestamp()
    return None


def _positive_id(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{field} must be a positive integer')
    return value


def _backfill(data, config, requested):
    """Fill missing cells by competition, season, fixture, team, period and statistic."""
    config = normalize_overlay_selection(config)
    if set(config) != {'manifest_path', 'manifest_sha256'}:
        raise ValueError('Overlay requires manifest_path and manifest_sha256')
    manifest_path = Path(config['manifest_path']).resolve()
    expected = config['manifest_sha256']
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('Manifest sha256 must be lowercase hexadecimal')
    raw_manifest = manifest_path.read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != expected:
        raise ValueError('Overlay manifest hash mismatch')
    manifest = json.loads(raw_manifest)
    if set(manifest) != {'version', 'partitions'} or not isinstance(manifest['version'], str) or not manifest['version'].strip() or not isinstance(manifest['partitions'], list) or not manifest['partitions']:
        raise ValueError('Invalid overlay release manifest')
    version = manifest['version']
    payloads = []
    covered = set()
    for part in manifest['partitions']:
        if set(part) != {'path', 'sha256', 'keys'} or not isinstance(part['keys'], list) or not part['keys'] or len(set(part['keys'])) != len(part['keys']) or set(part['keys']) - set(STAT_KEYS):
            raise ValueError('Invalid overlay partition')
        # Sources can supply disjoint cells of the same statistic. Check cell identities below.
        covered.update(part['keys'])
        relative = Path(part['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Overlay partition must remain within manifest directory')
        path = (manifest_path.parent / relative).resolve()
        if not path.is_relative_to(manifest_path.parent):
            raise ValueError('Overlay partition escapes manifest directory')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != part['sha256']:
            raise ValueError('Overlay partition hash mismatch')
        payloads.append((part, gzip.decompress(raw) if path.suffix == '.gz' else raw))
    requested = set(requested)
    if not requested or requested - set(STAT_KEYS) or requested - covered:
        raise ValueError('Requested keys must be present in the overlay manifest')
    previous = data.provenance.get('statistics_overlay')
    if previous:
        normalize_overlay_selection(previous)
        if any(previous.get(k) != v for k, v in {'version': version, 'manifest_sha256': expected}.items()):
            raise ValueError('A different statistics overlay is already applied')
        requested -= set(previous['applied_keys'])
    matches = data.matches
    stats = data.statistics
    if matches is None or stats is None:
        raise ValueError('Statistics overlay requires loaded matches and statistics')
    if matches.event_id.duplicated().any():
        raise ValueError('Overlay requires unique native fixture IDs')
    if not requested:
        return data, previous
    if stats.empty and not {'event_id', 'period', 'group_name', 'key', 'side', 'value'} <= set(stats.columns):
        import pyarrow as pa
        from .statistics_backfill import statistics_schema
        stats = pa.Table.from_batches([], schema=statistics_schema()).to_pandas(types_mapper=pd.ArrowDtype)
        for column in ('source_league', 'source_season'):
            if column in matches.columns:
                stats[column] = pd.Series(dtype='string')
    events = set(matches.event_id)
    records, seen = [], set()
    for part, payload in payloads:
        for line_number, line in enumerate(payload.splitlines(), 1):
            if not line.strip():
                raise ValueError(f'Blank overlay line {line_number}')
            try:
                cell = json.loads(line)
                key = tuple(_positive_id(cell[k], k) for k in ('competition_id', 'season_id', 'event_id', 'team_id')) + (cell['period'], cell['key'])
                value = cell['value']
                if not isinstance(cell['source_ref'], str) or not cell['source_ref'].strip():
                    raise ValueError('source_ref must be nonempty')
                if key[-2] not in ('ALL', '1ST', '2ND') or key[-1] not in part['keys']:
                    raise ValueError('Invalid statistic or period for partition')
                if key[-1] == 'ballPossession':
                    if cell.get('unit') != 'percentage_points' or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
                        raise ValueError('Possession requires finite 0–100 percentage_points')
                elif key[-1] == 'expectedGoals':
                    if (cell.get('unit') != 'expected_goals' or isinstance(value, bool)
                            or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                            or cell.get('source') not in ('Footiqo', 'SofaScore')
                            or not isinstance(cell.get('provider'), str) or not cell['provider'].strip()
                            or not isinstance(cell.get('expected_goals_model'), str) or not cell['expected_goals_model'].strip()):
                        raise ValueError('Expected goals requires valid source, provider, model, unit and finite nonnegative value')
                elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError('Count statistics must be nonnegative integers')
                if key in seen:
                    raise ValueError('Duplicate overlay cell')
                seen.add(key)
                if key[-1] in requested and key[2] in events:
                    records.append((key, cell))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f'Invalid overlay line {line_number}: {exc}') from exc
    fixture = {int(row.event_id): row for row in matches.itertuples(index=False)}
    output = stats.reset_index(drop=True).copy(deep=True)
    for column in ('source_key', 'raw_hash', 'observed_at', 'source_item_json'):
        if column not in output.columns:
            output[column] = None
    stat_index = {}
    for index, event, period, key, side in output[['event_id', 'period', 'key', 'side']].itertuples(index=True, name=None):
        if key in requested:
            stat_index.setdefault((int(event), period, side, key), []).append(index)
    drop_rows = set()
    appended = []
    counts = {k: {'filled': 0, 'protected_existing': 0} for k in requested}
    providers = {key: Counter() for key in requested}
    shot_policies = {key: Counter() for key in requested}
    for (competition, season, event, team, period, stat_key), cell in records:
        match = fixture[event]
        if (competition, season) != (int(match.competition_id), int(match.season_id)) or team not in (int(match.home_id), int(match.away_id)):
            raise ValueError(f'Overlay fixture/team identity mismatch: {event}')
        if ('home_id' in cell and cell['home_id'] != match.home_id
                or 'away_id' in cell and cell['away_id'] != match.away_id
                or 'side' in cell and cell['side'] != ('home' if team == match.home_id else 'away')):
            raise ValueError(f'Overlay home/away identity mismatch: {event}')
        if getattr(match, 'status', None) != 'finished':
            raise ValueError(f'Overlay targets a non-finished fixture: {event}')
        awarded = getattr(match, 'is_awarded', None)
        if pd.isna(awarded) or bool(awarded):
            raise ValueError(f'Overlay targets an awarded fixture: {event}')
        if getattr(match, 'status_code', None) not in (100, 110, 120):
            raise ValueError(f'Overlay targets a fixture without a supported finished status code: {event}')
        play_status = getattr(match, 'play_status', '')
        if isinstance(play_status, str) and any(term in play_status.lower() for term in ('award', 'annul', 'abandon', 'unplayed', 'cancel')):
            raise ValueError(f'Overlay targets a non-sporting fixture: {event}')
        side = 'home' if team == int(match.home_id) else 'away'
        indices = stat_index.get((event, period, side, stat_key), [])
        if indices:
            rows = output.loc[indices]
            if 'team_id' in rows and any(pd.isna(t) or int(t) != team for t in rows.team_id):
                raise ValueError(f'Native statistic team identity mismatch: {event}')
            if any(has_observation(row) for row in rows.to_dict('records')):
                counts[stat_key]['protected_existing'] += 1
                continue
        # Replace null aliases with a source-stamped row, never a falsely native fill.
        drop_rows.update(indices)
        row = {column: None for column in output.columns}
        name, group, direction = STAT_KEYS[stat_key]
        row.update(event_id=event, team_id=team, period=period, key=stat_key, side=side,
                   group_name=group, name=name, display=((str(int(cell['value'])) if isinstance(cell['value'], float) and cell['value'].is_integer() else str(cell['value'])) + '%' if stat_key == 'ballPossession' else str(cell['value'])),
                   value=cell['value'], value_type='event', statistics_type=direction,
                   source_key='statistics_overlay|' + expected, raw_hash=expected,
                   observed_at=_observed_at(cell), source_item_json=json.dumps(cell, sort_keys=True),
                   group_order=0, stat_order=0, total=None)
        for col in ('source_league', 'source_season'):
            if col in output.columns:
                row[col] = getattr(match, col)
        appended.append({k:v for k,v in row.items() if k in output.columns})
        counts[stat_key]['filled'] += 1
        providers[stat_key][cell.get('source', 'unspecified')] += 1
        selection = cell.get('identity_evidence', {}).get('shot_selection', {})
        if selection.get('mode'):
            shot_policies[stat_key][selection['mode']] += 1
    if drop_rows:
        output = output.drop(index=list(drop_rows))
    if appended:
        output = pd.concat([output, pd.DataFrame(appended, columns=output.columns)], ignore_index=True)
    output['value'] = pd.to_numeric(output['value'], errors='raise').astype('Float64')
    overlay_rows = output.source_key.fillna('').astype(str).str.startswith('statistics_overlay|')
    native_rows = output.loc[~overlay_rows]
    sorted_overlay = output.loc[overlay_rows].sort_values(['event_id', 'period', 'key', 'side', 'team_id'], kind='stable')
    output = pd.concat([native_rows, sorted_overlay], ignore_index=True)
    by_key = dict(previous.get('by_key', {}) if previous else {})
    for key in sorted(requested):
        by_key[key] = dict(counts[key],
                           by_source=dict(providers[key]), shot_selection_modes=dict(shot_policies[key]))
    receipt = {'version': version, 'manifest_sha256': expected,
               'applied_keys': sorted(by_key), 'by_key': {key: by_key[key] for key in sorted(by_key)}}
    tables = dict(data.tables, statistics=output)
    provenance = dict(data.provenance, statistics_overlay=receipt)
    return SeasonData(tables, provenance), receipt


def backfill_corners(data, config, recipe=None):
    return _backfill(data, config, {'cornerKicks'})


def backfill_shots(data, config, recipe=None, keys=None):
    selected = set(SHOT_KEYS if keys is None else keys)
    if not selected or selected - SHOT_KEYS:
        raise ValueError('backfill_shots keys must select supported shot statistics')
    return _backfill(data, config, selected)


def backfill_possession(data, config, recipe=None):
    return _backfill(data, config, {'ballPossession'})


def backfill_expected_goals(data, config, recipe=None):
    return _backfill(data, config, {'expectedGoals'})


backfill_xg = backfill_expected_goals


def backfill_statistics(data, config, recipe=None, keys=None):
    if keys is None:
        raise ValueError('Select statistic keys for the overlay')
    return _backfill(data, config, keys)
