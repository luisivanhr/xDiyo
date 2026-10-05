"""Missing-only statistics enrichment; native observations always take priority."""

from collections import defaultdict
import copy
import json
import math


# Workbook fields mapped to the existing native statistic identities and units.
STATISTICS = {
    'corners': ('cornerKicks', 'Corner kicks', 'Match overview', 'positive'),
    'yellow_cards': ('yellowCards', 'Yellow cards', 'Match overview', 'negative'),
    'red_cards': ('redCards', 'Red cards', 'Match overview', 'negative'),
    'xg': ('expectedGoals', 'Expected goals', 'Match overview', 'positive'),
    'ball_possession': ('ballPossession', 'Ball possession', 'Match overview', 'positive'),
    'total_shots': ('totalShotsOnGoal', 'Total shots', 'Match overview', 'positive'),
    'shots_on_target': ('shotsOnGoal', 'Shots on target', 'Shots', 'positive'),
    'shots_off_target': ('shotsOffGoal', 'Shots off target', 'Shots', 'negative'),
    'fouls': ('fouls', 'Fouls', 'Match overview', 'negative'),
    'goalkeeper_saves': ('goalkeeperSaves', 'Goalkeeper saves', 'Match overview', 'positive'),
}
PERIODS = {'ft': 'ALL', '1h': '1ST', '2h': '2ND'}


def statistics_schema():
    """Native schema-2 layout for publications that have no statistics table."""
    import pyarrow as pa
    names = ['event_id', 'source_key', 'raw_hash', 'observed_at', 'period',
             'group_name', 'key', 'name', 'side', 'display', 'value_type',
             'statistics_type', 'source_item_json', 'team_id', 'group_order',
             'stat_order', 'value', 'total']
    integers = {'event_id', 'team_id', 'group_order', 'stat_order'}
    decimals = {'observed_at', 'value', 'total'}
    return pa.schema([(n, pa.int64() if n in integers else pa.float64() if n in decimals else pa.string())
                      for n in names], metadata={b'xdiyo_schema_version': b'2'})


def statistic_value(raw, field):
    """Blanks stay missing, zeros remain observations, invalid values are rejected."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'Invalid {field}: {raw!r}')
    if field == 'ball_possession':
        if value > 100:
            raise ValueError('Possession must be on the native 0-100 scale')
    elif field != 'xg' and not value.is_integer():
        raise ValueError(f'{field} must be an integer count')
    return value


def has_observation(row):
    """Protect even display-only native values rather than overwrite uncertain data."""
    value = row.get('value')
    return ((value is not None and math.isfinite(float(value)))
            or row.get('display') not in (None, '', '-', '—'))


def fill_missing_statistics(rows, candidates):
    """Return merged rows and a cell audit; never change a populated native field.

    candidates are native-shaped rows with a separate ``evidence`` dictionary.
    Identity across native display groups is event/period/key/side. Existing
    aliases of a statistic protect that observation even if a different group's
    row is absent. Conflicting donor copies raise instead of choosing one.
    """
    result = copy.deepcopy(rows)
    index = defaultdict(list)
    def identity(row):
        return tuple(row[k] for k in ('event_id', 'period', 'key', 'side'))
    for i, row in enumerate(result):
        index[identity(row)].append(i)
    donors = {}
    for row in candidates:
        key = identity(row)
        if row['period'] not in PERIODS.values() or row['side'] not in ('home', 'away'):
            raise ValueError('Invalid donor period or side')
        if row.get('value') is None or not math.isfinite(float(row['value'])):
            raise ValueError('Donor value must be finite')
        if key in donors and (donors[key]['value'] != row['value'] or donors[key]['team_id'] != row['team_id']):
            raise ValueError(f'Conflicting donor copies for {key}')
        donors[key] = row
    audit = []
    for key, donor in donors.items():
        existing = index[key]
        had_existing = bool(existing)
        if any(has_observation(result[i]) for i in existing):
            continue
        if any(result[i]['team_id'] != donor['team_id'] for i in existing):
            raise ValueError(f'Statistic team identity mismatch: {key}')
        payload = {k: v for k, v in donor.items() if k != 'evidence'}
        if existing:
            for i in existing:
                # Metadata from the native source is preserved; donor provenance
                # is retained separately in the cell audit.
                result[i]['value'] = donor['value']
                if result[i].get('display') in (None, '', '-', '—'):
                    result[i]['display'] = donor['display']
        else:
            result.append(payload)
            index[key].append(len(result) - 1)
        audit.append(dict(zip(('event_id', 'period', 'key', 'side'), key),
                          value=donor['value'], action='fill_null' if had_existing else 'append',
                          **donor.get('evidence', {})))
    return result, audit


def make_candidate(match, *, field, period, side, value, evidence, observed_at):
    key, name, group, direction = STATISTICS[field]
    display = f'{value:g}' + ('%' if field == 'ball_possession' else '')
    return dict(event_id=int(match['event_id']), team_id=int(match[side + '_id']),
                source_key=f"imported_statistics|{evidence['workbook_sha256']}|{evidence['sheet']}|{evidence['row']}|{evidence['column']}",
                raw_hash=evidence['workbook_sha256'], observed_at=observed_at,
                period=PERIODS[period], group_name=group, key=key, name=name,
                side=side, display=display, value_type='event', statistics_type=direction,
                source_item_json=json.dumps(evidence, sort_keys=True),
                group_order=0 if group == 'Match overview' else 1,
                stat_order=list(STATISTICS).index(field), value=value, total=None,
                evidence=evidence)
