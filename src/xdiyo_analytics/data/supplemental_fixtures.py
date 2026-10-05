"""Append reviewed supplemental fixtures without masquerading as provider IDs."""
from datetime import datetime, timezone

import pyarrow as pa


def supplemental_event_id(source_match_id):
    """Reserved workbook-results namespace; positive provider IDs stay untouched."""
    if type(source_match_id) is not int or not 0 < source_match_id < 10**12:
        raise ValueError('A positive source match ID below 10**12 is required')
    return 10**12 + source_match_id


def append_fixture_tables(matches, statistics, nested, fixture, additions):
    """Validate and append one reviewed fixture on both dataset surfaces.

    Existing rows (including enrichment) and Arrow schemas remain unchanged.
    Unavailable components use empty lists; they are never invented.
    """
    eid = fixture['event_id']
    if type(eid) is not int or not 10**12 < eid < 2*10**12:
        raise ValueError('Use the reserved supplemental event ID namespace')
    if fixture.get('custom_id') is not None:
        raise ValueError('A supplemental fixture must not invent a provider custom ID')
    if fixture.get('status') != 'finished' or fixture.get('status_code') != 100:
        raise ValueError('Only independently verified played fixtures can be appended')
    if fixture.get('kickoff_utc') is None or not fixture.get('round'):
        raise ValueError('Reviewed kickoff and original round are required')
    old = matches.to_pylist()
    if eid in matches['event_id'].to_pylist() or eid in nested['event_id'].to_pylist():
        raise ValueError('Supplemental fixture already exists; refusing duplicate append')
    day = datetime.fromtimestamp(fixture['kickoff_utc'], timezone.utc).date()
    for r in old:
        if (r['competition_id'], r['season_id']) != (fixture['competition_id'], fixture['season_id']):
            raise ValueError('Fixture belongs to a different competition or season')
        same = (r['home_id'], r['away_id']) == (fixture['home_id'], fixture['away_id'])
        if same and (r.get('round') == fixture['round'] or (r.get('kickoff_utc') is not None and
                datetime.fromtimestamp(r['kickoff_utc'], timezone.utc).date() == day)):
            raise ValueError('An existing record already occupies this fixture; review it instead of appending')
    if matches.num_rows != nested.num_rows or not matches.equals(nested.select(matches.column_names), check_metadata=False):
        raise ValueError('Native and nested match records disagree before repair')
    if any(r['event_id'] != eid or r['team_id'] != fixture.get(r['side']+'_id') for r in additions):
        raise ValueError('Supplemental statistic identity mismatch')
    nested_row = dict(fixture)
    for field in nested.schema:
        if field.name not in nested_row:
            if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
                nested_row[field.name] = additions if field.name == 'statistics' else []
            else:
                raise ValueError(f'Unspecified match field: {field.name}')
    result = (
        pa.concat_tables([matches, pa.Table.from_pylist([fixture], schema=matches.schema)]),
        pa.concat_tables([statistics, pa.Table.from_pylist(additions, schema=statistics.schema)]),
        pa.concat_tables([nested, pa.Table.from_pylist([nested_row], schema=nested.schema)]),
    )
    for before, after in zip((matches, statistics, nested), result):
        if not before.equals(after.slice(0, len(before)), check_metadata=True):
            raise AssertionError('Existing rows changed during fixture append')
    return result
