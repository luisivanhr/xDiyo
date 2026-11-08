"""Prepare a reviewed directory from a supplied CSV and its original workbook.

Source join IDs remain transient. This only writes the explicitly requested
decision file; it neither changes datasets nor fetches images.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pandas as pd

from xdiyo_analytics.odds.workbook import _Workbook, NS
from xdiyo_analytics.data.catalogue.reviewed_teams import _validate
from xdiyo_analytics.data.catalogue.team_identities import identity_digest
from xdiyo_analytics.reporting import TeamCatalog


def prepare(mapping, workbook, *, catalog=None):
    catalog = TeamCatalog.builtin() if catalog is None else catalog
    with Path(mapping).open(encoding='utf-8-sig', newline='') as stream:
        supplied = list(csv.DictReader(stream))
    source_ids, target_ids = set(), set()
    for row in supplied:
        for field, seen in [('footiqo_team_id', source_ids), ('sofascore_team_id', target_ids)]:
            value = row[field]
            if not re.fullmatch('[1-9][0-9]*', value) or value in seen:
                raise ValueError(f'Duplicate or inexact mapping identifier in {field}')
            seen.add(value)
    observations = defaultdict(lambda: defaultdict(list))
    countries = defaultdict(set)
    source = _Workbook(workbook)
    try:
        with source.archive.open('xl/sharedStrings.xml') as stream:
            index = 0
            for _, node in ET.iterparse(stream, events=('end',)):
                if node.tag == NS + 'si':
                    source.strings[index] = ''.join(t.text or '' for t in node.iter(NS + 't'))
                    index += 1
                    node.clear()
        headers = source.headers('Matches_Results')
        required = {'home_team_id', 'away_team_id', 'home_team', 'away_team', 'start_datetime', 'country'}
        selected = {column: name for column, name in headers.items() if name in required}
        if set(selected.values()) != required:
            raise ValueError('Workbook identity columns changed')
        for index, cells in source.rows('Matches_Results'):
            if index == 1:
                continue
            row = {name: source.decode(cells.get(column)) for column, name in selected.items()}
            day = (pd.Timestamp('1899-12-30') + pd.to_timedelta(float(row['start_datetime']), unit='D')).round('s').date().isoformat()
            for side in ('home', 'away'):
                identity = row[side + '_team_id']
                if identity in source_ids:
                    observations[identity][row[side + '_team']].append(day)
                    countries[identity].add(row['country'])
    finally:
        source.archive.close()
    teams = []
    for row in supplied:
        aliases = observations[row['footiqo_team_id']]
        if row['team_name_footiqo'] not in aliases:
            raise ValueError('Supplied team name/ID is absent from the original workbook')
        if row['country'] not in countries[row['footiqo_team_id']]:
            raise ValueError('Supplied competition country is absent from the workbook team evidence')
        identity = int(row['sofascore_team_id'])
        teams.append(dict(team_id=identity,
                          canonical_name=catalog.entries.get(str(identity), {}).get('name', row['team_name_footiqo']),
                          country=row['country'], names=[dict(name=name, start=min(days), end=max(days)) for name, days in sorted(aliases.items())],
                          profile_url=row['sofascore_team_url'], evidence_url=row['sofascore_source_url'],
                          review_status=row['sofascore_match_status']))
    def digest(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    document = dict(schema=1, policy='supplied_country_team_directory_v1', input_sha256=digest(mapping),
                    workbook_sha256=digest(workbook), teams=sorted(teams, key=lambda r: r['team_id']))
    document['sha256'] = identity_digest(document)
    _validate(document)
    return document


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--workbook', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output path; existing decision files are not replaced')
    result = prepare(args.mapping, args.workbook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(teams=len(result['teams']), sha256=result['sha256'])))
