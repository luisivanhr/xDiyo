"""Apply supplied, country-scoped team identities without rewriting publications.

Identity evidence and artwork availability are independent. A reviewed identity
can be usable before its badge has been acquired. Existing IDs are never rebound.
"""
from datetime import date
from pathlib import Path
import json
import re
from urllib.parse import urlsplit

from .registry import Registry, name_key
from .team_identities import identity_digest


def load_reviewed_team_directory(path=None):
    path = Path(path) if path is not None else Path(__file__).with_name('data') / 'reviewed_team_directory.json'
    document = json.loads(path.read_text(encoding='utf-8'))
    _validate(document)
    return document


def _validate(document):
    if (type(document.get('schema')) is not int or document['schema'] != 1
            or document.get('policy') != 'supplied_country_team_directory_v1'
            or document.get('sha256') != identity_digest(document)):
        raise ValueError('Invalid reviewed team directory version or SHA-256')
    if not re.fullmatch('[a-f0-9]{64}', document.get('input_sha256', '')):
        raise ValueError('Reviewed team directory requires its input digest')
    if not isinstance(document.get('teams'), list):
        raise ValueError('Reviewed teams must be a list')
    ids = set()
    for row in document['teams']:
        if set(row) != {'team_id', 'canonical_name', 'country', 'names', 'profile_url', 'evidence_url', 'review_status'}:
            raise ValueError('Unexpected reviewed team fields')
        identity = row['team_id']
        if type(identity) is not int or identity <= 0 or identity in ids:
            raise ValueError('Reviewed team IDs must be unique exact positive integers')
        ids.add(identity)
        for field in ('canonical_name', 'country', 'review_status'):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f'Missing reviewed team {field}')
        profile = urlsplit(row['profile_url'])
        if row['profile_url'] and (profile.scheme != 'https' or profile.netloc != 'www.sofascore.com'
                or not re.fullmatch(r'(?:/[a-z]{2}(?:-[a-z]{2})?)?/football/team/[^/]+/' + str(identity), profile.path)
                or profile.query or profile.fragment):
            raise ValueError(f'Team {identity}: profile URL identity mismatch')
        if not row['profile_url'] and row['review_status'] != 'verified_external_mapping':
            raise ValueError(f'Team {identity}: missing reviewed profile URL')
        evidence = urlsplit(row['evidence_url'])
        if evidence.scheme != 'https' or not evidence.netloc or evidence.username or evidence.password:
            raise ValueError('Invalid reviewed identity evidence URL')
        if not isinstance(row['names'], list) or not row['names']:
            raise ValueError('Reviewed team requires dated names')
        for alias in row['names']:
            if set(alias) != {'name', 'start', 'end'} or not isinstance(alias['name'], str) or not alias['name'].strip():
                raise ValueError('Invalid reviewed team alias')
            for field in ('start', 'end'):
                value = alias[field]
                if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
                    raise ValueError('Invalid reviewed alias date')
                date.fromisoformat(value)
            if alias['start'] > alias['end']:
                raise ValueError('Reversed reviewed alias dates')


def apply_reviewed_team_directory(registry, document, *, catalog=None):
    """Return a copy with reviewed IDs and aliases for already-known leagues.

No competition, season, squad eligibility, membership or fixture is created.
Newly reviewed leagues can pick up their aliases by reapplying this directory.
Missing artwork does not invalidate an independently supplied team identity.
"""
    _validate(document)
    if catalog is None:
        from ...reporting.teams import TeamCatalog
        catalog = TeamCatalog.builtin()
    result = Registry(registry.payload)
    # These native records predate explicit country metadata. This is an exact
    # compatibility mapping, not a country inferred from an arbitrary name.
    legacy_regions = {'J1': 'Japan', 'J2': 'Japan',
                      'Liga_MX_Apertura': 'Mexico', 'Liga_MX_Clausura': 'Mexico'}
    scopes = {}
    for identity, facts in result.payload['entities'].get('competition', {}).items():
        if facts.get('entity_kind') == 'club_league':
            region = legacy_regions.get(facts.get('region'), facts.get('region'))
            scopes.setdefault(region, []).append(identity)
    for row in document['teams']:
        identity, name = row['team_id'], row['canonical_name']
        entry = catalog.entries.get(str(identity))
        if entry is not None and entry.get('name') != name:
            raise ValueError(f'Team {identity}: catalogue identity conflicts with supplied review')
        result.seed('team', identity, name=name, entity_kind='club')
        for scope in scopes.get(row['country'], []):
            for alias in row['names']:
                previous = result.aliases.get(('team', str(scope), name_key(alias['name'])), [])
                if any(a['id'] != identity and alias['start'] <= a['end'] and a['start'] <= alias['end'] for a in previous):
                    raise ValueError(f'Team {identity}: conflicting dated alias {alias["name"]!r} in {scope}')
                result.alias('team', identity, alias['name'], scope=scope,
                             start=alias['start'], end=alias['end'],
                             decision=f'supplied_team_review:{document["sha256"]}; {row["evidence_url"]}')
    return result
