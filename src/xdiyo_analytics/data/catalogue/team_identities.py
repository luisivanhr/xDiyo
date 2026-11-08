"""Reviewed, dated team-name bridges into the existing canonical registry.

This changes identity resolution only. It cannot establish squad eligibility,
competition membership, schedule completeness or result availability.
"""
from datetime import date
import hashlib
import json
from pathlib import Path
import re

from .registry import Registry, name_key


def identity_digest(document):
    payload = {key: value for key, value in document.items() if key != 'sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def load_team_identity_decisions(path=None):
    """Load the shipped reviewed names, or an explicitly supplied decision file."""
    path = Path(path) if path is not None else Path(__file__).with_name('data') / 'team_identities.json'
    document = json.loads(path.read_text(encoding='utf-8'))
    _validate_document(document)
    return document


def _validate_document(document):
    if document.get('schema') != 1 or document.get('policy') != 'dated_reviewed_native_team_aliases_v1':
        raise ValueError('Unsupported team identity decision format')
    if document.get('sha256') != identity_digest(document):
        raise ValueError('Team identity decisions changed: SHA-256 mismatch')
    if not isinstance(document.get('records'), list):
        raise ValueError('Team identity records must be a list')


def apply_team_identity_decisions(registry, document, *, catalog=None):
    """Return a new registry with validated aliases; never mutate the input.

    Existing IDs are retained. New club IDs require an independently reviewed
    catalogue entry. Overlapping aliases for different IDs fail before any
    result is returned. Date ranges are inclusive, as in Registry.resolve.
    Old dataset publications and feature/model artifacts are not rewritten.
    """
    _validate_document(document)
    if catalog is None:
        from ...reporting.teams import TeamCatalog
        catalog = TeamCatalog.builtin()
    result = Registry(registry.payload)
    checked_badges = set()
    for row in document['records']:
        required = {'team_id', 'canonical_name', 'name', 'competition_id', 'start', 'end', 'evidence'}
        if set(row) != required:
            raise ValueError('Unexpected team identity decision fields')
        identity, competition = row['team_id'], row['competition_id']
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in (identity, competition)):
            raise ValueError('Team and competition IDs must be exact positive integers')
        if str(competition) not in result.payload['entities'].get('competition', {}):
            raise ValueError(f'Unknown competition {competition}')
        start, end = row['start'], row['end']
        if (not isinstance(start, str) or not isinstance(end, str)
                or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', start)
                or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', end)
                or date.fromisoformat(start) > date.fromisoformat(end)):
            raise ValueError('Invalid effective team identity dates')
        if not isinstance(row['name'], str) or not row['name'].strip():
            raise ValueError('A team identity decision needs a nonempty name')
        evidence = row['evidence']
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError('A team identity decision needs review evidence')
        entry = catalog.entries.get(str(identity))
        if entry is None or entry.get('name') != row['canonical_name']:
            raise ValueError(f'Team {identity}: catalogue identity does not match the review')
        # Check the artwork too: identity success must not hide a broken badge.
        if identity not in checked_badges:
            _, image, error = catalog.display(identity, badges=True)
            if error or image is None:
                raise ValueError(f'Team {identity}: reviewed badge unavailable ({error})')
            digest = hashlib.sha256(Path(entry['badge_path']).read_bytes()).hexdigest()
            if entry.get('sha256') != digest:
                raise ValueError(f'Team {identity}: reviewed badge hash mismatch')
            checked_badges.add(identity)
        existing = result.payload['entities'].get('team', {}).get(str(identity))
        if existing is None:
            result.seed('team', identity, name=row['canonical_name'], entity_kind='club')
        elif existing.get('name') != row['canonical_name'] or existing.get('entity_kind') != 'club':
            raise ValueError(f'Team {identity}: existing canonical identity conflicts with review')
        key = ('team', str(competition), name_key(row['name']))
        for previous in result.aliases.get(key, []):
            if previous['id'] != identity and start <= previous['end'] and previous['start'] <= end:
                raise ValueError(f'Conflicting team alias {row["name"]!r} in {competition}: '
                                 f'{identity} versus {previous["id"]}, {start} through {end}')
        result.alias('team', identity, row['name'], scope=str(competition),
                     start=start, end=end,
                     decision=f'reviewed_team_identity:{document["sha256"]}; {evidence}')
    return result
