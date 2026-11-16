"""Additive annual calendar metadata; native fixture and tournament IDs stay intact."""

import hashlib
import json
from numbers import Integral
from pathlib import Path
import re

POLICY = 'native_annual_calendar_v1'


def validate_calendar(calendar, scope):
    if (calendar.get('policy') != POLICY or
            type(calendar.get('competition_id')) is not int or type(calendar.get('season_id')) is not int or
            calendar.get('competition_id') != scope['league_id'] or
            calendar.get('season_id') != scope['season_id']):
        raise ValueError('Season calendar policy or native identity mismatch')
    for key in ('annual_league', 'annual_season', 'phase'):
        if not isinstance(calendar.get(key), str) or not calendar[key]:
            raise ValueError(f'Season calendar needs {key}')
    if 'half' not in calendar or calendar['half'] is not None and (type(calendar['half']) is not int or calendar['half'] not in (1, 2)):
        raise ValueError('Season half must be 1, 2 or missing')
    for tid, entry in calendar.get('tournaments', {}).items():
        if (not re.fullmatch(r'[1-9][0-9]*', tid) or not isinstance(entry.get('phase'), str)
                or not entry['phase'] or 'half' not in entry
                or entry['half'] is not None and (type(entry['half']) is not int or entry['half'] not in (1, 2))):
            raise ValueError('Invalid tournament calendar entry')


def attach_calendar(matches, calendar):
    """Expose annual grouping alongside, never instead of, native identities."""
    import pandas as pd
    result = matches.copy()
    values = dict(annual_league=calendar['annual_league'], annual_season=calendar['annual_season'])
    mapping = calendar.get('tournaments', {})
    if mapping:
        if 'tournament_id' not in matches or matches.tournament_id.isna().any():
            raise ValueError('Calendar requires exact tournament IDs')
        if any(not isinstance(t, Integral) or isinstance(t, bool) or t <= 0 for t in matches.tournament_id):
            raise ValueError('Calendar requires exact positive integer tournament IDs')
        entries = [mapping.get(str(int(t))) for t in matches.tournament_id]
        if any(e is None for e in entries):
            raise ValueError('Unreviewed tournament in annual calendar')
    else:
        entries = [calendar] * len(matches)
    # season_phase is an existing numeric feature input (notably Japan 2016).
    # Annual descriptive labels must never overwrite that separate contract.
    values['annual_phase'] = [e['phase'] for e in entries]
    values['season_half'] = [e['half'] for e in entries]
    for key, value in values.items():
        if key in result:
            raise ValueError(f'Reserved calendar column already exists: {key}')
        result[key] = pd.Series(value, index=result.index, dtype='Int8' if key == 'season_half' else 'string')
    return result


def publish_calendar(data_root, stem, calendar):
    """Recoverable two-manifest publication, without rewriting observations.

    A prepared journal pins old and new bytes. A retry completes interrupted
    publication only if every live/staged file still matches those fingerprints.
    """
    from ._source import _open_source
    root = Path(data_root).resolve()
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*_[0-9]{2}_[0-9]{2}', stem):
        raise ValueError('Invalid season stem')
    journal = root / '_team_seasons' / 'calendars' / (stem + '.pending.json')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

    def finish(plan):
        files = []
        for item in plan['files']:
            path = (root / item['path']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Calendar publication must stay inside dataset')
            temp = path.with_suffix('.calendar-tmp')
            current = digest(path)
            if current not in (item['before'], item['after']):
                raise ValueError(f'Calendar publication changed concurrently: {path}')
            if current != item['after'] and (not temp.exists() or digest(temp) != item['after']):
                raise ValueError(f'Invalid staged calendar: {path}')
            files.append((path, temp, current == item['after']))
        for path, temp, complete in files:
            if not complete:
                temp.replace(path)
        journal.with_suffix('.audit.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
        journal.unlink()

    if journal.exists():
        plan = json.loads(journal.read_bytes())
        if plan['calendar'] != calendar:
            raise ValueError('Pending calendar differs from requested calendar')
        finish(plan)
        return
    source = _open_source(root, stem)
    validate_calendar(calendar, source.manifest['scope'])
    publication_path = root / (stem + '.manifest.json')
    files = []
    for path in (source.path, publication_path):
        value = json.loads(path.read_bytes())
        if value.get('season_calendar') == calendar:
            continue
        if 'season_calendar' in value:
            raise ValueError('Existing annual calendar differs; explicit review required')
        before = digest(path)
        value['season_calendar'] = calendar
        temp = path.with_suffix('.calendar-tmp')
        temp.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
        files.append(dict(path=path.relative_to(root).as_posix(), before=before, after=digest(temp)))
    if files:
        plan = dict(calendar=calendar, files=files)
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
        finish(plan)
