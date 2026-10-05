"""Optional, independently fingerprinted season-entry enrichment."""

import hashlib
import json
from pathlib import Path
import re


def assert_movement_parity(expected, persisted, *, season='season'):
    """Compare all evidence as well as flags, ignoring serialization dtypes/order."""
    import pandas as pd
    keys = ['competition_id', 'season_id', 'team_id']
    try:
        pd.testing.assert_frame_equal(
            expected.sort_values(keys).reset_index(drop=True),
            persisted.sort_values(keys).reset_index(drop=True), check_dtype=False, check_like=True)
    except AssertionError as exc:
        raise ValueError(f'{season}: persisted movement flags/evidence differ from reviewed derivation; '
                         'use explicit reviewed rematerialization before rebuilding the audit.') from exc


def validate_movement_flags(frame):
    import pandas as pd
    import numpy as np
    for row in frame.to_dict('records'):
        movement = row.get('movement')
        if movement not in ('retained', 'promoted', 'relegated', 'other_entry', 'unknown'):
            raise ValueError(f'Unknown season movement: {movement}')
        for flag, meaning in [('got_promoted', 'promoted'), ('got_demoted', 'relegated')]:
            value = row.get(flag)
            if pd.isna(value):
                continue
            if not isinstance(value, (bool, np.bool_)) or movement == 'unknown' or bool(value) != (movement == meaning):
                raise ValueError(f'{flag} contradicts movement={movement!r}.')
        pc, ps = row.get('previous_competition_id'), row.get('previous_season_id')
        if pd.isna(pc) != pd.isna(ps):
            raise ValueError('Supply both predecessor IDs or null both.')
        if pd.notna(pc) and (movement in ('unknown', 'other_entry') or
            (movement == 'retained' and pc != row['competition_id']) or
            (movement in ('promoted', 'relegated') and pc == row['competition_id']) or
            (pc == row['competition_id'] and ps == row['season_id'])):
            raise ValueError('Predecessor is incompatible with the declared movement.')


def derive_movements(rosters, competitions, *, reviewed_seasons=(), boundaries=()):
    """Compare ID rosters, with explicit membership review and boundary evidence.

    A new entrant absent from the reviewed previous upper and same division is
    inferred promoted from below. This is a membership assumption, recorded in
    evidence; administrative exceptions belong in boundary overrides. Flags
    describe entry INTO a season, never relegation at its end.
    """
    import pandas as pd

    rosters = list(rosters)
    by_year = {(r['league'], r['year']): r for r in rosters}
    if len(by_year) != len(rosters):
        raise ValueError('Duplicate league-year roster')
    reviewed = set(reviewed_seasons)
    overrides = {}
    for boundary in boundaries:
        b = dict(boundary)
        if b['stem'] in overrides:
            raise ValueError('Duplicate boundary season stem')
        if b['stem'] not in {r['stem'] for r in rosters}:
            raise ValueError('Boundary evidence refers to an absent season')
        for key in ('promoted_ids', 'relegated_ids', 'other_entry_ids'):
            ids = b.get(key, [])
            if any(isinstance(t, bool) or not re.fullmatch(r'[1-9][0-9]*', str(t)) for t in ids):
                raise ValueError('Boundary team IDs must be positive integers or integer strings')
            b[key] = [int(t) for t in ids]
        overrides[b['stem']] = b
    rows = []
    for current in rosters:
        league, year = current['league'], current['year']
        info = competitions[league]
        previous = by_year.get((league, year - 1))
        adjacent = [r for r in rosters if r['year'] == year - 1
                    and competitions[r['league']]['system'] == info['system']
                    and abs(competitions[r['league']]['tier'] - info['tier']) == 1]
        upper_names = [name for name, c in competitions.items()
                       if c['system'] == info['system'] and c['tier'] == info['tier'] - 1]
        upper_reviewed = info['tier'] == 1 or bool(upper_names) and all(
            (r := by_year.get((name, year - 1))) is not None and r['stem'] in reviewed
            for name in upper_names)
        boundary = overrides.get(current['stem'])
        if boundary:
            marked = [str(t) for key in ('promoted_ids', 'relegated_ids', 'other_entry_ids')
                      for t in boundary.get(key, [])]
            if len(marked) != len(set(marked)) or set(marked) - set(current['teams']):
                raise ValueError('Boundary evidence has duplicate or absent team IDs')
            if not boundary.get('sources'):
                raise ValueError('Boundary evidence needs sources')
        for team, name in sorted(current['teams'].items(), key=lambda item: int(item[0])):
            movement, evidence, prior = 'unknown', 'Missing or unreviewed predecessor roster', None
            candidates = [r for r in adjacent if team in r['teams']]
            if previous and team in previous['teams']:
                movement, evidence, prior = 'retained', 'Same team ID in consecutive league seasons', previous
            elif len(candidates) == 1:
                prior = candidates[0]
                movement = 'promoted' if competitions[prior['league']]['tier'] > info['tier'] else 'relegated'
                evidence = 'Team ID observed in previous adjacent division'
            elif previous and previous['stem'] in reviewed and upper_reviewed and not candidates:
                movement = 'promoted'
                evidence = 'Inferred entry from below: absent from reviewed same/upper division rosters'
            if boundary:
                inferred_movement = movement
                applied = False
                for key, value in [('promoted_ids', 'promoted'), ('relegated_ids', 'relegated'),
                                   ('other_entry_ids', 'other_entry')]:
                    if int(team) in boundary.get(key, []):
                        movement = value
                        applied = True
                        break
                else:
                    if boundary.get('complete_review'):
                        movement = 'retained'
                        applied = True
                if applied:
                    evidence = 'Boundary review: ' + '; '.join(
                        s['url'] if isinstance(s, dict) else s for s in boundary['sources'])
                    if prior is not None and movement not in ('retained', 'promoted', 'relegated'):
                        prior = None
                    elif prior is not None and movement != inferred_movement:
                        raise ValueError('Boundary movement contradicts the observed predecessor division.')
            rows.append(dict(competition_id=current['competition_id'], season_id=current['season_id'],
                             team_id=int(team), team_name=name, movement=movement,
                             got_promoted=None if movement == 'unknown' else movement == 'promoted',
                             got_demoted=None if movement == 'unknown' else movement == 'relegated',
                             previous_competition_id=prior['competition_id'] if prior else None,
                             previous_season_id=prior['season_id'] if prior else None,
                             evidence=evidence, season_stem=current['stem']))
    frame = pd.DataFrame(rows)
    for col in ('competition_id', 'season_id', 'team_id', 'previous_competition_id', 'previous_season_id'):
        # Construct from Python ints to avoid a float intermediate for nullable IDs.
        frame[col] = pd.array([r[col] for r in rows], dtype='UInt64')
    for col in ('got_promoted', 'got_demoted'):
        frame[col] = frame[col].astype('boolean')
    validate_movement_flags(frame)
    return frame


def _movement_path(data_root, stem, source):
    # Bind enrichment to the exact matches file, including pinned old versions.
    digest = source.manifest['tables']['matches']['sha256']
    return Path(data_root) / '_team_seasons' / stem / digest / 'team_seasons.parquet'


def attach_movement_flags(matches, flags):
    """Attach both sides without coercing signed/unsigned IDs through floats."""
    import pandas as pd

    keys = ['competition_id', 'season_id', 'team_id']
    validate_movement_flags(flags)
    if flags[keys].isna().any().any() or flags.duplicated(keys).any():
        raise ValueError('Team-season enrichment needs distinct nonmissing identities')
    fields = ['got_promoted', 'got_demoted', 'movement']
    lookup = {tuple(row[:3]): row[3:] for row in flags[keys + fields].itertuples(index=False, name=None)}
    result = matches.copy()
    for side in ('home', 'away'):
        values = [lookup.get(tuple(row), (None, None, 'unknown')) for row in
                  matches[['competition_id', 'season_id', f'{side}_id']].itertuples(index=False, name=None)]
        for i, field in enumerate(fields):
            output = 'season_entry' if field == 'movement' else field
            result[f'{side}_{output}'] = pd.array([v[i] for v in values],
                                                  dtype='string' if i == 2 else 'boolean')
    return result


def _read_movements(data_root, stem, source):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = _movement_path(data_root, stem, source)
    if not path.exists():
        raise KeyError(f'Team-season enrichment unavailable for this version of {stem}')
    manifest = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest['sha256']:
        raise ValueError('Team-season enrichment fingerprint mismatch')
    frame = pq.read_table(pa.BufferReader(payload)).to_pandas(types_mapper=pd.ArrowDtype)
    keys = ['competition_id', 'season_id', 'team_id']
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError('Team-season enrichment needs distinct nonmissing identities')
    scope = source.manifest['scope']
    if not (frame.competition_id.eq(scope['league_id']) & frame.season_id.eq(scope['season_id'])).all():
        raise ValueError('Team-season enrichment belongs to another season')
    return frame, {'path': str(path.resolve()), 'sha256': manifest['sha256']}


def save_movement_enrichment(data_root, season_stem, frame):
    """Write an additive table; leave every existing export and manifest intact.

    Re-running identical content is allowed. Changed enrichment needs explicit
    removal/review instead of silently replacing a table used by an experiment.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ._source import _open_source

    source = _open_source(data_root, season_stem)
    scope = source.manifest['scope']
    selected = frame.loc[frame.competition_id.eq(scope['league_id']) & frame.season_id.eq(scope['season_id'])]
    path = _movement_path(data_root, season_stem, source)
    buffer = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pandas(selected.reset_index(drop=True), preserve_index=False), buffer,
                   compression='zstd')
    payload = buffer.getvalue().to_pybytes()
    if path.exists() and path.read_bytes() != payload:
        raise FileExistsError(f'Existing enrichment differs: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(payload)
    metadata = {'sha256': hashlib.sha256(payload).hexdigest(), 'rows': len(selected),
                'matches_sha256': source.manifest['tables']['matches']['sha256']}
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return metadata
