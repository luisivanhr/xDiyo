"""Reviewed season-entry evidence for immutable named publications.

Never infer league membership from an imported fixture subset. Derivation uses
the ordinary data.movements API on separately reviewed rosters/boundaries.
"""
import pandas as pd
from ..movements import attach_movement_flags, validate_movement_flags

KEYS = ['competition_id', 'season_id', 'team_id']
MOVEMENT_POLICY = 'reviewed_season_entry_v1'
MOVEMENT_COLUMNS = (*KEYS, 'team_name', 'movement', 'got_promoted', 'got_demoted',
                    'previous_competition_id', 'previous_season_id', 'evidence', 'season_stem')
SIDE_COLUMNS = tuple(f'{s}_{f}' for s in ('home', 'away')
                     for f in ('got_promoted', 'got_demoted', 'season_entry'))


def fixture_membership(matches):
    return {tuple(row) for side in ('home', 'away') for row in
            matches[['competition_id', 'season_id', side + '_id']].itertuples(index=False, name=None)}


def validate_team_seasons(matches, flags):
    missing = set(MOVEMENT_COLUMNS) - set(flags)
    if missing:
        raise ValueError(f'Missing movement evidence columns: {sorted(missing)}')
    for col in KEYS + ['previous_competition_id', 'previous_season_id']:
        if not pd.api.types.is_integer_dtype(flags[col]) or flags[col].dropna().le(0).any():
            raise ValueError(f'{col} requires exact positive integer movement identities')
    if flags[KEYS].isna().any().any() or flags.duplicated(KEYS).any():
        raise ValueError('Movement evidence requires unique nonmissing team-season keys')
    if set(flags[KEYS].itertuples(index=False, name=None)) != fixture_membership(matches):
        raise ValueError('Movement evidence must cover exactly the published team-season membership')
    validate_movement_flags(flags)
    if flags.evidence.isna().any() or flags.evidence.astype(str).str.strip().eq('').any():
        raise ValueError('Movement records require evidence, including reasons for unknown entries')
    known = flags.movement.ne('unknown')
    if flags.loc[known, ['got_promoted', 'got_demoted']].isna().any().any():
        raise ValueError('Known movement requires both flags')
    expected = attach_movement_flags(matches, flags)
    for col in SIDE_COLUMNS:
        if col in matches:
            a, b = matches[col], expected[col]
            if not ((a.isna() & b.isna()) | (a.notna() & b.notna() & a.eq(b))).fillna(False).all():
                raise ValueError(f'Conflicting materialized movement field: {col}')
    return expected


def enrich_movements(root, name, version, flags, *, evidence_checksums=()):
    """Publish a new version with reviewed flags; preserve the old snapshot.

    Evidence can include a full reviewed roster: restrict it to the publication
    only after derivation, never derive movements from this restricted roster.
    """
    from .storage import load_snapshot, publish_dataset
    tables, manifest, registry = load_snapshot(root, name, version)
    membership = fixture_membership(tables['matches'])
    selected = flags.loc[[key in membership for key in
                          flags[KEYS].itertuples(index=False, name=None)]].copy()
    # Existing enrichment can only be revised explicitly in a new publication.
    matches = tables['matches'].drop(columns=list(SIDE_COLUMNS), errors='ignore')
    tables['matches'] = validate_team_seasons(matches, selected)
    tables['team_seasons'] = selected
    return publish_dataset(root, name, tables, registry,
                           input_checksums=[*manifest['input_checksums'], version, *evidence_checksums],
                           summary={k: manifest['coverage'][k] for k in
                                    ('conflicting_fixtures', 'input_counts') if k in manifest['coverage']})
