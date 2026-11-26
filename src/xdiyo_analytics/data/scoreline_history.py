"""Materialize xDiyo paired goals history without fitting or running ratings."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ._source import _open_source
from .movements import validate_movement_flags
from .scorelines import sha

KEYS = ['competition_id', 'season_id', 'team_id']
PREVIOUS = ['previous_competition_id', 'previous_season_id']
NATIVE = 'SofaScore'
ARCHIVE_COMPETITION = 1 << 60
ARCHIVE_SEASON = 2 << 60
ARCHIVE_EVENT = 3 << 60


def local_id(namespace, *key):
    digest = hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).digest()
    return namespace + (int.from_bytes(digest[:8], 'big') & ((1 << 60) - 1))


def _check_ids(rows, id_column, key_columns):
    names = list(dict.fromkeys([id_column, *key_columns]))
    pairs = rows[names].drop_duplicates()
    if pairs[id_column].duplicated().any() or pairs.duplicated(key_columns).any():
        raise ValueError(f'Non-bijective {id_column} for {key_columns}')


def read_native_flags(root, pins):
    """Use existing match-level flags and roster members from the pinned publications."""
    import pyarrow.parquet as pq
    fields = ['competition_id', 'season_id', 'home_id', 'away_id']
    extra = [f'{side}_season_entry' for side in ('home', 'away')]
    flags = {}
    for pin in pins:
        opened = _open_source(Path(root), pin['stem'])
        publication = Path(root) / (pin['stem'] + '.manifest.json')
        path = opened.table_path('matches')
        if (sha(publication) != pin['publication_sha256'] or
                sha(opened.path) != pin['manifest_sha256'] or sha(path) != pin['matches_sha256']):
            raise ValueError(f"Native tables differ from scorelines input: {pin['stem']}")
        table = pq.ParquetFile(path)
        frame = table.read(columns=[x for x in fields + extra if x in table.schema.names]).to_pandas()
        for side in ('home', 'away'):
            col = f'{side}_season_entry'
            if col not in frame:
                frame[col] = 'unknown'
            for c, s, t, label in frame[fields[:2] + [f'{side}_id', col]].drop_duplicates().itertuples(index=False, name=None):
                if pd.isna(t):
                    continue
                k = int(c), int(s), int(t)
                label = 'unknown' if pd.isna(label) else label
                old = flags.get(k, 'unknown')
                if old != 'unknown' and label not in (old, 'unknown'):
                    raise ValueError(f'Inconsistent native flags for {k}')
                flags[k] = old if label == 'unknown' else label
    return flags


def _add_fixture_keys(scores):
    d = scores.copy()
    native = d.source.eq(NATIVE)
    native_keys = d.loc[native, ['league', 'competition_id']].drop_duplicates()
    candidates = native_keys.groupby('league').competition_id.agg(list).to_dict()
    if set(d.league) - set(candidates):
        raise ValueError('Scorelines contains an unrecognized league')
    for name, namespace in [('competition_id', ARCHIVE_COMPETITION),
                            ('season_id', ARCHIVE_SEASON), ('event_id', ARCHIVE_EVENT)]:
        if d.loc[native, name].isna().any() or d.loc[native, name].ge(ARCHIVE_COMPETITION).any():
            raise ValueError(f'Invalid or colliding native {name}')
        vals = []
        for row in d.itertuples(index=False):
            raw = getattr(row, name)
            if row.source == NATIVE:
                value = int(raw)
            elif name == 'competition_id':
                known = candidates[row.league]
                if pd.notna(raw) and raw in known:
                    value = int(raw)
                elif pd.isna(raw) and len(known) == 1:
                    value = int(known[0])
                else:
                    value = local_id(namespace, 'annual_archive', row.league)
            elif name == 'season_id':
                value = local_id(namespace, int(row.competition_id), row.season)
            else:
                value = local_id(namespace, row.fixture_key)
            vals.append(value)
        d[name] = pd.array(vals, dtype='Int64')
    _check_ids(d, 'event_id', ['fixture_key'])
    _check_ids(d.loc[~native], 'season_id', ['competition_id', 'season'])
    if d.groupby('competition_id').league.nunique().gt(1).any():
        raise ValueError('Competition has conflicting league labels')
    return d


def _movement_table(membership, flags, leagues):
    """Resolve prior membership without bridging a skipped season."""
    scopes = {}
    origins = defaultdict(list)
    for (c, s), g in membership.groupby(['competition_id', 'season_id'], sort=False):
        if g.league.nunique() != 1 or g.season.nunique() != 1:
            raise ValueError(f'A competition-season has inconsistent canonical labels: {(c, s)}')
        league, season = g[['league', 'season']].iloc[0]
        start, _ = map(int, season.split('/'))
        system, tier = leagues[league]
        anchor = g.prediction_at.min()
        scopes[(c, s)] = (league, start, system, tier, anchor)
        for t in g.team_id.unique():
            origins[(system, t)].append((c, s, start, tier, anchor))
    columns = KEYS + ['movement', 'got_promoted', 'got_demoted', *PREVIOUS]
    records = []
    for c, s, t in membership[KEYS].drop_duplicates().itertuples(index=False, name=None):
        league, year, system, tier, anchor = scopes[(c, s)]
        old = [r for r in origins[(system, t)] if r[2] < year and r[4] < anchor]
        previous = [r for r in old if r[2] == year - 1]
        hint = flags.get((c, s, t), 'unknown')
        prior = None

        def relation(r):
            if r[0] == c and r[3] == tier:
                return 'retained'
            if r[3] == tier + 1:
                return 'promoted'
            if r[3] == tier - 1:
                return 'relegated'
            return 'unknown'

        opts = {relation(r) for r in previous}
        # Different provider phases or an archive-only annual competition can
        # verify same-tier membership, but cannot be a retained-state bridge.
        same_league_prior = any(scopes[r[:2]][0] == league and r[3] == tier for r in previous)
        if hint != 'unknown':
            movement = hint
        elif len(opts) == 1 and 'unknown' not in opts:
            movement = next(iter(opts))
        elif same_league_prior and opts == {'unknown'}:
            movement = 'retained'
        elif not previous and tier == 2 and any(r[3] == tier for r in old):
            movement = 'promoted'
        else:
            movement = 'unknown'

        matching = [r for r in previous if relation(r) == movement and movement != 'unknown']
        if matching and movement in ('retained', 'promoted', 'relegated'):
            newest = max(r[4] for r in matching)
            preferred = [r for r in matching if r[4] == newest]
            if len(preferred) == 1:
                prior = preferred[0][:2]
        records.append((c,s,t,movement,None if movement == 'unknown' else movement == 'promoted',
                        None if movement == 'unknown' else movement == 'relegated', *(prior or (None,None))))
    # Build nullable integer arrays directly rather than inferring float64 for
    # nullable 62-bit predecessor IDs.
    d = pd.DataFrame({col: pd.array([r[i] for r in records], dtype=(
        'Int64' if col in KEYS + PREVIOUS else 'boolean' if col.startswith('got_') else 'string'))
        for i, col in enumerate(columns)})
    validate_movement_flags(d)
    return d


def prepare(scores, flags, leagues):
    """Return paired history, movement table, and deferred fixture keys/reasons."""
    native = scores.source.eq(NATIVE)
    if not scores.season.str.fullmatch(r'\d{4}/\d{4}').all() or scores.fixture_key.duplicated().any():
        raise ValueError('Expected unique fixtures and canonical two-year season labels')
    ids = ['home_model_team_id', 'away_model_team_id']
    date = pd.to_datetime(scores.source_reported_date, utc=True)
    bad_ids = scores[ids].isna().any(axis=1) | scores[ids[0]].eq(scores[ids[1]]).fillna(False)
    # Do not turn a format conversion into an eligibility or identity review.
    eligible = scores.replay_eligible & ~bad_ids
    if (scores.replay_eligible & bad_ids).any():
        raise ValueError('Previously eligible fixture has missing or self-opposed model IDs')
    allowed = scores.replay_available_at.ge(date + pd.Timedelta(days=1)) & date.notna()
    if (eligible & ~allowed).any():
        raise ValueError('Previously eligible fixture violates the prior-day rule')
    scope_cols = ['competition_id', 'season_id', 'league', 'season', 'prediction_at']
    d = _add_fixture_keys(scores)
    d['prediction_at'] = date
    d['kickoff_at'] = d.kickoff_utc.where(native, date)
    d['status'] = 'finished'
    # xDiyo interprets this column as a not-before bound, not a publication
    # timestamp. This also guards consumers that omit available_at=.
    d['result_available_at'] = d.replay_available_at
    if (eligible & (d.kickoff_at.isna() | d.kickoff_at.lt(date) |
                    d.kickoff_at.ge(date + pd.Timedelta(days=1)))).any():
        raise ValueError('Eligible fixture kickoff is inconsistent with its analysis day')
    for field in ['home_goals', 'away_goals']:
        goals = d.loc[eligible, field]
        if not (goals.notna() & goals.ge(0) & goals.mod(1).eq(0)).all():
            raise ValueError(f'Invalid goals on eligible fixture: {field}')
    # Dates are day buckets for older rows, never written back as kickoff UTC.
    base_columns = scope_cols + ['event_id', 'fixture_key', 'phase', 'source', 'source_fixture_id',
        'status', 'kickoff_at', 'kickoff_utc', 'replay_available_at', 'result_available_at',
        'date_precision', 'score_basis']
    rosters = []
    history = []
    for side, other in [('home','away'), ('away','home')]:
        r = d[scope_cols + [f'{side}_model_team_id']].rename(columns={f'{side}_model_team_id':'team_id'})
        rosters.append(r.dropna(subset=['team_id', 'prediction_at']).drop_duplicates())
        renamed = {f'{side}_model_team_id':'team_id', f'{other}_model_team_id':'opponent_id',
                   f'{side}_team':'team_name', f'{other}_team':'opponent_name',
                   f'{side}_goals':'goals_for', f'{other}_goals':'goals_against'}
        h = d.loc[eligible, base_columns + list(renamed)].rename(columns=renamed)
        h['side'] = side
        h['team_id'] = h.team_id.astype('Int64')
        h['opponent_id'] = h.opponent_id.astype('Int64')
        history.append(h)
    paired = pd.concat(history, ignore_index=True).rename(columns={'season':'source_season'})
    paired.sort_values(['kickoff_at','competition_id','season_id','event_id','side'],
                       kind='stable', ignore_index=True, inplace=True)
    roster = pd.concat(rosters, ignore_index=True)
    roster['team_id'] = roster.team_id.astype('Int64')
    # Supplement observed native membership from its full schedule, including
    # clubs not in the finished goals-only sample.
    ns = d.loc[native, scope_cols].groupby(scope_cols[:-1], as_index=False).prediction_at.min()
    scopes = {(r[0], r[1]): r for r in ns[scope_cols].itertuples(index=False, name=None)}
    extra = [(*scopes[(c,s)], t) for c,s,t in flags if (c,s) in scopes]
    if extra:
        roster = pd.concat([roster, pd.DataFrame(extra, columns=scope_cols + ['team_id'])],
                           ignore_index=True)
    moves = _movement_table(roster, flags, leagues)
    movements = paired[KEYS].drop_duplicates().merge(moves, on=KEYS, validate='one_to_one')
    # Membership can source a movement label even if the predecessor has no
    # eligible score rows for this club. No state pointer exists in that case:
    # TransitionContext rejects a predecessor scope that lacks the team.
    observed = set(movements[KEYS].itertuples(index=False, name=None))
    absent = [(pc, ps, t) not in observed for pc, ps, t in
              movements[PREVIOUS + ['team_id']].itertuples(index=False, name=None)]
    movements.loc[absent, PREVIOUS] = pd.NA
    deferred = d.loc[~eligible, ['fixture_key','league','season','source','replay_basis','team_identity']].copy()
    deferred['reason'] = 'not_in_reviewed_replay_view'
    return paired, movements, deferred


def inventory(path):
    d = pd.read_csv(path, keep_default_na=False)
    lookup = {}
    for row in d.itertuples(index=False):
        for name in row.dataset_prefixes.split(';'):
            k = (row.country, int(row.division_level))
            if name in lookup and lookup[name] != k:
                raise ValueError(f'Conflicting league inventory for {name}')
            lookup[name] = k
    if 'Liga_MX_Apertura' in lookup and 'Liga_MX_Clausura' in lookup:
        if lookup['Liga_MX_Apertura'] != lookup['Liga_MX_Clausura']:
            raise ValueError('Liga MX phase inventory differs')
        lookup['Liga_MX'] = lookup['Liga_MX_Apertura']
    return lookup


def load_scoreline_history(snapshot, *, verify_hashes=True):
    """Read xDiyo-ready inputs without remapping or joining the original datasets."""
    root = Path(snapshot)
    files = ('team_history.parquet', 'team_seasons.parquet')
    if verify_hashes:
        manifest = json.loads((root / 'manifest.json').read_bytes())
        if manifest.get('schema') != 1 or any(sha(root / name) != manifest['files'].get(name) for name in files):
            raise ValueError('Scoreline history manifest or input hashes do not match')
    return tuple(pd.read_parquet(root / name) for name in files)


def build(scorelines, parent_manifest, native_root, league_inventory, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    parent = json.loads(Path(parent_manifest).read_bytes())
    file_hash = sha(Path(scorelines))
    scores = pd.read_parquet(scorelines)
    flags = read_native_flags(native_root, parent['native_publications'])
    paired, moves, deferred = prepare(scores, flags, inventory(league_inventory))
    if len(paired) != 2 * parent['replay_rows']:
        raise ValueError('Prepared history differs from the pinned replay fixture count')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        staging = Path(tmp) / 'snapshot'
        staging.mkdir()
        for name, rows in [('team_history', paired), ('team_seasons', moves), ('deferred_fixtures', deferred)]:
            rows.to_parquet(staging / f'{name}.parquet', index=False, compression='zstd')
        receipt = dict(schema=1, scorelines_sha256=file_hash,
                       parent_manifest_sha256=sha(Path(parent_manifest)),
                       inventory_sha256=sha(Path(league_inventory)),
                       builder_sha256=sha(Path(__file__)),
                       fixtures_by_source=(paired.loc[paired.side.eq('home')].source.value_counts().to_dict()),
                       team_rows=len(paired), team_seasons=len(moves), deferred_fixtures=len(deferred),
                       movement_counts=moves.movement.value_counts().to_dict(),
                       score_basis='as_reported_not_regulation_certified',
                       files={p.name: sha(p) for p in staging.iterdir()})
        (staging / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        staging.rename(output)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scorelines','parent-manifest','native-root','league-inventory','output'):
        p.add_argument(f'--{name}', required=True)
    print(json.dumps(build(**vars(p.parse_args())), indent=2))


if __name__ == '__main__':
    main()
