"""Reviewable provider-to-native fixture matching, using identity evidence only."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import pandas as pd

from .workbook import _hash, load_odds

MATCH_KEYS = ('source_league', 'source_season', 'competition_id', 'season_id', 'event_id')
CROSSWALK_VERSION = 1


def exact_id(value):
    if pd.isna(value):
        raise ValueError('Fixture identity cannot be missing')
    if isinstance(value, float):
        if not value.is_integer() or abs(value) > 2**53:
            raise ValueError('Unsafe floating-point fixture ID')
        return str(int(value))
    return str(value)


def read_manifest(directory, *, crosswalk=False):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if crosswalk:
        if manifest.get('crosswalk_version') != CROSSWALK_VERSION:
            raise ValueError('Unsupported crosswalk version')
    else:
        if manifest.get('adapter_version') != '1':
            raise ValueError('Unsupported odds database version')
        identity = {k: manifest[k] for k in ('adapter_version', 'source_files', 'league_seasons', 'league_aliases')}
        expected = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        if manifest.get('snapshot') != expected:
            raise ValueError('Snapshot identity does not match its manifest')
    return manifest


def verified_path(directory, item):
    root = Path(directory).resolve()
    path = (root / item['path']).resolve()
    if not path.is_relative_to(root) or _hash(path) != item['sha256']:
        raise ValueError('Snapshot payload was modified or leaves its directory')
    return path


def vendor_fixtures(snapshot, *, seasons, leagues=None, team_aliases=None):
    """Consolidate selected fixture metadata; disagreements stay quarantined."""
    if not seasons:
        raise ValueError('Select seasons explicitly')
    manifest = read_manifest(snapshot)
    pieces = []
    for row in manifest['coverage']:
        if row['kind'] != 'fixtures' or row['source_season'] not in seasons:
            continue
        if leagues is not None and row['source_league'] not in leagues:
            continue
        pieces.append(load_odds(snapshot, league=row['source_league'], season=row['source_season'],
                                   sheet=row['sheet'], kind='fixtures'))
    if not pieces:
        raise ValueError('No fixture metadata for the selected league-seasons')
    frame = pd.concat(pieces, ignore_index=True)
    raw = frame.start_datetime
    numeric = pd.to_numeric(raw, errors='coerce')
    dates = pd.to_datetime(raw.where(numeric.isna()), errors='coerce')
    serial = numeric.notna()
    dates.loc[serial] = pd.to_datetime(numeric.loc[serial], unit='D', origin='1899-12-30')
    # Excel serialization differences below one second are not fixture conflicts.
    frame['vendor_scheduled_at'] = dates.dt.round('s')
    rows = []
    for _, group in frame.groupby(['provider', 'snapshot', 'match_id'], sort=False):
        row = group.iloc[0].to_dict()
        evidence = ['source_league', 'source_season', 'country', 'home_team', 'away_team', 'vendor_scheduled_at']
        aliases = team_aliases or {}
        def identities(column):
            values = group[column]
            if column in ('home_team', 'away_team'):
                values = values.map(lambda name: ('native', exact_id(aliases[(row['source_league'], name)]))
                    if (row['source_league'], name) in aliases else ('name', str(name).strip().casefold()))
            return values
        row['metadata_conflict'] = any(identities(c).nunique(dropna=False) > 1 for c in evidence)
        row['source_evidence'] = json.dumps(group[['source_file', 'source_sheet', 'source_row', 'start_datetime',
            'home_team', 'away_team', 'home_team_id', 'away_team_id']].to_dict('records'))
        rows.append(row)
    return pd.DataFrame(rows)


def build_fixture_crosswalk(snapshot, native, *, seasons, leagues=None, team_aliases=None,
                            date_tolerance_days=0, native_timezone='UTC', fixture_overrides=None):
    """Match ordered teams and calendar dates, never prices or outcomes.

    native contains MATCH_KEYS, home_id/away_id, home_name/away_name, kickoff_at.
    team_aliases maps (native league, vendor team name) to exact native team IDs.
    Exact unique native names are accepted automatically. Unknown vendor timezone
    remains unknown: dates are compared to native kickoff in native_timezone.
    Unmatched, conflicting and ambiguous rows are retained for review.
    """
    if isinstance(date_tolerance_days, bool) or not isinstance(date_tolerance_days, int) or not 0 <= date_tolerance_days <= 7:
        raise ValueError('date_tolerance_days must be an explicit integer from 0 to 7')
    saved_aliases, saved_overrides = {}, []
    for directory in (Path(snapshot).parent, Path(snapshot).parent.parent):
        if (directory/'team_aliases.csv').exists():
            saved_aliases, saved_overrides = load_mapping_rules(directory)
            break
    if fixture_overrides is None:
        fixture_overrides = saved_overrides
    columns = [*MATCH_KEYS, 'home_id', 'away_id', 'home_name', 'away_name', 'kickoff_at']
    missing = set(columns) - set(native)
    if missing:
        raise ValueError(f'Native fixture metadata is missing {sorted(missing)}')
    data = native.loc[native.source_season.isin(seasons), columns].copy()
    if leagues is not None:
        data = data.loc[data.source_league.isin(leagues)]
    for key in (*MATCH_KEYS, 'home_id', 'away_id'):
        data[key] = data[key].map(exact_id)
    if data.duplicated(list(MATCH_KEYS)).any():
        raise ValueError('Native fixture identities must be unique')
    data['native_kickoff_at'] = pd.to_datetime(data.kickoff_at, utc=True, errors='raise')
    data['comparison_date'] = data.native_kickoff_at.dt.tz_convert(native_timezone).dt.tz_localize(None).dt.normalize()
    names = defaultdict(set)
    for _, row in data.iterrows():
        for side in ('home', 'away'):
            if pd.notna(row[side+'_name']):
                names[(row.source_league, str(row[side+'_name']).strip().casefold())].add(row[side+'_id'])
    aliases = {**saved_aliases, **dict(team_aliases or {})}
    def resolve(league, name):
        value = aliases.get((league, name))
        if value is not None:
            return exact_id(value)
        ids = names.get((league, str(name).strip().casefold()), set())
        return next(iter(ids)) if len(ids) == 1 else None
    groups = {key: group for key, group in data.groupby(['source_league', 'source_season', 'home_id', 'away_id'], sort=False)}
    rows = []
    for _, vendor in vendor_fixtures(snapshot, seasons=seasons, leagues=leagues, team_aliases=aliases).iterrows():
        record = dict(provider=vendor.provider, snapshot=vendor.snapshot, vendor_match_id=vendor.match_id,
                      source_league=vendor.source_league, source_season=vendor.source_season,
                      vendor_home=vendor.home_team, vendor_away=vendor.away_team,
                      vendor_scheduled_at=vendor.vendor_scheduled_at, vendor_timezone=None,
                      native_kickoff_at=None, date_tolerance_days=date_tolerance_days,
                      comparison_timezone=native_timezone, source_evidence=vendor.source_evidence,
                      competition_id=None, season_id=None, event_id=None,
                      status='unmatched', reason='No ordered-team/date candidate', candidate_count=0)
        home, away = resolve(vendor.source_league, vendor.home_team), resolve(vendor.source_league, vendor.away_team)
        record['native_home_id'], record['native_away_id'] = home, away
        record['alias_evidence'] = json.dumps({side: {'vendor_name':name, 'native_id':team,
            'method':'explicit_alias' if (vendor.source_league,name) in aliases else 'unique_exact_name'}
            for side,name,team in [('home',vendor.home_team,home),('away',vendor.away_team,away)]}, sort_keys=True)
        if vendor.metadata_conflict:
            record.update(status='conflict', reason='Provider sheets disagree on fixture metadata')
        elif home is None or away is None:
            record.update(reason='Unknown or ambiguous team alias')
        elif pd.isna(vendor.vendor_scheduled_at):
            record.update(reason='Missing provider scheduled date')
        else:
            candidates = groups.get((vendor.source_league, vendor.source_season, home, away), data.iloc[:0])
            distance = (candidates.comparison_date - vendor.vendor_scheduled_at.normalize()).abs()
            candidates = candidates.loc[distance <= pd.Timedelta(days=date_tolerance_days)]
            record['candidate_count'] = len(candidates)
            if len(candidates) == 1:
                candidate = candidates.iloc[0]
                record.update({k: candidate[k] for k in MATCH_KEYS})
                record.update(native_kickoff_at=candidate.native_kickoff_at, status='matched',
                              reason='Unique ordered-team/calendar-date match; vendor timezone unknown')
            elif len(candidates) > 1:
                record.update(status='ambiguous', reason='Multiple ordered-team/date candidates')
        rows.append(record)
    result = pd.DataFrame(rows)
    result = apply_fixture_overrides(result, data, fixture_overrides or [])
    matched = result.status.eq('matched')
    conflicts = result.loc[matched].duplicated(list(MATCH_KEYS), keep=False)
    result.loc[conflicts.index[conflicts], ['status', 'reason']] = ['ambiguous', 'Multiple vendor fixtures map to one native fixture']
    return result


def apply_fixture_overrides(frame, native, overrides):
    """Apply reviewed schedule exceptions without relaxing matching globally.

    Each record pins both dates, league-season, vendor ID and native event ID,
    plus a review reason and supporting source. Ordered team IDs must agree.
    This prevents a playoff meeting being substituted for a regular-season one.
    """
    result = frame.copy()
    result['schedule_evidence'] = result.get('schedule_evidence', None)
    seen = set()
    for item in overrides:
        vendor_id = exact_id(item['vendor_match_id'])
        if vendor_id in seen:
            raise ValueError('Duplicate reviewed fixture override')
        seen.add(vendor_id)
        selected = result.vendor_match_id.astype(str).eq(vendor_id)
        if not selected.any():
            continue  # A review can cover seasons outside this request.
        if selected.sum()!=1 or not item.get('reason') or not item.get('evidence'):
            raise ValueError('Fixture override requires a unique fixture and review evidence')
        index = result.index[selected][0]
        row = result.loc[index]
        target = native.loc[native.source_league.eq(item['source_league']) &
            native.source_season.eq(item['source_season']) & native.event_id.map(exact_id).eq(exact_id(item['event_id']))]
        if len(target)!=1 or row.source_league!=item['source_league'] or row.source_season!=item['source_season']:
            raise ValueError('Reviewed fixture override does not identify the same league-season')
        match = target.iloc[0]
        date = pd.to_datetime(match['kickoff_at'], utc=True)
        if (exact_id(match.home_id)!=row.native_home_id or exact_id(match.away_id)!=row.native_away_id or
            pd.Timestamp(row.vendor_scheduled_at)!=pd.Timestamp(item['vendor_scheduled_at']) or
            date!=pd.to_datetime(item['native_kickoff_at'],utc=True)):
            raise ValueError('Reviewed fixture override teams or dates have changed')
        if row.status not in ('unmatched','matched') or (row.status=='matched' and exact_id(row.event_id)!=exact_id(match.event_id)):
            raise ValueError('Reviewed fixture override conflicts with existing matching evidence')
        for key in MATCH_KEYS:
            result.loc[index,key] = exact_id(match[key])
        result.loc[index,'native_kickoff_at'] = date
        result.loc[index,'status'] = 'matched'
        result.loc[index,'reason'] = 'Reviewed schedule exception: '+item['reason']
        result.loc[index,'candidate_count'] = 1
        result.loc[index,'schedule_evidence'] = json.dumps(item,sort_keys=True)
    accepted = result.loc[result.status.eq('matched')]
    if accepted.duplicated(list(MATCH_KEYS)).any() and overrides:
        raise ValueError('Reviewed overrides would assign multiple prices to the same native fixture')
    return result


def load_mapping_rules(directory):
    """Load saved team aliases and individually reviewed schedule exceptions."""
    directory=Path(directory)
    path=directory/'team_aliases.csv'
    aliases={}
    if path.exists():
        table=pd.read_csv(path,dtype=str)
        if table.duplicated(['source_league','vendor_team']).any() or table.native_team_id.isna().any():
            raise ValueError('Saved team aliases must have unique keys and native IDs')
        aliases={(r.source_league,r.vendor_team):r.native_team_id for r in table.itertuples()}
    path=directory/'fixture_overrides.json'
    overrides=json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    return aliases,overrides


def save_crosswalk(frame, *, snapshot, output_root):
    """Persist a content-addressed crosswalk, including its quarantine rows."""
    manifest = read_manifest(snapshot)
    if frame.empty or not frame.snapshot.eq(manifest['snapshot']).all():
        raise ValueError('Crosswalk must belong to the chosen snapshot')
    if frame.vendor_match_id.duplicated().any():
        raise ValueError('Crosswalk vendor identities must be unique')
    matched = frame.loc[frame.status.eq('matched')]
    if matched[list(MATCH_KEYS)].isna().any().any() or matched.duplicated(list(MATCH_KEYS)).any():
        raise ValueError('Accepted native identities must be complete and unique')
    identity = frame.to_json(orient='records', date_format='iso')
    version = hashlib.sha256(identity.encode()).hexdigest()[:24]
    directory = Path(output_root) / version
    if (directory / 'manifest.json').exists():
        read_crosswalk(directory, manifest['snapshot'])
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    payload = directory / 'fixtures.parquet'
    frame.to_parquet(payload, index=False)
    record = dict(crosswalk_version=CROSSWALK_VERSION, version=version, snapshot=manifest['snapshot'],
                  snapshot_manifest_sha256=_hash(Path(snapshot)/'manifest.json'),
                  files=[dict(path=payload.name, sha256=_hash(payload))],
                  status_counts=frame.status.value_counts().to_dict())
    (directory/'manifest.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    return directory


def read_crosswalk(directory, snapshot_id):
    manifest = read_manifest(directory, crosswalk=True)
    if manifest['snapshot'] != snapshot_id:
        raise ValueError('Crosswalk belongs to another odds snapshot')
    frame = pd.read_parquet(verified_path(directory, manifest['files'][0]))
    version = hashlib.sha256(frame.to_json(orient='records', date_format='iso').encode()).hexdigest()[:24]
    if version != manifest['version']:
        raise ValueError('Crosswalk content identity was modified')
    return frame


def native_fixture_metadata(data_root, *, seasons, leagues=None):
    """Read only native fixture identity/time fields, never score/statistic cells."""
    from ..data._source import _open_source
    import pyarrow.parquet as pq
    frames = []
    for path in sorted(Path(data_root).glob('*.manifest.json')):
        stem = path.name.removesuffix('.manifest.json')
        league, first, second = stem.rsplit('_', 2)
        season = first+'_'+second
        if season not in seasons or (leagues is not None and league not in leagues):
            continue
        source = _open_source(data_root, stem)
        columns = ['competition_id','season_id','event_id','home_id','away_id','home_name','away_name','kickoff_utc']
        path = source.table_path('matches')
        available = pq.read_schema(path).names
        frame = pd.read_parquet(path, columns=[c for c in columns if c in available])
        for column, key in [('competition_id','league_id'),('season_id','season_id')]:
            if column not in frame:
                frame[column] = source.manifest['scope'][key]
        for column in ('home_name','away_name'):
            if column not in frame:
                frame[column] = None
        frame['source_league'], frame['source_season'] = league, season
        frame['kickoff_at'] = pd.to_datetime(frame.pop('kickoff_utc'), unit='s', utc=True)
        frames.append(frame)
    if not frames:
        raise ValueError('No native fixtures for the selected seasons/leagues')
    return pd.concat(frames, ignore_index=True)
