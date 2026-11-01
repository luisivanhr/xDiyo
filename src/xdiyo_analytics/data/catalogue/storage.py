"""Schema-3 publications: immutable ZSTD payloads and one atomic commit point."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import pandas as pd
from .registry import Registry

DATASETS=('comprehensive_dataset','international_dataset')
POLICY='canonical_squad_exclusion_v1'
# Whitelists, not a blacklist of provider names. Auxiliary data has its own schema.
MATCH_COLUMNS=('event_id','competition_id','season_id','season_year','stage','round','home_id','away_id','home_name','away_name',
 'date','local_datetime','time_precision','timezone','kickoff_at','result_available_at','status','score_basis',
 'home_score_current','away_score_current','home_score_1h','away_score_1h','home_score_2h','away_score_2h',
 'identity_state','conflict','membership_verified','schedule_complete','squad_policy','dataset')
STAT_COLUMNS=('event_id','team_id','side','period','group_name','key','value','validity','available_at','definition')
ODDS_COLUMNS=('event_id','market','selection','period','line','quote_type','decimal_odds','validity','available_at','bookmaker')
SCHEMAS={'matches':MATCH_COLUMNS,'statistics':STAT_COLUMNS,'odds':ODDS_COLUMNS}


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')


def digest(value): return hashlib.sha256(encoded(value)).hexdigest()

def file_hash(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.pending-')
    try:
        with os.fdopen(fd,'wb') as f: f.write(encoded(value));f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if Path(tmp).exists(): Path(tmp).unlink()


def squad_disposition(row,registry):
    states=[registry.squad(row[k],row['date']) for k in ('home_id','away_id')]
    if 'reserve_or_development_squad' in states: return 'reserve_excluded'
    if states != ['not_reserve','not_reserve']: return 'squad_needs_review'
    return 'eligible'


def validate_matches(frame,registry):
    required={'event_id','competition_id','season_id','home_id','away_id','date','identity_state','conflict','score_basis','status'}
    if required-set(frame): raise ValueError(f'Missing canonical fields: {sorted(required-set(frame))}')
    if frame.event_id.duplicated().any(): raise ValueError('Duplicate canonical fixtures in publication')
    for col,kind in [('event_id','fixture'),('competition_id','competition'),('season_id','season'),('home_id','team'),('away_id','team')]:
        if not pd.api.types.is_integer_dtype(frame[col]) or frame[col].isna().any() or frame[col].le(0).any():
            raise ValueError(f'{col} must contain exact positive integer identities')
        if not set(map(str,frame[col]))<=set(registry.payload['entities'].get(kind,{})): raise ValueError(f'Unregistered {kind}')
    for row in frame.to_dict('records'):
        expected=registry.fixture(row['competition_id'],row['season_id'],row['home_id'],row['away_id'],row['date'],row.get('stage',''),create=False)
        if expected!=row['event_id']: raise ValueError(f"Fixture identity mismatch: {row['event_id']}")
        if row['home_id']==row['away_id']: raise ValueError('A fixture cannot play itself')
        from datetime import date
        date.fromisoformat(row['date'])
        for side in ('home','away'):
            score=row.get(side+'_score_current')
            if pd.notna(score) and (not isinstance(score,(int,float)) or not __import__('math').isfinite(score) or score<0 or score%1):
                raise ValueError(f'Invalid canonical goal value: fixture {row["event_id"]}')
        kinds={registry.payload['entities']['team'][str(row[k])]['entity_kind'] for k in ('home_id','away_id')}
        if len(kinds)!=1:raise ValueError('Club and national-team identities cannot share a fixture')


def publish_dataset(root,name,tables,registry,*,input_checksums=(),summary=None):
    if name not in DATASETS: raise ValueError(f'Unknown dataset: {name}')
    if set(summary or {})-{'conflicting_fixtures','input_counts'}:raise ValueError('Noncanonical summary fields')
    for checksum in input_checksums:
        if not isinstance(checksum,str) or len(checksum)!=64 or any(c not in '0123456789abcdef' for c in checksum):
            raise ValueError('Inputs retain whole checksums only, never paths or attribution')
    if set(tables)-set(SCHEMAS): raise ValueError('Unsupported canonical table')
    frames={k:v.copy().reset_index(drop=True) for k,v in tables.items()}
    for k,v in frames.items():
        if set(v)-set(SCHEMAS[k]): raise ValueError(f'Noncanonical publication fields: {sorted(set(v)-set(SCHEMAS[k]))}')
    matches=frames['matches'];validate_matches(matches,registry)
    dispositions=matches.apply(lambda r:squad_disposition(r,registry),axis=1)
    allowed=dispositions.eq('eligible') & matches.identity_state.eq('resolved') & ~matches.conflict
    exclusions=dispositions[~allowed].value_counts().to_dict()
    conflicts=sorted(map(int,matches.loc[matches.conflict,'event_id']))
    matches=matches.loc[allowed].copy();matches['dataset']=name;matches['squad_policy']=POLICY
    frames['matches']=matches.sort_values('event_id').reset_index(drop=True)
    for kind in set(frames)-{'matches'}:
        f=frames[kind]
        if len(f):
            from ..loading import _check_statistics
            if kind=='statistics':_check_statistics(f, tables['matches'])
            if 'validity' not in f or not f.validity.isin(['observed','missing','not_collected','unsupported','not_applicable','invalid','conflicting','derived']).all():
                raise ValueError('Invalid field validity state')
            value='value' if kind=='statistics' else 'decimal_odds'
            observed=f.validity.eq('observed')
            numeric=pd.to_numeric(f[value],errors='coerce')
            if (observed & (~__import__('numpy').isfinite(numeric) | numeric.lt(0 if kind=='statistics' else 1))).any():
                raise ValueError('Observed field values must be finite and valid')
            if (kind=='odds' and (observed & numeric.le(1)).any()) or (observed & numeric.isna()).any():raise ValueError('Invalid observed price/value')
        if not set(f.event_id)<=set(tables['matches'].event_id): raise ValueError(f'Orphan {kind} fixture')
        frames[kind]=f.loc[f.event_id.isin(matches.event_id)].sort_values(list(f.columns[:min(4,len(f.columns))])).reset_index(drop=True)
    root=Path(root)/name;root.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.pending-',dir=root))
    try:
        files={}
        for kind,frame in frames.items():
            path=stage/(kind+'.parquet');frame.attrs={};frame.to_parquet(path,index=False,compression='zstd')
            files[kind]=dict(file=path.name,sha256=file_hash(path),rows=len(frame))
        (stage/'registry.json').write_bytes(encoded(registry.payload))
        manifest=dict(schema=3,dataset=name,policy=POLICY,registry_sha256=file_hash(stage/'registry.json'),
            input_checksums=sorted(set(input_checksums)),tables=files,
            coverage=dict(import_complete=True,real_schedule_complete=False,inspected=len(tables['matches']),published=len(matches),excluded=exclusions,conflicting_fixtures=conflicts,**{k:v for k,v in (summary or {}).items() if k!='conflicting_fixtures'}))
        manifest['coverage']['conflicting_fixtures']=sorted(set(conflicts)|set((summary or {}).get('conflicting_fixtures',[])))
        version=digest(manifest);manifest['version']=version
        # Import time is physical, separate from historical availability; not part of content version.
        manifest['imported_at']=datetime.now(timezone.utc).isoformat()
        (stage/'manifest.json').write_bytes(encoded(manifest))
        destination=root/version
        if destination.exists():
            load_snapshot(root.parent,name,version)
        else: os.replace(stage,destination)
        return version
    finally:
        if stage.exists(): shutil.rmtree(stage)


def load_snapshot(root,name,version):
    if name not in DATASETS or len(version)!=64 or any(c not in '0123456789abcdef' for c in version): raise ValueError('Exact dataset/version required')
    path=Path(root)/name/version;m=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    identity={k:v for k,v in m.items() if k not in ('version','imported_at')}
    if m.get('schema')!=3 or m.get('dataset')!=name or m.get('policy')!=POLICY or digest(identity)!=version or m.get('version')!=version:
        raise ValueError('Dataset manifest identity changed')
    if file_hash(path/'registry.json')!=m['registry_sha256']: raise ValueError('Registry changed')
    registry=Registry.load(path/'registry.json');tables={}
    for kind,info in m['tables'].items():
        if kind not in SCHEMAS or info['file']!=kind+'.parquet': raise ValueError('Invalid canonical table path')
        p=path/info['file']
        if file_hash(p)!=info['sha256']: raise ValueError(f'Dataset payload changed: {kind}')
        tables[kind]=pd.read_parquet(p)
        if len(tables[kind])!=info['rows'] or set(tables[kind])-set(SCHEMAS[kind]): raise ValueError('Invalid published table')
    validate_matches(tables['matches'],registry)
    if any(squad_disposition(r,registry)!='eligible' for r in tables['matches'].to_dict('records')): raise ValueError('Published reserve/review fixture')
    return tables,m,registry


def list_datasets(root):
    return [dict(dataset=name,version=p.name) for name in DATASETS for p in sorted((Path(root)/name).glob('*')) if (p/'manifest.json').is_file() and not p.name.startswith('.')]


def inspect_coverage(root,name,version):
    tables,m,_=load_snapshot(root,name,version)
    return dict(m['coverage'],dataset=name,version=version,fixtures_without_statistics=len(set(tables['matches'].event_id)-set(tables.get('statistics',pd.DataFrame(columns=['event_id'])).event_id)),fields={k:{c:int(v[c].notna().sum()) for c in v} for k,v in tables.items()},time_precision=tables['matches'].time_precision.value_counts().to_dict())
