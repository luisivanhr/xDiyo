"""Build an immutable, private goals-only snapshot; training never joins sources."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import parse_qsl,urlsplit
from zoneinfo import ZoneInfo

from ._source import _open_source
from .catalogue.registry import name_key
from .catalogue.reviewed_teams import load_reviewed_team_directory
from .loading import load_season

# ISDB v2 competition identities. Unlisted codes are not inferred.
ISDB = dict(x.split(':') for x in """
AUS1:A_League SWE1:Allsvenskan AUT1:Austrian_Bundesliga BRA1:Brazil_Serie_A BRA2:Brazil_Serie_B
GER1:Bundesliga GER2:Bundesliga_2 ENG2:Championship CHL1:Chile_Primera_Division CHN1:Chinese_Super_League
DNK1:Danish_Superliga NOR1:Eliteserien HOL1:Eredivisie GRE1:Greek_Super_League JPN1:J1 JPN2:J2
SPA1:La_Liga SPA2:La_Liga_2 ECU1:LigaPro MEX1:Liga_MX POR1:Liga_Portugal ARG1:Liga_Profesional
FRA1:Ligue_1 FRA2:Ligue_2 USA1:MLS ENG1:Premier_League SCO1:Premiership BEL1:Pro_League
RUS1:Russian_Premier_League SCO2:Scottish_Championship ITA1:Serie_A ITA2:Serie_B TUR1:Super_Lig
CHE1:Swiss_Super_League
""".split())
COUNTRIES = {
    'A_League':'Australia','Allsvenskan':'Sweden','Austrian_Bundesliga':'Austria',
    'Brazil_Serie_A':'Brazil','Brazil_Serie_B':'Brazil','Bundesliga':'Germany','Bundesliga_2':'Germany',
    'Championship':'England','Chile_Primera_Division':'Chile','Chinese_Super_League':'China',
    'Colombia_Primera_A':'Colombia','Czech_First_League':'Czech Republic','Danish_Superliga':'Denmark',
    'Eerste_Divisie':'Netherlands','Egyptian_Premier_League':'Egypt','Ekstraklasa':'Poland','Eliteserien':'Norway',
    'Eredivisie':'Netherlands','Greek_Super_League':'Greece','HNL':'Croatia','J1':'Japan','J2':'Japan',
    'La_Liga':'Spain','La_Liga_2':'Spain','LigaPro':'Ecuador','Liga_MX':'Mexico','Liga_Portugal':'Portugal',
    'Liga_Portugal_2':'Portugal','Liga_Profesional':'Argentina','Ligue_1':'France','Ligue_2':'France',
    'MLS':'USA','Parva_Liga':'Bulgaria','Peru_Liga_1':'Peru','Premier_League':'England','Premiership':'Scotland',
    'Pro_League':'Belgium','PrvaLiga':'Slovenia','Romanian_Superliga':'Romania','Russian_Premier_League':'Russia',
    'Saudi_Pro_League':'Saudi Arabia','Scottish_Championship':'Scotland','Serbian_Superliga':'Serbia',
    'Serie_A':'Italy','Serie_B':'Italy','Super_Lig':'Turkey','Swiss_Super_League':'Switzerland',
    'Ukrainian_Premier_League':'Ukraine',
}
CALENDAR = {'Allsvenskan','Brazil_Serie_A','Brazil_Serie_B','Chinese_Super_League','Eliteserien','J1','J2','LigaPro','MLS'}
TRANSITIONS = {('Chile_Primera_Division','2017/2017'), ('Liga_Profesional','2017/2018')}
PUBLICATION = re.compile(r'[A-Za-z][A-Za-z0-9_]*_\d\d_\d\d\.manifest\.json')
SEASON = re.compile(r'(20\d\d)/(20\d\d)')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def stable_key(parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def score(value):
    if isinstance(value, bool):
        raise ValueError('Boolean goals are not a score')
    n = float(value)
    if not n.is_integer() or n < 0:
        raise ValueError(f'Invalid goal count: {value!r}')
    return int(n)


def aware(value):
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError('A timestamp must have an explicit UTC offset')
    return dt.astimezone(timezone.utc)


def next_utc_day(dt):
    return datetime.combine(dt.date()+timedelta(days=1),datetime.min.time(),timezone.utc)


def season_year(start, end):
    if not 2000 <= start <= 2099 or end not in (start,start+1):
        raise ValueError(f'Invalid two-year season {start}/{end}')
    return f'{start:04}/{end:04}'


def historical_season(league, raw, day):
    if not re.fullmatch(r'\d\d-\d\d',raw):
        raise ValueError(f'Invalid ISDB season code {raw!r}')
    start,end=(2000+int(p) for p in raw.split('-'))
    if league in CALENDAR or (league=='Chile_Primera_Division' and day.year<2013):
        return season_year(day.year,day.year)
    if league=='Russian_Premier_League' and day.year<=2010:
        return season_year(day.year,day.year)
    return season_year(start,end)


def native_score(row):
    if row.get('status')!='finished' or row.get('is_awarded') is not False:
        return None
    if re.search('abandon|unplayed|annul|cancel',str(row.get('play_status') or ''),re.I):
        return None
    if row.get('status_code') in (110,120) or any(
        row.get(f'{side}_score_{extra}') is not None
        for side in ('home','away') for extra in ('extra1','extra2','overtime','penalties')
    ):
        return None
    fields=('home_score_current','away_score_current')
    if any(row.get(k) is None for k in fields):
        return None
    return tuple(score(row[k]) for k in fields)


def read_native(root):
    import pandas as pd

    def native_time(value):
        if value is None:
            return None
        if isinstance(value, (int,float)):
            return datetime.fromtimestamp(value,timezone.utc)
        return aware(str(value))

    manifests = sorted(p for p in root.glob('*.manifest.json') if PUBLICATION.fullmatch(p.name))
    if not manifests:
        raise ValueError('No native publications found')
    pins=[]; records=[]
    for file in manifests:
        stem=file.name.removesuffix('.manifest.json')
        opened=_open_source(root,stem)
        loaded=load_season(root,stem,tables=('matches',),include_awarded=True,verify_hashes=True)
        if (loaded.provenance['version']!=opened.manifest['version'] or
                sha(file)!=opened.record['publication_sha256'] or sha(opened.path)!=opened.record['manifest_sha256']):
            raise RuntimeError(f'Publication changed during the build: {stem}; retry from an immutable snapshot')
        scope=opened.manifest['scope']
        pins.append(dict(stem=stem,version=loaded.provenance['version'],publication_sha256=opened.record['publication_sha256'],
                         manifest_sha256=opened.record['manifest_sha256'],matches_sha256=opened.manifest['tables']['matches']['sha256']))
        frame=loaded.matches
        for raw in frame.astype(object).where(pd.notna(frame),None).to_dict('records'):
            league=raw.get('annual_league') or scope['league_name']
            label=raw.get('annual_season') or season_year(scope['season_start'],scope['season_end'])
            provider_annual=label
            if re.fullmatch(r'20\d\d',label):
                label=season_year(int(label),int(label))
            if label==f"{scope['season_start']}_transition" and raw.get('annual_phase')=='transition':
                label=season_year(scope['season_start'],scope['season_end'])
            m=SEASON.fullmatch(label)
            if not m or season_year(*(int(x) for x in m.groups()))!=label:
                raise ValueError(f'{stem}: unexpected season {label!r}')
            k=native_time(raw.get('kickoff_utc'))
            ident=raw['event_id']
            goals=native_score(raw)
            release=native_time(raw.get('result_available_at'))
            if release is not None and (k is None or release<k):
                raise ValueError(f'Invalid native result availability on event {ident}')
            records.append(dict(league=league,season=label,phase=raw.get('annual_phase'),
                competition_id=scope['league_id'],season_id=scope['season_id'],event_id=ident,
                league_name=raw.get('competition_name') or league,season_name=raw.get('season_name'),
                home_id=raw.get('home_id'),away_id=raw.get('away_id'),
                home_team=raw.get('home_display_name') or raw.get('home_name'),
                away_team=raw.get('away_display_name') or raw.get('away_name'),
                home_goals=goals[0] if goals else None,away_goals=goals[1] if goals else None,
                source='SofaScore',source_fixture_id=str(ident),source_url=None,
                source_reported_date=k.date() if k else None,source_kickoff_utc=k,kickoff_utc=k,
                source_manifest=stem,source_season=provider_annual,score_basis='native_current_no_extra_time_evidence',
                result_available_at=release,
                date_precision='second' if k else 'unknown',
                fixture_key=f'sofascore:{ident}',team_identity='sofascore_native'))
    return records,pins


def read_external(path, historical):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f,delimiter='\t')
        required={'Sea','Lge','Date','HT','AT','HS','AS','GD','WDL'} if historical else {
            'league','season','home_id','away_id','home_team','away_team','home_goals','away_goals',
            'source','source_fixture_id','source_reported_date','kickoff_utc','source_url'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Unexpected external score schema: {path.name}')
        for r in reader:
            if historical:
                h,a=score(r['HS']),score(r['AS'])
                if int(r['GD'])!=h-a or r['WDL']!=('W' if h>a else 'L' if h<a else 'D'):
                    raise ValueError('ISDB goal difference/outcome mismatch')
                day=datetime.strptime(r['Date'],'%d/%m/%Y').date()
                league=ISDB.get(r['Lge'])
                label=historical_season(league,r['Sea'],day) if league is not None else None
                hold=None
                if not league: hold='unmapped_isdb_competition'
                if (r['Lge']=='AUS1' and day<date(2005,8,26)) or (r['Lge']=='CHN1' and day.year<2004):
                    hold='predecessor_competition'
                if ((league,label) in TRANSITIONS or
                    (league=='Chile_Primera_Division' and 2013<=day.year<=2017) or
                    (league=='Liga_Profesional' and 2014<=day.year<=2017)):
                    hold='season_transition_needs_review'
                source='ISDBv2'
                names=(r['HT'],r['AT'])
                keys=names
                source_id=None
                token=(r['Lge'],r['Sea'],str(day),*keys)
                basis='reported_unknown'
            else:
                h,a=score(r['home_goals']),score(r['away_goals'])
                day=date.fromisoformat(r['source_reported_date'])
                league,label=r['league'],r['season']
                mm=SEASON.fullmatch(label)
                if not mm or season_year(*(int(g) for g in mm.groups()))!=label:
                    raise ValueError(f'Unexpected historical season: {label}')
                source=r['source']
                if not source or source in ('SofaScore','ISDBv2'):
                    raise ValueError('Invalid archive source label')
                names=(r['home_team'],r['away_team']);keys=(r['home_id'],r['away_id'])
                source_id=r['source_fixture_id'] or None
                if r['source_url']:
                    u=urlsplit(r['source_url'])
                    if (u.scheme!='https' or not u.hostname or u.username or u.password or
                        any(k.casefold() in {'key','api_key','apikey','access_token','token','password'}
                            for k,_ in parse_qsl(u.query,keep_blank_values=True))):
                        raise ValueError('Archive provenance URL must not contain credentials')
                hold='season_transition_needs_review' if (league,label) in TRANSITIONS else None
                token=(source,source_id) if source_id else (source,league,label,str(day),*keys)
                basis='historical_coverage_checked'
            if not all(str(v).strip() for v in (*names,*keys)) or keys[0]==keys[1]:
                raise ValueError(f'Invalid source teams: {token}')
            yield dict(fixture_key=f'{"isdb" if historical else "archive"}:{stable_key(token)}',
                league=league,season=label,home_team=names[0],away_team=names[1],home_goals=h,away_goals=a,
                source=source,source_fixture_id=source_id,source_home_key=keys[0],source_away_key=keys[1],
                source_competition=r['Lge'] if historical else league,
                source_season=r['Sea'] if historical else label,
                source_reported_date=day,source_kickoff_utc=None if historical else aware(r['kickoff_utc']),
                kickoff_utc=None,date_precision='date_only' if historical else r.get('date_precision','date_only'),
                source_url=None if historical else r['source_url'],score_basis=basis),hold


def materialize(native, historical, archive, directory, reviews=()):
    """Return scorelines, accepted overlap links, and a review queue."""
    comp=defaultdict(set); seasons=defaultdict(set); names=defaultdict(set)
    observed=defaultdict(set); fingerprints=defaultdict(list); cells={}
    ids={}; base={}; links=[]; held=[]
    for r in native:
        key=r['fixture_key']
        home,away=r['home_id'],r['away_id']
        pair=r['league'],r['season']
        cells[pair]='native'
        comp[r['league']].add((r['competition_id'],r['league_name']))
        seasons[pair].add((r['competition_id'],r['season_id'],r['season_name']))
        for side in ('home','away'):
            if r.get(side+'_id') is not None and r.get(side+'_team'):
                observed[r['league'],name_key(r[side+'_team'])].add(r[side+'_id'])
        if key in ids:
            fields=('competition_id','home_id','away_id','kickoff_utc','home_goals','away_goals','league','season','result_available_at')
            if any(ids[key][f]!=r[f] for f in fields):
                raise ValueError(f'Conflicting native fixture copies: {key}')
            links.append(dict(fixture_key=key,source_key=key,source='SofaScore',source_fixture_id=r['source_fixture_id'],
                              source_reported_date=r['source_reported_date'],source_manifest=r['source_manifest'],decision='same_native_event'))
            continue
        ids[key]=r
        if not all((home,away,r['kickoff_utc'])) or home==away or r['home_goals'] is None:
            continue
        fingerprints[(*pair,home,away,r['source_reported_date'])].append(r)
        base[key]=r
    for t in directory['teams']:
        for a in t['names']:
            names[t['country'],name_key(a['name'])].add((a['start'],a['end'],t['team_id'],t['canonical_name']))

    def resolve(league, team, day):
        native_ids=observed[league,name_key(team)]
        found={(tid,display) for lo,hi,tid,display in names[COUNTRIES.get(league),name_key(team)] if lo<=day<=hi}
        if len(found)==1 and (not native_ids or native_ids=={next(iter(found))[0]}):
            return next(iter(found))
        return None,team

    def hold(r,reason):
        held.append(dict(r,reason=reason))

    for source_rows in (historical,archive):
        seen=set()
        pending=defaultdict(list)
        for r,why in source_rows:
            key=r['fixture_key']
            if key in seen:
                raise ValueError(f'Duplicate external source identity: {key}')
            seen.add(key)
            pair=r['league'],r['season']
            if why:
                hold(r,why)
                continue
            if r['league'] not in comp:
                hold(r,'league_not_in_pinned_native_snapshot')
                continue
            comps=comp[r['league']]
            season_ids=seasons[pair]
            opts={i for i,_ in comps}
            cid=next(iter(opts)) if len(opts)==1 else None
            ls={n for i,n in comps}
            native_season=next(iter(season_ids)) if len(season_ids)==1 else (cid,None,None)
            hd,ht=resolve(r['league'],r['home_team'],str(r['source_reported_date']))
            ad,at=resolve(r['league'],r['away_team'],str(r['source_reported_date']))
            if hd is not None and hd==ad:
                hold(r,'mapped_teams_collapse')
                continue
            r.update(competition_id=native_season[0],league_name=next(iter(ls)) if len(ls)==1 else r['league'],
                     season_id=native_season[1],season_name=native_season[2],home_id=hd,away_id=ad,
                     home_team=ht,away_team=at,team_identity='reviewed' if hd is not None and ad is not None else 'needs_review')
            if pair in cells:
                matches=fingerprints[(*pair,hd,ad,r['source_reported_date'])] if hd and ad else []
                if len(matches)==1 and all(matches[0][f]==r[f] for f in ('home_goals','away_goals')):
                    links.append(dict(fixture_key=matches[0]['fixture_key'],source_key=key,source=r['source'],
                                      source_fixture_id=r['source_fixture_id'],source_reported_date=r['source_reported_date'],
                                      source_manifest=None,decision='same_ordered_teams_day_and_score'))
                else:
                    hold(r,f'{cells[pair]}_overlap_score_conflict' if len(matches)==1 else f'{cells[pair]}_overlap_needs_review')
            else:
                pending[pair].append(r)
        # Claim whole league-season scopes deterministically. An unlinked
        # external match in an occupied scope is not proof of a new fixture.
        for pair,rs in sorted(pending.items()):
            cells[pair]='external'
            occurrences=defaultdict(list)
            for r in rs:
                p=(r.get('home_id') or name_key(r['home_team']),r.get('away_id') or name_key(r['away_team']),r['source_reported_date'])
                occurrences[p].append(r)
            for group in occurrences.values():
                if len(group)!=1:
                    for r in group:hold(r,'ambiguous_external_occurrence')
                    continue
                r=group[0]
                base[r['fixture_key']]=r
                if r['home_id'] is not None and r['away_id'] is not None:
                    fingerprints[(*pair,r['home_id'],r['away_id'],r['source_reported_date'])].append(r)

    review_map={}
    for review in reviews:
        if set(review)!={'fixture_key','completion_date','timezone','regulation_verified','evidence'}:
            raise ValueError('Unexpected reviewed result fields')
        key=review['fixture_key']
        if key not in base or key in review_map:
            raise ValueError(f'Unknown or duplicated review fixture {key}')
        if review['regulation_verified'] is not True or not str(review['evidence']).strip():
            raise ValueError(f'Review must verify regulation score and cite evidence: {key}')
        review_map[key]=review
    for r in base.values():
        r['home_team_key']=f"sofascore:{r['home_id']}" if r.get('home_id') is not None else f"{r['source']}:{r['league']}:{r['source_home_key']}"
        r['away_team_key']=f"sofascore:{r['away_id']}" if r.get('away_id') is not None else f"{r['source']}:{r['league']}:{r['source_away_key']}"
        release=r.get('result_available_at')
        guarded=None
        if release is not None and r.get('kickoff_utc') is not None:
            guarded=next_utc_day(release)
        review=review_map.get(r['fixture_key'])
        if review:
            day=date.fromisoformat(review['completion_date'])
            zone=ZoneInfo(review['timezone'])
            k=r.get('kickoff_utc')
            if (k is not None and day<k.astimezone(zone).date()) or (
                    k is None and day<r['source_reported_date']-timedelta(days=1)):
                raise ValueError(f"Review completion predates the recorded fixture: {r['fixture_key']}")
            # End of the verified local completion day is the conservative
            # bound. Round up to the first UTC analysis day not intersecting it.
            end=datetime.combine(day+timedelta(days=1),datetime.min.time(),zone).astimezone(timezone.utc)
            bound=end if end==end.replace(hour=0,minute=0,second=0,microsecond=0) else next_utc_day(end)
            r.update(completion_date=day,completion_timezone=review['timezone'],
                     review_evidence=str(review['evidence']),score_basis='reviewed_regulation')
            guarded=max(guarded,bound) if guarded else bound
        r['replay_available_at']=guarded
        r['replay_eligible']=guarded is not None and r['team_identity'] in ('sofascore_native','reviewed')
    return list(base.values()),links,held


DATA_FIELDS=(
    'fixture_key league league_name competition_id season season_id season_name phase event_id home_id away_id '
    'home_team away_team home_team_key away_team_key home_goals away_goals source source_fixture_id source_url '
    'source_home_key source_away_key source_competition source_season source_manifest source_reported_date '
    'source_kickoff_utc kickoff_utc date_precision result_available_at completion_date completion_timezone review_evidence replay_available_at replay_eligible score_basis team_identity'
).split()
HELD_FIELDS=[*DATA_FIELDS,'reason']
LINK_FIELDS='fixture_key source_key source source_fixture_id source_reported_date source_manifest decision'.split()


def write_parquet(path,rows,columns):
    import pyarrow as pa
    import pyarrow.parquet as pq
    types={
        **dict.fromkeys(('competition_id','season_id','event_id','home_id','away_id','home_goals','away_goals'),pa.int64()),
        **dict.fromkeys(('source_kickoff_utc','kickoff_utc','result_available_at','replay_available_at'),pa.timestamp('us',tz='UTC')),
        'source_reported_date':pa.date32(),'completion_date':pa.date32(),'replay_eligible':pa.bool_(),
    }
    schema=pa.schema((c,types.get(c,pa.string())) for c in columns)
    pq.write_table(pa.Table.from_pylist(rows,schema=schema),path,compression='zstd')


def build(sofa_root,isdb,archive,out,*,team_directory=None,source_revision=None,reviewed_results=None):
    sofa_root,isdb,archive,out=(Path(p).resolve() for p in (sofa_root,isdb,archive,out))
    if out.exists() or out.is_relative_to(sofa_root):
        raise ValueError('Choose a new output directory outside the SofaScore source')
    input_hashes={'isdb':sha(isdb),'archive':sha(archive)}
    review_path=Path(reviewed_results) if reviewed_results else None
    review_bytes=review_path.read_bytes() if review_path else None
    reviews=json.loads(review_bytes) if review_bytes else []
    if not isinstance(reviews,list):raise ValueError('Reviewed results must be a JSON array')
    directory=load_reviewed_team_directory(team_directory)
    native,pubs=read_native(sofa_root)
    rows,links,held=materialize(native,read_external(isdb,True),read_external(archive,False),directory,reviews)
    if {'isdb':sha(isdb),'archive':sha(archive)}!=input_hashes:
        raise RuntimeError('An external input changed during the build')
    if not rows:raise ValueError('No eligible goals in the selected inputs')
    rows.sort(key=lambda r:(r['source_reported_date'],r['league'],r['fixture_key']))
    links.sort(key=lambda r:(r['fixture_key'],r['source_key'],r.get('source_manifest') or ''))
    held.sort(key=lambda r:(r['fixture_key'],r['reason']))
    out.parent.mkdir(parents=True,exist_ok=True)
    working=Path(tempfile.mkdtemp(prefix='.scorelines-',dir=out.parent))
    try:
        write_parquet(working/'scorelines.parquet',rows,DATA_FIELDS)
        write_parquet(working/'scorelines_replay.parquet',[r for r in rows if r['replay_eligible']],DATA_FIELDS)
        write_parquet(working/'provenance.parquet',links,LINK_FIELDS)
        write_parquet(working/'review.parquet',held,HELD_FIELDS)
        receipt=dict(schema_version=1,native_revision=source_revision,native_publications=pubs,
                     reviewed_team_directory_sha256=directory['sha256'],inputs=input_hashes,
                     reviewed_result_sha256=hashlib.sha256(review_bytes).hexdigest() if review_bytes else None,
                     builder_sha256=sha(Path(__file__)),
                     rows_by_source=Counter(r['source'] for r in rows),
                     native_input_rows=len(native),review_by_reason=Counter(r['reason'] for r in held),
                     replay_rows=sum(r['replay_eligible'] for r in rows),linked_observations=len(links),
                     data_sha256={f.name:sha(f) for f in sorted(working.iterdir())})
        (working/'manifest.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        if out.exists():
            raise FileExistsError(out)
        working.rename(out)
    finally:
        if working.exists():shutil.rmtree(working)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for f in ('sofascore-root','isdb','archive','output'):
        parser.add_argument('--'+f,required=True,type=Path)
    parser.add_argument('--reviewed-team-directory',type=Path)
    parser.add_argument('--reviewed-results',type=Path)
    parser.add_argument('--native-revision',help='Optional Git commit or pinned data release used as input')
    args=parser.parse_args(argv)
    r=build(args.sofascore_root,args.isdb,args.archive,args.output,team_directory=args.reviewed_team_directory,
            source_revision=args.native_revision,reviewed_results=args.reviewed_results)
    print(json.dumps({k:r[k] for k in ('rows_by_source','replay_rows','review_by_reason','linked_observations')},indent=2))


if __name__=='__main__':
    main()
