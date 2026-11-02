"""Explicit non-mutating bridge from schema-2 to the canonical publications."""
from pathlib import Path
import json
import math
from numbers import Real
import pandas as pd
from ..loading import load_season
from .registry import Registry
from .storage import MATCH_COLUMNS, STAT_COLUMNS, ODDS_COLUMNS
from .importers import ImportResult
from ...odds.workbook import LEAGUES


def _bind_entity(registry,kind,identity,facts,*,scope,day):
    """Native numbers are candidates, never proof of an existing canonical identity."""
    entities=registry.payload['entities'].get(kind,{})
    label=facts.get('name',facts.get('label'))
    reviewed=registry.resolve(kind,label,scope,day)
    chosen=reviewed if reviewed is not None else identity
    existing=entities.get(str(chosen))
    if existing is not None:
        stable={k:v for k,v in facts.items() if k not in ('name','label')}
        if any(existing.get(k)!=v for k,v in stable.items()) or (existing.get('name',existing.get('label'))!=label and reviewed!=chosen):
            raise ValueError(f'Unreviewed native {kind} ID collision {identity}: {label}; add a reviewed canonical alias/mapping')
    else:registry.seed(kind,chosen,**facts)
    return chosen


def _bind_fixture(registry,identity,comp,season,home,away,day):
    key=[comp,season,home,away,day,'']
    resolved=registry.fixture(comp,season,home,away,day,create=False)
    existing=registry.payload['entities'].get('fixture',{}).get(str(identity))
    if resolved is not None:
        if existing is not None and resolved!=identity:
            decisions=registry.payload['entities']['fixture'][str(resolved)].get('decisions',[])
            if not any(d.get('occurrence')==key for d in decisions):
                raise ValueError(f'Native fixture ID collision {identity}; explicitly review distinct fixture {resolved}')
        return resolved
    if existing is not None or registry.nearby_fixture(comp,season,home,away,day):
        raise ValueError(f'Native fixture {identity} occurrence {key} requires map_occurrence or an explicit reviewed distinct-fixture decision')
    registry.seed('fixture',identity,occurrences=[key]);registry.fixture_keys[tuple(key)]=identity
    registry._pairs.setdefault(tuple(key[:4]),[]).append((day,'',identity))
    return identity


def seed_comprehensive(data_root,*,reviewed_competitions=('Premier_League',),registry=None):
    """Seed existing canonical IDs. Only the explicitly reviewed first batch is publishable.

    Senior-only Premier League participation is the initial reviewed squad scope.
    Other competitions are inventoried but cannot acquire senior classification
    from a team-name suffix or presence/absence in the downloaded matches.
    """
    registry=registry or Registry();raw=[];checksums=[];statistics=[]
    known={canonical:region for (region,_),canonical in LEAGUES.items()}
    for p in sorted(Path(data_root).glob('*.manifest.json')):
        stem=p.name.removesuffix('.manifest.json');league=stem.rsplit('_',2)[0]
        data=load_season(data_root,stem,tables=['matches'],include_awarded=True,verify_hashes=True)
        frame=data.matches;scope=data.provenance.get('scope',{})
        publication=json.loads(p.read_text());manifest=json.loads((Path(data_root)/publication['manifest']).read_text());scope=manifest['scope']
        comp=int(scope['league_id']);season=int(scope['season_id']);country=known.get(league,league)
        day=f"{scope['season_start']}-01-01"
        comp=_bind_entity(registry,'competition',comp,dict(name=league,region=country,entity_kind='club_league'),scope=country,day=day)
        registry.alias('competition',comp,league,scope=country,decision='existing canonical competition registry')
        for (region,alias),canonical in LEAGUES.items():
            if canonical==league:registry.alias('competition',comp,alias,scope=region,decision='reviewed canonical competition spelling')
        label=f"{scope['season_start']}/{scope['season_end']}"
        season=_bind_entity(registry,'season',season,dict(competition=comp,label=label,start_year=scope['season_start'],end_year=scope['season_end']),scope=str(comp),day=day)
        for alias in (label,f"{scope['season_start']}/{scope['season_end']%100:02}",f"{scope['season_start']%100:02}-{scope['season_end']%100:02}"):
            registry.alias('season',season,alias,scope=str(comp),decision='existing canonical season definition')
        # Source team numbers can be reused across dated canonical identities.
        # Bind every fixture perspective on its actual date, never a season anchor.
        eventmap={};perspectives={};team_cache={}
        for row in frame.to_dict('records'):
            external_event=int(row['event_id'])
            try:
                seconds=row.get('kickoff_utc')
                if isinstance(seconds,bool) or not isinstance(seconds,Real) or not math.isfinite(seconds):
                    raise ValueError('kickoff_utc must supply finite Unix seconds')
                t=pd.Timestamp(seconds,unit='s',tz='UTC')
                if pd.isna(t):raise ValueError('missing kickoff')
                day=t.date().isoformat()
            except (ValueError,TypeError,OverflowError) as error:
                raise ValueError(f'Native fixture {external_event} date is missing or invalid; identity review required') from error
            teams={}
            for side in ('home','away'):
                team,name=row.get(side+'_id'),row.get(side+'_name')
                if pd.isna(team) or pd.isna(name):
                    raise ValueError(f'Native fixture {external_event} {side} identity requires review')
                native_team=int(team);key=(native_team,str(name),day)
                if key not in team_cache:
                    reviewed=registry.resolve('team',str(name),str(comp),day)
                    team=_bind_entity(registry,'team',native_team,dict(name=str(name),entity_kind='club'),scope=str(comp),day=day)
                    # Reviewed intervals already supply the authority. Do not extend
                    # them. New native evidence supports only this fixture date.
                    if reviewed is None:
                        registry.alias('team',team,str(name),scope=str(comp),start=day,end=day,
                            decision='native fixture-day identity')
                    team_cache[key]=team
                    if league in reviewed_competitions:
                        if league!='Premier_League':raise ValueError('Additional squad batches need reviewed classification decisions')
                        registry.classify(team,'not_reserve',decision='reviewed senior Premier League participant')
                team=team_cache[key];teams[side]=team
                perspective=(external_event,side);binding=(native_team,team)
                if perspective in perspectives and perspectives[perspective]!=binding:
                    raise ValueError(f'Conflicting native fixture perspective: {perspective}')
                perspectives[perspective]=binding
            home,away=teams['home'],teams['away']
            event=_bind_fixture(registry,external_event,comp,season,home,away,day)
            eventmap[external_event]=event
            raw.append(dict(event_id=event,competition_id=comp,season_id=season,season_year=label,stage='',round=row.get('round'),home_id=home,away_id=away,
                home_name=registry.payload['entities']['team'][str(home)]['name'],away_name=registry.payload['entities']['team'][str(away)]['name'],date=day,local_datetime=None,time_precision='second',timezone='UTC',kickoff_at=t.isoformat(),
                result_available_at=None if pd.isna(row.get('result_available_at')) else str(row['result_available_at']),status='awarded' if row.get('is_awarded') is True else row['status'],
                score_basis='reported_current',home_score_current=row.get('home_score_current'),away_score_current=row.get('away_score_current'),identity_state='resolved',conflict=False,
                membership_verified=False,schedule_complete=False))
        if league in reviewed_competitions and 'statistics' in manifest['tables']:
            stats=load_season(data_root,stem,tables=['matches','statistics'],include_awarded=True,verify_hashes=True).statistics
            stats=stats.loc[stats.event_id.isin(eventmap),[c for c in STAT_COLUMNS if c in stats]].copy()
            mapped_teams=[]
            for event,side,team in stats[['event_id','side','team_id']].itertuples(index=False,name=None):
                binding=perspectives.get((int(event),side))
                if binding is None or pd.isna(team) or int(team)!=binding[0]:
                    raise ValueError(f'Statistic identity disagrees with native fixture {event} {side}; review required')
                mapped_teams.append(binding[1])
            stats['event_id']=stats.event_id.map(eventmap)
            stats['team_id']=mapped_teams
            stats['validity']=stats.value.map(lambda v:'observed' if pd.notna(v) and v>=0 else 'missing' if pd.isna(v) else 'invalid')
            stats.loc[stats.validity.eq('invalid'),'value']=None
            stats['available_at']=None;stats['definition']='native_reported_period'
            statistics.append(stats)
        checksums.append(manifest['tables']['matches']['sha256'])
    rows=pd.DataFrame(raw,columns=MATCH_COLUMNS)
    if rows.event_id.duplicated().any():
        if any(len(group.drop_duplicates())!=1 for _,group in rows.groupby('event_id')):
            raise ValueError('Conflicting canonical fixture copies in legacy publications')
        rows=rows.drop_duplicates('event_id')
    registry._index()
    return registry,dict(matches=rows,statistics=pd.concat(statistics,ignore_index=True) if statistics else pd.DataFrame(columns=STAT_COLUMNS)),checksums


def review_premier_league(registry):
    """Initial approved canonical aliases/calendar; no fuzzy entity matching."""
    comp=17;registry.alias('competition',comp,'ENG1',scope='historical_league_code',decision='English top division code reviewed')
    for start in range(2000,2027):
        label=f'{start}/{start+1}'
        season=registry.resolve('season',label,str(comp),f'{start}-08-01')
        if season is None:season=registry.allocate('season',competition=comp,label=label,start_year=start,end_year=start+1)
        for alias in (label,f'{start}/{(start+1)%100:02}',f'{start%100:02}-{(start+1)%100:02}'):
            registry.alias('season',season,alias,scope=str(comp),decision='reviewed English August-May season calendar')
    aliases={'Manchester Utd':'Manchester United','Man United':'Manchester United','Man City':'Manchester City',
             'Wolves':'Wolverhampton','Wolverhampton Wanderers':'Wolverhampton','Nottm Forest':'Nottingham Forest',
             'Nottingham':'Nottingham Forest','West Brom':'West Bromwich Albion','West Bromwich':'West Bromwich Albion',
             'Newcastle':'Newcastle United','West Ham':'West Ham United','Tottenham':'Tottenham Hotspur',
             'Leicester':'Leicester City','Norwich':'Norwich City','Swansea':'Swansea City','Cardiff':'Cardiff City',
             'Huddersfield':'Huddersfield Town','Hull':'Hull City','Stoke':'Stoke City','Wigan':'Wigan Athletic',
             'Bolton':'Bolton Wanderers','Blackburn':'Blackburn Rovers','Birmingham':'Birmingham City',
             'QPR':'Queens Park Rangers','Ipswich':'Ipswich Town','Charlton':'Charlton Athletic','Leeds':'Leeds United',
             'Brighton':'Brighton & Hove Albion','Sheffield Utd':'Sheffield United','Liverpool':'Liverpool FC',
             'Bournemouth':'Bournemouth','Bradford':'Bradford City','Coventry':'Coventry City'}
    # Aliases use canonical names already in the registry; never allocate an existing club under a second ID.
    names={facts['name']:int(key) for key,facts in registry.payload['entities']['team'].items()}
    for alias,canonical in aliases.items():
        team=names.get(canonical)
        if team is None:continue
        registry.alias('team',team,alias,scope=str(comp),decision='reviewed English senior club spelling')
        registry.alias('team',team,canonical,scope=str(comp),decision='reviewed English senior club spelling')
        registry.classify(team,'not_reserve',decision='reviewed senior Premier League participant')

def review_native_reserves(registry):
    """Reviewed canonical reserve identities, kept distinct from their parent clubs."""
    decisions={24324:('Athletic Club B U21','Athletic official reserve-team description'),
               24343:('Barcelona Atlètic','Barcelona official reserve-team description'),
               24360:('Real Sociedad B','Real Sociedad official filial description'),
               5069:('Real Madrid Castilla U21','Real Madrid official reserve-team description')}
    for team,(name,decision) in decisions.items():
        if registry.payload['entities'].get('team',{}).get(str(team),{}).get('name')==name:
            registry.classify(team,'reserve_or_development_squad',decision=decision)
