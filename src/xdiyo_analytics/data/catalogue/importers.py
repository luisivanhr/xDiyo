"""Read-only full results/statistics adapters. Input join keys live in memory only."""
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import math
import re
import xml.etree.ElementTree as ET
import pandas as pd
from ..statistics_backfill import STATISTICS, PERIODS, statistic_value
from ...odds.workbook import _Workbook, NS, IDENTITY, MARKETS
from .storage import MATCH_COLUMNS, STAT_COLUMNS, ODDS_COLUMNS, file_hash, squad_disposition


@dataclass
class ImportResult:
    matches: pd.DataFrame
    statistics: pd.DataFrame
    odds: pd.DataFrame
    summary: dict
    checksum: str


def empty_result(rows,stats,odds,summary,path):
    frame=pd.DataFrame(rows,columns=MATCH_COLUMNS)
    for c in ('event_id','competition_id','season_id','home_id','away_id'): frame[c]=frame[c].astype('int64')
    return ImportResult(frame,pd.DataFrame(stats,columns=STAT_COLUMNS),pd.DataFrame(odds,columns=ODDS_COLUMNS),dict(summary),file_hash(path))


def canonical_row(registry, *, competition, season, home, away, day, score, basis, local=None, stage=''):
    home_id=registry.resolve('team',home,str(competition),day)
    away_id=registry.resolve('team',away,str(competition),day)
    if home_id is None or away_id is None: return None,'identity_needs_review'
    row=dict(competition_id=competition,season_id=season,season_year=registry.payload['entities']['season'][str(season)]['label'],
             stage=stage,round=None,home_id=home_id,away_id=away_id,
             home_name=registry.payload['entities']['team'][str(home_id)]['name'],away_name=registry.payload['entities']['team'][str(away_id)]['name'],
             date=day,local_datetime=local,time_precision='minute' if local else 'date',timezone='unknown',kickoff_at=None,
             result_available_at=None,status='reported_result',score_basis=basis,home_score_current=score[0],away_score_current=score[1],
             identity_state='resolved',conflict=False,membership_verified=False,schedule_complete=False)
    reason=squad_disposition(row,registry)
    if reason!='eligible':return None,reason
    existing=registry.fixture(competition,season,home_id,away_id,day,stage,create=False)
    if existing is None and registry.nearby_fixture(competition,season,home_id,away_id,day):
        return None,'fixture_identity_needs_review'
    row['event_id']=existing or registry.fixture(competition,season,home_id,away_id,day,stage)
    return row,'accepted'


def read_historical_results(path,registry,*,revision='v2'):
    if revision!='v2': raise ValueError('Only reviewed v2 baseline is admitted; v1 requires explicit revision review, never concatenation')
    frame=pd.read_csv(path,sep='\t',dtype=str,keep_default_na=False)
    if list(frame)!=['Sea','Lge','Date','HT','AT','HS','AS','GD','WDL']:raise ValueError('Unexpected historical results schema')
    dates=pd.to_datetime(frame.Date,format='%d/%m/%Y',errors='raise').dt.strftime('%Y-%m-%d')
    home=pd.to_numeric(frame.HS,errors='raise');away=pd.to_numeric(frame.AS,errors='raise')
    if (home.lt(0)|away.lt(0)|home.mod(1).ne(0)|away.mod(1).ne(0)).any():raise ValueError('Invalid historical goal score')
    if pd.to_numeric(frame.GD,errors='raise').ne(home-away).any():raise ValueError('Historical goal difference mismatch')
    result=pd.Series(['W' if a>b else 'D' if a==b else 'L' for a,b in zip(home,away)])
    if result.ne(frame.WDL).any():raise ValueError('Historical WDL mismatch')
    if frame.assign(Date=dates).duplicated(['Lge','Date','HT','AT']).any():raise ValueError('Duplicate historical fixture; do not concatenate revisions')
    rows=[];summary=Counter(inspected=len(frame));seen=set()
    for raw,day in zip(frame.itertuples(index=False),dates):
        comp=registry.resolve('competition',raw.Lge,'historical_league_code',day)
        if comp is None:summary['competition_needs_review']+=1;continue
        season=registry.resolve('season',raw.Sea,str(comp),day)
        if season is None:summary['season_needs_review']+=1;continue
        row,reason=canonical_row(registry,competition=comp,season=season,home=raw.HT,away=raw.AT,day=day,score=(int(raw.HS),int(raw.AS)),basis='reported_unknown')
        summary[reason]+=1
        if row is not None:
            if row['event_id'] in seen:raise ValueError('Historical aliases duplicate a physical fixture')
            seen.add(row['event_id']);rows.append(row)
    return empty_result(rows,[],[],summary,path)


def _rows(workbook,sheet):
    headers=workbook.headers(sheet)
    for index,cells in workbook.rows(sheet):
        if index==1:continue
        values={headers[c]:workbook.decode(cell) for c,cell in cells.items() if c in headers}
        if values.get('match_id') in (None,''):continue
        identity=Decimal(values['match_id'])
        if not identity.is_finite() or identity<=0 or identity!=identity.to_integral_value():raise ValueError('Invalid exact workbook join ID')
        values['match_id']=str(int(identity))
        yield values


def _numbers(raw,names):
    numbers=[]
    for name in names:
        d=Decimal(raw[name])
        if not d.is_finite() or d<0 or d!=d.to_integral_value():raise ValueError(f'Invalid goal field {name}')
        numbers.append(int(d))
    return numbers


def read_results_workbook(path,registry):
    """Validate all five sheet populations, map supported fields, quarantine unknown identities.

    Price labels have no availability evidence; they never become executable quotes.
    Negative/invalid optional cells are null with explicit invalid states.
    """
    w=_Workbook(path);rows=[];stats=[];odds=[];summary=Counter();matches={};accepted={}
    try:
        # Resolve shared strings once. No repeatedly rescanning the shared-string archive.
        if 'xl/sharedStrings.xml' in w.archive.namelist():
            with w.archive.open('xl/sharedStrings.xml') as stream:
                index=0
                for _,element in ET.iterparse(stream,events=('end',)):
                    if element.tag==NS+'si':
                        w.strings[index]=''.join(t.text or '' for t in element.iter(NS+'t'));index+=1;element.clear()
        for raw in _rows(w,'Matches_Results'):
            key=raw['match_id']
            if key in matches:raise ValueError('Duplicate workbook match ID')
            matches[key]=raw;summary['inspected']+=1
            # Numeric Excel timestamps are local civil times, never UTC.
            stamp=pd.Timestamp('1899-12-30')+pd.to_timedelta(float(raw['start_datetime']),unit='D') if re.fullmatch(r'\d+(\.\d+)?',raw['start_datetime']) else pd.Timestamp(raw['start_datetime'])
            if stamp.tzinfo is not None:raise ValueError('Unexpected zoned workbook datetime; review definition')
            stamp=stamp.round('s');day=stamp.date().isoformat()
            goals=_numbers(raw,[f'{side}_goals_{period}' for period in ('ft','1h','2h') for side in ('home','away')])
            if goals[0]!=goals[2]+goals[4] or goals[1]!=goals[3]+goals[5]:raise ValueError('Half goals do not reconcile with full time')
            for period,(h,a) in zip(('ft','1h','2h'),zip(goals[::2],goals[1::2])):
                if raw.get('result_'+period) not in (None,'', 'H' if h>a else 'D' if h==a else 'A','1' if h>a else 'X' if h==a else '2'):
                    raise ValueError('Workbook outcome does not agree with goals')
            comp=registry.resolve('competition',raw['league'],raw['country'],day)
            if comp is None:summary['competition_needs_review']+=1;continue
            season=registry.resolve('season',str(raw['season']),str(comp),day)
            if season is None:summary['season_needs_review']+=1;continue
            row,reason=canonical_row(registry,competition=comp,season=season,home=raw['home_team'],away=raw['away_team'],day=day,
                 score=goals[:2],basis='regulation',local=stamp.isoformat())
            summary[reason]+=1
            if row is not None:
                row.update(home_score_1h=goals[2],away_score_1h=goals[3],home_score_2h=goals[4],away_score_2h=goals[5])
                accepted[key]=row;rows.append(row)
        for sheet in ('Statistics_FT','Statistics_1H','Statistics_2H','Odds'):
            seen=set()
            for raw in _rows(w,sheet):
                key=raw['match_id']
                if key not in matches or key in seen:raise ValueError(f'Non one-to-one workbook join in {sheet}')
                seen.add(key)
                if any(raw.get(c)!=matches[key].get(c) for c in ('start_datetime','country','competition_type','league','season','home_team','away_team')):
                    raise ValueError(f'Conflicting workbook identity descriptors in {sheet}')
                row=accepted.get(key)
                if sheet=='Odds':
                    for field,(market,selection,line,timing) in MARKETS['Odds'].items():
                        raw_value=raw.get(field);value=None;validity='missing'
                        if raw_value not in (None,''):
                            try:value=float(raw_value)
                            except ValueError:pass
                            validity='observed' if value is not None and math.isfinite(value) and value>1 else 'invalid'
                            if validity=='invalid':value=None;summary['invalid_odds_cells']+=1
                        if row is not None:odds.append(dict(event_id=row['event_id'],market=market,selection=selection,period='ALL',line=line,quote_type=timing,decimal_odds=value,validity=validity,available_at=None,bookmaker=None))
                else:
                    period=sheet.split('_')[1].lower()
                    for field,(stat_key,_,group,_) in STATISTICS.items():
                        for side in ('home','away'):
                            raw_value=raw.get(f'{side}_{field}_{period}')
                            try:value=statistic_value(raw_value,field);validity='missing' if value is None else 'observed'
                            except (ValueError,TypeError):value=None;validity='invalid';summary['invalid_statistics_cells']+=1
                            if row is not None:stats.append(dict(event_id=row['event_id'],team_id=row[side+'_id'],side=side,period=PERIODS[period],group_name=group,key=stat_key,value=value,validity=validity,available_at=None,definition='regulation_period'))
            if seen!=matches.keys():raise ValueError(f'Workbook populations differ: {sheet}')
            summary[sheet.lower()+'_rows']=len(seen)
        if len({r['event_id'] for r in rows})!=len(rows):raise ValueError('Workbook aliases duplicate canonical fixtures')
        return empty_result(rows,stats,odds,summary,path)
    finally:w.archive.close()


def combine_imports(imports):
    """Coherent rows only. Conflicting scores quarantine the entire fixture."""
    rows=pd.concat([r.matches.assign(_input=i) for i,r in enumerate(imports)],ignore_index=True)
    chosen=[];conflicts=[]
    for identity,group in rows.groupby('event_id',sort=True):
        if len(group[['home_score_current','away_score_current']].drop_duplicates())>1:
            conflicts.append(int(identity));continue
        # A documented regulation-time row is preferable to unknown score basis.
        chosen.append(group.assign(_priority=group.score_basis.ne('regulation')).sort_values('_priority',kind='stable').drop(columns='_priority').iloc[0])
    selected=pd.DataFrame(chosen,columns=[*MATCH_COLUMNS,'_input'])
    stats=[];odds=[]
    for i,item in enumerate(imports):
        ids=selected.loc[selected._input.eq(i),'event_id']
        stats.append(item.statistics.loc[item.statistics.event_id.isin(ids)])
        odds.append(item.odds.loc[item.odds.event_id.isin(ids)])
    return dict(matches=selected.drop(columns='_input').astype({c:'int64' for c in ('event_id','competition_id','season_id','home_id','away_id')}),statistics=pd.concat([f for f in stats if not f.empty],ignore_index=True) if any(not f.empty for f in stats) else pd.DataFrame(columns=STAT_COLUMNS),odds=pd.concat([f for f in odds if not f.empty],ignore_index=True) if any(not f.empty for f in odds) else pd.DataFrame(columns=ODDS_COLUMNS)),dict(conflicting_fixtures=conflicts,input_counts=[r.summary for r in imports])

def inspect_historical_revisions(v1_path,v2_path):
    """Aggregate revision audit only; no v1 supplements or persisted source keys."""
    frames=[pd.read_csv(p,sep='\t',dtype=str,keep_default_na=False) for p in (v1_path,v2_path)]
    keys=['Lge','Date','HT','AT'];scores=['HS','AS']
    a,b=frames
    for frame in frames:
        if frame.groupby(keys)[scores].nunique().gt(1).any().any():raise ValueError('Conflicting scores within historical revision')
    aa=a.drop_duplicates(keys).set_index(keys);bb=b.drop_duplicates(keys).set_index(keys)
    common=aa.index.intersection(bb.index)
    return dict(row_counts=[len(a),len(b)],duplicate_excess=[len(a)-len(aa),len(b)-len(bb)],common_raw_keys=len(common),
                only_v1=len(aa.index.difference(bb.index)),only_v2=len(bb.index.difference(aa.index)),
                score_revisions=int(aa.loc[common,scores].ne(bb.loc[common,scores]).any(axis=1).sum()),
                input_checksums=[file_hash(p) for p in (v1_path,v2_path)],supplements_admitted=0)
