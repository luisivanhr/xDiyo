"""Audit both complete international sources against published native fixtures.

Read-only discovery. Does not publish matches, change existing IDs or accept
unreviewed candidates. Private outputs contain names/dates and native evidence,
not workbook join IDs. Re-run after changes to any source.
"""
from pathlib import Path
import argparse, hashlib, xml.etree.ElementTree as ET
from xdiyo_analytics.odds.workbook import _Workbook, NS
from collections import defaultdict, Counter
import json, unicodedata
import pandas as pd

HIST={'ENG1':('England','Premier_League'),'ENG2':('England','Championship'),'GER1':('Germany','Bundesliga'),'GER2':('Germany','Bundesliga_2'),'SPA1':('Spain','La_Liga'),'SPA2':('Spain','La_Liga_2'),'ITA1':('Italy','Serie_A'),'ITA2':('Italy','Serie_B'),'FRA1':('France','Ligue_1'),'FRA2':('France','Ligue_2'),'HOL1':('Netherlands','Eredivisie'),'SCO1':('Scotland','Premiership'),'BEL1':('Belgium','Pro_League'),'JPN1':('Japan','J1'),'JPN2':('Japan','J2'),'MEX1':('Mexico','Liga_MX')}
COUNTRY={'ENG':'England','GER':'Germany','SPA':'Spain','ITA':'Italy','FRA':'France','HOL':'Netherlands','SCO':'Scotland','BEL':'Belgium','JPN':'Japan','MEX':'Mexico','ARG':'Argentina','AUS':'Australia','AUT':'Austria','BRA':'Brazil','CHE':'Switzerland','CHL':'Chile','CHN':'China','DNK':'Denmark','DZA':'Algeria','ECU':'Ecuador','FIN':'Finland','GRE':'Greece','ISR':'Israel','KOR':'South Korea','MAR':'Morocco','NOR':'Norway','NZL':'New Zealand','POR':'Portugal','RUS':'Russia','SWE':'Sweden','TUN':'Tunisia','TUR':'Turkey','USA':'USA','VEN':'Venezuela','ZAF':'South Africa'}
BOOK={'Premier League':'Premier_League','Championship':'Championship','Bundesliga':'Bundesliga','2. Bundesliga':'Bundesliga_2','LaLiga':'La_Liga','LaLiga2':'La_Liga_2','Serie A':'Serie_A','Serie B':'Serie_B','Ligue 1':'Ligue_1','Ligue 2':'Ligue_2','Eredivisie':'Eredivisie','Premiership':'Premiership','Jupiler Pro League':'Pro_League','J1 League':'J1','Liga MX':'Liga_MX'}
def norm(x):return ' '.join(unicodedata.normalize('NFC',x).casefold().split())

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--historical',type=Path,required=True)
 parser.add_argument('--workbook',type=Path,required=True)
 parser.add_argument('--native-root',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args()
 HERE=args.output.resolve()
 if HERE==args.native_root.resolve() or HERE.is_relative_to(args.native_root.resolve()):
  raise ValueError('Audit output must be outside the native dataset')
 HERE.mkdir(parents=True,exist_ok=True)
 CAT=Path(__file__).resolve().parents[1]/'docs/analytics/team_assets/catalog.json'
 cat=json.loads(CAT.read_text(encoding='utf-8'));tables=[];native_files=[]
 for publication in sorted(args.native_root.glob('*.manifest.json')):
  reference=json.loads(publication.read_text(encoding='utf-8'))
  manifest_path=publication.parent/reference['manifest']
  manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
  path=manifest_path.parent/manifest['tables']['matches']['file']
  with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
  if sha!=manifest['tables']['matches']['sha256']:raise ValueError(f'Native table changed: {publication.name}')
  table=pd.read_parquet(path,columns=['event_id','competition_id','home_id','away_id','home_name','away_name','kickoff_utc'])
  table['league']=manifest['scope']['league_name'];tables.append(table)
  native_files.append(dict(publication=publication.name,sha256=sha))
 if not tables:raise ValueError('No published native match tables')
 native=pd.concat(tables,ignore_index=True).drop_duplicates()
 if native.duplicated(['competition_id','event_id']).any():raise ValueError('Conflicting native fixture identities')
 league_country={league:country for country,league in HIST.values()};league_country.update(Liga_MX_Apertura='Mexico',Liga_MX_Clausura='Mexico',J1_Transition='Japan')
 names=defaultdict(set);events=defaultdict(list);native_country={}
 for row in native.itertuples():
  league='Liga_MX' if row.league.startswith('Liga_MX') else row.league
  if row.league=='J1_Transition':continue
  country=league_country[row.league]
  day=pd.Timestamp(row.kickoff_utc,unit='s',tz='UTC').date().isoformat()
  events[(country,league,day)].append((row.home_id,row.away_id,row.event_id))
  for id,name in [(row.home_id,row.home_name),(row.away_id,row.away_name)]:names[(country,norm(name))].add(id);native_country[id]=country
 for id,item in cat.items():
  for name in [item['name'],*item.get('aliases',[])]:
   if int(id) in native_country:names[(native_country[int(id)],norm(name))].add(int(id))
 sources=[];members=defaultdict(list)
 def add(source,key,country,league,day,home,away,hn,an):
  sources.append((source,country,league,day,home,away))
  for token,name in [(home,hn),(away,an)]:members[token].append(dict(source=source,country=country,league=league,day=day,name=name))
 raw=pd.read_csv(args.historical,sep='\t',dtype=str)
 dates=pd.to_datetime(raw.Date,format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
 for row,day in zip(raw.itertuples(),dates):
  country=COUNTRY[row.Lge[:3]];league=HIST.get(row.Lge,(country,row.Lge))[1]
  add('historical','',country,league,day,('historical',country,row.HT,row.Sea),('historical',country,row.AT,row.Sea),row.HT,row.AT)
 book=[];workbook=_Workbook(args.workbook)
 try:
  with workbook.archive.open('xl/sharedStrings.xml') as stream:
   index=0
   for _,element in ET.iterparse(stream,events=('end',)):
    if element.tag==NS+'si':
     workbook.strings[index]=''.join(t.text or '' for t in element.iter(NS+'t'));index+=1;element.clear()
  headers=workbook.headers('Matches_Results')
  wanted={'country','league','season','home_team','away_team','home_team_id','away_team_id','start_datetime'}
  columns={k:v for k,v in headers.items() if v in wanted}
  if set(columns.values())!=wanted:raise ValueError('Workbook identity schema changed')
  for index,cells in workbook.rows('Matches_Results'):
   if index>1:book.append({name:workbook.decode(cells.get(column)) for column,name in columns.items()})
 finally:workbook.archive.close()
 book_dates=pd.to_datetime([float(x['start_datetime']) for x in book],unit='D',origin='1899-12-30').round('s').strftime('%Y-%m-%d')
 for row,day in zip(book,book_dates):
  country=row['country'];league=BOOK.get(row['league'],row['league'])
  add('workbook','',country,league,day,('workbook',row['home_team_id'],row['season']),('workbook',row['away_team_id'],row['season']),row['home_team'],row['away_team'])
 # Exact names propose candidates. Only independent native fixture support accepts them.
 seeds={}
 for token,values in members.items():
  candidates=set().union(*(names[(v['country'],norm(v['name']))] for v in values))
  if len(candidates)==1:seeds[token]=next(iter(candidates))
 accepted={};evidence={};proposals={}
 # Reviewed aliases already backed by the native statistics crosswalk.
 for filename in ['statistics_backfill_aliases','japan_statistics_aliases','mexico_statistics_aliases']:
  for alias in json.loads((CAT.parents[1]/'data'/f'{filename}.json').read_text(encoding='utf-8')):
   country=league_country[alias['league']]
   for token,values in members.items():
    if token[0]=='workbook' and any(v['country']==country and v['name']==alias['name'] for v in values):seeds[token]=int(alias['team_id'])
 # Globally consistent source names can seed season-specific checks, but each
 # accepted season must still have its own fixture evidence.
 for iteration in range(12):
  lookup={**seeds,**accepted};support=defaultdict(lambda:defaultdict(set))
  for source,country,league,day,h,a in sources:
   for nh,na,event in events.get((country,league,day),[]):
    # Known opponent + exact date, competition and side; no score-based matching.
    if lookup.get(h)==nh:support[a][na].add((event,nh,day))
    if lookup.get(a)==na:support[h][nh].add((event,na,day))
  new={}
  for token,candidates in support.items():
   if len(candidates)!=1:continue
   # Source keys are only hints: conflicting domestic identities cannot be
   # laundered through a shared workbook key into a known club's alias.
   domestic={v['country'] for v in members[token]}-{'Europe','World','South America'}
   if len(domestic)>1:continue
   id,proof=next(iter(candidates.items()))
   if len({e for e,o,d in proof})>=3 and len({o for e,o,d in proof})>=2:
    new[token]=id;evidence[token]=sorted(proof)
  proposals=support
  if new==accepted:break
  accepted=new
 result=[]
 for token,values in members.items():
  id=accepted.get(token);proof=evidence.get(token,[]) if id else []
  result.append(dict(source=token[0],source_season=token[-1],source_names=sorted({v['name'] for v in values}),countries=sorted({v['country'] for v in values}),leagues=sorted({v['league'] for v in values}),first_date=min(v['day'] for v in values),last_date=max(v['day'] for v in values),native_team_id=id,native_name=cat[str(id)]['name'] if id else None,supporting_fixtures=len(proof),supporting_opponents=len({p[1] for p in proof}),evidence=proof,unresolved_candidates={str(k):len(v) for k,v in proposals.get(token,{}).items()} if not id else {}))
 (HERE/'full-reconciliation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 pd.DataFrame([{k:v for k,v in row.items() if k!='evidence'} for row in result]).to_csv(HERE/'full-reconciliation.csv',index=False)
 summary=dict(source_identities=len({token[:-1] for token in members}),source_season_identity_records=len(result),historical_matches=len(raw),workbook_matches=len(book),native_matches=len(native),native_team_count=len(native_country),by_source=dict(Counter(r['source'] for r in result)),matched_records=sum(r['native_team_id'] is not None for r in result),distinct_native_ids=len(set(accepted.values())),unresolved_records=sum(r['native_team_id'] is None for r in result),iterations=iteration+1,native_files=native_files)
 summary['inputs']={}
 for name,path in [('historical',args.historical),('workbook',args.workbook),('script',Path(__file__)),('catalogue',CAT)]:
  with path.open('rb') as stream:summary['inputs'][name]=hashlib.file_digest(stream,'sha256').hexdigest()
 (HERE/'audit-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in summary.items() if k not in ('native_files','inputs')},indent=2))

if __name__=='__main__':main()
