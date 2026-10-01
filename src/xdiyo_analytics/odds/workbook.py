"""Price-only odds workbook projections into private Parquet snapshots.

Only explicitly mapped identity and price cells are decoded. This extractor
does not create a native fixture crosswalk or evaluate betting outcomes.
"""
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
VERSION = '1'
LEAGUES = {
    ('England', 'Premier League'): 'Premier_League',
    ('England', 'Championship'): 'Championship',
    ('Germany', 'Bundesliga'): 'Bundesliga',
    ('Germany', '2. Bundesliga'): 'Bundesliga_2',
    ('Spain', 'LaLiga'): 'La_Liga',
    ('Spain', 'LaLiga2'): 'La_Liga_2',
    ('Italy', 'Serie A'): 'Serie_A',
    ('Italy', 'Serie B'): 'Serie_B',
    ('France', 'Ligue 1'): 'Ligue_1',
    ('France', 'Ligue 2'): 'Ligue_2',
    ('Netherlands', 'Eredivisie'): 'Eredivisie',
    ('Scotland', 'Premiership'): 'Premiership',
    ('Belgium', 'Jupiler Pro League'): 'Pro_League',
}
IDENTITY = ('match_id','start_datetime','competition_type','country','league','league_id',
            'season','home_team_id','home_team','away_team_id','away_team')


def _markets():
    result = {}
    for sheet,market,lines in [('Corners_Closing_Odds','corners',(7.5,8.5,9.5,10.5)),
                               ('Cards_Closing_Odds','yellow_cards',(2.5,3.5,4.5,5.5))]:
        result[sheet] = {f'{side}_{str(line).replace(".","_")}_{market}_ft_closing_odds':
            (market,side,str(line),'closing') for line in lines for side in ('over','under')}
    main = {}
    for side in ('home','draw','away'):
        for timing in ('opening','closing'):
            column = f'{side + "_win" if side != "draw" else side}_{timing}_odds'
            main[column] = ('1x2',side,None,timing)
    for line in (.5,1.5,2.5,3.5,4.5):
        for side in ('over','under'):
            for timing in ('opening','closing'):
                main[f'{side}_{str(line).replace(".","_")}_goals_{timing}_odds']=('goals',side,str(line),timing)
    for side in ('yes','no'):
        for timing in ('opening','closing'):
            main[f'btts_{side}_{timing}_odds']=('btts',side,None,timing)
    # Retain the vendor's handicap-column convention; native settlement is not
    # inferred from the column name, particularly for the away side.
    for token,line in [('minus_1_5','-1.5'),('minus_1','-1'),('0','0'),('plus_1','1'),('plus_1_5','1.5')]:
        for side in ('home','away'):
            main[f'{side}_ah_{token}_closing_odds']=('asian_handicap',side,line,'closing')
    result['Odds']=main
    return result


MARKETS = _markets()


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


class _Workbook:
    """Selective cell decoder with a shared-string cache limited to requested cells."""
    def __init__(self,path):
        self.archive=ZipFile(path)
        rel={r.attrib['Id']:r.attrib['Target'].lstrip('/') for r in
             ET.fromstring(self.archive.read('xl/_rels/workbook.xml.rels'))}
        self.sheets={}
        for item in ET.fromstring(self.archive.read('xl/workbook.xml')).find(NS+'sheets'):
            path=rel[item.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]
            self.sheets[item.attrib['name']]=path if path.startswith('xl/') else 'xl/'+path
        self.strings={}

    def rows(self,sheet):
        with self.archive.open(self.sheets[sheet]) as stream:
            parser=ET.iterparse(stream,events=('start','end'))
            parent=None
            for event,element in parser:
                if event=='start' and element.tag==NS+'sheetData':parent=element
                if event=='end' and element.tag==NS+'row':
                    yield int(element.attrib['r']),{re.sub(r'\d','',c.attrib['r']):c for c in element if c.tag==NS+'c'}
                    element.clear()
                    if parent is not None:parent.clear()

    def need_strings(self,cells):
        return {int(c.find(NS+'v').text) for c in cells if c is not None and c.attrib.get('t')=='s'}

    def resolve(self,indices):
        needed=set(indices)-self.strings.keys()
        if not needed:return
        with self.archive.open('xl/sharedStrings.xml') as stream:
            index=0
            for _,element in ET.iterparse(stream,events=('end',)):
                if element.tag==NS+'si':
                    if index in needed:
                        self.strings[index]=''.join(t.text or '' for t in element.iter(NS+'t'))
                        needed.remove(index)
                    element.clear();index+=1
                    if not needed:break
        if needed:raise ValueError('Invalid shared-string references')

    def decode(self,cell):
        if cell is None:return None
        if cell.find(NS+'f') is not None:raise ValueError('Formula in selected source cell is unsupported')
        kind=cell.attrib.get('t')
        if kind=='inlineStr':return ''.join(t.text or '' for t in cell.iter(NS+'t'))
        value=cell.find(NS+'v')
        if value is None:return None
        return self.strings[int(value.text)] if kind=='s' else value.text

    def headers(self,sheet):
        rows=self.rows(sheet)
        try:
            _,cells=next(rows)
            self.resolve(self.need_strings(cells.values()))
            headers={col:self.decode(cell) for col,cell in cells.items()}
        finally:rows.close()
        if len(set(headers.values()))!=len(headers):raise ValueError('Duplicate worksheet headers')
        return headers


def _season(value):
    match=re.fullmatch(r'(20\d{2})/(\d{2}|20\d{2})',str(value))
    if not match:return None
    start=int(match[1]);end=int(match[2])%100
    return f'{start%100:02}_{end:02}' if (start+1)%100==end else None


def _native_pairs(root,seasons):
    pairs=set()
    for path in Path(root).glob('*.manifest.json'):
        match=re.fullmatch(r'(.+)_(\d{2}_\d{2})\.manifest\.json',path.name)
        if match and match[2] in seasons:pairs.add((match[1],match[2]))
    if not pairs:raise ValueError('No native league-season manifests match the explicit seasons')
    return pairs


def _identity(value,name):
    if value is None:return None
    if name.endswith('_id'):
        d=Decimal(value)
        if not d.is_finite() or d!=d.to_integral_value():raise ValueError(f'Invalid exact vendor ID: {name}')
        return str(int(d))
    return value


def extract_odds(workbooks, *, data_root, output_root, seasons, league_aliases=None):
    """Export only odds/identity columns for native league-season pairs.

    seasons is explicit, e.g. ['25_26', '26_27']. Outputs remain provider data,
    not native-fixture-aligned bets. A content-addressed snapshot is immutable.
    """
    seasons=sorted(set(seasons))
    if not seasons or any(not re.fullmatch(r'\d{2}_\d{2}',s) for s in seasons):raise ValueError('Supply explicit seasons as YY_YY')
    pairs=_native_pairs(data_root,seasons)
    aliases=dict(LEAGUES if league_aliases is None else league_aliases)
    paths=[Path(p) for p in workbooks]
    if not paths or len({p.name for p in paths})!=len(paths):
        raise ValueError('Supply workbooks with distinct filenames')
    sources=[dict(file=p.name,bytes=p.stat().st_size,sha256=_hash(p)) for p in paths]
    identity=dict(adapter_version=VERSION,source_files=sources,league_seasons=sorted(pairs),
                  league_aliases=sorted((a,b,c) for (a,b),c in aliases.items()))
    snapshot=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    destination=Path(output_root)/snapshot
    manifest=destination/'manifest.json'
    if manifest.exists():
        recorded=json.loads(manifest.read_text(encoding='utf-8'))
        if any(recorded.get(k)!=json.loads(json.dumps(v)) for k,v in identity.items()):
            raise ValueError('Snapshot manifest identity was modified')
        for item in recorded['files']:
            if _hash(destination/item['path'])!=item['sha256']:raise ValueError('Snapshot payload was modified')
        return destination
    if destination.exists():raise ValueError(f'Incomplete snapshot already exists: {destination}')
    destination.mkdir(parents=True)
    buffers=defaultdict(list);parts=Counter();files=[];summary=Counter();excluded=Counter();headers_record={}
    fixture_schema=pa.schema([(n,pa.string()) for n in ('provider','snapshot','source_league','source_season','source_file','source_sheet',*IDENTITY)]+[('source_row',pa.int64())])
    quote_schema=pa.schema([(n,pa.string()) for n in ('provider','snapshot','source_league','source_season','match_id','market','period','selection','line','quote_type','bookmaker','quote_timestamp','timing_certainty','source_file','source_sheet','source_column','raw_price','status','settlement_definition')]+[('source_row',pa.int64()),('decimal_odds',pa.float64())])

    def flush():
        for (league,season,sheet,kind),records in buffers.items():
            if not records:continue
            key=(league,season,sheet,kind)
            rel=Path(f'league={league}')/f'season={season}'/sheet/f'{kind}-{parts[key]:04}.parquet'
            path=destination/rel;path.parent.mkdir(parents=True,exist_ok=True)
            pq.write_table(pa.Table.from_pylist(records,schema=fixture_schema if kind=='fixtures' else quote_schema),path,compression='zstd')
            files.append(dict(path=rel.as_posix(),rows=len(records),sha256=_hash(path)))
            parts[key]+=1
        buffers.clear()

    for path in paths:
        book=_Workbook(path)
        try:
            selected=[s for s in MARKETS if s in book.sheets]
            if not selected:raise ValueError(f'No supported odds sheets in {path.name}')
            for sheet in selected:
                headers=book.headers(sheet);cols={name:col for col,name in headers.items()}
                required={'match_id','country','league','season','home_team','away_team',*MARKETS[sheet]}
                if required-cols.keys():raise ValueError(f'Missing columns in {sheet}: {sorted(required-cols.keys())}')
                headers_record[path.name+'/'+sheet]=headers
                gate=[cols[n] for n in ('country','league','season')]
                needed=set()
                for row,cells in book.rows(sheet):
                    if row==1:continue
                    needed.update(book.need_strings(cells.get(c) for c in gate))
                book.resolve(needed)
                kept=[cols[n] for n in (*IDENTITY,*MARKETS[sheet]) if n in cols]
                needed=set()
                for row,cells in book.rows(sheet):
                    if row==1:continue
                    country,league,season=[book.decode(cells.get(c)) for c in gate]
                    native=aliases.get((country,league));season=_season(season)
                    if (native,season) in pairs:needed.update(book.need_strings(cells.get(c) for c in kept))
                book.resolve(needed)
                count=0;seen=set()
                for row,cells in book.rows(sheet):
                    if row==1:continue
                    country,league,rawseason=[book.decode(cells.get(c)) for c in gate]
                    native=aliases.get((country,league));season=_season(rawseason)
                    if (native,season) not in pairs:
                        excluded[(path.name,sheet,str(country),str(league),str(rawseason))]+=1;continue
                    fixture={name:_identity(book.decode(cells.get(cols[name])),name) if name in cols else None for name in IDENTITY}
                    if not fixture['match_id']:raise ValueError(f'Missing match ID in {sheet} row {row}')
                    if fixture['match_id'] in seen:raise ValueError(f'Duplicate provider match ID in {sheet}: {fixture["match_id"]}')
                    seen.add(fixture['match_id'])
                    base=dict(provider='imported_odds',snapshot=snapshot,source_league=native,source_season=season,
                              source_file=path.name,source_sheet=sheet,source_row=row)
                    buffers[(native,season,sheet,'fixtures')].append({**base,**fixture})
                    for column,(market,side,line,timing) in MARKETS[sheet].items():
                        raw=book.decode(cells.get(cols[column]));value=None;status='missing'
                        if raw not in (None,''):
                            try:
                                d=Decimal(raw)
                                if d.is_finite() and d>1 and float(d)<float('inf'):value=float(d);status='valid'
                                else:status='invalid_decimal_odds'
                            except InvalidOperation:status='invalid_decimal_odds'
                        buffers[(native,season,sheet,'quotes')].append({**base,'match_id':fixture['match_id'],
                            'market':market,'period':'FT','selection':side,'line':line,'quote_type':timing,
                            'bookmaker':None,'quote_timestamp':None,'timing_certainty':'vendor_prematch_label_only',
                            'source_column':column,'raw_price':raw,'decimal_odds':value,'status':status,
                            'settlement_definition':'regulation_time_v1_unmapped'})
                        summary[(native,season,sheet,'quotes_'+status)]+=1
                    summary[(native,season,sheet,'fixtures')]+=1
                    count+=1
                    if count%250==0:flush()
                flush()
        finally:book.archive.close()
    report={**identity,'snapshot':snapshot,'headers':headers_record,'files':files,
        'coverage':[dict(source_league=a,source_season=b,sheet=c,kind=d,rows=n) for (a,b,c,d),n in sorted(summary.items())],
        'excluded':[dict(source_file=a,sheet=b,country=c,league=d,season=e,rows=n) for (a,b,c,d,e),n in sorted(excluded.items())],
        'caveats':['Provider fixture IDs are not native IDs; no fixture crosswalk has been applied.',
                   'Source start_datetime retains the Excel serial/text exactly; timezone is unknown.',
                   'Bookmaker and actual quote timestamps are unavailable. Opening/closing are vendor labels.',
                   'No outcomes or statistics were decoded or exported. Unsupported settlement markets remain price-only.',
                   'Paid data is private and excluded from Git. Native forecast coverage is not established.']}
    manifest.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return destination


def load_odds(snapshot, *, league, season, sheet, kind='quotes'):
    """Read one provider partition; no automatic native fixture alignment."""
    if kind not in ('quotes','fixtures'):raise ValueError('kind must be quotes or fixtures')
    snapshot=Path(snapshot)
    report=json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'))
    prefix=f'league={league}/season={season}/{sheet}/{kind}-'
    tables=[]
    for item in report['files']:
        if item['path'].startswith(prefix):
            path=snapshot/item['path']
            if _hash(path)!=item['sha256']:raise ValueError('Snapshot payload was modified')
            tables.append(pq.ParquetFile(path).read())
    if not tables:raise ValueError('No matching provider partition')
    return pa.concat_tables(tables).to_pandas()
