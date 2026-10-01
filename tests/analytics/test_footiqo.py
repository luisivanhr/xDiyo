"""Synthetic workbook contracts; no purchased data is needed for tests."""
import json
import shutil
from xml.sax.saxutils import escape
from zipfile import ZipFile

import pytest

from xdiyo_analytics.odds.footiqo import MARKETS, extract_footiqo, load_footiqo


def workbook(path, sheet='Corners_Closing_Odds', *, duplicate=False, missing_header=False,
             seasons=('2026/2027','2025/2026'), missing_opening=False):
    columns=['match_id','country','league','season','home_team','away_team','start_datetime',
             'total_corners_ft',*MARKETS[sheet]]
    if missing_header: columns.pop()
    strings=[]
    def cell(col, row, value):
        # Shared strings exercise the same storage as the supplied workbooks.
        if value=='FORMULA':return f'<c r="{col}{row}"><f>DO_NOT_READ()</f><v>999</v></c>'
        strings.append(str(value))
        return f'<c r="{col}{row}" t="s"><v>{len(strings)-1}</v></c>'
    def letter(i):
        result=''
        while i: i,r=divmod(i-1,26);result=chr(65+r)+result
        return result
    rows=[columns]
    for country,season,mid,price in [
        ('England',seasons[0],'9007199254740993','2.10'),
        ('England',seasons[1],'9007199254740993' if duplicate else '42','bad'),
        ('England','2027/2028','43','FORMULA'),
        ('Elsewhere','2026/2027','44','FORMULA')]:
        row=[mid,country,'Premier League',season,'A','B',season[:4]+'-09-01 12:00:00','FORMULA']
        row += [price] * (len(columns)-len(row))
        if price=='2.10':row[-1]=''
        if missing_opening and sheet=='Odds':row[columns.index('home_win_opening_odds')]=''
        rows.append(row)
    xml=''.join(f'<row r="{n}">'+''.join(cell(letter(i),n,v) for i,v in enumerate(row,1))+'</row>' for n,row in enumerate(rows,1))
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    with ZipFile(path,'w') as z:
        z.writestr('xl/workbook.xml',f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="{sheet}" sheetId="1" r:id="r1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr('xl/worksheets/sheet1.xml',f'<worksheet xmlns="{ns}"><dimension ref="A1:X1048576"/><sheetData>{xml}</sheetData></worksheet>')
        z.writestr('xl/sharedStrings.xml',f'<sst xmlns="{ns}">'+''.join('<si><t>'+escape(s)+'</t></si>' for s in strings)+'</sst>')
    return path


def extract(tmp_path, **kwargs):
    data=tmp_path/'native';data.mkdir(exist_ok=True)
    for season in ('25_26','26_27'):(data/f'Premier_League_{season}.manifest.json').write_text('{}')
    path=workbook(tmp_path/'source.xlsx',**kwargs)
    args=dict(data_root=data,output_root=tmp_path/'odds',seasons=['25_26','26_27'])
    return extract_footiqo([path],**args),path,args


def test_price_projection_and_season_scope(tmp_path):
    snapshot,_,_=extract(tmp_path)
    quotes=load_footiqo(snapshot,league='Premier_League',season='26_27',sheet='Corners_Closing_Odds')
    assert len(quotes)==8
    assert set(quotes.match_id)=={'9007199254740993'}
    assert set(quotes.status)=={'valid','missing'}
    assert quotes.decimal_odds.dropna().eq(2.1).all()
    assert quotes.raw_price.dropna().str.contains('2.10').sum()==7
    assert quotes.bookmaker.isna().all() and quotes.quote_timestamp.isna().all()
    fixtures=load_footiqo(snapshot,league='Premier_League',season='26_27',sheet='Corners_Closing_Odds',kind='fixtures')
    assert 'total_corners_ft' not in fixtures
    invalid=load_footiqo(snapshot,league='Premier_League',season='25_26',sheet='Corners_Closing_Odds')
    assert invalid.status.eq('invalid_decimal_odds').all()
    report=json.loads((snapshot/'manifest.json').read_text())
    assert len(report['excluded'])==2


def test_reuse_relocation_and_tamper(tmp_path):
    snapshot,path,args=extract(tmp_path)
    assert extract_footiqo([path],**args)==snapshot
    moved=tmp_path/'moved';shutil.copytree(snapshot,moved)
    assert len(load_footiqo(moved,league='Premier_League',season='26_27',sheet='Corners_Closing_Odds'))==8
    manifest=json.loads((snapshot/'manifest.json').read_text())
    payload=snapshot/manifest['files'][0]['path']
    payload.write_bytes(b'changed')
    with pytest.raises(ValueError,match='modified'):extract_footiqo([path],**args)


def test_main_opening_and_closing_remain_distinct(tmp_path):
    snapshot,_,_=extract(tmp_path,sheet='Odds')
    quotes=load_footiqo(snapshot,league='Premier_League',season='26_27',sheet='Odds')
    assert len(quotes)==40
    assert set(quotes.quote_type)=={'opening','closing'}
    assert not quotes.duplicated(['match_id','source_column']).any()


@pytest.mark.parametrize('option,message', [('duplicate','Duplicate provider match ID'),('missing_header','Missing columns')])
def test_ambiguous_source_rejected(tmp_path,option,message):
    with pytest.raises(ValueError,match=message):extract(tmp_path,**{option:True})
