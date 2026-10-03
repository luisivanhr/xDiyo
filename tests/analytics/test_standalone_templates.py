"""Standalone export contracts and real file:// browser checks. Never fit models."""
import base64
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import runpy
import tempfile

import pytest

from xdiyo_analytics.reporting.templates import build_tickets, build_curves, build_portfolio
from xdiyo_analytics.reporting.templates.common import exact_sum, decimal, read_csv
from xdiyo_analytics.reporting.templates.tickets import badge_uri

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = runpy.run_path(str(ROOT / 'examples/standalone_reports.py'))


@pytest.fixture
def ticket_input(tmp_path):
    m = EXAMPLE['create_inputs'](tmp_path / 'input')
    return tmp_path / 'input', m


def save(root, m, name='tickets.json'):
    p = root / name
    p.write_text(json.dumps(m), encoding='utf-8')
    return p


def chunk(output, index=0):
    text = (output / f'policies/policy-{index:05d}.js').read_text()
    return json.loads(text[text.index(',')+1:-2])


def edit_csv(root, edit):
    _, headers, rows = read_csv(root / 'tickets.csv')
    edit(headers, rows)
    with (root/'tickets.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(headers);writer.writerows(rows)


def test_general_tickets_exact_source_and_css(ticket_input, tmp_path):
    root,m=ticket_input
    before=(root/'tickets.csv').read_bytes()
    index=build_tickets(save(root,m),tmp_path/'result')
    p=chunk(index.parent)
    assert base64.b64decode(p['csv_base64'])==before
    assert [len(r['legs']) for r in p['rows'][:3]]==[1,2,3]
    assert p['rows'][2]['legs'][-1]['label']=='Non-team example'
    assert p['rows'][0]['source'][-1]=='00001'
    assert p['rows'][0]['legs'][0]['event_id']=='9223372036854775808'
    assert p['rows'][0]['legs'][0]['probability']=='0.80000000000000000001'
    assert (root/'tickets.csv').read_bytes()==before
    assert hashlib.sha256((index.parent/'tickets.css').read_bytes()).hexdigest()=='e7c2513eceb34b0bc1370457501bb54de3974ce09884323f10953524638cb508'
    rebuilt=build_tickets(index.parent/'manifest.json',tmp_path/'rebuilt')
    assert base64.b64decode(chunk(rebuilt.parent)['csv_base64'])==before
    with pytest.raises(ValueError,match='empty'):build_tickets(save(root,m),index.parent)


@pytest.mark.parametrize('change,match',[
    (lambda m:m.update(schema_version='2'), 'schema_version'),
    (lambda m:m['policies'].append(deepcopy(m['policies'][0])), 'Duplicate policy'),
    (lambda m:m['policies'][0]['fields'].update(unknown='result'), 'mapping'),
    (lambda m:m['policies'][0]['fields'].update(result='absent'), 'missing mapped'),
    (lambda m:m['policies'][0]['summary'].update(rows=1), 'summary rows'),
    (lambda m:m['policies'][0]['summary'].update(profit_units='0'), 'summary profit'),
    (lambda m:m.update(accounting_tolerance='-1'), 'nonnegative'),
    (lambda m:m['policies'][0]['summary'].update(stake_units='NaN'), 'decimal'),
])
def test_manifest_failures_before_writes(ticket_input,tmp_path,change,match):
    root,m=ticket_input;change(m)
    with pytest.raises(ValueError,match=match):build_tickets(save(root,m),tmp_path/'result')
    assert not (tmp_path/'result').exists()


@pytest.mark.parametrize('field,value,match',[
    ('ticket_id','', 'nonempty'),('stake_units','inf','decimal'),
    ('payout_units','3','payout - stake'),('first_kickoff_utc','2025-01-01','UTC'),
    ('legs_json','[]','nonempty'),('legs_json','[{','property'),
    ('legs_json','[{"event_id":42}]','event_id'),
])
def test_ticket_row_failures(ticket_input,tmp_path,field,value,match):
    root,m=ticket_input
    edit_csv(root,lambda h,r:r[0].__setitem__(h.index(field),value))
    with pytest.raises(ValueError,match=match):build_tickets(save(root,m),tmp_path/'result')


@pytest.mark.parametrize('kind',['duplicate_header','width','duplicate_id','probability','duplicate_json'])
def test_bad_shapes(ticket_input,tmp_path,kind):
    root,m=ticket_input
    def edit(h,r):
        if kind=='duplicate_header':h[-1]=h[0]
        if kind=='width':r[0].pop()
        if kind=='duplicate_id':r[1][0]=r[0][0]
        if kind=='duplicate_json':r[0][4]='[{"event_id":"a","event_id":"b"}]'
        if kind=='probability':
            legs=json.loads(r[0][4]);legs[0]['probability']='1.1';r[0][4]=json.dumps(legs)
    edit_csv(root,edit)
    with pytest.raises(ValueError):build_tickets(save(root,m),tmp_path/'result')


def test_exact_decimal_long_cancellation():
    assert str(exact_sum([decimal('1000000000000000000000000000000000000.000000000000001','x'),
                          decimal('-1000000000000000000000000000000000000','x')]))=='1E-15'


def test_badges_and_untrusted_text(ticket_input,tmp_path):
    root,m=ticket_input
    svg=b'<svg xmlns="http://www.w3.org/2000/svg"><circle cx="15" cy="15" r="12" fill="red"/></svg>'
    m['badges']={'0001':{'name':'Badge team','badge':'data:image/svg+xml;base64,'+base64.b64encode(svg).decode()}}
    with pytest.raises(ValueError,match='attribution'):build_tickets(save(root,m),tmp_path/'result')
    (root/'LICENSE.txt').write_text('Synthetic circle asset. CC0 fixture.')
    m['links']=[{'label':'Source guide','file':'LICENSE.txt'}]
    m['attribution_files']=['LICENSE.txt'];m['title']='</script><script>alert(1)</script>{{footer}}'
    html=build_tickets(save(root,m),tmp_path/'result').read_text()
    assert '<script>alert' not in html and '{{footer}}' in html
    assert (tmp_path/'result/attribution/000-LICENSE.txt').read_bytes()==(root/'LICENSE.txt').read_bytes()
    assert (tmp_path/'result/attachments/000-LICENSE.txt').read_bytes()==(root/'LICENSE.txt').read_bytes()
    for uri in ['https://example.com/badge.png','data:image/png;base64,YWJj',
                'data:image/svg+xml;base64,'+base64.b64encode(b'<svg><script>alert(1)</script></svg>').decode()]:
        with pytest.raises(ValueError):badge_uri(uri)


def test_curves_preserve_json_tables_selectors_and_portfolio(tmp_path):
    root=tmp_path/'input';root.mkdir();m=EXAMPLE['create_curves'](root)
    index=build_curves(root/'curves.json',tmp_path/'curves')
    assert (index.parent/'evidence/figure-0000.plotly.json').read_bytes()==(root/'curve.plotly.json').read_bytes()
    copied=list((index.parent/'evidence').glob('*.csv'))
    assert all(p.read_bytes()==(root/'curve.csv').read_bytes() for p in copied)
    html=index.read_text()
    assert 'Study explorer' in html and 'plotly.min.js' in html and 'synthetic-example-v1' in html
    p=build_portfolio(root/'portfolio.json',tmp_path/'portfolio')
    assert 'nothing is summed by the renderer' in p.read_text()
    raw=json.loads((root/'curve.plotly.json').read_text())
    assert len(raw['data'])==25 and len(raw['layout']['updatemenus'][0]['buttons'])==25
    rebuilt=build_curves(index.parent/'manifest.json',tmp_path/'rebuilt')
    assert (rebuilt.parent/'evidence/figure-0000.plotly.json').read_bytes()==(root/'curve.plotly.json').read_bytes()


@pytest.mark.parametrize('kind',['endpoint','calendar','mask','runtime','time','rows'])
def test_invalid_curve_evidence(tmp_path,kind):
    root=tmp_path/'input';root.mkdir();m=EXAMPLE['create_curves'](root)
    fig=m['sections'][0]['figures'][0]
    if kind=='endpoint':fig['endpoints'][3]='999'
    if kind=='calendar':fig['curve_tables'][0]['calendar']['end']='2025-01-04T00:00:00Z'
    if kind=='runtime':m['plotly_js_version']='0.0.0'
    if kind=='mask':
        spec=json.loads((root/'curve.plotly.json').read_text());spec['layout']['updatemenus'][0]['buttons'][0]['args'][0]['visible']=[True]
        (root/'curve.plotly.json').write_text(json.dumps(spec))
    if kind in ('time','rows'):
        text=(root/'curve.csv').read_text()
        text=text.replace('2025-01-02','2025-01-01') if kind=='time' else text.splitlines()[0]+'\n'
        (root/'curve.csv').write_text(text)
    with pytest.raises(ValueError):build_curves(save(root,m,'curves.json'),tmp_path/'result')


@pytest.fixture
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser=p.chromium.launch()
        yield browser
        browser.close()


def test_file_browser_filters_downloads_paging_and_switches(ticket_input,tmp_path,browser):
    root,m=ticket_input
    index=build_tickets(save(root,m),tmp_path/'result')
    page=browser.new_page(viewport={'width':1440,'height':1000},accept_downloads=True)
    errors=[];requests=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('console',lambda msg:errors.append(msg.text) if msg.type=='error' else None)
    page.on('request',lambda r:requests.append(r.url))
    page.goto(index.as_uri());page.wait_for_function('window.TICKET_VIEWER_QA?.loaded')
    assert page.title()==m['title'] and page.url==index.as_uri()
    assert page.locator('article.ticket').count()==50
    assert not any('policy-00001.js' in u for u in requests)
    assert not any(u.startswith(('http:','https:')) for u in requests)
    page.locator('#next').click();assert page.locator('article.ticket').count()==11
    page.locator('#size').select_option('20');assert page.locator('article.ticket').count()==20
    page.locator('#size').select_option('100');assert page.locator('article.ticket').count()==61
    page.locator('#league').select_option('Second league');assert page.locator('article.ticket').count()==31
    page.locator('#search').fill('impossible-match');assert page.locator('.empty').is_visible()
    page.locator('#reset').click();assert page.locator('article.ticket').count()==61
    page.locator('#search').fill('non-team');assert page.locator('article.ticket').count()==1
    assert page.locator('article.ticket .match').count()==3
    with page.expect_download() as downloaded:page.locator('#filteredCsv').click()
    data=Path(downloaded.value.path()).read_bytes()
    original=read_csv(root/'tickets.csv')
    assert list(csv.reader(io.StringIO(data.decode(),newline='')))==[original[1],original[2][2]]
    with page.expect_download() as downloaded:page.locator('#csv').click()
    assert Path(downloaded.value.path()).read_bytes()==(root/'tickets.csv').read_bytes()
    assert page.evaluate("TICKET_VIEWER_QA.decimalSum(['0.10000000000000000001','0.2'])")=='0.30000000000000000001'
    page.evaluate("""() => {const p=document.getElementById('policy');p.value='example/1';p.dispatchEvent(new Event('change'));p.value='example/0';p.dispatchEvent(new Event('change'));}""")
    page.wait_for_function("TICKET_VIEWER_QA.loaded && TICKET_VIEWER_QA.currentPolicy==='example/0'")
    assert page.evaluate('TICKET_VIEWER_QA.getRows().length')==61
    assert page.locator('script[src*="policy-"]').count()==0
    for width,name in [(1440,'desktop'),(390,'mobile')]:
        page.set_viewport_size({'width':width,'height':1000})
        page.screenshot(path=str(Path(tempfile.gettempdir())/f'xdiyo-sharing-{name}.png'),full_page=False)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    assert not errors
    page.close()


def test_policy_failure_and_retry(ticket_input,tmp_path,browser):
    root,m=ticket_input;index=build_tickets(save(root,m),tmp_path/'result')
    missing=index.parent/'policies/policy-00001.js'
    content=missing.read_bytes();missing.unlink()
    page=browser.new_page();page.goto(index.as_uri());page.wait_for_function('TICKET_VIEWER_QA.loaded')
    page.locator('#policy').select_option('example/1')
    page.wait_for_function('document.getElementById("loading").className==="error"')
    assert page.locator('#csv').is_disabled()
    assert page.locator('article.ticket').count()==0
    missing.write_bytes(content)
    page.locator('#policy').select_option('example/0')
    page.wait_for_function('TICKET_VIEWER_QA.loaded')
    page.locator('#policy').select_option('example/1')
    page.wait_for_function("TICKET_VIEWER_QA.loaded && TICKET_VIEWER_QA.currentPolicy==='example/1'")
    assert page.locator('article.ticket').count()==50
    page.close()


def test_legacy_flat_pairs_never_recalculate(ticket_input,tmp_path):
    root,m=ticket_input
    def edit(h,rows):
        fields=['event_id','kickoff_utc','home_id','away_id','opening_draw_odds','draw_probability','draw_settlement']
        for leg in (1,2):
            h.extend(f'leg{leg}_{f}' for f in fields)
            for row in rows:row.extend(['001','2025-01-01T12:00:00Z','0001','0002','3.3','0.9','unknown'])
    edit_csv(root,edit)
    for p in m['policies']:p['adapter']='legacy_draw_pairs';p['fields'].pop('legs',None)
    index=build_tickets(save(root,m),tmp_path/'result')
    data=chunk(index.parent)
    assert len(data['rows'][0]['legs'])==2
    assert data['rows'][0]['legs'][0]['probability']=='0.9'
    assert 'combined_probability' not in data['rows'][0]
    assert base64.b64decode(data['csv_base64'])==(root/'tickets.csv').read_bytes()


def test_curve_browser_all_25_policies(tmp_path,browser):
    root=tmp_path/'input';root.mkdir();EXAMPLE['create_curves'](root)
    index=build_curves(root/'curves.json',tmp_path/'result')
    page=browser.new_page(viewport={'width':1440,'height':1000})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('console',lambda msg:errors.append(msg.text) if msg.type=='error' else None)
    page.goto(index.as_uri());page.wait_for_function("document.querySelector('.plot')?._fullLayout !== undefined")
    assert page.locator('.updatemenu-header').count()==1
    for i in range(25):
        page.locator('.updatemenu-header').click()
        page.locator('.updatemenu-dropdown-button').nth(i).click()
        page.wait_for_function('(i)=>document.querySelector(".plot").data[i].visible===true',arg=i)
        assert page.evaluate('Array.from(document.querySelector(".plot").data).filter(t=>t.visible===true).length')==1
    assert page.evaluate('document.querySelector(".plot").data[24].y')==['0','0','1.25']
    for width,name in [(1440,'desktop'),(390,'mobile')]:
        page.set_viewport_size({'width':width,'height':1000})
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(Path(tempfile.gettempdir())/f'xdiyo-sharing-curves-{name}.png'))
    assert not errors
    page.close()
