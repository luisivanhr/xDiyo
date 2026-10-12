"""Optional real-browser regressions for native reporter controls and downloads.

Uses an already installed Playwright Chromium; never installs dependencies.
Fixtures are presentation-only and do not fit models or projections.
"""
import base64
from pathlib import Path
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

playwright = pytest.importorskip('playwright.sync_api')
from plotly.offline import get_plotlyjs
from xdiyo_analytics.reporting.embedding_view import embedding_figure, JS as EMBEDDING_JS
from xdiyo_analytics.reporting.attention_view import attention_figure, JS as ATTENTION_JS
from xdiyo_analytics.reporting import TeamCatalog

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jfN8AAAAASUVORK5CYII=')
SVG = b'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><circle cx="10" cy="10" r="9" fill="red"/></svg>'


@pytest.fixture
def browser_page(tmp_path):
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(tmp_path)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except playwright.Error as error:
                pytest.skip(f'Installed Playwright browser unavailable: {error}')
            context = browser.new_context(viewport={'width':1000, 'height':1000}, accept_downloads=True)
            page = context.new_page()
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            yield page, f'http://127.0.0.1:{server.server_port}', errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def write_page(path, figure, initializer):
    path.write_text('<!doctype html><meta charset="utf-8"><title>Native reporter regression</title>'
        '<style>#plot{width:720px;max-width:100%;height:650px}body{margin:0}</style>'
        '<details open><summary>Chart</summary><div id="plot"></div></details><script>'+get_plotlyjs()+'</script><script>'+
        EMBEDDING_JS+ATTENTION_JS+'\nconst spec='+figure.to_json()+
        f';{initializer}(document.getElementById("plot"),spec).then(()=>window.ready=true);</script>', encoding='utf-8')


def test_packaged_svg_badges_decode_and_paint_offline(browser_page, tmp_path):
    page, url, errors = browser_page
    catalog = TeamCatalog.from_json(Path(__file__).resolve().parents[2] / 'docs/analytics/team_assets/catalog.json')
    images = []
    for key in catalog.entries:
        _, source, error = catalog.display(key, badges=True)
        assert source and not error, key
        images.append(f'<img data-team="{key}" src="{source}">')
    (tmp_path / 'badges.html').write_text('<!doctype html><meta charset="utf-8">'+''.join(images), encoding='utf-8')
    page.route('https://**/*', lambda route: route.abort())
    page.goto(url+'/badges.html')
    failures = page.evaluate('''async()=>{
      const results=await Promise.all([...document.images].map(async img=>{
        try{await img.decode();const c=document.createElement('canvas');c.width=c.height=64;
          const ctx=c.getContext('2d');ctx.drawImage(img,0,0,64,64);
          const pixels=ctx.getImageData(0,0,64,64).data;
          for(let i=3;i<pixels.length;i+=4)if(pixels[i])return null;
          return img.dataset.team+': empty pixels';
        }catch(e){return img.dataset.team+': '+e.message;}
      }));return results.filter(Boolean);
    }''')
    assert not failures
    assert not errors


@pytest.mark.parametrize('projection', ['PCA', 'tSNE'])
def test_repeated_find_reset_pan_zoom_and_offline_badge_export(browser_page, tmp_path, projection):
    page, url, errors = browser_page
    # Coincident embeddings used to be silently deduplicated by Plotly's
    # position/size image key, in the live plot and the SVG export.
    data = pd.DataFrame(dict(x=[-2.,-2.,3.], y=[1.,1.,2.], league=[1,2,1],
                            team_name=['One','Two','Three'], support=[1.,2.,3.], gate=[.1,.2,.3]))
    (tmp_path/'vector.png').write_bytes(SVG)  # Actual SVG with PNG extension.
    # Incorrect data-URI MIME is also repaired by inspecting the exact bytes.
    png_uri = 'data:image/svg+xml;base64,'+base64.b64encode(PNG).decode()
    fig = embedding_figure(data, ['vector.png', png_uri, None], projection)
    write_page(tmp_path/'index.html', fig, 'initializeEmbedding')
    page.goto(url+'/index.html'); page.wait_for_function('window.ready')
    assert page.locator('#plot .imagelayer image').count() == 2
    assert page.evaluate('plot.layout.images.every(i=>i.x===-2 && i.y===1)')
    baseline = page.evaluate('({x:[...plot.layout.xaxis.range],y:[...plot.layout.yaxis.range]})')
    for _ in range(3):
        page.get_by_label('Find team').select_option('0:0')
        page.wait_for_function('plot.layout.images.at(-1).name==="badge:0:0"')
        page.wait_for_function('(b)=>plot.layout.xaxis.range[0]!==b.x[0]', arg=baseline)
        page.get_by_role('button', name='Reset view', exact=True).click()
        page.wait_for_function('(b)=>JSON.stringify({x:plot.layout.xaxis.range,y:plot.layout.yaxis.range})===JSON.stringify(b)', arg=baseline)
        bounds = page.locator('#plot .nsewdrag').bounding_box()
        x, y = bounds['x']+bounds['width']/2, bounds['y']+bounds['height']/2
        page.mouse.move(x,y); page.mouse.down(); page.mouse.move(x+45,y+20,steps=5); page.mouse.up()
        page.wait_for_function('(b)=>plot.layout.xaxis.range[0]!==b.x[0]', arg=baseline)
        page.locator('[data-title="Zoom in"]').click()
        page.get_by_role('button', name='Reset view', exact=True).click()
        page.wait_for_function('(b)=>JSON.stringify({x:plot.layout.xaxis.range,y:plot.layout.yaxis.range})===JSON.stringify(b)', arg=baseline)
    page.get_by_role('button', name='League 1', exact=True).click()
    page.wait_for_function('plot.layout.images.filter(i=>i.visible).length===1')
    page.get_by_role('button', name='League 1', exact=True).click()
    page.wait_for_function('plot.layout.images.filter(i=>i.visible).length===2')
    with page.expect_download() as download:
        page.locator('[data-title*="Download plot"]').click()
    exported = tmp_path/'download.svg'; download.value.save_as(exported)
    root = ET.parse(exported).getroot()
    images = [e.attrib['{http://www.w3.org/1999/xlink}href'] for e in root.iter() if e.tag.endswith('}image')]
    assert len(images) == 2 and all(s.startswith('data:') for s in images)
    assert {s.split(';')[0] for s in images} == {'data:image/png', 'data:image/svg+xml'}
    assert {base64.b64decode(s.split(',')[1]) for s in images} == {PNG, SVG}
    offline = page.context.new_page()
    offline.route('http://**/*', lambda route: route.abort())
    offline.goto(exported.as_uri())
    assert offline.locator('image').count() == 2
    offline.close()
    page.locator('summary').click()
    page.set_viewport_size({'width':390,'height':844})
    page.locator('summary').click()
    page.wait_for_function('plot.layout.width<=390 && plot.layout.width===plot.layout.height')
    assert not errors


def test_attention_native_selectors_include_cold_teams_and_last_option(browser_page, tmp_path):
    page, url, errors = browser_page
    roster = [dict(team=str(i), name=f'Team {i:03}', league=str(i//40), league_name=f'Competition {i//40}', badge=None) for i in range(80)]
    table = pd.DataFrame([dict(team_id=0, layer=0, head=0, opponent_id=1, event_id=i, kickoff_at=pd.Timestamp(f'2020-01-0{i}',tz='UTC'), attention=w) for i,w in [(1,.6),(2,.4)]])
    other = table.iloc[[0]].assign(team_id=1, event_id=99, kickoff_at=pd.Timestamp('2021-01-01', tz='UTC'))
    table = pd.concat([table, other], ignore_index=True)
    fig = attention_figure(table, roster, TeamCatalog({1:'One'}), 20, 0)
    write_page(tmp_path/'index.html', fig, 'initializeAttention')
    page.goto(url+'/index.html'); page.wait_for_function('window.ready')
    assert page.get_by_label('Team', exact=True).locator('option').count() == 80
    page.get_by_label('League', exact=True).select_option('1')
    assert page.get_by_label('Team', exact=True).locator('option').count() == 40
    page.get_by_label('Team', exact=True).focus(); page.keyboard.press('End'); page.keyboard.press('Enter')
    assert page.get_by_label('Team', exact=True).input_value() == '79'
    page.wait_for_function('plot.layout.annotations.length===1')
    assert page.get_by_text('No eligible historical edges for this team at this issuance.', exact=True).is_visible()
    page.get_by_label('League', exact=True).select_option('')
    page.get_by_label('Team', exact=True).select_option('0')
    page.wait_for_function('plot.layout.annotations.length===0 && plot.data[0].visible')
    assert page.evaluate('plot._fullLayout.xaxis._categories') == list(fig.layout.xaxis.categoryarray)[1:]
    assert page.evaluate('plot.data.length') == 1  # No hidden-team bars or slots.
    assert not errors
