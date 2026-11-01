"""Real Chromium checks using the native authenticated builder endpoints."""
import json
import pytest
from test_named_datasets import published
from xdiyo_analytics.ui.server import launch_ui
playwright=pytest.importorskip('playwright.sync_api')


@pytest.mark.parametrize('width',[1200,390])
def test_named_dataset_preview_accept_and_legacy_restore(tmp_path,width):
    _,_,snapshots=published(tmp_path/'catalogue')
    (tmp_path/'data/xDiyo_data').mkdir(parents=True)
    handle=launch_ui(tmp_path,open_browser=False)
    try:
        with playwright.sync_playwright() as p:
            browser=p.chromium.launch()
            page=browser.new_page(viewport={'width':width,'height':900})
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
            page.goto(handle.url)
            page.get_by_role('heading',name='Named datasets',exact=True).wait_for()
            assert 'Football' in page.title()
            page.get_by_label('Dataset catalogue folder').fill(str(tmp_path/'catalogue'))
            page.get_by_role('button',name='List dataset versions',exact=True).click()
            page.get_by_label('international_dataset',exact=True).select_option(snapshots['international_dataset'])
            page.get_by_role('button',name='Preview selection',exact=True).click()
            button=page.get_by_role('button',name='Use previewed selection',exact=True)
            button.wait_for();page.wait_for_function("!Array.from(document.querySelectorAll('button')).find(e=>e.textContent==='Use previewed selection').disabled")
            assert 'conditional_retrospective_research' in page.locator('section.named-datasets').filter(has=page.get_by_role('heading',name='Named datasets',exact=True)).inner_text()
            with page.expect_response(lambda r: 'named-choices' in r.url) as response:
                button.click()
            assert response.value.ok
            choices=response.value.json()
            assert choices['teams']==[{'value':'1','label':'Alpha'},{'value':'2','label':'Beta'}]
            saved=json.loads(page.evaluate("localStorage.getItem('football-builder-draft-v2')"))
            selection=json.loads(saved['data_selection']['selection'])
            assert selection['target_ids']==[1,2,3,4]
            assert page.get_by_role('heading',name='Source data',exact=True).count()==0
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            page.screenshot(path=str(tmp_path/f'named-data-{width}.png'),full_page=True)
            page.get_by_role('button',name='Use legacy loader',exact=True).click()
            assert json.loads(page.evaluate("localStorage.getItem('football-builder-draft-v2')"))['data_selection'] is None
            assert not errors
            browser.close()
    finally:handle.close()
