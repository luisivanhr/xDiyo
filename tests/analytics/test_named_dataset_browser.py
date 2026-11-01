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



def test_pending_preview_cannot_rebind_changed_inputs_or_root(tmp_path):
    (tmp_path/'data/xDiyo_data').mkdir(parents=True)
    handle=launch_ui(tmp_path,open_browser=False)
    try:
        with playwright.sync_playwright() as p:
            browser=p.chromium.launch();page=browser.new_page()
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(handle.url)
            page.get_by_role('heading',name='Named datasets',exact=True).wait_for()
            page.evaluate("""async()=>{
              const {namedDataPanel}=await import('/named-data.js');
              window.pending=[];window.recipe={};
              const api=(name,args)=>new Promise((resolve,reject)=>pending.push({name,args,resolve,reject}));
              document.body.replaceChildren(namedDataPanel(recipe,api,()=>{},async()=>{}));
            }""")
            root=page.get_by_label('Dataset catalogue folder')
            train=page.get_by_label('Fixed training fixture IDs')
            preview=page.get_by_role('button',name='Preview selection',exact=True)
            accept=page.get_by_role('button',name='Use previewed selection',exact=True)
            root.fill('original');train.fill('9007199254740993');preview.click()
            train.fill('999')
            page.evaluate("pending[0].resolve({fingerprint:'old',selection_json:'old'})")
            assert accept.is_disabled()
            preview.click();root.fill('other-root')
            page.evaluate("pending[1].resolve({fingerprint:'wrong-root',selection_json:'wrong-root'})")
            assert accept.is_disabled()
            preview.click();preview.click()
            page.evaluate("pending[3].resolve({fingerprint:'latest',selection_json:'latest'})")
            accept.wait_for();page.wait_for_function("!Array.from(document.querySelectorAll('button')).find(b=>b.textContent==='Use previewed selection').disabled")
            page.evaluate("pending[2].resolve({fingerprint:'stale',selection_json:'stale'})")
            accept.click()
            assert page.evaluate('recipe.data_selection')=={'root':'other-root','selection':'latest'}
            assert page.evaluate('pending[0].args.options.train_ids')==['9007199254740993']
            preview.click();page.get_by_role('button',name='Use legacy loader',exact=True).click()
            page.evaluate("pending[4].resolve({fingerprint:'after-cancel',selection_json:'after-cancel'})")
            assert accept.is_disabled() and page.evaluate('recipe.data_selection') is None
            # A stale catalogue response must not populate another root's versions.
            listing=page.get_by_role('button',name='List dataset versions',exact=True)
            listing.click();root.fill('third');listing.click()
            page.evaluate("pending[6].resolve([{dataset:'international_dataset',version:'current'}])")
            page.get_by_label('international_dataset',exact=True).wait_for()
            page.evaluate("pending[5].resolve([{dataset:'international_dataset',version:'obsolete'}])")
            assert page.get_by_label('international_dataset',exact=True).locator('option').all_text_contents()==['Not selected','current']
            assert not errors
            browser.close()
    finally:handle.close()
