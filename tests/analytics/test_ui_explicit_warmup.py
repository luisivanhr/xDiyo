"""Browser and synthetic preparation checks; no run/train/predict calls."""
import json
from pathlib import Path
import runpy
import tempfile
from copy import deepcopy

from test_ui_odds_roundtrip import browser_ui, import_recipe, save_recipe, export_payload
from test_ui_workflow import ui_recipe
from xdiyo_analytics.ui import default_recipe,prepare_recipe
from xdiyo_analytics.ui.catalog import default_catalog
from xdiyo_analytics.features import WarmStart,SeededEMA,RollingStd,LinearFade,TeamMovement,Stat


def test_explicit_modes_controls_roundtrip_exports(browser_ui):
    page,handle,calls=browser_ui
    assert page.url==handle.url and page.title()=='Football experiment builder'
    assert page.get_by_role('link',name='Data',exact=True).is_visible()
    recipe=default_recipe('/synthetic/data')
    recipe['features']=default_catalog().encode({
        'dispersion':WarmStart(RollingStd(Stat('ALL','Match overview','cornerKicks'),5,ddof=0),
            SeededEMA(mode='uniform',alpha=.25,handoff=LinearFade(1,2))),
        'was_promoted':TeamMovement(), 'was_relegated':TeamMovement('relegated')})
    # Ordinary statistics omit the default venue field in the builder; venue
    # controls are reserved for spatial features by the existing UI contract.
    recipe['features']['dispersion']['params']['source']['params'].pop('venue')
    import_recipe(page,recipe)
    page.get_by_role('link',name='Features & ratings',exact=True).click()
    assert save_recipe(page,handle)==recipe
    mode=page.get_by_role('combobox',name='Mode',exact=True)
    assert mode.locator('option:checked').inner_text()=='Uniform — own rolling state'
    assert page.get_by_role('spinbutton',name='Bottom',exact=True).count()==0
    assert page.get_by_role('spinbutton',name='League weight',exact=True).count()==0
    mode.select_option(label='With league prior — mover cohort')
    p=recipe['features']['dispersion']['params']['policy']['params'];p['mode']='w_league_prior'
    bottom=page.get_by_role('spinbutton',name='Bottom',exact=True)
    bottom.fill('-1');bottom.blur();p['bottom']=-1
    estimator=page.get_by_role('combobox',name='Variance estimator',exact=True)
    estimator.select_option(label='weighted sample');p['variance_estimator']='weighted_sample'
    page.get_by_role('checkbox',name='Enable Prior strength',exact=True).check()
    strength=page.get_by_role('spinbutton',name='Prior strength',exact=True)
    strength.fill('4');strength.blur();p['prior_strength']=4.
    page.get_by_role('spinbutton',name='Ddof',exact=True).fill('1')
    page.get_by_role('spinbutton',name='Ddof',exact=True).blur()
    recipe['features']['dispersion']['params']['source']['params']['ddof']=1
    assert save_recipe(page,handle)==recipe
    page.get_by_role('button',name='Open recipe',exact=True).click()
    page.get_by_role('button',name='Open',exact=True).click()
    page.get_by_text('Recipe opened.',exact=True).wait_for()
    page.get_by_role('link',name='Features & ratings',exact=True).click()
    assert save_recipe(page,handle)==recipe
    mode.evaluate("element => element.scrollIntoView({block: 'center'})")
    page.screenshot(path=str(Path(tempfile.gettempdir())/'xdiyo-explicit-warmup-desktop.png'))
    for fmt in ('Python','notebook'):
        with page.expect_download() as downloading:
            page.get_by_role('button',name=f'Export {fmt}',exact=True).click()
        text=Path(downloading.value.path()).read_text()
        source=text if fmt=='Python' else ''.join(json.loads(text)['cells'][1]['source'])
        compile(source,'export','exec')
        assert export_payload(source)==handle.state.recipe_paths(recipe)
    page.set_viewport_size({'width':480,'height':850})
    mode.evaluate("element => element.scrollIntoView({block: 'center'})")
    assert mode.is_visible()
    page.screenshot(path=str(Path(tempfile.gettempdir())/'xdiyo-explicit-warmup-mobile.png'))
    assert not any(route in ('run','prepare','predict') for route,_ in calls)


def test_compact_sources_recipe_preparation_only(ui_recipe):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import hashlib
    # Enrich only the temporary synthetic export with the requested two shots identities.
    root=Path(ui_recipe['data']['root']) if 'root' in ui_recipe['data'] else Path(ui_recipe['data']['data_root'])
    for manifest_path in root.glob('_tables/**/manifest.json'):
        manifest=json.loads(manifest_path.read_text())
        path=manifest_path.parent/'statistics.parquet'
        stats=pq.read_table(path).to_pandas()
        shots=stats.copy();shots['group_name']='Shots';shots['key']='totalShotsOnGoal'
        target=shots.copy();target['key']='shotsOnGoal'
        import pandas as pd
        pq.write_table(pa.Table.from_pandas(pd.concat([stats,shots,target],ignore_index=True),preserve_index=False),path)
        manifest['tables']['statistics'].update(rows=len(stats)*3,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        manifest_path.write_text(json.dumps(manifest))
    factory=runpy.run_path(str(Path(__file__).resolve().parents[2]/'examples/explicit_warmup.py'))['feature_definitions']
    for mode in ('uniform','w_league_prior'):
        recipe=deepcopy(ui_recipe)
        features=factory(mode=mode,alpha=.5,fade_start=1,fade_rounds=2,bottom=-1,top=-1,
                         variance_estimator='population',prior_strength=None)
        recipe['features']=default_catalog().encode(features)
        prepared=prepare_recipe(recipe)
        assert 'home::was_promoted' in prepared.dataset.X
        assert 'home::goals_for' in prepared.dataset.X
        assert prepared.dataset.X['home::was_promoted'].isna().all()
        assert prepared.dataset.definitions['warm_start_audit']
