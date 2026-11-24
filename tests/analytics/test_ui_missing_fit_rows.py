"""Actual browser controls roundtrip without preparing or training."""
from test_ui_odds_roundtrip import browser_ui, import_recipe, save_recipe
from xdiyo_analytics.ui import default_recipe
from xdiyo_analytics.ui.recipe import node


def test_browser_policy_preserves_and_disables_explicit_recipe(browser_ui):
    page, handle, calls = browser_ui
    recipe = default_recipe('/synthetic/data')
    recipe['candidate']['missing_fit_rows'] = node('training.MissingFitRows', required_features=['home::corners_mean', 'away::corners_mean'])
    import_recipe(page, recipe)
    page.get_by_role('link', name='Model & training', exact=True).click()
    assert save_recipe(page, handle) == recipe
    enabled = page.get_by_role('checkbox', name='Enable Missing fit rows', exact=True)
    assert enabled.is_checked()
    assert page.get_by_text('Required features', exact=True).is_visible()
    enabled.uncheck()
    saved = save_recipe(page, handle)
    assert saved['candidate'].get('missing_fit_rows') is None
    assert saved['features'] == recipe['features']
    assert saved['preprocessors'] == recipe['preprocessors']
    assert not any(route in ('run', 'prepare', 'predict') for route, _ in calls)


# Use an installed system Chromium when the optional Playwright download is absent.
import pytest
import shutil
from pathlib import Path

@pytest.fixture(autouse=True)
def available_chromium(monkeypatch):
    api = pytest.importorskip('playwright.sync_api')
    original = api.BrowserType.launch
    def launch(browser_type, **kwargs):
        if not Path(browser_type.executable_path).exists() and shutil.which('chromium'):
            kwargs.setdefault('executable_path', shutil.which('chromium'))
        try:
            return original(browser_type, **kwargs)
        except api.Error as error:
            if "socket() failed: Operation not permitted" in str(error):
                pytest.skip("Browser integration blocked by runtime IPC socket restriction")
            raise
    monkeypatch.setattr(api.BrowserType, 'launch', launch)
