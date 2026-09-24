"""Compatibility defaults for legacy graphical experiment recipes."""

import base64
from copy import deepcopy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from test_ui_workflow import ui_recipe
from xdiyo_analytics.ui.inventory import inventory
from xdiyo_analytics.ui.recipe import run_recipe, validate_recipe


def test_legacy_match_report_defaults_are_added_without_mutating_input(ui_recipe):
    params = ui_recipe['post_reporters']['matches']['params']
    params.pop('show_badges')
    params.pop('catalog')
    before = deepcopy(ui_recipe)
    migrated = validate_recipe(ui_recipe)
    assert ui_recipe == before
    assert migrated['post_reporters']['matches']['params']['show_badges'] is True
    assert migrated['post_reporters']['matches']['params']['catalog'] == {'ref': 'team_catalog'}
    migrated['post_reporters']['matches']['params']['catalog']['ref'] = 'changed'
    assert ui_recipe == before
    field = next(f for f in inventory()['components']['reporting.MatchResultReporter']['fields']
                 if f['name'] == 'show_badges')
    assert field['default'] is True


def test_explicit_badge_opt_out_and_custom_catalog_survive_validation(ui_recipe):
    params = ui_recipe['post_reporters']['matches']['params']
    params.update(show_badges=False, catalog={'ref': 'custom_catalog'})
    before = deepcopy(ui_recipe)
    migrated = validate_recipe(ui_recipe)
    assert migrated == before == ui_recipe


@pytest.mark.parametrize('explicit_opt_out', [False, True])
def test_legacy_recipe_renders_local_badges_unless_explicitly_disabled(ui_recipe, tmp_path, explicit_opt_out):
    badge = tmp_path / 'badge.png'
    badge.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jfN8AAAAASUVORK5CYII='))
    catalog = tmp_path / 'teams.json'
    catalog.write_text(json.dumps({str(2**53 + 3): {'name': 'Old source name', 'badge_path': badge.name}}),
                       encoding='utf-8')
    ui_recipe['team_badges'] = str(catalog)
    params = ui_recipe['post_reporters']['matches']['params']
    params.pop('catalog')
    params.pop('show_badges')
    if explicit_opt_out:
        params['show_badges'] = False
    before = deepcopy(ui_recipe)
    result = run_recipe(ui_recipe)
    html = result.to_html()
    assert ('data:image/png;base64,' in html) is not explicit_opt_out
    assert 'Alpha Home' in html
    assert ui_recipe == before


def test_reopening_legacy_report_refreshes_badges_without_mutating_saved_artifacts(ui_recipe, tmp_path):
    from xdiyo_analytics.ui.server import render_result_report

    badge = tmp_path / 'badge.png'
    badge.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jfN8AAAAASUVORK5CYII='))
    catalog = tmp_path / 'teams.json'
    catalog.write_text(json.dumps({str(2**53 + 3): {'name': 'Old name', 'badge_path': badge.name}}), encoding='utf-8')
    ui_recipe['team_badges'] = str(catalog)
    ui_recipe['post_reporters']['matches']['params']['show_badges'] = False
    report = run_recipe(ui_recipe).report
    artifact = next(a for study in report.studies for a in study.result.artifacts if a.kind == 'match_results')
    original_options = deepcopy(artifact.options)
    original_data = artifact.data.copy(deep=True)
    legacy_recipe = deepcopy(ui_recipe)
    legacy_recipe['post_reporters']['matches']['params'].pop('show_badges')
    legacy_recipe['post_reporters']['matches']['params'].pop('catalog')
    before_recipe = deepcopy(legacy_recipe)
    # Only a report is supplied: no model/training object is available for a refit.
    html = render_result_report(SimpleNamespace(report=report), legacy_recipe)
    assert 'data:image/png;base64,' in html
    assert artifact.options == original_options
    pd.testing.assert_frame_equal(artifact.data, original_data)
    assert legacy_recipe == before_recipe

    present = deepcopy(report)
    present_artifact = next(a for study in present.studies for a in study.result.artifacts if a.kind == 'match_results')
    present_artifact.options['teams'][str(2**53 + 3)]['badge'] = 'data:image/png;base64,retained'
    present_options = deepcopy(present_artifact.options)
    hidden = render_result_report(SimpleNamespace(report=present), ui_recipe)
    assert 'data:image/png;base64,' not in hidden
    assert present_artifact.options == present_options
