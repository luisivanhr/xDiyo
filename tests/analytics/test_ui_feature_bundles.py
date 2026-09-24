"""Pure browser bundle output remains an ordinary executable Python recipe."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from test_ui_workflow import ui_recipe
from xdiyo_analytics.ui.inventory import inventory
from xdiyo_analytics.ui.recipe import node, prepare_recipe


def test_node_bundle_expression_prepares_through_normal_recipe(ui_recipe):
    executable = shutil.which('node')
    if executable is None:
        pytest.skip('Node is needed to verify the actual browser bundle helper.')
    module = Path(__file__).resolve().parents[2] / 'src/xdiyo_analytics/ui/static/feature-bundles.js'
    payload = {
        'template': node('features.RollingMean', source=node('features.Stat', period='ALL',
                         group='Match overview', key='cornerKicks'), window=3),
        'stats': [{'period': 'ALL', 'group_name': 'Match overview', 'key': 'cornerKicks'}],
        'components': list(inventory()['components'].values()),
    }
    script = """
      import {readFileSync} from 'node:fs';
      const {expandFeatureBundle}=await import(process.argv[1]);
      const input=JSON.parse(readFileSync(0,'utf8'));
      process.stdout.write(JSON.stringify(expandFeatureBundle(input.template,input.stats,[],input.components)));
    """
    completed = subprocess.run([executable, '--input-type=module', '-e', script, module.as_uri()],
                               input=json.dumps(payload), text=True, capture_output=True, check=True)
    expanded = json.loads(completed.stdout)
    assert len(expanded) == 1
    name = expanded[0]['name']
    ui_recipe['features'] = {name: expanded[0]['expression']}
    prepared = prepare_recipe(ui_recipe)
    assert len(prepared.dataset.X) == 23
    metadata = prepared.dataset.metadata
    third = metadata.index[(metadata.source_season == '22_23') & (metadata['round'] == 3)][0]
    home = [c for c in prepared.dataset.X if c.startswith('home::') and name in c]
    away = [c for c in prepared.dataset.X if c.startswith('away::') and name in c]
    assert len(home) == len(away) == 1
    # Match three uses only the first two results: home 3,4 and away 2,3.
    assert prepared.dataset.X.loc[third, home[0]] == pytest.approx(3.5)
    assert prepared.dataset.X.loc[third, away[0]] == pytest.approx(2.5)
