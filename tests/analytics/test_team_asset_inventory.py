"""The offline catalog covers published identities, including historical clubs."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

from xdiyo_analytics.reporting import TeamCatalog

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'docs/analytics/team_assets'


def test_all_published_teams_have_hashed_svg_badges():
    spec = importlib.util.spec_from_file_location('team_inventory', ASSETS / 'inventory_teams.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stored = json.loads((ASSETS / 'catalog.json').read_text(encoding='utf-8'))
    _, current = module.inventory(ROOT / 'data/xDiyo_data', stored)
    assert current['team_ids']
    catalog = TeamCatalog.from_json(ASSETS / 'catalog.json')
    for key in current['team_ids']:
        entry = stored[key]
        path = ASSETS / entry['badge_path']
        assert entry['format'] == 'svg' and path.suffix == '.svg'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], key
        _, image, error = catalog.display(key, badges=True)
        assert error is None and image.startswith('data:image/svg+xml;base64,'), key


def test_inventory_includes_old_seasons_and_preserves_badge_provenance(tmp_path):
    spec = importlib.util.spec_from_file_location('team_inventory', ASSETS / 'inventory_teams.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for season, home, name in [('10_11', 1, 'Historical FC'), ('25_26', 3, 'Current FC')]:
        publication = tmp_path / season
        publication.mkdir()
        pd.DataFrame(dict(home_id=[home], home_name=[name], away_id=[2], away_name=['Away FC'])).to_parquet(publication / 'matches.parquet')
        (publication / 'manifest.json').write_text(json.dumps(dict(scope=dict(league_name='League'), tables=dict(matches=dict(file='matches.parquet')))))
        (tmp_path / f'League_{season}.manifest.json').write_text(json.dumps(dict(manifest=f'{season}/manifest.json')))
    provenance = dict(name='Old spelling', badge_path='badges/1.svg', sha256='unchanged', source_url='https://example.com/crest.svg')
    previous = {'1': provenance.copy()}
    result, summary = module.inventory(tmp_path, previous)
    assert previous == {'1': provenance}
    assert summary['team_ids'] == ['1', '2', '3']
    assert result['1']['name'] == 'Historical FC'
    assert result['1']['seasons'] == ['10_11']
    assert result['1']['badge_path'] == provenance['badge_path']
    assert result['1']['sha256'] == provenance['sha256']
    assert result['1']['source_url'] == provenance['source_url']
