"""Portable badge bundles must not silently retain an older league subset."""
import base64
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from xdiyo_analytics.reporting import TeamCatalog


REPORTED_MISSING = {str(x) for x in (47, 64, 1665, 1672, 1675, 1711, 2363,
    2595, 2730, 2741, 2868, 2869, 2972, 7935, 24338, 24360, 43680, 113755, 368506)}


def test_complete_builtin_bundle_survives_relocation(tmp_path):
    source = TeamCatalog.builtin()
    original = json.dumps(source.entries, sort_keys=True)
    path = source.export_bundle(tmp_path / 'bundle')
    records = json.loads(path.read_text('utf-8'))
    assert REPORTED_MISSING <= records.keys()
    for league in ('J1', 'J2', 'Liga_MX_Apertura', 'Liga_MX_Clausura'):
        ids = {key for key, row in source.entries.items() if league in row.get('leagues', [])}
        assert ids and ids <= records.keys()
    moved = tmp_path / 'relocated'
    shutil.move(path.parent, moved)
    loaded = TeamCatalog.from_json(moved / 'catalog.json')
    assert loaded.entries.keys() == source.entries.keys()
    for key, record in records.items():
        assert not Path(record['badge_path']).is_absolute()
        data = (moved / record['badge_path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == source.entries[key]['sha256']
        name, uri, note = loaded.display(key, badges=True)
        assert note is None and name == source.entries[key]['name']
        assert uri.startswith('data:image/svg+xml;base64,')
        assert base64.b64decode(uri.split(',', 1)[1]) == data
    assert json.dumps(source.entries, sort_keys=True) == original


@pytest.mark.parametrize('problem', ['absent', 'path', 'hash', 'unsafe'])
def test_bad_bundle_fails_before_creating_output(tmp_path, problem):
    asset = tmp_path / 'source.svg'
    asset.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    record = dict(name='Test', badge_path=str(asset))
    if problem == 'absent':
        record.pop('badge_path')
    elif problem == 'path':
        asset.unlink()
    elif problem == 'hash':
        record['sha256'] = 'wrong'
    else:
        asset.write_text('<svg><script/></svg>')
    destination = tmp_path / 'output'
    with pytest.raises(ValueError, match='Team 47:'):
        TeamCatalog({47: record}).export_bundle(destination)
    assert not destination.exists()


def test_bundle_subset_and_content_mime(tmp_path):
    asset = tmp_path / 'wrong.png'
    asset.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    catalog = TeamCatalog({47: dict(name='One', badge_path=str(asset)), 64: dict(name='Two')})
    path = catalog.export_bundle(tmp_path / 'bundle', team_ids=[47, '47'])
    result = TeamCatalog.from_json(path)
    assert list(result.entries) == ['47']
    assert result.entries['47']['badge_path'].endswith('.svg')
    with pytest.raises(ValueError, match='missing catalog entry'):
        catalog.export_bundle(tmp_path / 'unknown', team_ids=[999999999])
    with pytest.raises(FileExistsError):
        catalog.export_bundle(path.parent, team_ids=[47])


def test_embedding_retains_exact_ids_for_badge_refresh():
    from xdiyo_analytics.reporting.embedding_view import embedding_figure
    table = pd.DataFrame(dict(team_id=[47, 2**53 + 1, 64], league=[1, 2, 1],
                              team_name=['Same', 'Same', 'Other'], x=[0., 0., 1.],
                              y=[0., 0., 1.], support=[1., 2., 3.], gate=[.1, .2, .3]))
    figure = embedding_figure(table, [None] * 3, 'Saved coordinates')
    assert tuple(figure.data[0].ids) == ('47', '64')
    assert tuple(figure.data[1].ids) == (str(2**53 + 1),)
    assert tuple(figure.data[0].x) == (0., 1.)
