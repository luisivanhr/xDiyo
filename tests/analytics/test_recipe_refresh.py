from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from xdiyo_analytics.data import load_season, derive_movements, save_movement_enrichment
from xdiyo_analytics.data.publish_movements import materialize_movement_flags
from xdiyo_analytics.ui import default_recipe, refresh_recipe_data


STEM = 'Premier_League_24_25'


def test_materialize_then_refresh_stale_recipe_without_old_dataset(export, tmp_path, monkeypatch):
    original = load_season(export.root, STEM)
    native_before = export.table('matches')
    nested_path = export.root / f'{STEM}.parquet'
    pq.write_table(native_before, nested_path)
    records = tmp_path / 'old_records'
    load_season(export.root, STEM, record_path=records / f'{STEM}.json')
    old_pin = (records / f'{STEM}.json').read_bytes()
    recipe = default_recipe(str(export.root))
    recipe['data'].update(seasons=['24_25'], tables=['matches'], record_dir=str(records))
    before = deepcopy(recipe)
    ids = set(original.matches.home_id) | set(original.matches.away_id)
    frame = derive_movements([dict(stem=STEM, league='A', year=2024,
        competition_id=17, season_id=61627, teams={str(i): str(i) for i in ids})],
        {'A': {'tier': 1, 'system': 'A'}}, boundaries=[dict(stem=STEM,
        sources=['test'], promoted_ids=[max(ids)], complete_review=True)])
    save_movement_enrichment(export.root, STEM, frame)
    replace = Path.replace
    def interrupted(path, target):
        if Path(target) == export.publication_path:
            raise OSError('simulated interruption before publication update')
        return replace(path, target)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'replace', interrupted)
        with pytest.raises(OSError, match='simulated interruption'):
            materialize_movement_flags(export.root, STEM)
    audit = materialize_movement_flags(export.root, STEM)  # resume journal
    assert audit['new_manifest_sha256'] != audit['old_manifest_sha256']
    assert materialize_movement_flags(export.root, STEM)['already_materialized']
    for table in (export.table('matches'), pq.read_table(nested_path)):
        assert table.select(native_before.column_names).equals(native_before)
        assert 'home_got_promoted' in table.column_names
    publication = json.loads(export.publication_path.read_bytes())
    assert publication['sha256'] == hashlib.sha256(nested_path.read_bytes()).hexdigest()
    assert 'team_seasons' in export.manifest()['tables']
    with pytest.raises(ValueError, match='manifest has changed'):
        load_season(export.root, STEM, record_path=records / f'{STEM}.json')
    result = refresh_recipe_data(recipe, tmp_path / 'updated.json')
    updated = result['recipe']
    loaded = load_season(export.root, STEM,
                        record_path=f"{updated['data']['record_dir']}/{STEM}.json", verify_hashes=True)
    assert 'home_got_promoted' in loaded.matches
    assert result['changes'][0]['changed']
    assert recipe == before
    assert (records / f'{STEM}.json').read_bytes() == old_pin
    assert {k:v for k,v in updated.items() if k != 'data'} == {k:v for k,v in recipe.items() if k != 'data'}
    with pytest.raises(FileExistsError):
        refresh_recipe_data(recipe, tmp_path / 'updated.json')


def test_missing_season_fails_without_creating_output(export, tmp_path):
    recipe = default_recipe(str(export.root))
    recipe['data']['seasons'] = ['24_25', '25_26']
    output = tmp_path / 'updated.json'
    with pytest.raises(ValueError, match='no current publication'):
        refresh_recipe_data(recipe, output)
    assert not output.exists() and not output.with_name('updated_data_records').exists()


def test_ui_refresh_endpoint_preserves_settings(export, tmp_path):
    from xdiyo_analytics.ui.server import BuilderState
    from xdiyo_analytics.ui.recipe import catalog_for_ui
    state = BuilderState(tmp_path, catalog_for_ui())
    recipe = default_recipe(str(export.root))
    recipe['data'].update(seasons=['24_25'], tables=['matches'])
    try:
        result = state.dispatch('refresh-data', {'recipe': recipe})
        assert result['recipe']['model'] == recipe['model']
        assert result['changes'][0]['season'] == STEM
    finally:
        state.executor.shutdown()
