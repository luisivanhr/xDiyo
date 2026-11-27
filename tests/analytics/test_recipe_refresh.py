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

    from xdiyo_analytics.data.movements import assert_movement_parity
    before_correction = load_season(export.root, STEM, tables='team_seasons')['team_seasons']
    frame.loc[:, 'evidence'] = 'Corrected reviewed source URL'
    with pytest.raises(ValueError, match='persisted movement'):
        assert_movement_parity(frame, before_correction)
    materialize_movement_flags(export.root, STEM, reviewed_flags=frame)
    corrected = load_season(export.root, STEM, tables='team_seasons', verify_hashes=True)['team_seasons']
    assert_movement_parity(frame, corrected)
    assert materialize_movement_flags(export.root, STEM, reviewed_flags=frame)['already_materialized']


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


def test_refresh_query_addition_and_missing_existing_pin(export,tmp_path):
    from test_native_query_preflight import publication, recipe
    import pyarrow as pa

    publication(export)
    original = recipe(export, query={'competitions': 17})
    first = refresh_recipe_data(original, tmp_path/'first.json')['recipe']
    # A genuinely new season of the same competition starts matching the saved query.
    extra = type(export)(export.root)
    extra.version = export.version.parent.parent.parent/'season=70000'/'versions'/'v1'
    extra.version.mkdir(parents=True)
    extra.manifest_path = extra.version/'manifest.json'
    extra.publication_path = export.root/'Premier_League_25_26.manifest.json'
    manifest = export.manifest()
    manifest['scope'].update(season_id=70000,season_start=2025,season_end=2026)
    manifest['native_integration']['season'].update(id=70000,name='Premier League 25/26',year='25/26')
    manifest['season_calendar'].update(season_id=70000,annual_season='2025/2026')
    manifest['tables'] = {}
    extra.manifest_path.write_text(json.dumps(manifest))
    for table in ('matches','team_seasons'):
        rows = export.table(table).to_pylist()
        for row in rows:
            row['season_id']=70000
            if table=='matches':
                row['event_id']+=1000
        extra.replace(table,pa.Table.from_pylist(rows))
    pub = json.loads(export.publication_path.read_bytes())
    pub.update(season_file='Premier_League_25_26.parquet',
               manifest=extra.manifest_path.relative_to(export.root).as_posix(),
               native_integration=manifest['native_integration'],season_calendar=manifest['season_calendar'])
    extra.publication_path.write_text(json.dumps(pub))
    result = refresh_recipe_data(first,tmp_path/'second.json')
    assert [(v['season'],v['change_kind']) for v in result['changes'] if v['change_kind']=='new_selection'] == [
        ('Premier_League_25_26','new_selection')]
    assert json.loads(Path(result['report_path']).read_bytes())['native_selection']['added'] == ['Premier_League_25_26']
    (Path(first['data']['record_dir'])/(STEM+'.json')).unlink()
    with pytest.raises(FileNotFoundError,match='Missing previous selection record'):
        refresh_recipe_data(first,tmp_path/'third.json')


def test_refresh_checks_overlay_without_repinning(export,tmp_path):
    from hashlib import sha256
    part=tmp_path/'corners.jsonl'
    part.write_bytes(b'')
    manifest=tmp_path/'overlay.json'
    manifest.write_text(json.dumps(dict(version='test',partitions=[dict(path=part.name,
        keys=['cornerKicks'],sha256=sha256(part.read_bytes()).hexdigest())])))
    r=default_recipe(str(export.root))
    r['data'].update(seasons=['24_25'],tables=['matches'])
    r['statistics_overlay']=dict(manifest_path=manifest.name,manifest_sha256=sha256(manifest.read_bytes()).hexdigest(),
                                 keys=['cornerKicks'])
    result=refresh_recipe_data(r,'v2.json',workspace=tmp_path,dry_run=True)
    assert result['report']['verified_statistics_overlay']==r['statistics_overlay']
    part.write_bytes(b'changed')
    for preview in (True, False):
        with pytest.raises(ValueError,match='partition hash mismatch'):
            refresh_recipe_data(r,'v2.json',workspace=tmp_path,dry_run=preview)
    assert not (tmp_path/'v2.json').exists()
