"""Regression cases from the dataset flags and refresh review; no model fitting."""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import json
import runpy
from pathlib import Path

from xdiyo_analytics.data import load_season, derive_movements, save_movement_enrichment
from xdiyo_analytics.data.publish_movements import materialize_movement_flags
from xdiyo_analytics.ui import default_recipe, refresh_recipe_data
from test_movement_enrichment import roster, COMPETITIONS
from transition_samples import history_from_games, mover_games, A
from xdiyo_analytics.features import TransitionContext, build_team_seasons

STEM='Premier_League_24_25'


@pytest.mark.parametrize('change',[dict(seasons=['24_25','24_25']),dict(tables=[]),
    dict(tables=['matches','matches']),dict(include_awarded='false'),dict(leagues=[]),dict(seasons=[])])
def test_refresh_rejects_invalid_options(export,tmp_path,change):
    recipe=default_recipe(str(export.root))
    recipe['data'].update(seasons=['24_25'],tables=['matches'])
    recipe['data'].update(change)
    with pytest.raises((ValueError,TypeError)):
        refresh_recipe_data(recipe,tmp_path/'invalid.json')
    assert not (tmp_path/'invalid.json').exists()
    assert not (tmp_path/'invalid_data_records').exists()


def test_refresh_discovery_and_interruption(export,tmp_path,monkeypatch):
    import xdiyo_analytics.ui.refresh as module
    (export.root/'misc.manifest.json').write_text('{}')
    recipe=default_recipe(str(export.root))
    recipe['data'].update(seasons=['24_25'],tables=['matches'])
    before=export.snapshot()
    def interrupted(*a,**kw): raise OSError('simulated final publication failure')
    with monkeypatch.context() as patch:
        patch.setattr(module.os,'link',interrupted)
        with pytest.raises(OSError,match='simulated'):
            refresh_recipe_data(recipe,tmp_path/'retry.json')
    assert not (tmp_path/'retry_data_records').exists()
    assert not (tmp_path/'retry.json').exists()
    refresh_recipe_data(recipe,tmp_path/'retry.json')
    assert export.snapshot()==before


def test_refresh_verifies_table_bytes(export,tmp_path):
    recipe=default_recipe(str(export.root))
    recipe['data'].update(seasons=['24_25'],tables=['matches'],verify_hashes=True)
    from xdiyo_analytics.data._source import _open_source
    path=_open_source(export.root,STEM).table_path('matches')
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='fingerprint'):
        refresh_recipe_data(recipe,tmp_path/'corrupt.json')
    assert not (tmp_path/'corrupt_data_records').exists()


def test_compression_and_nested_schema(export):
    original=load_season(export.root,STEM).matches
    table=export.table('matches')
    nested=table.append_column('sample_nested',pa.array([[1,2]]*len(table),type=pa.list_(pa.field('item',pa.int64()))))
    path=export.root/f'{STEM}.parquet'
    pq.write_table(nested,path,compression='zstd',use_compliant_nested_type=False)
    before=pq.read_table(path)
    ids=set(original.home_id)|set(original.away_id)
    flags=derive_movements([dict(stem=STEM,league='A',year=2024,competition_id=17,season_id=61627,
        teams={str(i):str(i) for i in ids})],{'A':dict(tier=1,system='A')})
    save_movement_enrichment(export.root,STEM,flags)
    from xdiyo_analytics.data._source import _open_source
    enrichment = next((export.root/'_team_seasons'/STEM).rglob('team_seasons.parquet'))
    assert pq.ParquetFile(enrichment).metadata.row_group(0).column(0).compression == 'ZSTD'
    materialize_movement_flags(export.root,STEM)
    actual=pq.read_table(path).select(before.column_names)
    assert actual.schema.equals(before.schema,check_metadata=True)
    assert actual.equals(before)
    assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression=='ZSTD'
    published = _open_source(export.root, STEM).table_path('team_seasons')
    assert pq.ParquetFile(published).metadata.row_group(0).column(0).compression == 'ZSTD'


def test_boundary_id_normalization_and_duplicates():
    rows=[roster('A',2024,[1,2])]
    boundary=dict(stem='A_2024',promoted_ids=['1'],sources=['test'],complete_review=True)
    result=derive_movements(rows,COMPETITIONS,boundaries=[boundary]).set_index('team_id')
    assert result.loc[1,'movement']=='promoted'
    with pytest.raises(ValueError,match='Duplicate boundary'):
        derive_movements(rows,COMPETITIONS,boundaries=[boundary,boundary])


@pytest.mark.parametrize('string_type', [pa.string(), pa.large_string()])
def test_existing_flag_schema_survives_pandas_string_width_changes(export, string_type):
    """Revisiting old publications must retain their exact Arrow field types."""
    from xdiyo_analytics.data.movements import attach_movement_flags, assert_movement_parity
    table = export.table('matches')
    ids = sorted(set(table['home_id'].to_pylist()) | set(table['away_id'].to_pylist()))
    flags = derive_movements([dict(stem=STEM, league='A', year=2024,
        competition_id=17, season_id=61627, teams={str(i): str(i) for i in ids})],
        {'A': dict(tier=1, system='A')}, boundaries=[dict(stem=STEM,
            promoted_ids=[ids[0]], complete_review=True, sources=['reviewed entry'])])
    enriched = attach_movement_flags(table.to_pandas(), flags)
    for name in enriched.columns:
        if name not in table.column_names:
            dtype = string_type if name.endswith('season_entry') else pa.bool_()
            field = pa.field(name, dtype, metadata={b'purpose': b'season-entry'})
            table = table.append_column(field, pa.array(enriched[name], type=dtype))
    export.replace('matches', table)
    nested = export.root / f'{STEM}.parquet'
    pq.write_table(table, nested, compression='zstd')
    paths = [export.version / 'matches.parquet', nested]
    before = {p: p.read_bytes() for p in paths}
    save_movement_enrichment(export.root, STEM, flags)
    materialize_movement_flags(export.root, STEM, reviewed_flags=flags)
    # Evidence-only refresh follows the same path used for the 2015 boundary.
    corrected = flags.copy()
    corrected['evidence'] = 'Reviewed predecessor evidence'
    materialize_movement_flags(export.root, STEM, reviewed_flags=corrected)
    for path in paths:
        assert path.read_bytes() == before[path]
        assert pq.ParquetFile(path).read().equals(table, check_metadata=True)
    loaded = load_season(export.root, STEM, tables='team_seasons', verify_hashes=True)
    assert_movement_parity(corrected, loaded['team_seasons'], season=STEM)


def test_builder_checks_persisted_evidence_before_replacing_audit(export,tmp_path):
    build=runpy.run_path(str(Path(__file__).resolve().parents[2]/'examples/build_team_movements.py'))['build']
    table=export.table('matches')
    for side in ('home','away'):
        table=table.append_column(f'{side}_name',pa.array([str(i) for i in table[f'{side}_id'].to_pylist()]))
    export.replace('matches',table)
    scope=export.manifest()['scope']
    matches=load_season(export.root,STEM).matches
    ids=sorted(set(matches.home_id)|set(matches.away_id))
    evidence=tmp_path/'evidence'; evidence.mkdir()
    review=dict(competitions={scope['league_name']:dict(tier=1,system='A')},seasons=[dict(
        stem=STEM,team_ids=ids,matches_sha256=export.manifest()['tables']['matches']['sha256'])])
    (evidence/'team_movement_roster_review.json').write_text(json.dumps(review))
    boundary=[dict(stem=STEM,complete_review=True,sources=['old source'])]
    path=evidence/'team_movement_boundary_evidence.json'
    path.write_text(json.dumps(boundary))
    pq.write_table(export.table('matches'),export.root/f'{STEM}.parquet')
    build(export.root,evidence)
    materialize_movement_flags(export.root,STEM)
    old_audit=(export.root/'_team_seasons/audit.json').read_bytes()
    boundary[0]['sources']=['corrected source']
    path.write_text(json.dumps(boundary))
    with pytest.raises(ValueError,match='persisted movement'):
        build(export.root,evidence)
    assert (export.root/'_team_seasons/audit.json').read_bytes()==old_audit
    from xdiyo_analytics.data._source import _open_source
    paths = [_open_source(export.root, STEM).table_path('matches'), export.root/f'{STEM}.parquet']
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    build(export.root,evidence,rematerialize=True)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    assert build(export.root,evidence).evidence.str.contains('corrected source').all()


def test_boundary_cannot_contradict_known_predecessor():
    with pytest.raises(ValueError,match='contradicts'):
        derive_movements([roster('A',2023,[1]),roster('A',2024,[1])],COMPETITIONS,
            boundaries=[dict(stem='A_2024',promoted_ids=[1],sources=['test'])])


def test_contradictory_flags_rejected():
    from xdiyo_analytics.data.movements import attach_movement_flags
    flags=derive_movements([roster('A',2024,[1,2])],COMPETITIONS,
        boundaries=[dict(stem='A_2024',promoted_ids=[1],sources=['test'],complete_review=True)])
    flags.loc[0,'got_demoted']=True
    with pytest.raises(ValueError,match='contradicts'):
        attach_movement_flags(pd.DataFrame(),flags)
    history=history_from_games(mover_games())
    record=dict(competition_id=10,season_id=2,team_id=A,movement='promoted',got_promoted=False,got_demoted=True)
    with pytest.raises(ValueError,match='contradicts'):
        TransitionContext(history,team_seasons=[record])
    override=dict(competition_id=10,season_id=2,team_id=A,movement='retained',evidence='test',
                  previous_competition_id=20,previous_season_id=11)
    with pytest.raises(ValueError,match='incompatible'):
        build_team_seasons(history,{10:dict(tier=1,system='A'),20:dict(tier=2,system='A')},overrides=[override])
