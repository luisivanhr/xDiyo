"""Multi-season discovery, independent concatenation and optional coverage."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from xdiyo_analytics.data import inspect_season, load_season, load_seasons, load_season_table


EVENT, HOME, AWAY = 2**53 + 1, 2**53 + 3, 2**53 + 5


def _publish(root, league, label, *, version='v1', extra=False, coverage=False):
    start, end = (2000 + int(piece) for piece in label.split('_'))
    league_id = {'Alpha': 101, 'Beta': 102, 'Gamma': 103}[league]
    season_id = 1000 + start
    stem = f'{league}_{label}'
    folder = root / f'_tables/competition={league_id}/season={season_id}/versions/{version}'
    folder.mkdir(parents=True)
    matches = pa.table({
        'event_id': [EVENT, 7], 'home_id': [HOME, 42], 'away_id': [AWAY, 50],
        'competition_id': [league_id] * 2, 'season_id': [season_id] * 2,
        'kickoff_utc': [1700000000.0, 1700001000.0],
    })
    if extra:
        matches = matches.append_column('extra_flag', pa.array([False, None], type=pa.bool_()))
    tables = {
        'matches': matches,
        'statistics': pa.table({
            'event_id': [EVENT, EVENT], 'period': ['ALL', 'ALL'],
            'group_name': ['Match overview', 'Shots'], 'key': ['totalShotsOnGoal'] * 2,
            'side': ['home', 'home'], 'team_id': [HOME, HOME],
            'value': pa.array([0.0, None], type=pa.float64()), 'display': ['0', 'unknown'],
        }),
        'pregame': pa.table({'event_id': [EVENT], 'side': ['home'], 'position': [3]}),
        'shots': pa.table({
            'event_id': [EVENT, EVENT, EVENT],
            'shot_id': pa.array([2**63 + 9, 2**63 + 10, 2**63 + 11], type=pa.uint64()),
            'xg': pa.array([0.0, None, float('nan')], type=pa.float64(), from_pandas=False),
        }),
    }
    if coverage:
        tables['coverage'] = pa.table({'event_id': [EVENT], 'component': ['statistics'], 'status': ['present']})
    descriptors = {}
    for name, table in tables.items():
        path = folder / f'{name}.parquet'
        pq.write_table(table, path)
        payload = path.read_bytes()
        descriptors[name] = {'file': path.name, 'rows': table.num_rows, 'bytes': len(payload),
                             'sha256': hashlib.sha256(payload).hexdigest()}
    manifest = {'schema_version': '2', 'parser_version': '2', 'version': version,
                'scope': {'league_id': league_id, 'league_name': league, 'season_id': season_id,
                          'season_start': start, 'season_end': end}, 'tables': descriptors}
    manifest_path = folder / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    publication = {'schema_version': '2', 'complete': True, 'matches': 2,
                   'season_file': f'{stem}.parquet', 'manifest': manifest_path.relative_to(root).as_posix()}
    (root / f'{stem}.manifest.json').write_text(json.dumps(publication), encoding='utf-8')
    return manifest_path


def _snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def population(tmp_path):
    root = tmp_path / 'exports'
    _publish(root, 'Beta', '22_23')
    _publish(root, 'Alpha', '23_24')
    _publish(root, 'Alpha', '22_23', extra=True)
    # A matching nested sidecar and an unselected top-level sidecar must be ignored.
    nested = root / 'nested'
    nested.mkdir()
    (nested / 'Gamma_22_23.manifest.json').write_text('not a publication', encoding='utf-8')
    (root / 'Gamma_21_22.manifest.json').write_text('not a publication', encoding='utf-8')
    return root


def test_discovery_order_labels_optional_schema_and_exact_source_data(population):
    before = _snapshot(population)
    result = load_seasons(population, ['23_24', '22_23'], tables=['matches', 'statistics', 'pregame'])
    stems = [source['season'] for source in result.provenance['sources']]
    assert stems == ['Alpha_23_24', 'Alpha_22_23', 'Beta_22_23']
    assert tuple(result.tables) == ('matches', 'statistics', 'pregame')
    assert [len(result[name]) for name in result.tables] == [6, 6, 3]
    assert result.shots is None
    assert result.matches['event_id'].tolist() == [EVENT, 7] * 3  # No global deduplication.
    assert result.matches['source_league'].tolist() == ['Alpha'] * 4 + ['Beta'] * 2
    assert result.matches['source_season'].tolist() == ['23_24'] * 2 + ['22_23'] * 4
    for frame in result.tables.values():
        assert frame.index.equals(pd.RangeIndex(len(frame)))
        for label in ('source_league', 'source_season'):
            assert frame[label].dtype.storage == 'pyarrow'
    assert result.matches['extra_flag'].isna().tolist() == [True, True, False, True, True, True]
    assert result.matches['extra_flag'].iloc[2] == False
    for source in result.provenance['sources']:
        league, start, end = source['season'].rsplit('_', 2)
        for name, combined in result.tables.items():
            with pq.ParquetFile(Path(source['manifest_path']).with_name(f'{name}.parquet')) as parquet:
                expected = parquet.read().to_pandas(types_mapper=pd.ArrowDtype, ignore_metadata=True)
            selected = combined.loc[combined.source_league.eq(league) & combined.source_season.eq(f'{start}_{end}')]
            pd.testing.assert_frame_equal(selected[expected.columns].reset_index(drop=True), expected, check_exact=True)
    assert _snapshot(population) == before


def test_exact_league_filter_and_missing_combinations_are_not_fabricated(population):
    result = load_seasons(population, ['22_23', '23_24'], leagues=['Beta', 'Alpha'])
    assert [s['season'] for s in result.provenance['sources']] == ['Alpha_22_23', 'Beta_22_23', 'Alpha_23_24']
    beta = load_seasons(population, '22_23', leagues='Beta', tables='statistics')
    assert len(beta.statistics) == 2 and beta.matches is None
    assert beta.provenance['leagues'] == ('Beta',)
    with pytest.raises(ValueError, match='No publications'):
        load_seasons(population, '22_23', leagues='alpha')


def test_shots_remain_independently_selectable_with_large_nullable_values(population):
    result = load_seasons(population, ['22_23', '23_24'], tables='shots')
    assert tuple(result.tables) == ('shots',) and result.matches is None
    assert result.shots['shot_id'].tolist() == [2**63 + 9, 2**63 + 10, 2**63 + 11] * 3
    assert result.shots['xg'].array.__arrow_array__().null_count == 3
    assert len(result.shots) == 9


@pytest.mark.parametrize('kwargs', [
    {'seasons': []}, {'seasons': '2022/23'}, {'seasons': '22_24'},
    {'seasons': ['22_23', '22_23']}, {'seasons': '22_23', 'leagues': []},
    {'seasons': '22_23', 'leagues': ['Alpha', 'Alpha']},
    {'seasons': '22_23', 'tables': ['matches', 'matches']},
])
def test_invalid_selections_fail_clearly(population, kwargs):
    with pytest.raises(ValueError, match='Choose'):
        load_seasons(population, **kwargs)


@pytest.mark.parametrize('kwargs', [
    {'seasons': ['22_23', '24_25']}, {'seasons': '22_23', 'leagues': ['Alpha', 'Gamma']},
])
def test_missing_entire_requested_labels_raise(population, kwargs):
    with pytest.raises(ValueError, match='No publications'):
        load_seasons(population, **kwargs)


@pytest.mark.parametrize('column', ['source_league', 'source_season'])
def test_existing_origin_column_is_never_overwritten(population, column):
    sidecar = json.loads((population / 'Alpha_22_23.manifest.json').read_bytes())
    path = population / sidecar['manifest']
    table_path = path.with_name('shots.parquet')
    with pq.ParquetFile(table_path) as parquet:
        table = parquet.read()
    pq.write_table(table.append_column(column, pa.array(['original'] * len(table))), table_path)
    manifest = json.loads(path.read_bytes())
    payload = table_path.read_bytes()
    manifest['tables']['shots'].update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    path.write_text(json.dumps(manifest), encoding='utf-8')
    before = _snapshot(population)
    with pytest.raises(ValueError, match='already contains reserved'):
        load_seasons(population, '22_23', tables='shots')
    assert _snapshot(population) == before


def test_unavailable_requested_table_is_an_error(population):
    with pytest.raises(KeyError, match="coverage.*unavailable"):
        load_seasons(population, ['22_23', '23_24'], tables=['matches', 'coverage'])


def test_records_pin_versions_and_allow_new_discovery_membership(population, tmp_path):
    records = tmp_path / 'selections'
    before = _snapshot(population)
    first = load_seasons(population, ['22_23', '23_24'], tables=['matches', 'statistics'], record_dir=records)
    assert _snapshot(population) == before
    saved = {p.name: p.read_bytes() for p in records.glob('*.json')}
    assert len(saved) == 3
    _publish(population, 'Alpha', '22_23', version='v2', extra=True)
    replay = load_seasons(population, ['22_23', '23_24'], tables=['matches', 'statistics'], record_dir=records)
    pd.testing.assert_frame_equal(replay.matches, first.matches)
    assert replay.provenance['sources'][0]['version'] == 'v1'
    assert {p.name: p.read_bytes() for p in records.glob('*.json')} == saved
    _publish(population, 'Beta', '23_24')
    expanded = load_seasons(population, ['22_23', '23_24'], record_dir=records)
    assert len(expanded.provenance['sources']) == 4 and len(expanded.matches) == 8
    assert all((records / name).read_bytes() == payload for name, payload in saved.items())
    assert len(list(records.glob('*.json'))) == 4


def test_failed_later_publication_raises_and_retains_only_earlier_records(population, tmp_path):
    path = population / 'Alpha_23_24.manifest.json'
    publication = json.loads(path.read_bytes())
    publication['complete'] = False
    path.write_text(json.dumps(publication), encoding='utf-8')
    records = tmp_path / 'selections'
    with pytest.raises(ValueError, match='not a complete'):
        load_seasons(population, ['22_23', '23_24'], record_dir=records)
    assert sorted(p.name for p in records.glob('*.json')) == ['Alpha_22_23.json', 'Beta_22_23.json']


def test_record_directory_cannot_be_inside_source(population):
    with pytest.raises(ValueError, match='outside the source'):
        load_seasons(population, '22_23', record_dir=population / 'selections')
    assert not (population / 'selections').exists()


def test_opt_in_hash_audit_reaches_every_partition(population):
    publication = json.loads((population / 'Beta_22_23.manifest.json').read_bytes())
    path = population / publication['manifest']
    manifest = json.loads(path.read_bytes())
    manifest['tables']['shots']['sha256'] = '0' * 64
    path.write_text(json.dumps(manifest), encoding='utf-8')
    assert len(load_seasons(population, '22_23', tables='shots').shots) == 6
    with pytest.raises(ValueError, match='File fingerprint'):
        load_seasons(population, '22_23', tables='shots', verify_hashes=True)


@pytest.mark.parametrize('previous_coverage', [False, True])
def test_loaders_and_inspector_work_when_coverage_is_omitted(tmp_path, previous_coverage):
    root = tmp_path / 'exports'
    old_manifest = _publish(root, 'Alpha', '22_23', coverage=previous_coverage)
    previous_bytes = old_manifest.read_bytes()
    if previous_coverage:
        selected_manifest = _publish(root, 'Alpha', '22_23', version='v2', coverage=False)
        assert old_manifest.with_name('coverage.parquet').exists()
        assert old_manifest.read_bytes() == previous_bytes
    else:
        selected_manifest = old_manifest
    assert 'coverage' not in json.loads(selected_manifest.read_bytes())['tables']
    assert not selected_manifest.with_name('coverage.parquet').exists()
    assert 'coverage' not in inspect_season(root, 'Alpha_22_23').index
    assert len(load_season(root, 'Alpha_22_23', tables=['matches', 'statistics', 'pregame']).matches) == 2
    assert len(load_season_table(root, 'Alpha_22_23', table='statistics')) == 2
    assert len(load_seasons(root, '22_23', tables=['matches', 'statistics', 'pregame']).statistics) == 2
    with pytest.raises(KeyError, match="coverage.*unavailable"):
        load_season(root, 'Alpha_22_23', tables='coverage')
    with pytest.raises(KeyError, match="coverage.*unavailable"):
        load_seasons(root, '22_23', tables='coverage')
