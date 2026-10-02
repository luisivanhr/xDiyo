import pandas as pd
import pytest

from xdiyo_analytics.data import (derive_movements, save_movement_enrichment,
                                  load_season, load_seasons, inspect_season, select_stats)
from xdiyo_analytics.histories import build_team_history


COMPETITIONS = {'A': {'tier': 1, 'system': 'country'}, 'B': {'tier': 2, 'system': 'country'}}


def roster(league, year, teams):
    return dict(league=league, year=year, stem=f'{league}_{year}',
                competition_id=1 if league == 'A' else 2, season_id=year,
                teams={str(t): f'Team {t}' for t in teams})


def test_cross_tier_missing_below_and_boundary_exceptions():
    rosters = [roster('A', 2023, [1, 2]), roster('B', 2023, [3, 4]),
               roster('A', 2024, [1, 3]), roster('B', 2024, [2, 4, 5, 6])]
    boundaries = [dict(stem='B_2024', other_entry_ids=[6], sources=['reviewed administrative entry'])]
    result = derive_movements(rosters, COMPETITIONS,
                              reviewed_seasons=['A_2023', 'B_2023'], boundaries=boundaries)
    current = result[result.season_id.eq(2024)].set_index('team_id')
    assert current.movement.to_dict() == {1: 'retained', 3: 'promoted', 2: 'relegated',
                                          4: 'retained', 5: 'promoted', 6: 'other_entry'}
    assert current.loc[2, 'previous_competition_id'] == 1
    assert current.loc[5, 'evidence'].startswith('Inferred entry from below:')
    assert result.loc[result.season_id.eq(2023), 'got_promoted'].isna().all()


def test_incomplete_previous_upper_and_missing_season_do_not_imply_promotion():
    result = derive_movements([roster('B', 2023, [1]), roster('B', 2024, [2]),
                               roster('A', 2025, [3])], COMPETITIONS, reviewed_seasons=['B_2023'])
    assert result.movement.eq('unknown').all()


def test_additive_loading_preserves_existing_files_records_and_exact_ids(export, tmp_path):
    stem = 'Premier_League_24_25'
    saved = tmp_path / 'selection.json'
    original = load_season(export.root, stem, record_path=saved)
    before, record = export.snapshot(), saved.read_bytes()
    matches = original.matches
    ids = set(matches.home_id.tolist()) | set(matches.away_id.tolist())
    r = dict(stem=stem, league='A', year=2024, competition_id=17, season_id=61627,
             teams={str(t): str(t) for t in ids})
    frame = derive_movements([r], COMPETITIONS, boundaries=[dict(
        stem=stem, promoted_ids=[max(ids)], sources=['test review'], complete_review=True)])
    save_movement_enrichment(export.root, stem, frame)
    save_movement_enrichment(export.root, stem, frame)  # idempotent
    loaded = load_season(export.root, stem, tables=['matches', 'team_seasons'], record_path=saved)
    history = build_team_history(loaded)
    assert history.loc[history.team_id.eq(max(ids)), 'team_got_promoted'].all()
    assert set(loaded['team_seasons'].team_id) == ids
    assert 'team_seasons' in inspect_season(export.root, stem).index
    assert 'team_seasons' in loaded.provenance
    with_stats = load_season(export.root, stem, tables=['matches', 'statistics', 'team_seasons'])
    selected = select_stats(with_stats, bundles='all_stats')
    pd.testing.assert_frame_equal(selected['team_seasons'], with_stats['team_seasons'])
    assert load_seasons(export.root, ['24_25'], tables=['team_seasons'])['team_seasons'].shape[0] == 4
    pd.testing.assert_frame_equal(original.matches, load_season(export.root, stem).matches)
    assert saved.read_bytes() == record
    after = export.snapshot()
    assert all(after[k] == v for k, v in before.items())
    frame.loc[0, 'evidence'] = 'changed'
    with pytest.raises(FileExistsError):
        save_movement_enrichment(export.root, stem, frame)


def test_enrichment_checksum_and_version_binding(export):
    stem = 'Premier_League_24_25'
    frame = derive_movements([dict(stem=stem, league='A', year=2024,
        competition_id=17, season_id=61627, teams={str(2**63+5): 'Big ID'})], COMPETITIONS)
    save_movement_enrichment(export.root, stem, frame)
    assert int(load_season(export.root, stem, tables='team_seasons')['team_seasons'].team_id.iloc[0]) == 2**63+5
    path = next((export.root / '_team_seasons').rglob('*.parquet'))
    path.write_bytes(path.read_bytes() + b'x')
    with pytest.raises(ValueError, match='fingerprint'):
        load_season(export.root, stem, tables='team_seasons')
    export.change_manifest(lambda m: m['tables']['matches'].update(sha256='new'))
    with pytest.raises(KeyError, match='unavailable'):
        load_season(export.root, stem, tables='team_seasons')


def test_ui_exposes_optional_movement_table():
    from xdiyo_analytics.ui.inventory import inventory
    fields = {f['name']: f for f in inventory()['stages']['data']['fields']}
    assert 'team_seasons' in fields['tables']['choices']
    assert 'team_seasons' not in fields['tables']['default']
