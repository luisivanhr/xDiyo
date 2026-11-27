import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from xdiyo_analytics.data import statistics_overlay as module
from xdiyo_analytics.data.loading import SeasonData
apply = module.backfill_corners


def cell(team=10, value=0, period='ALL', evidence_at='2020-01-01T00:00:00Z'):
    return dict(competition_id=1, season_id=2, event_id=3, team_id=team, period=period,
                value=value, evidence_at=evidence_at, source_ref='source:row', key='cornerKicks')


def dataset(value=None, display=None):
    matches = pd.DataFrame([dict(competition_id=1, season_id=2, event_id=3, home_id=10, away_id=11,
                                 status='finished', status_code=100, is_awarded=False, kickoff_utc=1609459200)])
    stats = pd.DataFrame([dict(event_id=3, team_id=10, period='ALL', key='cornerKicks', side='home',
                               group_name='Match overview', source_key='sofascore|v1|en|/event/3/statistics', value=value, display=display)])
    return SeasonData({'matches': matches, 'statistics': stats}, {})


def config(tmp_path, rows):
    path = tmp_path / 'overlay.jsonl'
    raw = b''.join((json.dumps(row) + '\n').encode() for row in rows)
    path.write_bytes(raw)
    manifest = tmp_path / 'release.json'
    manifest.write_text(json.dumps(dict(version='v1', partitions=[dict(path=path.name, sha256=hashlib.sha256(raw).hexdigest(), keys=['cornerKicks'])])))
    return dict(manifest_path=str(manifest), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())



def test_missing_only_and_zeros(tmp_path):
    original = dataset()
    result, receipt = apply(original, config(tmp_path, [cell(), cell(11, 4)]), {})
    assert dict(zip(result.statistics.team_id, result.statistics.value)) == {10: 0, 11: 4}
    assert pd.isna(original.statistics.value.iloc[0])
    assert receipt['by_key']['cornerKicks']['filled'] == 2
    result, receipt = apply(dataset(0, '0'), config(tmp_path, [cell(value=9)]), {})
    assert result.statistics.value.iloc[0] == 0 and receipt['by_key']['cornerKicks']['protected_existing'] == 1
    result, receipt = apply(dataset(None, '7'), config(tmp_path, [cell(value=9)]), {})
    assert pd.isna(result.statistics.value.iloc[0]) and receipt['by_key']['cornerKicks']['protected_existing'] == 1


@pytest.mark.parametrize('capture', [None, '2040-01-01T00:00:00Z', 'undated'])
def test_capture_time_is_metadata_only(tmp_path, capture):
    row = cell(value=7)
    if capture is None:
        del row['evidence_at']
    else:
        row['evidence_at'] = capture
    options = config(tmp_path, [row])
    filled, receipt = apply(dataset(), options)
    actual = filled.statistics.iloc[0]
    assert actual.value == 7 and json.loads(actual.source_item_json) == row
    assert ('mode' in receipt, 'as_of' in receipt) == (False, False)
    if capture and capture.startswith('2040'):
        assert actual.observed_at == pd.Timestamp(capture).timestamp()
    else:
        assert pd.isna(actual.observed_at)


@pytest.mark.parametrize('rows', [[cell(), cell()], [cell(period='HT')], [cell(value=-1)], [cell(value=1.5)],
                                  [cell(team=99)]])
def test_invalid_overlay_rejected(tmp_path, rows):
    with pytest.raises(ValueError):
        apply(dataset(), config(tmp_path, rows), {})


def test_hash_pin(tmp_path):
    c = config(tmp_path, [cell()])
    c['manifest_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash mismatch'):
        apply(dataset(), c, {})

def test_real_history_entrypoint(tmp_path):
    from xdiyo_analytics.histories import build_team_history
    result, _ = apply(dataset(), config(tmp_path, [cell(value=3), cell(11, 4)]), {})
    history = build_team_history(result)
    assert dict(zip(history.team_id, history['team::ALL::Match overview::cornerKicks::value'])) == {10: 3, 11: 4}




def test_external_populated_values_preserved(tmp_path):
    data = dataset(7, '7')
    data.statistics.loc[0, 'source_key'] = 'imported_statistics|old'
    result, receipt = apply(data, config(tmp_path, [cell(value=5)]), {})
    assert result.statistics.value.iloc[0] == 7 and receipt['by_key']['cornerKicks']['protected_existing'] == 1
    data.statistics.loc[0, 'source_key'] = 'unknown|old'
    result, _ = apply(data, config(tmp_path, [cell(value=5)]), {})
    assert result.statistics.value.iloc[0] == 7


def test_repeat_is_idempotent(tmp_path):
    cfg = config(tmp_path, [cell(value=5)])
    once, receipt = apply(dataset(), cfg, {})
    twice, second = apply(once, cfg, {})
    pd.testing.assert_frame_equal(once.statistics, twice.statistics)
    assert receipt == second and once.provenance == twice.provenance


def multi_config(tmp_path, rows):
    raw = b''.join((json.dumps(row) + '\n').encode() for row in rows)
    path = tmp_path / 'combined.jsonl'
    path.write_bytes(raw)
    manifest = tmp_path / 'combined-release.json'
    keys = list(module.STAT_KEYS)
    manifest.write_text(json.dumps(dict(version='v2', partitions=[dict(path=path.name, sha256=hashlib.sha256(raw).hexdigest(), keys=keys)])))
    return dict(manifest_path=str(manifest), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())


def shot(team, key, value):
    row = cell(team=team, value=value)
    row['key'] = key
    return row


def test_helpers_compose_both_orders_and_preserve_native(tmp_path):
    rows = [cell(value=0), shot(10, 'totalShotsOnGoal', 12), shot(11, 'totalShotsOnGoal', 9),
            shot(10, 'shotsOnGoal', 0), shot(11, 'shotsOffGoal', 7)]
    cfg = multi_config(tmp_path, rows)
    original = dataset(value=0, display='0')
    existing = dict(event_id=3, team_id=10, period='ALL', key='totalShotsOnGoal', side='home',
                    group_name='Match overview', source_key='sofascore|v1|en|/event/3/statistics', value=8, display='8')
    original.tables['statistics'] = pd.concat([original.statistics, pd.DataFrame([existing])], ignore_index=True)
    a, _ = module.backfill_corners(original, cfg, {})
    a, receipt = module.backfill_shots(a, cfg, {})
    b, _ = module.backfill_shots(original, cfg, {})
    b, _ = module.backfill_corners(b, cfg, {})
    pd.testing.assert_frame_equal(a.statistics, b.statistics)
    assert a.provenance['statistics_overlay'] == b.provenance['statistics_overlay']
    assert sorted(receipt['applied_keys']) == sorted({'cornerKicks'} | module.SHOT_KEYS)
    assert a.statistics.loc[a.statistics.key.eq('totalShotsOnGoal') & a.statistics.team_id.eq(10), 'value'].iloc[0] == 8
    assert a.statistics.loc[a.statistics.key.eq('shotsOnGoal'), 'value'].iloc[0] == 0
    twice, same = module.backfill_shots(a, cfg, {})
    assert same == receipt
    pd.testing.assert_frame_equal(twice.statistics, a.statistics)
    bad = dict(cfg, mode='strict')
    with pytest.raises(ValueError, match='time filter'):
        module.backfill_shots(a, bad, {})


def test_subset_mask_does_not_delete_other_statistics(tmp_path):
    data = dataset(3, '3')
    data.statistics.loc[0, 'source_key'] = 'imported_statistics|old'
    imported_shot = dict(event_id=3, team_id=10, period='ALL', key='totalShotsOnGoal', side='home',
                    group_name='Match overview', source_key='imported_statistics|old', value=10, display='10')
    data.tables['statistics'] = pd.concat([data.statistics, pd.DataFrame([imported_shot])], ignore_index=True)
    cfg = multi_config(tmp_path, [cell(value=4), shot(10, 'totalShotsOnGoal', 11)])
    corners, _ = module.backfill_corners(data, cfg, {})
    assert corners.statistics.loc[corners.statistics.key.eq('totalShotsOnGoal'), 'value'].iloc[0] == 10
    both, _ = module.backfill_shots(corners, cfg, {})
    assert both.statistics.loc[both.statistics.key.eq('totalShotsOnGoal'), 'value'].iloc[0] == 10


def test_missing_statistics_table_rejected(tmp_path):
    data = dataset()
    del data.tables['statistics']
    with pytest.raises(ValueError, match='requires loaded matches and statistics'):
        module.backfill_shots(data, multi_config(tmp_path, []), {})


def possession(team, value, unit='percentage_points'):
    row = shot(team, 'ballPossession', value)
    row['unit'] = unit
    return row


def test_three_helpers_compose_all_orders(tmp_path):
    from itertools import permutations
    rows = [cell(value=0), shot(11, 'shotsOnGoal', 2), possession(10, 0), possession(11, 61.5)]
    cfg = multi_config(tmp_path, rows)
    original = dataset()
    result_views = []
    helpers = [module.backfill_corners, module.backfill_shots, module.backfill_possession]
    for order in permutations(helpers):
        data = original
        for helper in order:
            data, receipt = helper(data, cfg, {})
        columns = ['team_id', 'period', 'key', 'value', 'display']
        result_views.append(data.statistics)
        data_again, receipt_again = order[-1](data, cfg, {})
        assert receipt_again == receipt
        pd.testing.assert_frame_equal(data_again.statistics, data.statistics)
    for view in result_views[1:]:
        pd.testing.assert_frame_equal(result_views[0], view)
    p = result_views[0].loc[result_views[0].key.eq('ballPossession')].sort_values('team_id')
    assert p.value.tolist() == [0, 61.5] and p.display.tolist() == ['0%', '61.5%']


@pytest.mark.parametrize('row', [possession(10, -0.1), possession(10, 100.1), possession(10, True),
                                 possession(10, float('nan')), possession(10, 50, 'fraction')])
def test_invalid_possession_values_rejected(tmp_path, row):
    with pytest.raises(ValueError, match='percentage_points'):
        module.backfill_possession(dataset(), multi_config(tmp_path, [row]), {})


def test_possession_native_zero_and_one_sided(tmp_path):
    data = dataset()
    native = dict(event_id=3, team_id=10, period='ALL', key='ballPossession', side='home',
                  group_name='Match overview', source_key='sofascore|v1|en|/event/3/statistics', value=0, display='0%')
    data.tables['statistics'] = pd.concat([data.statistics, pd.DataFrame([native])], ignore_index=True)
    cfg = multi_config(tmp_path, [possession(10, 30), possession(11, 55.5)])
    result, _ = module.backfill_possession(data, cfg, {})
    p = result.statistics.loc[result.statistics.key.eq('ballPossession')].sort_values('team_id')
    assert p.value.tolist() == [0, 55.5]
    assert p.display.tolist() == ['0%', '55.5%']


def expected_goals(team, value, source='Footiqo', provider='unknown', model='unknown', unit='expected_goals'):
    row = shot(team, 'expectedGoals', value)
    row.update(source=source, provider=provider, expected_goals_model=model, unit=unit)
    return row


def test_expected_goals_native_and_provenance(tmp_path):
    data = dataset()
    existing = dict(event_id=3, team_id=10, period='ALL', key='expectedGoals', side='home',
                    group_name='Shots', source_key='sofascore|v1|en|/event/3/statistics', value=0, display='0')
    data.tables['statistics'] = pd.concat([data.statistics, pd.DataFrame([existing])], ignore_index=True)
    cfg = multi_config(tmp_path, [expected_goals(10, 2.4), expected_goals(11, 1.27)])
    result, receipt = module.backfill_expected_goals(data, cfg, {})
    vals = result.statistics.loc[result.statistics.key.eq('expectedGoals')].sort_values('team_id')
    assert vals.value.tolist() == [0, 1.27]
    inserted = json.loads(vals.iloc[1].source_item_json)
    assert inserted['provider'] == 'unknown' and inserted['expected_goals_model'] == 'unknown'
    assert receipt['by_key']['expectedGoals']['protected_existing'] == 1


@pytest.mark.parametrize('row', [expected_goals(10, -1), expected_goals(10, float('inf')),
                                 expected_goals(10, True), expected_goals(10, 1, source='Flashscore'),
                                 expected_goals(10, 1, provider=''), expected_goals(10, 1, model=''),
                                 expected_goals(10, 1, unit='goals')])
def test_expected_goals_bad_source_or_values(tmp_path, row):
    with pytest.raises(ValueError, match='Expected goals'):
        module.backfill_expected_goals(dataset(), multi_config(tmp_path, [row]), {})


def test_empty_statistics_schema_supported(tmp_path):
    data = dataset()
    data.tables['statistics'] = pd.DataFrame()
    result, _ = module.backfill_expected_goals(data, multi_config(tmp_path, [expected_goals(10, 0.52)]), {})
    assert result.statistics.value.iloc[0] == 0.52
    assert result.statistics.source_key.iloc[0].startswith('statistics_overlay|')


def test_possession_display_preserves_precision(tmp_path):
    result, _ = module.backfill_possession(dataset(), multi_config(tmp_path, [possession(10, 12.3456789012345)]), {})
    row = result.statistics.loc[result.statistics.key.eq('ballPossession')].iloc[0]
    assert row.display == '12.3456789012345%'


def test_four_family_orders_and_receipts_identical(tmp_path):
    from itertools import permutations
    cfg = multi_config(tmp_path, [cell(value=0), shot(11, 'shotsOnGoal', 2), possession(10, 25.125), expected_goals(11, 1.235)])
    views = []
    receipts = []
    for order in permutations([module.backfill_corners, module.backfill_shots, module.backfill_possession, module.backfill_expected_goals]):
        data = dataset()
        for helper in order:
            data, receipt = helper(data, cfg, {})
        views.append(data.statistics)
        receipts.append(receipt)
    for view, receipt in zip(views[1:], receipts[1:]):
        pd.testing.assert_frame_equal(views[0], view)
        assert receipt == receipts[0]


def test_xg_alias(tmp_path):
    cfg = multi_config(tmp_path, [expected_goals(10, 0.2)])
    data = dataset()
    a, receipt_a = module.backfill_xg(data, cfg, {})
    b, receipt_b = module.backfill_expected_goals(data, cfg, {})
    pd.testing.assert_frame_equal(a.statistics, b.statistics)
    assert receipt_a == receipt_b


def test_legacy_zero_one_sided_gap_and_unrelated_external_survive(tmp_path):
    data = dataset(0, '0')
    data.statistics.loc[0, 'source_key'] = 'imported_statistics|old'
    gap = dict(event_id=3, team_id=11, period='ALL', key='cornerKicks', side='away',
               group_name='Match overview', source_key='sofascore|v1|en|/event/3/statistics', value=None, display=None)
    other = dict(event_id=3, team_id=10, period='1ST', key='ballPossession', side='home',
                 group_name='Match overview', source_key='unknown|old', value=62, display='62%')
    data.tables['statistics'] = pd.concat([data.statistics, pd.DataFrame([gap, other])], ignore_index=True)
    cfg = multi_config(tmp_path, [cell(value=9), cell(11, 4)])
    result, receipt = module.backfill_corners(data, cfg, {})
    own = result.statistics.loc[result.statistics.key.eq('cornerKicks')].set_index('team_id')
    assert own.loc[10, 'value'] == 0 and own.loc[10, 'source_key'] == 'imported_statistics|old'
    assert own.loc[11, 'value'] == 4 and own.loc[11, 'source_key'].startswith('statistics_overlay|')
    assert result.statistics.loc[result.statistics.key.eq('ballPossession'), 'value'].iloc[0] == 62
    assert receipt['by_key']['cornerKicks']['protected_existing'] == 1
    again, _ = module.backfill_corners(result, cfg, {})
    pd.testing.assert_frame_equal(result.statistics, again.statistics)


def test_legacy_missing_only_selection_and_fitted_receipt(tmp_path):
    from xdiyo_analytics.ui.recipe import _verify_overlay_selection, default_recipe, validate_recipe

    opts = config(tmp_path, [cell()])
    plain = dict(opts, keys=['cornerKicks'])
    old = dict(plain, mode='retrospective', as_of=None)
    recipe = dict(default_recipe('.'), statistics_overlay=old)
    assert validate_recipe(recipe)['statistics_overlay'] == plain
    _, receipt = apply(dataset(), dict(opts, mode='retrospective', as_of=None))
    _verify_overlay_selection(dict(statistics_overlay=plain),
                              dict(statistics_overlay=dict(receipt, mode='retrospective', as_of=None)))
    with pytest.raises(ValueError, match='selection differs'):
        _verify_overlay_selection(dict(statistics_overlay=plain), dict(statistics_overlay=dict(receipt, mode='strict')))
    with pytest.raises(ValueError, match='rebuild'):
        validate_recipe(dict(recipe, statistics_overlay=dict(old, mode='strict', as_of='2020-01-01T00:00:00Z')))


def test_combined_partitions_check_exact_cells(tmp_path):
    opts = config(tmp_path, [cell()])
    manifest = Path(opts['manifest_path'])
    parts = json.loads(manifest.read_bytes())
    other = tmp_path / 'second.jsonl'
    parts['partitions'].append(dict(parts['partitions'][0], path=other.name))
    for row, error in [(cell(11, 7), None), (cell(), 'Duplicate overlay cell'),
                       (dict(cell(11), side='home'), 'home/away identity')]:
        other.write_text(json.dumps(row))
        parts['partitions'][1]['sha256'] = hashlib.sha256(other.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(parts))
        opts['manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if error:
            with pytest.raises(ValueError, match=error):
                apply(dataset(), opts, {})
        else:
            result, _ = apply(dataset(), opts, {})
            assert result.statistics.set_index('team_id').value.to_dict() == {10: 0, 11: 7}

def test_compressed_partition_checked_before_use(tmp_path):
    options = config(tmp_path, [cell(value=4)])
    manifest_path = Path(options['manifest_path'])
    manifest = json.loads(manifest_path.read_bytes())
    part, = manifest['partitions']
    raw = (tmp_path / part['path']).read_bytes()
    compressed = tmp_path / 'corners.jsonl.gz'
    compressed.write_bytes(gzip.compress(raw, mtime=0))
    part.update(path=compressed.name, sha256=hashlib.sha256(compressed.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest))
    options['manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    result, _ = apply(dataset(), options)
    assert result.statistics.value.tolist() == [4]
    compressed.write_bytes(b'invalid')
    with pytest.raises(ValueError, match='partition hash mismatch'):
        apply(dataset(), options)
