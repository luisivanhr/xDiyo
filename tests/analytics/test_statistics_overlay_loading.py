import hashlib
import json

import pandas as pd
import pyarrow as pa
import pytest
from xdiyo_analytics.data import load_season


def test_missing_statistics_is_only_allowed_explicitly(export):
    export.change_manifest(lambda m: m['tables'].pop('statistics'))
    stem = 'Premier_League_24_25'
    with pytest.raises(KeyError):
        load_season(export.root, stem, tables=['matches', 'statistics'])
    data = load_season(export.root, stem, tables=['matches', 'statistics'], allow_missing_statistics=True)
    assert data.statistics.empty
    assert {'event_id', 'key', 'period', 'value'} <= set(data.statistics.columns)
    assert data.provenance['missing_statistics_table'] is True
    without = load_season(export.root, stem, tables=['matches'], allow_missing_statistics=True)
    assert 'statistics' not in without.tables and not without.provenance['missing_statistics_table']

def test_recipe_backfills_before_history_labels_and_lag(export, tmp_path):
    from xdiyo_analytics.ui import default_recipe, prepare_recipe
    from xdiyo_analytics.ui.recipe import node

    export.replace('matches', pa.table(dict(
        competition_id=[17, 17], season_id=[61627, 61627], event_id=[11, 12],
        home_id=[1, 3], away_id=[2, 1], home_name=['One', 'Three'], away_name=['Two', 'One'],
        kickoff_utc=[1735752600, 1736017200], status=['finished'] * 2,
        status_code=[100] * 2, is_awarded=[False] * 2, round=[1, 2])))
    export.change_manifest(lambda manifest: manifest['tables'].pop('statistics'))
    original = export.snapshot()
    row = dict(competition_id=17, season_id=61627, event_id=11, team_id=1,
               key='shotsOffGoal', period='ALL', value=0, source='Test',
               evidence_at='2026-11-27T00:00:00Z', source_ref='test:report',
               identity_evidence={'shot_selection': {'mode': 'highest_reported_team_shots'}})
    partition = tmp_path / 'cells.jsonl'
    partition.write_text(json.dumps(row))
    manifest = tmp_path / 'overlay.json'
    manifest.write_text(json.dumps(dict(version='test', partitions=[dict(
        keys=[row['key']], path=partition.name, sha256=hashlib.sha256(partition.read_bytes()).hexdigest())])))
    overlay = dict(manifest_path=str(manifest), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   keys=[row['key']])
    stat = node('features.Stat', period='ALL', group='Shots', key=row['key'])
    recipe = default_recipe(export.root)
    recipe.update(data=dict(data_root=str(export.root), leagues=['Premier_League'], seasons=['24_25'], tables=['matches']),
                  statistics_overlay=overlay, features={'prior': node('features.Lag', source=stat, periods=1)},
                  labels={'shots': node('labels.TeamValue', source=stat)}, target='shots',
                  assembly=dict(layout='team_match', drop_missing_targets=False))
    loaded, ds, _ = prepare_recipe(recipe, prediction=True)
    i, j = ds.metadata.index[ds.metadata.team_id.eq(1)]
    assert ds.y.iloc[i, 0] == 0 and pd.isna(ds.X.at[i, 'prior'])
    assert pd.isna(ds.y.iloc[j, 0]) and ds.X.at[j, 'prior'] == 0
    counts = ds.definitions['statistics_overlay']['by_key'][row['key']]
    assert counts['by_source'] == {'Test': 1}
    assert counts['shot_selection_modes'] == {'highest_reported_team_shots': 1}
    assert json.loads(loaded.statistics.iloc[0].source_item_json) == row
    assert export.snapshot() == original
    with pytest.raises(ValueError, match='selection differs'):
        prepare_recipe(dict(recipe, statistics_overlay=None), prediction=True, preparation_state=ds.definitions)
