"""Basic statistics safety through load_season; no standalone diagnostic API."""

import pandas as pd
import pyarrow as pa
import pytest

from xdiyo_analytics.data import load_season

STEM = 'Premier_League_24_25'


def test_statistics_alone_preserve_group_separation_missingness_and_text(export):
    before = export.snapshot()
    season = load_season(export.root, STEM, tables='statistics')
    assert season.matches is None
    assert season.statistics['group_name'].tolist()[:2] == ['Match overview', 'Shots']
    assert len(season.statistics) == 4
    assert pd.isna(season.statistics['value'].iloc[3])
    assert season.statistics['display'].iloc[3] == 'not supplied'
    assert export.snapshot() == before


def test_only_full_grain_duplicates_are_rejected_and_create_no_record(export, tmp_path):
    table = export.table('statistics')
    export.replace('statistics', pa.concat_tables([table, table.slice(0, 1)]))
    record = tmp_path / 'unsaved/selection.json'
    with pytest.raises(ValueError, match='duplicate match/period/group/key/side'):
        load_season(export.root, STEM, tables=['matches', 'statistics'], record_path=record)
    assert not record.parent.exists()


@pytest.mark.parametrize('column,values,message', [
    ('event_id', [2**53 + 1, 2**53 + 1, 2**53 + 1, 999], 'outside the loaded season'),
    ('team_id', [42, 2**53 + 3, 2**53 + 5, None], 'does not match'),
])
def test_cross_table_checks_run_when_matches_are_selected(export, column, values, message):
    table = export.table('statistics')
    export.replace('statistics', table.set_column(table.schema.get_field_index(column), column, pa.array(values)))
    assert len(load_season(export.root, STEM, tables='statistics').statistics) == 4
    with pytest.raises(ValueError, match=message):
        load_season(export.root, STEM, tables=['statistics', 'matches'])


@pytest.mark.parametrize('column,values', [
    ('side', ['HOME', 'home', 'away', 'home']),
    ('group_name', ['', 'Shots', 'Shots', 'Attack']),
    ('event_id', [1.0, 1.0, 1.0, 7.0]),
])
def test_malformed_basic_statistics_are_rejected(export, column, values):
    table = export.table('statistics')
    export.replace('statistics', table.set_column(table.schema.get_field_index(column), column, pa.array(values)))
    with pytest.raises(ValueError):
        load_season(export.root, STEM, tables='statistics')
