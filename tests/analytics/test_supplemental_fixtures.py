import pyarrow as pa
import pandas as pd
import pytest

from xdiyo_analytics.data.supplemental_fixtures import append_fixture_tables, supplemental_event_id


def inputs():
    fixture = dict(event_id=supplemental_event_id(72241), competition_id=36, season_id=10379,
                   home_id=2355, away_id=2376, kickoff_utc=1442342700., round=3,
                   custom_id=None, status='finished', status_code=100)
    old = {**fixture, 'event_id': 42, 'round': 32, 'kickoff_utc': 1459683000., 'custom_id': 'fXsBX'}
    matches = pa.Table.from_pylist([old])
    stats = pa.Table.from_pylist([dict(event_id=42, team_id=2355, side='home', value=0.)])
    nested = matches.append_column('statistics', pa.array([stats.to_pylist()], type=pa.list_(pa.struct(stats.schema))))
    nested = nested.append_column('incidents', pa.array([[]], type=pa.list_(pa.int64())))
    additions = [dict(event_id=fixture['event_id'], team_id=2355, side='home', value=5.)]
    return matches, stats, nested, fixture, additions


def test_append_preserves_all_old_rows_and_populates_both_surfaces():
    args = inputs()
    result = append_fixture_tables(*args)
    for old, new in zip(args[:3], result):
        assert new.slice(0, len(old)).equals(old, check_metadata=True)
    assert result[2]['statistics'][1].as_py() == args[4]
    assert result[2]['incidents'][1].as_py() == []
    assert result[0]['event_id'][1].as_py() == 1000000072241
    with pytest.raises(ValueError, match='already exists'):
        append_fixture_tables(*result, *args[3:])
    from xdiyo_analytics.data.loading import _check_matches
    _check_matches(result[0].to_pandas(types_mapper=pd.ArrowDtype), {'league_id':36, 'season_id':10379})


@pytest.mark.parametrize('change', [{'event_id':72241}, {'custom_id':'fabricated'}, {'round':32}, {'kickoff_utc':1459683000.}])
def test_rejects_native_id_masquerading_and_existing_fixture(change):
    a,b,c,fixture,stats = inputs()
    with pytest.raises(ValueError):
        append_fixture_tables(a,b,c,{**fixture, **change},stats)


def test_rejects_inconsistent_existing_surfaces_and_wrong_stat_team():
    a,b,c,f,s = inputs()
    with pytest.raises(ValueError, match='disagree'):
        append_fixture_tables(a,b,c.set_column(0, 'event_id', pa.array([99])),f,s)
    with pytest.raises(ValueError, match='identity'):
        append_fixture_tables(a,b,c,f,[{**s[0], 'team_id':1}])
