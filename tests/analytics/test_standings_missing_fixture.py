"""Real repaired population: availability, preserved source rows, native UI path."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from xdiyo_analytics.data import load_seasons
from xdiyo_analytics.features import DerivedStandings
from xdiyo_analytics.features.history import availability_times
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.ratings import StandingsRun, build_standings

ROOT = Path(__file__).resolve().parents[2]
EVENT = 1000000011784
BOUND = pd.Timestamp('2022-03-24T23:00:00Z')


@pytest.fixture(scope='module')
def repaired():
    data = load_seasons(ROOT/'data/xDiyo_data', ['21_22'], leagues=['Bundesliga'],
        tables=['matches', 'pregame', 'team_seasons'], include_awarded=True, verify_hashes=True)
    h = build_team_history(data)
    h['release'] = availability_times(h, h.kickoff_at + pd.Timedelta(hours=3))
    return data, h


def test_repaired_source_identity_movements_and_compression(repaired):
    data, h = repaired
    assert len(data.matches) == 306 and data.matches.event_id.is_unique
    row = data.matches.loc[data.matches.event_id.eq(EVENT)].iloc[0]
    assert row.is_awarded and pd.isna(row.provider_is_awarded)
    assert row.play_status == 'abandoned' and row.result_classification == 'abandoned_then_awarded'
    assert row.home_id == 2542 and row.away_id == 2527 and row['round'] == 27
    assert row.home_got_promoted and row.home_season_entry == 'promoted'
    assert not row.away_got_promoted and row.away_season_entry == 'retained'
    assert row.home_score_played == row.home_score_awarded == row.home_standings_goals == 0
    assert row.away_score_played == row.away_score_awarded == row.away_standings_goals == 2
    assert row.result_available_at == row.standings_result_available_at == BOUND
    assert h.loc[h.event_id.eq(EVENT), 'release'].eq(BOUND).all()
    default = load_seasons(ROOT/'data/xDiyo_data', ['21_22'], leagues=['Bundesliga'], tables=['matches'])
    assert len(default.matches) == 305 and EVENT not in set(default.matches.event_id)
    pub = json.loads((ROOT/'data/xDiyo_data/Bundesliga_21_22.manifest.json').read_text())
    manifest_path = ROOT/'data/xDiyo_data'/pub['manifest']
    manifest = json.loads(manifest_path.read_text())
    paths = [ROOT/'data/xDiyo_data'/pub['season_file'], manifest_path.parent/'matches.parquet']
    tables = [pq.ParquetFile(p) for p in paths]
    assert tables[0].read().select(tables[1].schema_arrow.names).equals(tables[1].read(), check_metadata=False)
    for table in tables:
        assert {table.metadata.row_group(i).column(j).compression
                for i in range(table.num_row_groups)
                for j in range(table.metadata.num_columns)} == {'ZSTD'}


def test_ruling_boundary_never_erases_table_or_backdates_award(repaired, tmp_path):
    data, h = repaired
    query = h.loc[h.event_id.eq(9594354)]  # Bochum's next fixture, April 2.
    def at(q, source=h):
        return build_standings(source, team_seasons=data['team_seasons'], queries=query,
            cutoffs=q, normalized_ranks=[1,3,6,30], available_at='release')
    before = at(BOUND-pd.Timedelta(1, 'ns'))
    after = at(BOUND)
    assert before.snapshots.played.gt(0).any() and after.snapshots.played.gt(0).any()
    assert before.snapshots['rank'].notna().all()
    assert EVENT not in before.publications[0]['contributing_events']
    assert after.publications[0]['contributing_events'].count(EVENT) == 1
    a = before.snapshots.query('window == 0').set_index('team_id')
    b = after.snapshots.query('window == 0').set_index('team_id')
    for team, changes in {2542: [1,0,0,1,0,0,2,-2], 2527: [1,1,0,0,3,2,0,2]}.items():
        cols = ['played','wins','draws','losses','points','gf','ga','gd']
        assert (b.loc[team,cols]-a.loc[team,cols]).tolist() == changes
    # Mutating the unavailable result cannot change pre-ruling state.
    changed = h.copy()
    changed.loc[changed.event_id.eq(EVENT)&changed.side.eq('away'), 'standings_goals_for'] = 9
    changed.loc[changed.event_id.eq(EVENT)&changed.side.eq('home'), 'standings_goals_against'] = 9
    pd.testing.assert_frame_equal(at(BOUND-pd.Timedelta(1,'ns'), changed).snapshots, before.snapshots)
    unknown = h.copy(); unknown.loc[unknown.event_id.eq(EVENT), 'release'] = pd.NaT
    assert EVENT not in at(BOUND, unknown).publications[0]['contributing_events']
    after.save(tmp_path/'run')
    restored = StandingsRun.load(tmp_path/'run')
    pd.testing.assert_frame_equal(after.snapshots, restored.snapshots)
    pd.testing.assert_frame_equal(after.features(query, DerivedStandings([3,6]), cutoffs=BOUND),
                                  restored.features(query, DerivedStandings([3,6]), cutoffs=BOUND))


def test_repaired_all_subsequent_provider_positions_and_form_oracle(repaired):
    from test_derived_standings import oracle
    data, h = repaired
    queries = h.loc[h.kickoff_at.ge('2022-04-01') & h.team_position.notna()]
    run = build_standings(h, team_seasons=data['team_seasons'], queries=queries,
        cutoffs=queries.kickoff_at, normalized_ranks=[1,3,6,30], available_at='release')
    values = run.features(queries, DerivedStandings([], True), cutoffs=queries.kickoff_at)
    assert len(queries) == 128
    np.testing.assert_array_equal(values.iloc[:,0], queries.team_position)
    for (_, q, w), table in run.snapshots.groupby(['season_id','recorded_at','window']):
        query = queries.loc[queries.kickoff_at.eq(q)].iloc[0]
        totals, ranks, gate = oracle(h, data['team_seasons'], query, q, w)
        for row in table.itertuples(index=False):
            assert row.rank == ranks[row.team_id]
            assert [getattr(row, c) for c in ('played','wins','draws','losses','points','gf','ga','gd')] == totals[row.team_id]


def test_native_recipe_includes_award_only_in_standings_history(repaired):
    from examples.derived_standings import recipe
    from xdiyo_analytics.ui.recipe import export_python, prepare_recipe
    r = recipe(ROOT/'data/xDiyo_data')
    r['data'].update(seasons=['21_22'], leagues=['Bundesliga'])
    prepared = prepare_recipe(r)
    assert prepared.dataset.X.shape == (305, 6)
    assert prepared.dataset.X.notna().all().all()
    assert EVENT not in set(prepared.dataset.metadata.event_id)
    namespace = {}
    exec(export_python(r).split('result = run_recipe')[0], namespace)
    pd.testing.assert_frame_equal(namespace['prepared'].dataset.X, prepared.dataset.X)


def test_repair_is_idempotent_and_rejects_duplicate_fixture(repaired):
    from examples.repair_bundesliga_standings_20261027 import repair, repaired_tables, EVIDENCE
    assert repair(ROOT/'data/xDiyo_data')['status'] == 'already_applied'
    pub = json.loads((ROOT/'data/xDiyo_data/Bundesliga_21_22.manifest.json').read_text())
    folder = (ROOT/'data/xDiyo_data'/pub['manifest']).parent
    matches, statistics = [pq.ParquetFile(folder/(n+'.parquet')).read() for n in ('matches','statistics')]
    nested = pq.ParquetFile(ROOT/'data/xDiyo_data'/pub['season_file']).read()
    with pytest.raises(ValueError, match='already exists'):
        repaired_tables(matches, statistics, nested, json.loads(EVIDENCE.read_text()))
