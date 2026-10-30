"""Fixture attention uses retained fold models at native issuance cutoffs."""
from types import SimpleNamespace
import json
import pandas as pd
import pytest

from test_graph_integration import trained
from test_ui_workflow import ui_recipe
from test_temporal_graph import sample, config, deep_state
from xdiyo_analytics.reporting import GraphAttentionWeightReporter
from xdiyo_analytics.reporting.fixture_attention import fixture_population
from xdiyo_analytics.ui.recipe import catalog_for_ui, node, export_python


def report_context(training, folds=(0,)):
    model = training.folds[0].model
    metadata = training.folds[0].metadata.copy()
    return SimpleNamespace(partition='score', layout='match', pooling='occurrences',
        metadata=pd.concat({fold: metadata for fold in folds}, names=['fold_id']),
        models={fold: model for fold in folds}, resources={})


@pytest.mark.parametrize('pooled,per_fixture', [(True, False), (True, True), (False, True)])
def test_panels_queries_raw_weights_and_recovery(trained, monkeypatch, pooled, per_fixture):
    _, training = trained
    ctx = report_context(training)
    model = ctx.models[0]
    before = deep_state(model.backend_.model)
    original = model.attention_weights
    calls = []
    def inspect(queries):
        calls.append(queries.copy())
        return original(queries)
    monkeypatch.setattr(model, 'attention_weights', inspect)
    result = GraphAttentionWeightReporter(type='overall', partition='score', pooled=pooled, per_fixture=per_fixture).run(ctx)
    assert len(result.artifacts) == int(pooled)+int(per_fixture)
    assert len(calls) == 1 and not calls[0].duplicated().any()
    if pooled:
        latest = model.event_store.issue_times(training.folds[0].metadata, model.config).max()
        expected = original(pd.DataFrame(dict(team_id=[1,2], issuance_at=latest))).assign(fold_id=0)
        pd.testing.assert_frame_equal(result.tables['attention'].reset_index(drop=True), expected)
    if per_fixture:
        rows = result.tables['attention_fixtures']
        assert len(rows) == 3 and rows.issuance_at.nunique() == 3
        # This population has two home=1/away=2 fixtures at different kickoffs.
        assert rows.groupby(['home_id', 'away_id']).size().max() == 2
        raw = result.tables['fixture_attention']
        for row in rows.itertuples():
            expected = original(pd.DataFrame(dict(team_id=[row.home_id,row.away_id], issuance_at=row.issuance_at)))
            actual = raw[raw.issuance_at.eq(row.issuance_at)].drop(columns='fold_id')
            sort = ['team_id','layer','head','event_id']
            pd.testing.assert_frame_equal(actual.sort_values(sort).reset_index(drop=True), expected.sort_values(sort).reset_index(drop=True))
            assert (actual.kickoff_at < actual.history_cutoff).all()
            assert row.event_id not in set(actual.event_id)
        assert result.tables['fixture_concentration'].issuance_at.nunique() == 3
        spec = result.artifacts[-1].data.layout.meta
        assert spec['xdiyo_fixture_attention'] and len(spec['fixtures']) == 3
        assert len(spec['views']) == 6
        from xdiyo_analytics.experiments.recovery import pack, unpack
        restored = unpack(pack(result))
        assert json.loads(restored.artifacts[-1].data.to_json()) == json.loads(result.artifacts[-1].data.to_json())
    else:
        assert set(result.tables) == {'attention','concentration'}
    assert deep_state(model.backend_.model) == before


def test_folds_and_explicit_issuance_remain_separate_and_cold_teams_selectable(trained, monkeypatch):
    _, training = trained
    ctx = report_context(training, folds=(0,1))
    model = ctx.models[0]
    time = model.event_store.issue_times(training.folds[0].metadata, model.config).min()
    monkeypatch.setattr(model, 'attention_weights', lambda queries: pd.DataFrame())
    result = GraphAttentionWeightReporter(type='overall', partition='score', pooled=False, per_fixture=True, issuance_at=str(time)).run(ctx)
    assert len(result.artifacts) == 2
    assert set(result.tables['attention_fixtures'].fold_id) == {0,1}
    assert result.tables['attention_fixtures'].issuance_at.eq(time).all()
    for artifact in result.artifacts:
        meta = artifact.data.layout.meta
        assert len(meta['fixtures']) == 1 and len(meta['views']) == 2
        assert all(not v['data'] for v in meta['views'].values())


@pytest.mark.parametrize('policy', ['by_round','one_day_ahead'])
def test_native_issuance_and_shared_queries(policy):
    store = sample()
    model = SimpleNamespace(event_store=store, config=config(forecast_policy=policy))
    metadata = store.schedule.iloc[[6,7,8]]
    fixtures, queries = fixture_population(model, pd.concat([metadata,metadata]))
    assert len(fixtures) == 3 and len(queries) == 6
    pd.testing.assert_series_equal(fixtures.issuance_at.reset_index(drop=True),
        store.issue_times(metadata,model.config).reset_index(drop=True), check_names=False)


def test_distinct_fixtures_in_one_round_share_receiver_queries():
    from test_temporal_graph import history
    from xdiyo_analytics.graphs import TemporalEventStore
    store = sample()
    h = history([(1,'2020-01-01',1,2,1,0,10,900,1),
                 (2,'2020-01-02',1,3,0,1,10,900,1)])
    store = TemporalEventStore(h,store.team_seasons)
    model = SimpleNamespace(event_store=store,config=config(forecast_policy='by_round'))
    fixtures,queries = fixture_population(model,store.schedule)
    assert len(fixtures) == 2 and len(queries) == 3
    assert queries.issuance_at.nunique() == 1


def test_flags_ui_json_and_python_exports(ui_recipe):
    from xdiyo_analytics.ui.inventory import component_schema
    fields = {f['name']: f for f in component_schema('reporting.GraphAttentionWeightReporter')['fields']}
    assert fields['pooled']['default'] is True
    assert fields['per_fixture']['default'] is False
    assert fields['pooled']['kind'] == fields['per_fixture']['kind'] == 'boolean'
    for pooled, per_fixture in ((True,False),(False,True),(True,True)):
        spec = node('reporting.GraphAttentionWeightReporter',type='overall',partition='score',pooled=pooled,per_fixture=per_fixture)
        reporter = catalog_for_ui().build(json.loads(json.dumps(spec)))
        assert (reporter.pooled,reporter.per_fixture) == (pooled,per_fixture)
        # Export preserves these native constructor parameters.
        text = export_python({**ui_recipe, 'post_reporters': {'attention': spec}})
        namespace = {}
        exec(text.split('prepared =')[0], namespace)
        exported = namespace['recipe']['post_reporters']['attention']
        assert exported['component'] == spec['component']
        assert all(exported['params'][key] == value for key,value in spec['params'].items())
    for kwargs in ({'pooled':False}, {'pooled':1}, {'per_fixture':'yes'}):
        with pytest.raises(ValueError):
            GraphAttentionWeightReporter(type='overall',partition='score',**kwargs).run(None)
