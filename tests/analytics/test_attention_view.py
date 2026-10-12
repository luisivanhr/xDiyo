"""Attention display does not change inference, weights or population identity."""
from types import SimpleNamespace
import pandas as pd

from xdiyo_analytics.reporting.attention_view import attention_figure, attention_roster
from xdiyo_analytics.reporting import TeamCatalog


def test_top_edges_keep_strength_selection_but_all_heads_share_chronological_axis():
    table = pd.DataFrame([
        dict(team_id=1, opponent_id=2, layer=0, head=head, event_id=i,
             kickoff_at=pd.Timestamp(f'2020-01-0{i}', tz='UTC'), attention=weight)
        for head, weights in enumerate([[.4, .1, .5], [.1, .6, .3]])
        for i, weight in enumerate(weights, 1)
    ])
    before = table.copy(deep=True)
    catalog = TeamCatalog({1:'One', 2:'Two'})
    fig = attention_figure(table, [dict(team='1', name='One', league='10', league_name='Premier', badge=None)], catalog, 2, 0)
    assert list(fig.data[0].y) == [.5, .4]
    assert list(fig.data[1].y) == [.3, .6]
    assert [s.split(' · ')[-1] for s in fig.layout.xaxis.categoryarray] == ['3', '2', '1']
    pd.testing.assert_frame_equal(table, before)
    assert not fig.layout.updatemenus


def test_roster_includes_every_evaluation_team_without_edges_and_uses_current_membership():
    time = pd.Timestamp('2020-01-01', tz='UTC')
    members = pd.DataFrame(dict(team_id=[1,2,3,1], competition_id=[10,20,10,99],
                                entry_at=[time]*3+[time+pd.Timedelta(days=1)]))
    queries = pd.DataFrame(dict(team_id=[1,2,3], issuance_at=time))
    model = SimpleNamespace(event_store=SimpleNamespace(team_seasons=members))
    metadata = pd.DataFrame(dict(competition_id=[10,20], source_league=['Alpha','Beta']))
    roster = attention_roster(model, queries, metadata, TeamCatalog({1:'One',2:'Two',3:'Cold'}), False, [])
    assert {r['team']:r['league_name'] for r in roster} == {'1':'Alpha','2':'Beta','3':'Alpha'}
    fig = attention_figure(pd.DataFrame(), roster, TeamCatalog({}), 20, 0)
    assert len(fig.layout.meta['roster']) == 3 and not fig.data
