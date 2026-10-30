"""Fixture selectors over frozen attention at native issuance times."""
import pandas as pd

from .attention_view import attention_figure
from .teams import team_key


def fixture_population(model, metadata, issuance_at=None):
    from ..graphs.config import MATCH_KEYS
    fixtures = metadata[[*MATCH_KEYS, 'home_id', 'away_id', 'kickoff_at']].copy()
    fixtures['issuance_at'] = model.event_store.issue_times(metadata, model.config).to_numpy()
    fixtures = fixtures.drop_duplicates().reset_index(drop=True)
    if fixtures.duplicated(list(MATCH_KEYS)).any():
        raise ValueError('Conflicting fixture identities in attention reporting population.')
    if issuance_at is not None:
        fixtures = fixtures[fixtures.issuance_at.eq(pd.Timestamp(issuance_at))].reset_index(drop=True)
    fixtures = fixtures.sort_values(['kickoff_at', *MATCH_KEYS], kind='stable').reset_index(drop=True)
    queries = pd.concat([
        fixtures[[side+'_id', 'issuance_at']].rename(columns={side+'_id': 'team_id'})
        for side in ('home', 'away')
    ], ignore_index=True).drop_duplicates().reset_index(drop=True)
    return fixtures, queries


def fixture_attention_figure(table, fixtures, queries, catalog, show_badges, top_edges, fold, notes):
    """Store each team's top-edge view once per issuance, shared by fixtures."""
    go_figure = attention_figure(pd.DataFrame(), [], catalog, top_edges, fold)
    views, keys = {}, {}
    groups = {} if table.empty else {
        (team_key(team), pd.Timestamp(time)): group
        for (team, time), group in table.groupby(['team_id', 'issuance_at'], sort=False)
    }
    for query in queries.itertuples(index=False):
        key = (team_key(query.team_id), pd.Timestamp(query.issuance_at))
        name, badge, note = catalog.display(query.team_id, badges=show_badges)
        if note:
            notes.append(note)
        roster = [dict(team=key[0], name=name, badge=badge)]
        figure = attention_figure(groups.get(key, pd.DataFrame()), roster, catalog, top_edges, fold)
        view_id = str(len(views))
        keys[key] = view_id
        # The theme and axes are shared; do not duplicate Plotly's full template
        # for every team/issuance in a standalone report.
        views[view_id] = dict(data=figure.to_plotly_json()['data'], layout=dict(
            meta=dict(roster=roster), xaxis=dict(categoryarray=list(figure.layout.xaxis.categoryarray))))
    choices = []
    for row in fixtures.itertuples(index=False):
        choices.append(dict(
            home=team_key(row.home_id), away=team_key(row.away_id),
            home_name=catalog.display(row.home_id)[0], away_name=catalog.display(row.away_id)[0],
            kickoff=pd.Timestamp(row.kickoff_at).isoformat(), issuance=pd.Timestamp(row.issuance_at).isoformat(),
            event=team_key(row.event_id), competition=team_key(row.competition_id), season=team_key(row.season_id),
            home_view=keys[(team_key(row.home_id), pd.Timestamp(row.issuance_at))],
            away_view=keys[(team_key(row.away_id), pd.Timestamp(row.issuance_at))]))
    go_figure.update_layout(meta=dict(xdiyo_fixture_attention=True, fixtures=choices, views=views))
    return go_figure
