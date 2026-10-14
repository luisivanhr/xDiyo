"""Display-only league labels; never rewrite a model's causal membership."""
import pandas as pd

from .teams import team_key


def embedding_leagues(table, metadata):
    """Label recorded memberships, resolving cold teams from evaluation fixtures.

    ``league`` remains the model's as-of-issuance value. A cold team's unique
    competition in the selected evaluation fixtures supplies display context,
    not a new model prior or an earlier effective membership date.
    """
    names = {}
    if 'competition_id' in metadata:
        for column in ('source_league', 'league_name'):
            if column not in metadata:
                continue
            for key, name in metadata[['competition_id', column]].dropna().itertuples(index=False, name=None):
                if str(name).strip():
                    names.setdefault(team_key(key), str(name).strip().replace('_', ' '))
    fixtures = {}
    for side in ('home', 'away'):
        if {f'{side}_id', 'competition_id'} <= set(metadata):
            for team, league in metadata[[f'{side}_id', 'competition_id']].dropna().itertuples(index=False, name=None):
                fixtures.setdefault(team_key(team), set()).add(team_key(league))
    result = table.copy()
    leagues, labels, sources = [], [], []
    for row in table.itertuples():
        if pd.notna(row.league):
            league, source = team_key(row.league), 'model_membership'
        else:
            candidates = fixtures.get(team_key(row.team_id), set())
            if len(candidates) != 1:
                raise ValueError(f'Team {team_key(row.team_id)}: cannot resolve display league from '
                                 'evaluation fixtures; expected one recorded competition for a cold team.')
            league, source = next(iter(candidates)), 'evaluation_fixture'
        leagues.append(league)
        labels.append(names.get(league, f'League {league}'))
        sources.append(source)
    result['display_league'] = leagues
    result['league_name'] = labels
    result['league_label_source'] = sources
    return result
