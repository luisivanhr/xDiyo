"""Narrow, evidence-backed standings corrections; sporting labels stay intact."""
from copy import deepcopy
import json

import pandas as pd


BASTIA_RED_STAR_REVIEW = dict(
    review_version=1, reviewed=True,
    competition_id=182, season_id=77357, event_id=14064792,
    home_id=1655, away_id=52874, round=16, kickoff_at='2025-12-05T19:00:00Z',
    standings_goals_home=0, standings_goals_away=0,
    standings_points_home=0, standings_points_away=3,
    source='https://www.lfp.fr/article/commission-de-discipline-les-decisions-du-22-decembre-2025',
    rules='https://www.lfp.fr/assets/25_26_Reglement_Competitions_29_07_2024_5055b664e4.pdf',
    rule='2025/26 article 518 bis paragraph 3: penalty loss, winning points, actual goals retained',
    sporting_status='abandoned at 56 minutes, 0-0',
    sporting_evidence='https://www.redstar.fr/live/?ID=1237093',
    decision_date='2025-12-22',
    publication_time=None,
    label_policy='Standings only; never convert this administrative outcome into an observed full-time goal target or bookmaker settlement.'
)


def apply_known_standings_reviews(history):
    """Return owned history with verified catalog reviews, preserving raw goals.

    The catalog is deliberately narrow. Unknown awarded events still require
    explicit reviewed evidence; no score or award is guessed. The original
    kickoff, round, provider goals and result labels remain untouched. An exact
    identity/date mismatch or a conflicting approved score/points pair raises.
    This supplies no publication timestamp. Timestamp-based callers must still
    provide a verified release; daily retrospective callers need none.
    """
    keys = ('competition_id', 'season_id', 'event_id')
    if not set(keys) <= set(history):
        return history
    review = deepcopy(BASTIA_RED_STAR_REVIEW)
    mask = pd.Series(True, index=history.index)
    for key in keys:
        mask &= history[key].eq(review[key]).fillna(False)
    if not mask.any():
        return history
    pair = history.loc[mask]
    if len(pair) != 2 or set(pair.side) != {'home', 'away'}:
        raise ValueError('Known standings review needs exactly two paired fixture perspectives.')
    result = history.copy(deep=True)
    for side, own, other in [('home', review['home_id'], review['away_id']),
                             ('away', review['away_id'], review['home_id'])]:
        row = pair.loc[pair.side.eq(side)].iloc[0]
        if (row.team_id != own or row.opponent_id != other or row['round'] != review['round']
                or pd.to_datetime(row.kickoff_at, utc=True) != pd.Timestamp(review['kickoff_at'])):
            raise ValueError('Known standings review disagrees with fixture identity, original round or actual kickoff.')
        selected = mask & history.side.eq(side)
        for field in ('goals', 'points'):
            for role, perspective in [('for', side), ('against', 'away' if side == 'home' else 'home')]:
                column = f'standings_{field}_{role}'
                approved = review[f'standings_{field}_{perspective}']
                if column in history and (history.loc[selected, column].dropna() != approved).any():
                    raise ValueError(f'Conflicting approved {column} for known reviewed fixture {review["event_id"]}.')
                result.loc[selected, column] = approved
    if 'result_review_json' in result:
        previous = pair.result_review_json.dropna().unique()
        if len(previous) > 1:
            raise ValueError('Known standings review has inconsistent paired review metadata.')
        if len(previous):
            try:
                prior = json.loads(previous[0])
            except (TypeError, ValueError):
                prior = None
            if isinstance(prior, dict) and all(prior.get(k) == review[k] for k in ('review_version', 'event_id', 'source')):
                if 'previous_review_metadata' in prior:
                    review['previous_review_metadata'] = prior['previous_review_metadata']
            else:
                review['previous_review_metadata'] = [str(previous[0])]
    result.loc[mask, 'result_review_json'] = json.dumps(review, sort_keys=True)
    return result
