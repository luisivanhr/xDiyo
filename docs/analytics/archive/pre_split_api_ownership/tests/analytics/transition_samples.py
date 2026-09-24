"""Small paired match histories with explicit seasons, rounds and observations."""

import pandas as pd

from xdiyo_analytics.features import Stat

A, B, C, D, E = [2**63 + i for i in (11, 19, 27, 35, 43)]
START = pd.Timestamp('2024-01-01 12:00', tz='UTC')
STAT = Stat('ALL', 'Match overview', 'cornerKicks')
OWN = 'team::ALL::Match overview::cornerKicks::value'
OTHER = 'opponent::ALL::Match overview::cornerKicks::value'


def history_from_games(specifications):
    rows, metadata = [], {}
    for event, spec in enumerate(specifications):
        kickoff = START + pd.Timedelta(days=spec.get('day', event * 2))
        home, away = spec.get('home', A), spec.get('away', B)
        result = spec.get('result', 'W')
        for side, team, opponent in [('home', home, away), ('away', away, home)]:
            row = dict(event_id=2**53 + event, team_id=team, opponent_id=opponent,
                       competition_id=spec.get('competition', 10), season_id=spec.get('season', 1),
                       source_season=spec.get('label', '23_24' if spec.get('season', 1) == 1 else '24_25'),
                       kickoff_at=kickoff, available_at=START + pd.Timedelta(days=spec.get('release', spec.get('day', event * 2))),
                       side=side, status=spec.get('status', 'finished'), round=spec.get('round', event + 1),
                       stage=spec.get('stage', 'regular'),
                       result=result if side == 'home' else {'W': 'L', 'L': 'W', 'D': 'D'}[result],
                       team_position=spec.get('positions', (1, 2))[0 if side == 'home' else 1])
            for period, pair in [('ALL', spec.get('values', (2., 6.))), ('1ST', spec.get('half', (1., 3.)))]:
                for role in ('team', 'opponent'):
                    column = f'{role}::{period}::Match overview::cornerKicks::value'
                    row[column] = pair[0 if (side == 'home') == (role == 'team') else 1]
                    metadata[column] = dict(role=role, period=period, group_name='Match overview', key='cornerKicks', field='value')
            rows.append(row)
    history = pd.DataFrame(rows)
    for key in ('team_id', 'opponent_id', 'competition_id', 'season_id', 'event_id'):
        history[key] = pd.array(history[key], dtype='uint64[pyarrow]')
    for column in metadata:
        history[column] = pd.array(history[column], dtype='float64[pyarrow]')
    history.attrs = {'stat_columns': metadata, 'source': {'version': 'independent_synthetic'}}
    return history


def league_games():
    return [dict(day=0, round=1, home=A, away=B, values=(2, 8)),
            dict(day=1, round=1, home=C, away=D, values=(4, 6)),
            dict(day=3, round=2, home=A, away=C, values=(10, 20)),
            dict(day=4, round=2, home=B, away=D, values=(30, 40)),
            dict(day=6, round=3, home=A, away=D, values=(50, 60)),
            dict(day=8, round=3, home=B, away=C, values=(70, 80))]


def warm_games():
    return [dict(day=0, season=1, round=1, values=(2, 6)),
            dict(day=2, season=1, round=2, values=(4, 8)),
            dict(day=10, season=2, round=1, values=(10, 20)),
            dict(day=12, season=2, round=2, values=(14, 24)),
            dict(day=14, season=2, round=3, values=(18, 28)),
            dict(day=16, season=2, round=4, values=(1000, 2000), status='notstarted')]


def mover_games():
    return [dict(day=0, competition=20, season=11, label='23_24', home=A, away=C, values=(2, 6)),
            dict(day=2, competition=20, season=11, label='23_24', home=A, away=C, values=(4, 8)),
            dict(day=0, competition=10, season=1, home=B, away=D, values=(10, 20)),
            dict(day=2, competition=10, season=1, home=B, away=D, values=(30, 40)),
            dict(day=10, competition=10, season=2, round=1, home=A, away=B, values=(50, 60)),
            dict(day=12, competition=10, season=2, round=2, home=A, away=B, values=(70, 80))]


def movement():
    return [dict(competition_id=10, season_id=2, team_id=A, movement='promoted',
                 previous_competition_id=20, previous_season_id=11)]
