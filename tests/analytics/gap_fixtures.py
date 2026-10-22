"""Synthetic GAP fixture builder; no live data or training dependencies."""
import pandas as pd
from xdiyo_analytics.ratings import GAPParameters, GAPStatistic, GAPSeason, GAPVintage, build_gap_ratings

PARAMS = GAPParameters(.2, .75, .6)  # Explicitly synthetic, not a published value.


def fixture(event, day, h=1, a=2, yh=4., ya=2., *, league=10, season=2020, release=None, corners=(6.,3.), status='finished'):
    kickoff = pd.Timestamp(day, tz='UTC')
    release = kickoff + pd.Timedelta(hours=2) if release is None else pd.to_datetime(release, utc=True)
    return [dict(competition_id=league, season_id=season, event_id=event, team_id=t, opponent_id=o,
                 side=side, status=status, kickoff_at=kickoff, result_available_at=release,
                 goals_for=x, goals_against=y, corners_for=c, corners_against=d)
            for t,o,side,x,y,c,d in [(h,a,'home',yh,ya,*corners), (a,h,'away',ya,yh,*reversed(corners))]]


def statistic(name='goals'):
    return GAPStatistic(name, f'{name}_for', f'{name}_against', 'Synthetic full-time count definition, no extra time or penalties')


def season_plan(league=10, season=2020, stat='goals', entry='2020-01-01', members=((1,'initial'),(2,'initial')), previous=None, **kwargs):
    return GAPSeason(league, season, stat, entry, entry, tuple(members), 'Synthetic verified league and team membership', previous, **kwargs)


def vintage(stats=('goals',), start='2020-01-01', end='2022-01-01', fit='2019-12-31', parameters=PARAMS):
    return GAPVintage(tuple((s,parameters) for s in stats), fit, start, end, 'Synthetic test parameters, not paper constants')


def build(history, *, stats=('goals',), plans=None, vintages=None, cutoff='2021-12-31', **kwargs):
    if plans is None:
        plans = [season_plan(stat=s) for s in stats]
    return build_gap_ratings(history, statistics=tuple(statistic(s) for s in stats), seasons=plans,
                             vintages=vintages or (vintage(stats),), data_cutoff=cutoff, state_policy='replay', **kwargs)
