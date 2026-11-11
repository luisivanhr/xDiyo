"""Build daily rank lookups, retaining explicit legacy preparation support."""
import numpy as np
import pandas as pd
from .phase_context import (PhaseRankLookup, CORRECTED_POLICY, DAILY_POLICY,
                            REVIEWED_POLICY, REVIEWED_DAILY_POLICY, WINDOWS, KEYS)


def build_phase_context(history, team_seasons, fixtures, *, cutoffs, provenance, available_at=None,
                        rank_timing='kickoff_day', meetings_per_pair=None, allow_incomplete_schedule=False):
    """Create exact forecast rows with prior-calendar-day ranks by default.

    Unsupported split schedules raise through the current producer. Reviewed
    incomplete independent regular tables require allow_incomplete_schedule=True,
    an explicit standings_table_id/standings_rule and meetings_per_pair. Ranks
    then use the same eligible prior results and full declared roster; no future
    unplayed fixtures are invented or required. Countdown retains its native
    planned-round and available-schedule-cluster definition independently.
    Dataset-specific decisions about unsupported scopes belong in preparation
    examples; old tables are loaded directly and never rebuilt here. Use
    rank_timing='available_at_cutoff' to reproduce v2 timestamp preparation.
    Daily preparation requires cutoffs on each fixture's kickoff calendar date;
    it changes neither countdown semantics nor original round metadata.
    Pass meetings_per_pair=3 only for a verified independent triple round robin.
    """
    from ..features import DerivedStandings, SeasonCountdown
    from ..features.countdown import season_countdown
    from ..ratings import build_standings
    keys=pd.MultiIndex.from_frame(fixtures[list(KEYS)])
    home=history.loc[history.side.eq('home')].set_index(list(KEYS))
    if not home.index.is_unique or not keys.is_unique:
        raise ValueError('Phase preparation requires unique home-oriented fixture identities.')
    query=home.loc[keys].reset_index();query.index=fixtures.index
    if isinstance(cutoffs, pd.Series) and not cutoffs.index.equals(fixtures.index):
        raise ValueError("Phase cutoffs must align with the fixture population.")
    cuts=pd.Series(cutoffs,index=fixtures.index)
    for field,native in (('kickoff_at','kickoff_at'),('home_id','team_id'),('away_id','opponent_id'),('round','round')):
        if not np.array_equal(fixtures[field],query[native]):
            raise ValueError('Phase fixture identity/orientation differs from the native history.')
    if not cuts.index.equals(fixtures.index):
        raise ValueError('Phase cutoffs must align with the fixture population.')
    result=fixtures[[*KEYS,'home_id','away_id','kickoff_at']].copy()
    result['cutoff_at']=cuts
    result['countdown']=season_countdown(query,SeasonCountdown(),schedule_history=history,
                                        team_seasons=team_seasons,cutoffs=cuts).to_numpy()
    native=build_standings(history,team_seasons=team_seasons,normalized_ranks=WINDOWS,
                           queries=query,cutoffs=cuts,available_at=available_at,timing=rank_timing,
                           meetings_per_pair=meetings_per_pair,allow_incomplete_schedule=allow_incomplete_schedule)
    values=native.features(query,DerivedStandings(normalized_ranks=WINDOWS,include_raw=False,side='both'),cutoffs=cuts)
    for role,side in (('team','home'),('opponent','away')):
        for window in WINDOWS:
            result[f'{side}_rank_{window}']=values[f'{role}::form_rank_{window}'].to_numpy()
    policy = DAILY_POLICY if rank_timing == 'kickoff_day' else CORRECTED_POLICY
    if 'points_rule' in native.metadata:
        policy = REVIEWED_DAILY_POLICY if rank_timing == 'kickoff_day' else REVIEWED_POLICY
    if allow_incomplete_schedule:
        provenance = dict(provenance, standings_schedule=native.metadata['schedule'],
                          meetings_per_pair=meetings_per_pair)
    return PhaseRankLookup(result,provenance=provenance, policy=policy)
