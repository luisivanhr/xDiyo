"""Opt-in annual rating keys for split-phase leagues in prepared scorelines."""
import pandas as pd
from .scoreline_history import local_id, ARCHIVE_SEASON
from .movements import validate_movement_flags

CANONICAL={'Liga_MX':11621,'Colombia_Primera_A':11539}
KEYS=['competition_id','season_id','team_id']
PREV=['previous_competition_id','previous_season_id']

def unify_rating_leagues(history, team_seasons):
    """Return (history, movements, crosswalk) for a continuous annual filter.

    Apply to a whole prepared snapshot before choosing fit or replay windows.
    The scoreline fixture identity, provider phases and original keys survive.
    This view is not a replacement for native keys when joining provider tables.
    """
    h, m = history, team_seasons
    if {'source_competition_id', 'provider_season_id'} & set(h):
        raise ValueError('Use the original prepared history, not an existing rating view.')
    sel=h.league.isin(CANONICAL)
    source=h.loc[sel].copy()
    if set(h.loc[~sel,'competition_id']) & set(CANONICAL.values()): raise ValueError('Competition ID collision.')
    source['source_competition_id']=source.competition_id
    source['provider_season_id']=source.season_id
    source['competition_id']=source.league.map(CANONICAL).astype(h.competition_id.dtype)
    keyvals=source[['competition_id','source_season']].drop_duplicates()
    season={(int(c),str(s)):local_id(ARCHIVE_SEASON,'rating_annual_v1',int(c),str(s))
            for c,s in keyvals.itertuples(index=False,name=None)}
    if len(set(season.values()))!=len(season) or set(season.values()) & set(h.loc[~sel,'season_id']):
        raise ValueError('Rating season ID collision.')
    source['season_id']=pd.array([season[(c,s)] for c,s in zip(source.competition_id,source.source_season)], dtype=h.season_id.dtype)
    bindings=source[['source_competition_id','provider_season_id','competition_id','season_id','source_season','league','phase']].drop_duplicates()
    mapping=bindings[['source_competition_id','provider_season_id','competition_id','season_id']].drop_duplicates()
    if mapping.duplicated(['source_competition_id','provider_season_id']).any():
        raise ValueError('A provider competition-season maps to multiple rating seasons.')
    orig=m.set_index(KEYS,verify_integrity=True)[['movement',*PREV]].to_dict('index')
    oldsource=set(source.source_competition_id)
    other=m.loc[~m.competition_id.isin(oldsource)].copy()
    if other.previous_competition_id.isin(oldsource).any(): raise ValueError('Other league depends on a split league; review remap.')
    x=source.groupby(['competition_id','season_id','source_season','team_id'],as_index=False).agg(
        first=('prediction_at','min'),first_release=('replay_available_at','min'))
    lookup={(r.competition_id,r.source_season,r.team_id):r for r in x.itertuples()}
    origin=source[['source_competition_id','provider_season_id','team_id','competition_id','season_id']].drop_duplicates()
    flags={}
    for oc,os,t,c,s in origin.itertuples(index=False,name=None):
        record=orig[(oc,os,t)]
        movement=record['movement']
        if movement in ('promoted','relegated','other_entry'):
            if pd.notna(record[PREV[0]]) or pd.notna(record[PREV[1]]):
                raise ValueError('Review cross-tier predecessor before unifying this league.')
            flags.setdefault((c,s,t),set()).add(movement)
    rows=[]
    for r in x.itertuples():
        c,s,t=r.competition_id,r.season_id,r.team_id
        year,end=map(int,r.source_season.split('/'))
        previous=lookup.get((c,f'{year-1:04d}/{end-1:04d}',t))
        explicit=flags.get((c,s,t),set())
        if len(explicit)>1: raise ValueError(f'Conflicting movement evidence: {c} {r.source_season} {t} {explicit}')
        if previous is not None and previous.first_release<=r.first:
            if explicit: raise ValueError(f'Adjacent retained club also flagged as movement: {c} {r.source_season} {t} {explicit}')
            move, pc, ps='retained', c, previous.season_id
        else:
            move,pc,ps=next(iter(explicit),'unknown'),None,None
        rows.append((c,s,t,move,None if move=='unknown' else move=='promoted',
                     None if move=='unknown' else move=='relegated',pc,ps))
    rebased=pd.DataFrame({k:pd.array([row[i] for row in rows],dtype=('Int64' if k in KEYS+PREV else 'string' if k=='movement' else 'boolean'))
                         for i,k in enumerate(KEYS+['movement','got_promoted','got_demoted']+PREV)})
    result=pd.concat([h.loc[~sel].assign(source_competition_id=lambda f:f.competition_id,provider_season_id=lambda f:f.season_id),source],
                     ignore_index=True).sort_values(['kickoff_at','competition_id','season_id','event_id','side'],kind='stable').reset_index(drop=True)
    moves=pd.concat([other,rebased],ignore_index=True)
    if len(result)!=len(h) or result.duplicated(['competition_id','season_id','event_id','side']).any():
        raise ValueError('Unified rating keys change fixture identity or introduce duplicates.')
    if moves.duplicated(KEYS).any() or set(result[KEYS].itertuples(index=False,name=None))!=set(moves[KEYS].itertuples(index=False,name=None)):
        raise ValueError('Unified rating movement records do not match the history.')
    validate_movement_flags(moves)
    return result,moves,bindings.sort_values(['league','source_season','phase','source_competition_id']).reset_index(drop=True)
