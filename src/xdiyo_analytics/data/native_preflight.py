"""Inspectable cohort readiness and temporal assumptions, without training."""
from copy import deepcopy
import hashlib
import json

from .native_formats import POLICY


def explicit_times(values, label):
    import pandas as pd
    from numbers import Number
    for value in values.dropna():
        if isinstance(value, Number) or pd.Timestamp(value).tzinfo is None:
            raise ValueError(f'{label} requires timezone-aware datetimes, not ambiguous local times or numeric epochs')
    return pd.to_datetime(values,utc=True,errors='raise')


def validate_preprocessing(options=None):
    options = dict(options or {})
    if set(options)-{'availability','entry_information'}:
        raise ValueError('Unknown native preprocessing option')
    options.setdefault('availability','retrospective_finished')
    options.setdefault('entry_information','retrospective_season_entry')
    if options['availability'] not in ('retrospective_finished','explicit'):
        raise ValueError('availability must be retrospective_finished or explicit')
    if options['entry_information'] not in ('retrospective_season_entry','explicit'):
        raise ValueError('entry_information must be retrospective_season_entry or explicit')
    return options


def sporting_mask(frame):
    import numpy as np
    import pandas as pd
    scores = frame[['home_score_current','away_score_current']].to_numpy(dtype=float,na_value=np.nan)
    mask = frame.status.eq('finished').fillna(False) & np.isfinite(scores).all(axis=1)
    if 'is_awarded' in frame:mask &= ~frame.is_awarded.fillna(False)
    if 'play_status' in frame:mask &= ~frame.play_status.fillna('').str.contains('award|annul|abandon|unplayed|cancel',case=False)
    return pd.Series(mask,index=frame.index,dtype=bool)


def preflight_loaded(data, *, required_features=(), options=None, cutoff=None):
    import pandas as pd
    options = validate_preprocessing(options)
    if set(required_features)-{'countdown','standings','movements'}:
        raise ValueError('Required features must be countdown, standings or movements')
    if 'native_scope_rules' not in data.provenance:
        raise ValueError('Native preflight needs a resolved NativeLibrary/load_seasons query')
    stamp=None
    if cutoff is not None:
        stamp=pd.Timestamp(cutoff)
        if pd.isna(stamp) or stamp.tzinfo is None:raise ValueError('Preflight cutoff must be timezone-aware')
    rows=[];matches=data.matches
    for source in data.provenance['sources']:
        stem=source['season'];parts=stem.rsplit('_',2)
        population=matches.loc[matches.source_league.eq(parts[0]) & matches.source_season.eq('_'.join(parts[1:]))]
        for tid,frame in population.groupby('tournament_id',sort=True,dropna=False):
            phase_id=None if pd.isna(tid) else int(tid)
            spec=deepcopy(data.provenance['native_scope_rules'][stem]['missing' if phase_id is None else str(phase_id)])
            roster=data.tables.get('team_seasons')
            members=(roster.loc[roster.competition_id.eq(spec['competition_id']) & roster.season_id.eq(spec['season_id']) & roster.team_id.isin(spec['team_ids'])]
                     if roster is not None else None)
            usable=sporting_mask(frame);targets=frame.get('native_target',pd.Series(True,index=frame.index))
            published=(pd.to_datetime(frame.result_available_at,utc=True,errors='raise') if 'result_available_at' in frame else pd.Series(pd.NaT,index=frame.index,dtype='datetime64[ns, UTC]'))
            if options['availability']=='explicit' and 'result_available_at' in frame:
                published=explicit_times(frame.result_available_at,'Result availability')
            kick=pd.to_datetime(frame.kickoff_utc,unit='s',utc=True,errors='raise')
            eligible=usable & targets
            if stamp is not None:
                eligible &= kick.lt(stamp)
                if options['availability']=='explicit':eligible &= published.notna() & published.le(stamp)
                else:eligible &= published.isna() | published.le(stamp)
            unknown_moves=(len(spec['team_ids']) if members is None else
                           len(set(spec['team_ids'])-set(members.team_id)) + int(members[['got_promoted','got_demoted']].isna().any(axis=1).sum()))
            movement_times=(pd.to_datetime(members.entry_available_at,utc=True) if members is not None and 'entry_available_at' in members
                            else pd.Series(pd.NaT,index=[] if members is None else members.index,dtype='datetime64[ns, UTC]'))
            assumptions=[];blockers=[]
            if options['availability']=='retrospective_finished':
                assumptions.append('Earlier finished kickoff is a retrospective result/statistic availability proxy; reviewed completion bounds still apply')
            elif (usable & published.isna()).any():
                blockers.append('Finished results lack explicit result availability evidence')
            if options['entry_information']=='retrospective_season_entry':
                assumptions.append('Season-entry/membership information is retrospectively reviewed; no historical publication timestamps are asserted')
            if 'movements' in required_features:
                if unknown_moves:blockers.append('Unknown season-entry movement flags')
            if options['entry_information']=='explicit' and (members is None or len(members)!=len(spec['team_ids']) or movement_times.isna().any()):
                blockers.append('Season-entry publication times are missing')
            if options['availability']=='explicit' and any(name in data.tables and not data.tables[name].empty for name in ('statistics','heatmap_points')):
                blockers.append('Per-statistic/heatmap publication times are not established by result release times')
            for feature in set(required_features)&{'countdown','standings'}:
                if spec[feature]['status']!='supported':blockers.append(feature+': '+spec[feature]['reason'])
            if 'countdown' in required_features:
                assumptions.append('Countdown uses the revised published schedule and closest round clusters, not an archived issuance schedule')
                if options['availability']=='explicit':blockers.append('Historical schedule-at-issuance evidence is unavailable for countdown')
            if not eligible.any():blockers.append('No finished labeled sporting targets eligible for this selection/cutoff')
            metadata=source.get('native_integration',{})
            display=source.get('query_display_names',{})
            row=dict(publication=stem,competition_id=spec['competition_id'],competition=display.get('competition',metadata.get('competition',{}).get('name')),
                season_id=spec['season_id'],season=display.get('season',metadata.get('season',{}).get('name')),season_label=metadata.get('season',{}).get('year'),
                phase_id=phase_id,phase=frame.tournament_name.dropna().iloc[0] if 'tournament_name' in frame and frame.tournament_name.notna().any() else None,
                annual_group=source.get('season_calendar'),coverage=metadata.get('coverage',{}),membership=spec['membership'],
                countdown=spec['countdown'],standings=spec['standings'],table_treatment=spec['table_treatment'],
                movement=dict(unknown_teams=unknown_moves,known_teams=len(spec['team_ids'])-unknown_moves,
                              availability=options['entry_information'],publication_times=int(movement_times.notna().sum())),
                result_availability=dict(convention=options['availability'],known_bounds=int(published.notna().sum()),unknown=int(published.isna().sum())),
                counts=dict(loaded=len(frame),selected_targets=int(targets.sum()),finished_sporting=int(usable.sum()),eligible_targets=int(eligible.sum()),
                    excluded_unfinished=int(frame.status.ne('finished').fillna(True).sum()),
                    excluded_administrative=int(frame.get('is_awarded',pd.Series(False,index=frame.index)).fillna(False).sum()),
                    unknown_award_flags=int(frame.is_awarded.isna().sum()) if 'is_awarded' in frame else len(frame)),
                assumptions=assumptions,blockers=blockers,readiness='blocked' if blockers else 'conditional_retrospective' if assumptions else 'ready_with_supplied_evidence',
                sources=spec['evidence'],manifest_sha256=metadata.get('manifest_sha256'))
            rows.append(row)
    report=dict(policy=POLICY,selection=deepcopy(data.provenance['native_selection']),
                required_features=sorted(set(required_features)),options=options,cutoff=None if stamp is None else stamp.isoformat(),cohorts=rows,
                loader_excluded_awarded_event_ids={s['season']:s.get('excluded_awarded_event_ids',[]) for s in data.provenance['sources']})
    report['fingerprint']=hashlib.sha256(json.dumps(report,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return report


def native_preflight(library, *, query=None, required_features=(), availability='retrospective_finished',
                     entry_information='retrospective_season_entry', cutoff=None):
    data=library.load(query=query,tables=['matches','team_seasons'],include_awarded=True)
    return preflight_loaded(data,required_features=required_features,options=dict(availability=availability,entry_information=entry_information),cutoff=cutoff)


def prepare_native_history(data, history, options):
    """Mark unusable sporting observations in the derived history only.

    Original status/scores remain in loaded data. This status prevents every
    native rolling operator/rating from consuming administrative target scores.
    """
    import pandas as pd
    options=validate_preprocessing(options)
    mask=sporting_mask(data.matches)
    keys=['competition_id','season_id','event_id']
    allowed=set(data.matches.loc[mask,keys].itertuples(index=False,name=None))
    out=history.copy()
    eligible=pd.Series([k in allowed for k in out[keys].itertuples(index=False,name=None)],index=out.index)
    out['source_status']=out.status.copy()
    out.loc[~eligible & out.status.eq('finished'),'status']='excluded_non_sporting_result'
    from .native_formats import resumed_standings_evidence
    for identity,(scores,release) in resumed_standings_evidence(data.matches).items():
        selected=pd.Series([key==identity for key in out[keys].itertuples(index=False,name=None)],index=out.index)
        for name in ('standings_goals_for','standings_goals_against'):
            if name not in out:out[name]=float('nan')
        if 'standings_result_available_at' not in out:
            out['standings_result_available_at']=pd.Series(pd.NaT,index=out.index,dtype='datetime64[ns, UTC]')
        for side,own,other in (('home',scores[0],scores[1]),('away',scores[1],scores[0])):
            mask=selected & out.side.eq(side)
            for name,value in (('standings_goals_for',own),('standings_goals_against',other),('standings_result_available_at',release)):
                if (out.loc[mask,name].notna() & out.loc[mask,name].ne(value)).any():
                    raise ValueError(f'Conflicting resumed-result standings evidence for {identity}: {name}')
                out.loc[mask,name]=value
    if options['availability']=='explicit':
        if 'result_available_at' not in out:
            raise ValueError('Explicit native availability requires result_available_at evidence')
        times=explicit_times(out.result_available_at,'Result availability')
        if (eligible & times.isna()).any():
            raise ValueError('Explicit native availability has unknown result release times; inspect preflight')
    return out


def validate_entry_cutoffs(history, roster, cutoffs):
    """Require known season-entry evidence before every use, including warm starts."""
    from ..features.history import aligned_times
    import pandas as pd
    if 'entry_available_at' not in roster:
        raise ValueError('Explicit entry information requires entry_available_at evidence')
    keys=['competition_id','season_id','team_id']
    if roster.duplicated(keys).any():raise ValueError('Ambiguous season-entry availability identities')
    lookup=dict(zip(roster[keys].itertuples(index=False,name=None),explicit_times(roster.entry_available_at,'Entry availability')))
    times=aligned_times(history,cutoffs,default='kickoff_at')
    for key,cutoff in zip(history[keys].itertuples(index=False,name=None),times):
        available=lookup.get(key)
        if available is None or pd.isna(available) or pd.isna(cutoff) or available>cutoff:
            raise ValueError(f'Season-entry evidence unavailable at prediction cutoff for {key}: {available} > {cutoff}')


def validate_round_groups(history, keys, label):
    """Reject phase collisions without changing assigned provider rounds."""
    keys=list(dict.fromkeys(['competition_id','season_id',*keys]))
    if any(k not in history for k in keys):
        raise ValueError(f'{label}: round keys require complete native history columns')
    for identity,group in history.groupby(keys,sort=False,dropna=False):
        if (group.countdown_calendar_id.nunique(dropna=False)>1
                or group.duplicated(['team_id']).any()):
            raise ValueError(f'{label}: ambiguous native phase/round group {identity}; select an independent phase or configure explicit stage keys')
