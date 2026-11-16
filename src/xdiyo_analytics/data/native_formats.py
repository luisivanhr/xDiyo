"""Evidence-bound adapters to existing countdown and sporting-form engines.

These adapters are used by explicit native queries. Legacy storage-key loading
retains its pinned behavior. No data, official scores or source manifests change.
"""
from collections import Counter
from copy import deepcopy
import json
import pandas as pd

POLICY = 'native_scoped_preprocessing_v1'
FORMAT_REFERENCES = {
    170: 'https://hns.family/files/documents/33080/Propozicije%20natjecanja%20SuperSport%20HNL%2026-27.pdf',
    206: 'https://spfl.co.uk/admin/filemanager/images/shares/pdfs/MASTER%20-%20Rules%20and%20Regulations%20(CLEAN%20-%2011%20JUNE%202024).pdf',
    212: 'https://www.prvaliga.si/Doc/Info/2025/Navodila%20za%20organizacijo%20tekem%20v%201.%20SNL_V5.1_julij2025.pdf',
    196: 'https://www.jleague.jp/sp/special/2026specialseason/j1/',
}


def resumed_standings_evidence(matches):
    """Recognize identity-bound played results with a reviewed completion bound.

    The existing standings engine requires explicit approved scores for every
    reviewed record. Resumptions can supply those from their verified played
    result, without treating a ruling date as a sporting completion time.
    """
    if 'result_review_json' not in matches:return {}
    approved={}
    for row in matches.loc[matches.result_review_json.notna()].to_dict('records'):
        try:review=json.loads(row['result_review_json'])
        except (TypeError,ValueError):continue
        if not isinstance(review,dict) or review.get('classification')!='resumed_completed':continue
        identity=['event_id','home_id','away_id']
        scores=[row['home_score_current'],row['away_score_current']]
        if (any(review.get(k)!=row[k] for k in identity) or review.get('is_awarded') is not False
                or pd.isna(row.get('is_awarded')) or bool(row['is_awarded'])
                or review.get('play_status')!='resumed' or row.get('play_status')!='resumed'
                or any(pd.isna(s) for s in scores) or review.get('played_score')!=scores
                or not review.get('sources') or pd.isna(row.get('result_available_at'))):
            continue
        release=pd.Timestamp(row['result_available_at']);recorded=pd.Timestamp(review.get('result_available_at'))
        if pd.isna(release) or release.tzinfo is None or pd.isna(recorded) or recorded.tzinfo is None or release!=recorded:
            continue
        if release<=pd.to_datetime(row['kickoff_utc'],unit='s',utc=True):continue
        approved[(row['competition_id'],row['season_id'],row['event_id'])]=(scores,release)
    return approved


def scope_rules(publication, matches, roster):
    """Return per-provider-phase rules, without inferring rules from a name.

    Count topology validates a reviewed provider scope; alone it never proves
    a rule. Partial scopes need reviewed planned fixture counts or independent
    phase/table evidence, with the full known membership present.
    """
    cid, sid = publication['competition_id'], publication['season_id']
    meta = publication['manifest'].get('native_integration', {})
    cohort = publication['publication'].get('reconciliation', {}).get('archive_cohort', {})
    evidence = cohort.get('format_source') or meta.get('rules', {}).get('format_sources') or []
    evidence = [evidence] if isinstance(evidence, str) else list(evidence)
    phases = {int(p['tournament_id']):p for p in cohort.get('reviewed_phases', [])}
    member_ids = set(map(int, roster.team_id)) if roster is not None else set()
    resumptions=resumed_standings_evidence(matches)
    output = {}
    for tid, frame in matches.groupby('tournament_id', sort=True, dropna=False):
        if pd.isna(tid):
            reason='Provider tournament/phase identity is missing; no phase ID is invented'
            output['missing']=dict(policy=POLICY,competition_id=cid,season_id=sid,phase_id=None,
                team_ids=sorted(set(map(int,frame.home_id))|set(map(int,frame.away_id))),membership='missing_evidence',
                countdown=dict(status='unsupported',reason=reason),standings=dict(status='unsupported',reason=reason),
                evidence=evidence,table_treatment='unresolved',schedule_availability='unverified')
            continue
        tid = int(tid)
        ids = sorted(set(map(int, frame.home_id)) | set(map(int, frame.away_id)))
        n = len(ids); pairs = Counter(tuple(sorted((int(a),int(b)))) for a,b in zip(frame.home_id,frame.away_id))
        directed = Counter((int(a),int(b)) for a,b in zip(frame.home_id,frame.away_id))
        reason = 'Missing verified independent calendar/table rules or full membership evidence'
        block = dict(status='unsupported', reason=reason)
        result = dict(policy=POLICY, competition_id=cid, season_id=sid, phase_id=tid,
                      team_ids=ids, membership='missing_evidence', countdown=deepcopy(block), standings=deepcopy(block),
                      evidence=evidence, table_treatment='independent_3_1_0_sporting_form_not_official_table',
                      schedule_availability='retrospective_revised_schedule_not_historical_as_of')
        meetings = None; total = None; reviewed = False
        phase = phases.get(tid)
        if phase and cohort.get('schedule_verified') is True and evidence:
            if (phase.get('clubs') == n and phase.get('round_mode') == 'strict'
                    and len(phase.get('table_ids', [])) == 1 and set(phase.get('pair_multiplicities', {})) <= {'1'}):
                meetings = 1; reviewed = True
        elif (cohort.get('schedule_verified') is True and evidence and cohort.get('selection') == 'all'
              and matches.tournament_id.nunique() == 1 and n >= 2):
            expected = cohort.get('regular_matches')
            # The count is a reviewed *planned scope* count, not downloaded rows.
            if isinstance(expected, int) and expected % (n*(n-1)//2) == 0:
                m = expected // (n*(n-1)//2)
                if m in (1,2,3,4):
                    meetings = m; reviewed = True
        old = meta.get('rules', {})
        if old.get('countdown', {}).get('status') == 'supported' and matches.tournament_id.nunique()==1:
            known = old['countdown']; total = known['total_rounds']
            meetings = {'single':1,'double':2}.get(known['format'])
            reviewed = True
        # Legacy regular scopes retain established IDs; a complete schedule and
        # roster are still validated. No general whitelist based on file names.
        legacy_regular = {17:1,18:2,34:4,182:19,35:42,44:41,23:33,53:34,37:39,8:36,54:37,38:38,196:82,402:3034,11621:28,11620:16753}
        if old.get('countdown', {}).get('status') == 'legacy_verified' and tid == legacy_regular.get(cid):
            counts = set(pairs.values())
            if n>=2 and len(pairs)==n*(n-1)//2 and len(counts)==1 and counts <= {1,2}:
                meetings = next(iter(counts)); reviewed=True
        if cid==196 and sid==87931 and tid in (171852,171853) and n==10 and cohort.get('schedule_verified'):
            meetings=2; total=18; reviewed=True; result['evidence']=[*evidence,FORMAT_REFERENCES[196]]
        if cid==36 and tid in (54,92066,92067):
            # Continuous 33+5 countdown; split positions/points do not become an
            # independent double-round-robin form table by resetting at the split.
            result['countdown']=dict(status='supported',format='scotland',total_rounds=38,
                                     calendar_id='season',team_ids=sorted(member_ids),policy=POLICY)
            result['standings']['reason']='Scottish split carry-over and rank scope require an explicit implementation; regular pre-split table may be selected separately'
            if tid==54 and n==12 and frame['round'].max()<=33:
                meetings=3; total=38; reviewed=True
        if reviewed and n>=2 and set(ids)<=member_ids:
            # A separate reviewed phase declares its roster; otherwise require
            # all season members, not merely whichever teams played so far.
            membership_ok = phase is not None or (cid==196 and sid==87931) or set(ids)==member_ids
            if membership_ok:
                result['membership']='reviewed_retrospective_roster'
                if total is None:total=(n-1 if n%2==0 else n)*meetings
                rounds=frame['round'].astype('Float64')
                valid_rounds=rounds.notna().all() and rounds.ge(1).all() and rounds.mod(1).eq(0).all() and rounds.le(total).all()
                unique_rounds=all(not g[['home_id','away_id']].stack().duplicated().any() for _,g in frame.groupby('round'))
                topology=(meetings is not None and max(pairs.values(),default=0)<=meetings
                          and max(directed.values(),default=0)<=(meetings+1)//2)
                if valid_rounds and unique_rounds and topology:
                    fmt={1:'single',2:'double',3:'triple',4:'quadruple'}[meetings]
                    if not (cid==36):
                        result['countdown']=dict(status='supported',format=fmt,total_rounds=total,
                            calendar_id=tid,team_ids=ids,policy=POLICY)
                    result['standings']=dict(status='supported',rule='independent_3_1_0',meetings_per_pair=meetings,
                        allow_incomplete_schedule=True,table_id=tid,team_ids=ids,policy=POLICY,
                        official_table=False)
                else:
                    result['standings']['reason']='Reviewed format conflicts with observed rounds, teams or pair multiplicities'
        if cid==196 and sid==87931:
            result['standings']=dict(status='unsupported',reason='Transition uses shootout 3/2/1/0 points; timed regulation-score and shootout evidence is required, not ordinary 3/1/0')
        if cid==196 and publication['source_season']=='16_16' and tid==82:
            from ..features.countdown import _japan_phases
            try:
                _japan_phases(frame)
            except ValueError:
                pass
            else:
                result['countdown']=dict(status='supported',format='double',total_rounds=34,
                    calendar_id='season',team_ids=ids,policy=POLICY)
                result['standings']=dict(status='unsupported',reason='Japan two-stage standings need explicit stage tables; the verified countdown preserves original phase ordinals')
        if any(r.get('event_id') in set(frame.event_id) for r in meta.get('rulings', [])) or frame.get('is_awarded', pd.Series(False,index=frame.index)).fillna(False).any():
            result['standings']=dict(status='unsupported',reason='Administrative/awarded outcomes need timed approved standings treatment; played scores are not administrative results')
        if 'result_review_json' in frame:
            reviewed=frame.loc[frame.result_review_json.notna(),['competition_id','season_id','event_id']]
            if any(key not in resumptions for key in reviewed.itertuples(index=False,name=None)):
                result['standings']=dict(status='unsupported',reason='Reviewed results need verified played-resumption evidence or an explicit timed administrative standings treatment')
        if meta.get('coverage', {}).get('kind')=='curtailed' and cid==11653:
            result['standings']=deepcopy(old['standings'])
        output[str(tid)]=result
    if matches.tournament_id.isna().any():
        # Unassigned fixtures may belong to any table/calendar in this season.
        # Selecting a named phase must not silently erase their dependencies.
        for result in output.values():
            for feature in ('countdown','standings'):
                result[feature]=dict(status='unsupported',reason='Season contains fixtures without provider phase identity; resolve their table/calendar membership first')
    return output


def apply_native_scopes(data, publications, selection):
    """Apply scoped metadata and target selection; retain complete team history."""
    import pandas as pd
    from .loading import _read_table
    from ._source import _open_source
    from .movements import _read_movements
    rules={};selected_events=set();phase_selection=selection['phase_ids']
    for pub in publications:
        source=_open_source(_publication_root(data, pub),pub['stem'])
        raw=_read_table(source,'matches',True)
        try:
            roster=(_read_table(source,'team_seasons',True) if 'team_seasons' in source.manifest['tables']
                    else _read_movements(_publication_root(data,pub),pub['stem'],source)[0])
        except KeyError:
            roster=None
        specs=scope_rules(pub,raw,roster)
        rules[pub['stem']]=specs
        allowed=[int(tid) for tid in specs if tid!='missing' and (phase_selection is None or [pub['competition_id'],int(tid)] in phase_selection)]
        keep=raw.tournament_id.isin(allowed)
        if phase_selection is None:keep |= raw.tournament_id.isna()
        selected_events.update((pub['source_league'],pub['source_season'],eid) for eid in raw.loc[keep,'event_id'].tolist())
    for name,frame in data.tables.items():
        if 'event_id' in frame:
            mask=[key in selected_events for key in frame[['source_league','source_season','event_id']].itertuples(index=False,name=None)]
            data.tables[name]=frame.loc[mask].reset_index(drop=True)
    if data.matches is not None:
        frame=data.matches.copy()
        for col in ('native_countdown_rule','native_standings_rule','countdown_calendar_id','standings_table_id','standings_rule'):
            frame[col]=pd.Series(pd.NA,index=frame.index,dtype='object')
        for pub in publications:
            for tid,spec in rules[pub['stem']].items():
                phase_mask=frame.tournament_id.isna() if tid=='missing' else frame.tournament_id.eq(int(tid))
                mask=frame.source_league.eq(pub['source_league']) & frame.source_season.eq(pub['source_season']) & phase_mask
                frame.loc[mask,'native_countdown_rule']=json.dumps(spec['countdown'],sort_keys=True)
                frame.loc[mask,'native_standings_rule']=json.dumps(spec['standings'],sort_keys=True)
                frame.loc[mask,'countdown_calendar_id']=str(spec['countdown'].get('calendar_id',tid))
                frame.loc[mask,'standings_table_id']=None if tid=='missing' else int(tid)
                frame.loc[mask,'standings_rule']='independent_3_1_0'
        teams=selection['team_ids']
        frame['native_target']=True if teams is None else frame.home_id.isin(teams)|frame.away_id.isin(teams)
        if not frame.native_target.any():raise ValueError('No target fixtures for the selected team and phase')
        data.tables['matches']=frame
    data.provenance['native_scope_rules']=rules
    data.provenance['native_preprocessing_policy']=POLICY
    return data


def _publication_root(data, pub):
    from pathlib import Path
    # Resolve via the physical source record; never parse competition names.
    source=next(s for s in data.provenance['sources'] if s['season']==pub['stem'])
    path=Path(source['manifest_path']).resolve()
    relative=Path(pub['publication']['manifest'])
    for _ in relative.parts:path=path.parent
    return path
