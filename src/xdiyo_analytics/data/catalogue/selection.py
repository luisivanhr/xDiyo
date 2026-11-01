"""Frozen coherent selection, with separate training/evaluation/history membership."""
from copy import deepcopy
import json
from pathlib import Path
import pandas as pd
import numpy as np
from .storage import DATASETS, POLICY, atomic_json, digest, load_snapshot, squad_disposition
from ..loading import SeasonData

RETROSPECTIVE_ASSUMPTIONS=[
 'Reported results are accepted as retrospective observations, not proof of final sporting status.',
 'Unknown timezones use the full UTC-12..UTC+14 civil-date interval.',
 'History assumes results/statistics known by interval end plus 24 hours; publication delay is unverified.',
 'Later corrections may exist; this is not a historically archived as-of dataset.']


def _time(value):
    if value is None or pd.isna(value):return pd.NaT
    t=pd.Timestamp(value)
    if t.tzinfo is None:raise ValueError('Exact evidence timestamps must be timezone-aware')
    return t.tz_convert('UTC')


def _bounds(row,profile):
    kickoff=_time(row.get('kickoff_at'));release=_time(row.get('result_available_at'))
    if pd.notna(kickoff) and pd.notna(release) and release<kickoff:
        raise ValueError(f"Result predates kickoff: fixture {row['event_id']}")
    if profile=='strict_as_of':
        if pd.isna(kickoff) or pd.isna(release):return None
        return kickoff,release
    start=pd.Timestamp(row['date'],tz='UTC')-pd.Timedelta(hours=14)
    available=pd.Timestamp(row['date'],tz='UTC')+pd.Timedelta(hours=60)
    return (start if pd.isna(kickoff) else kickoff,max(available,release) if pd.notna(release) else available)


def requirement_closure(requirements):
    """Union every transitive raw capability, rejecting cyclic or unknown dependencies."""
    allowed={'goals','regulation_goals','membership','standings','heatmaps','odds'}
    out=set();visiting=set()
    def visit(node):
        if isinstance(node,str):
            if node not in allowed and not node.startswith('stat:'):raise ValueError(f'Unknown capability: {node}')
            out.add(node);return
        if not isinstance(node,dict) or set(node)-{'name','requires','dependencies'}:raise ValueError('Invalid dependency requirement')
        key=id(node)
        if key in visiting:raise ValueError('Cyclic feature requirements')
        visiting.add(key)
        for child in [*node.get('requires',[]),*node.get('dependencies',[])]:visit(child)
        visiting.remove(key)
    for requirement in requirements:visit(requirement)
    return sorted(out|{'goals'})


def _eligible(row,stats,odds,requirements,profile):
    if row['identity_state']!='resolved':return 'identity_needs_review'
    if row['conflict']:return 'material_conflict'
    if any(pd.isna(row.get(k)) for k in ('home_score_current','away_score_current')):return 'missing_score_pair'
    if row['status'] not in ('finished','reported_result'):return 'unsupported_status'
    if 'regulation_goals' in requirements and row['score_basis']!='regulation':return 'unknown_goal_basis'
    if profile=='strict_as_of' and (row['status']!='finished' or row['score_basis']!='regulation'):return 'unsupported_strict_semantics'
    if _bounds(row,profile) is None:return 'unknown_availability'
    if any(r in requirements for r in ('membership','standings','heatmaps')):
        # An approved canonical membership/schedule input is needed; never infer it from matches.
        return 'unsupported_dependency'
    if 'odds' in requirements:
        return 'unsupported_quote_availability'  # no bookmaker/publication contract in these inputs
    for requirement in requirements:
        if not requirement.startswith('stat:'):continue
        _,period,key=requirement.split(':',2)
        valid,known=stats.get(requirement,(set(),set()))
        if row['event_id'] not in valid:return 'missing_required_statistic'
        if profile=='strict_as_of' and row['event_id'] not in known:return 'unknown_statistic_availability'
    return None


def _identities(values):
    """Never repair rounded/fractional IDs through int() coercion."""
    result=set()
    for value in values:
        if isinstance(value,(bool,np.bool_)) or not (isinstance(value,(int,np.integer)) or isinstance(value,str) and value.isascii() and value.isdecimal()):
            raise ValueError('Fixture IDs must be exact positive integers or decimal strings')
        identity=int(value)
        if identity<=0:raise ValueError('Fixture IDs must be positive')
        result.add(identity)
    return result


def select_data(root,*,snapshots,mode='single_dataset',requirements=('goals',),profile='retrospective_reported_goals',
                preference=DATASETS,train_ids=(),evaluation_ids=(),history_ids=None,competitions=None,seasons=None,
                target_ids=None,fold_membership=None):
    """Preview and freeze an explicit population. Saving this object accepts the preview.

    extend_history requires fixed nonempty target memberships. Union/single modes
    use explicit train/evaluation IDs when provided, otherwise expose all eligible
    rows as a target population for an explicit downstream splitter.
    """
    if mode not in ('single_dataset','union','extend_history'):raise ValueError('Unknown selection mode')
    if profile not in ('retrospective_reported_goals','strict_as_of'):raise ValueError('Unknown timing/score profile')
    if not snapshots or set(snapshots)-set(DATASETS):raise ValueError('Choose exact named dataset snapshots')
    if mode=='single_dataset' and len(snapshots)!=1:raise ValueError('single_dataset requires one dataset')
    if set(preference)!=set(DATASETS) or len(preference)!=len(DATASETS):raise ValueError('Preference must rank both datasets once')
    train=_identities(train_ids);evaluation=_identities(evaluation_ids)
    folds=None
    if fold_membership is not None:
        if train or evaluation:raise ValueError('Use fold membership or global train/evaluation IDs, not both')
        folds=[]
        for fold in fold_membership:
            if set(fold)-{'train_ids','evaluation_ids','score_ids','metadata'}:raise ValueError('Unknown frozen fold fields')
            item={}
            for key in ('train_ids','evaluation_ids','score_ids'):
                values=fold.get(key,fold.get('evaluation_ids',[]) if key=='score_ids' else [])
                ids=_identities(values)
                if len(ids)!=len(values):raise ValueError('Duplicate fixture in frozen fold role')
                item[key]=[int(v) for v in values]
            if set(item['train_ids'])&set(item['evaluation_ids']):raise ValueError('A fixture crosses train/test within a fold')
            if not set(item['score_ids'])<=set(item['evaluation_ids']):raise ValueError('Frozen scoring fixtures must be in evaluation')
            item['metadata']=deepcopy(fold.get('metadata',{}));folds.append(item)
        if not folds:raise ValueError('Frozen folds cannot be empty')
    if train&evaluation:raise ValueError('A physical fixture cannot be in both training and evaluation')
    fold_targets=set() if folds is None else {e for f in folds for k in ('train_ids','evaluation_ids') for e in f[k]}
    fixed_targets=_identities(target_ids) if target_ids is not None else train|evaluation|fold_targets
    if not (train|evaluation|fold_targets)<=fixed_targets:raise ValueError('Fold membership lies outside fixed targets')
    if mode=='extend_history' and not fixed_targets:raise ValueError('Extended history requires fixed targets')
    required=requirement_closure(requirements);frames=[];aux={};exclusions=[];registry_facts={};known_conflicts=set();capabilities={};registries=[]
    for name,version in sorted(snapshots.items()):
        tables,manifest,registry=load_snapshot(root,name,version)
        registries.append((name,registry))
        known_conflicts.update(manifest['coverage'].get('conflicting_fixtures',[]))
        # Registry extensions may differ; overlapping canonical IDs must mean the same entity.
        for kind,entities in registry.payload['entities'].items():
            for key,facts in entities.items():
                identity=(kind,key)
                if identity in registry_facts and registry_facts[identity]!=facts:raise ValueError(f'Incompatible canonical registries: {identity}')
                registry_facts[identity]=facts
        frame=tables['matches'];aux[name]=tables
        capabilities[name]={}
        for requirement in required:
            if not requirement.startswith('stat:'):continue
            _,period,key=requirement.split(':',2)
            stats=tables.get('statistics')
            if stats is None or stats.empty:continue
            subset=stats.loc[stats.period.eq(period)&stats.key.eq(key)].copy()
            subset['_valid']=subset.value.notna()&subset.validity.eq('observed')&subset.definition.eq('regulation_period')
            subset['_known']=[pd.notna(_time(t)) for t in subset.available_at]
            groups=subset.groupby('event_id',sort=False).agg(count=('side','size'),sides=('side','nunique'),valid=('_valid','all'),known=('_known','all'))
            valid=groups['count'].eq(2)&groups.sides.eq(2)&groups.valid
            capabilities[name][requirement]=(set(groups.index[valid]),set(groups.index[valid&groups.known]))
        if competitions is not None:frame=frame.loc[frame.competition_id.isin(competitions)]
        if seasons is not None:frame=frame.loc[frame.season_id.isin(seasons)]
        frames.append(frame)
    all_rows=pd.concat(frames,ignore_index=True);chosen=[]
    # Even a publication with zero retained fixtures can carry a reserve correction.
    checked=set()
    for row in all_rows.to_dict('records'):
        for side in ('home_id','away_id'):
            team=int(row[side]);key=(team,row['date'])
            if key in checked:continue
            checked.add(key)
            decisions={name:r.squad(team,row['date']) for name,r in registries
                       if str(team) in r.payload['entities'].get('team',{})}
            if len(set(decisions.values()))>1:
                raise ValueError(f"Incompatible squad classifications: team {team}, fixture {row['event_id']}, date {row['date']}: {decisions}")
            if set(decisions.values())!={'not_reserve'}:
                raise ValueError(f"Union reserve policy rejects team {team} on {row['date']}: {decisions}")
    for event,group in all_rows.groupby('event_id',sort=True):
        # Contradictions remain disqualifying, even if an ineligible record is less preferred.
        material=['home_id','away_id','competition_id','season_id','date']
        conflict=event in known_conflicts or any(group[c].dropna().nunique()>1 for c in material)
        scores=group.loc[group[['home_score_current','away_score_current']].notna().all(axis=1)]
        conflict |= len(scores[['home_score_current','away_score_current']].drop_duplicates())>1
        conflict |= group.conflict.any()
        statuses=set(group.status)-{'reported_result'}
        conflict |= len(statuses)>1 or bool(statuses-{'finished'})
        if conflict:exclusions.append(dict(event_id=int(event),reason='material_conflict'));continue
        candidates=[]
        for row in group.to_dict('records'):
            tables=aux[row['dataset']]
            reason=_eligible(row,capabilities[row['dataset']],tables.get('odds'),required,profile)
            if reason:exclusions.append(dict(event_id=int(event),dataset=row['dataset'],reason=reason))
            else:candidates.append(row)
        if candidates:chosen.append(min(candidates,key=lambda r:list(preference).index(r['dataset'])))
    selected=pd.DataFrame(chosen,columns=all_rows.columns)
    eligible=set(map(int,selected.event_id))
    if fixed_targets-eligible:raise ValueError(f'Fixed targets are ineligible: {sorted(fixed_targets-eligible)[:20]}')
    targets=fixed_targets if target_ids is not None or fixed_targets else eligible
    history=eligible if history_ids is None else _identities(history_ids)
    if history-eligible:raise ValueError('History membership contains ineligible fixtures')
    if mode=='extend_history':
        max_cutoff=max(_bounds(r,profile)[0] for r in selected.to_dict('records') if r['event_id'] in targets)
        history={r['event_id'] for r in selected.to_dict('records') if r['event_id'] in history and _bounds(r,profile)[1]<=max_cutoff}
    payload=dict(schema=1,snapshots=dict(sorted(snapshots.items())),mode=mode,requirements=required,profile=profile,preference=list(preference),
       competitions=competitions,seasons=seasons,train_ids=sorted(train),evaluation_ids=sorted(evaluation),target_ids=sorted(targets),history_ids=sorted(history),
       records=[dict(event_id=int(r['event_id']),dataset=r['dataset']) for r in chosen if r['event_id'] in targets|history],
       exclusions=exclusions,policy=POLICY,history_scope=['team_id','competition_id'],
       classification='conditional_retrospective_research' if profile.startswith('retrospective') else 'verified_input_availability',
       assumptions=RETROSPECTIVE_ASSUMPTIONS if profile.startswith('retrospective') else [])
    if target_ids is not None or folds is not None:
        payload.update(schema=2,fold_membership=folds)
    return dict(payload,fingerprint=digest(payload))


def save_selection(selection,path):
    if digest({k:v for k,v in selection.items() if k!='fingerprint'})!=selection.get('fingerprint'):raise ValueError('Selection changed')
    atomic_json(path,selection)


def resolve_selection(root,selection):
    if isinstance(selection,str) and selection.lstrip().startswith('{'):selection=json.loads(selection)
    elif isinstance(selection,(str,Path)):selection=json.loads(Path(selection).read_text(encoding='utf-8'))
    if digest({k:v for k,v in selection.items() if k!='fingerprint'})!=selection.get('fingerprint'):raise ValueError('Selection changed')
    if selection.get('schema') not in (1,2) or selection.get('policy')!=POLICY:raise ValueError('Selection requires explicit migration')
    options={k:selection[k] for k in ('snapshots','mode','requirements','profile','preference','train_ids','evaluation_ids','history_ids','competitions','seasons')}
    if selection['schema']==2:options.update(target_ids=selection['target_ids'],fold_membership=selection['fold_membership'])
    fresh=select_data(root,**options)
    if fresh!=selection:raise ValueError('Selection dependencies or membership changed; rebuild explicitly')
    tables={};frames=[];all_stats=[]
    for name,version in selection['snapshots'].items():
        native,_,_=load_snapshot(root,name,version)
        ids={r['event_id'] for r in selection['records'] if r['dataset']==name}
        frame=native['matches'].loc[native['matches'].event_id.isin(ids)].copy()
        frames.append(frame)
        if 'statistics' in native:all_stats.append(native['statistics'].loc[native['statistics'].event_id.isin(ids)])
    matches=pd.concat(frames,ignore_index=True).sort_values('event_id').reset_index(drop=True)
    tables['matches']=matches
    if all_stats:tables['statistics']=pd.concat(all_stats,ignore_index=True)
    return SeasonData(tables,dict(data_selection=deepcopy(selection)))


def selected_history(data):
    """Native team rows with explicit logical bounds, never fictitious kickoff_utc."""
    from ...histories import build_team_history
    selection=data.provenance['data_selection'];matches=data.matches.copy()
    bounds=[_bounds(r,selection['profile']) for r in matches.to_dict('records')]
    # The legacy reshape needs a numeric column. Keep uncertain real timestamps missing.
    matches['kickoff_utc']=pd.Series([_time(v).timestamp() if pd.notna(_time(v)) else np.nan for v in matches.kickoff_at])
    required_statistics={tuple(r.split(':',2)[1:]) for r in selection['requirements'] if r.startswith('stat:')}
    if required_statistics and 'statistics' in data.tables:
        releases={}
        for stat in data.tables['statistics'].to_dict('records'):
            if (stat['period'],stat['key']) in required_statistics:
                t=_time(stat.get('available_at'))
                if pd.notna(t):releases[stat['event_id']]=max(releases.get(stat['event_id'],t),t)
        bounds=[(b[0],max(b[1],releases.get(int(e),b[1]))) for e,b in zip(matches.event_id,bounds)]
    history=build_team_history(SeasonData(dict(data.tables,matches=matches),data.provenance))
    timing={int(r.event_id):b for r,b in zip(matches.itertuples(),bounds)}
    # Add grouping metadata after the native statistics reshape (which joins by fixture).
    history['source_season']=history.event_id.map(dict(zip(matches.event_id,matches.season_year)))
    history['kickoff_at']=pd.to_datetime([timing[int(e)][0] for e in history.event_id],utc=True)
    history['result_available_at']=pd.to_datetime([timing[int(e)][1] for e in history.event_id],utc=True)
    history['catalogue_history_available_at']=history.result_available_at.where(history.event_id.isin(selection['history_ids']))
    history['selection_time_kind']='conservative_interval_lower_bound' if selection['profile'].startswith('retrospective') else 'actual_kickoff'
    # This view is explicitly an observed-results research stream, not invented raw match status.
    if selection['profile'].startswith('retrospective'):
        history['status']='finished'
        history['result']=np.where(history.goals_for>history.goals_against,'W',np.where(history.goals_for<history.goals_against,'L','D'))
    history.attrs['data_selection']=deepcopy(selection)
    return history.sort_values(['kickoff_at','event_id','side'],kind='stable').reset_index(drop=True)
