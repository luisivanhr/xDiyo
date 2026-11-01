"""Inspectable, fail-closed migration of legacy recipes; never fits or rebinds models."""
from copy import deepcopy
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import tempfile
import pandas as pd
from .storage import digest,load_snapshot
from .selection import select_data,resolve_selection,_identities
from .integration import feature_requirements

SCHEMA=1


def _json(value):
    def default(v):
        if hasattr(v,'tolist'):return v.tolist()
        if hasattr(v,'item'):return v.item()
        if isinstance(v,(pd.Timestamp,Path)):return str(v)
        raise TypeError(f'Unsupported migration metadata: {type(v).__name__}')
    return json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False,default=default)+'\n'


def _sealed(value):
    value=json.loads(_json(value))
    return dict(value,fingerprint=digest(value))


def _verify(value):
    if value.get('schema')!=SCHEMA or digest({k:v for k,v in value.items() if k!='fingerprint'})!=value.get('fingerprint'):
        raise ValueError('Migration plan changed or has an unsupported schema')


def validate_migration_envelope(recipe):
    """Unknown legacy settings remain opaque and hash-bound, never executable extensions."""
    from ...ui.recipe import default_recipe
    capsule=recipe.get('dataset_migration')
    if not capsule:return set()
    _verify(capsule)
    unknown=set(recipe)-set(default_recipe())
    if digest({k:recipe[k] for k in unknown})!=capsule['opaque_settings_hash']:
        raise ValueError('Preserved legacy settings changed; create a new migration preview')
    protected={k:recipe.get(k) for k in ('split','split_options','fold_ids','features','labels','target','assembly','data_selection')}
    if digest(protected)!=capsule['protected_settings_hash']:
        raise ValueError('Migrated population/split settings changed; create a new preview')
    return unknown


def _folds(prepared,mapping):
    meta=prepared.dataset.metadata
    def ids(indices):
        selected=meta.iloc[indices];events=list(dict.fromkeys(map(int,selected.event_id)))
        for event in events:
            if sum(int(e)==event for e in selected.event_id)!=sum(int(e)==event for e in meta.event_id):
                raise ValueError(f'Fold splits the paired rows of fixture {event}')
        return [mapping[e] for e in events]
    return [dict(train_ids=ids(f.train),evaluation_ids=ids(f.test),score_ids=ids(f.score),metadata=json.loads(_json(f.metadata))) for f in prepared.split_plan.folds]


def _membership_summary(members):
    return dict(targets=len(members['target_ids']),history=len(members['history_ids']),
                folds=[{k:len(f[k]) for k in ('train_ids','evaluation_ids','score_ids')} for f in members['folds']],
                fingerprint=digest(members))


def _check_retained_observations(old,new,mapping,required):
    columns=['goals_for','goals_against']
    columns += [c for c,info in old.attrs.get('stat_columns',{}).items()
                if f"stat:{info['period']}:{info['key']}" in required]
    columns=list(dict.fromkeys(columns))
    a=old.loc[[int(e) in mapping for e in old.event_id],['event_id','team_id',*columns]].copy()
    a['event_id']=[mapping[int(e)] for e in a.event_id]
    mapped=set(mapping.values())
    b=new.loc[[int(e) in mapped for e in new.event_id],['event_id','team_id',*columns]].copy()
    keys=['event_id','team_id']
    try:
        pd.testing.assert_frame_equal(a.sort_values(keys).reset_index(drop=True),b.sort_values(keys).reset_index(drop=True),check_dtype=False,check_exact=True)
    except AssertionError as error:
        raise ValueError(f'Retained source observations changed; review data corrections separately: {error}') from error


def _members(prepared,mapping):
    history=prepared.outputs['history']
    return dict(target_ids=[mapping[int(e)] for e in dict.fromkeys(prepared.dataset.metadata.event_id)],
                history_ids=sorted({mapping[int(e)] for e in history.loc[history.status.eq('finished'),'event_id']}),
                folds=_folds(prepared,mapping))


def _history_sets(history,targets,mapping,*,named=False):
    from ...features.history import _eligible_history_index
    rows=_eligible_history_index(history,available_at=history.catalogue_history_available_at if named else None)
    result={}
    for i,row in enumerate(history.itertuples()):
        if int(row.event_id) not in mapping:continue
        event=mapping[int(row.event_id)]
        if event in targets:
            key=(event,int(row.team_id))
            result[key]=tuple(mapping[int(e)] for e in history.iloc[rows[i]].event_id)
    return result


def plan_recipe_dataset_migration(recipe,*,catalogue_root,snapshots,mapping=None,mode='single_dataset',
                                  profile='retrospective_reported_goals',competitions=None,seasons=None):
    """Return a schema-versioned ready/blocked/unchanged plan without writing anything.

    Mapping contains exact legacy canonical fixture IDs to canonical catalogue IDs.
    Numeric equality is accepted only when the fixture facts also agree. Union and
    extended history are explicit, and applying any population change needs approval.
    Unsupported legacy timing/history overrides block instead of being deleted.
    """
    source_path=str(Path(recipe).resolve()) if isinstance(recipe,(str,Path)) else None
    report=dict(schema=SCHEMA,status='blocked',source_path=source_path,blockers=[],proposed_recipe=None,diff='',population_changes=False)
    try:
        raw=Path(source_path).read_bytes() if source_path else _json(recipe).encode('utf-8')
        original=json.loads(raw.decode('utf-8-sig'));report['source_sha256']=hashlib.sha256(raw).hexdigest();report['original_recipe']=deepcopy(original)
        if not isinstance(original,dict):raise ValueError('Recipe must be a JSON object')
        root=str(Path(catalogue_root).resolve())
        mapping={} if mapping is None else {str(next(iter(_identities([k])))):next(iter(_identities([v]))) for k,v in mapping.items()}
        options=dict(catalogue_root=root,snapshots=dict(snapshots),mapping=mapping,mode=mode,profile=profile,competitions=competitions,seasons=seasons)
        report['options']=options
        from ...ui.recipe import default_recipe,prepare_recipe,validate_recipe
        if original.get('data_selection'):
            current=resolve_selection(original['data_selection']['root'],original['data_selection']['selection']).provenance['data_selection']
            if (Path(original['data_selection']['root']).resolve()!=Path(root) or current['snapshots']!=snapshots or current['mode']!=mode or current['profile']!=profile
                or competitions is not None and competitions!=current['competitions'] or seasons is not None and seasons!=current['seasons']):
                raise ValueError('Existing named selection differs: use the original recipe for a fresh preview; root relocation is a separate explicit operation')
            validate_recipe(original)
            report.update(status='unchanged',proposed_recipe=original,selection_fingerprint=current['fingerprint'])
            return _sealed(report)
        unknown=set(original)-set(default_recipe());known={k:v for k,v in original.items() if k not in unknown}
        required=feature_requirements([known.get('features',{}),known.get('labels',{})])
        for key in ('stat_selection','history','feature_options','ratings','feature_preset','context_features','identity_features'):
            if known.get(key):raise ValueError(f'Legacy {key} requires explicit validated translation; it will not be deleted')
        if known.get('cutoff_hours') is not None:raise ValueError('Legacy cutoff_hours requires an explicit temporal translation')
        if known.get('model',{}).get('component') in ('graphs.TemporalGraphModel','neural.HeatmapCNNModel'):
            raise ValueError('Unsupported graph/spatial migration dependencies')
        # Prepare only: no fit, rating calibration, model update or experiment rebinding.
        before=prepare_recipe(known)
        if before.split_plan.paths is not None:raise ValueError('Path-based split topology requires a validated migration translation')
        matches=before.outputs['matches'];history=before.outputs['history']
        if matches.event_id.duplicated().any():raise ValueError('Ambiguous legacy fixture IDs across source scope; supply repaired canonical identities')
        candidates={}
        for name,version in snapshots.items():
            tables,_,_=load_snapshot(root,name,version)
            for row in tables['matches'].to_dict('records'):candidates.setdefault(int(row['event_id']),[]).append(row)
        mapped={};unresolved=[]
        used=set(map(int,before.dataset.metadata.event_id))|set(map(int,history.loc[history.status.eq('finished'),'event_id']))
        for row in matches.to_dict('records'):
            event=int(row['event_id'])
            if event not in used:continue
            target=mapping.get(str(event),event);choices=candidates.get(target,[])
            if not choices:unresolved.append(event);continue
            day=pd.Timestamp(row['kickoff_utc'],unit='s',tz='UTC').date().isoformat()
            facts=('competition_id','season_id','home_id','away_id')
            if not all(all(int(row[k])==int(c[k]) for k in facts) and c['date']==day for c in choices):
                raise ValueError(f'Fixture {event} to {target} has incompatible participants/competition/season/date; review canonical identity before migration')
            mapped[event]=target
        report['unresolved_mappings']=unresolved
        if unresolved:raise ValueError(f'Missing or excluded canonical fixtures: {unresolved}; exact population cannot be preserved')
        if len(set(mapped.values()))!=len(mapped):raise ValueError('Many legacy fixtures map to one physical fixture')
        original_members=_members(before,mapped);report['original_membership']=original_members
        report['original_summary']=_membership_summary(original_members)
        comp_scope=sorted(map(int,matches.competition_id.unique())) if competitions is None else list(competitions)
        season_scope=(None if mode=='extend_history' else sorted(map(int,matches.season_id.unique()))) if seasons is None else list(seasons)
        kwargs=dict(snapshots=snapshots,mode=mode,requirements=sorted(required),profile=profile,competitions=comp_scope,seasons=season_scope)
        if mode!='union':kwargs.update(target_ids=original_members['target_ids'],fold_membership=original_members['folds'])
        if mode=='single_dataset':kwargs['history_ids']=original_members['history_ids']
        selection=select_data(root,**kwargs)
        proposed=deepcopy(known);proposed['data_selection']=dict(root=root,selection=selection)
        after=prepare_recipe(proposed)
        if mode=='union':
            identity={int(e):int(e) for e in after.outputs['history'].event_id}
            kwargs.update(target_ids=list(dict.fromkeys(map(int,after.dataset.metadata.event_id))),fold_membership=_folds(after,identity))
            selection=select_data(root,**kwargs);proposed['data_selection']['selection']=selection
            after=prepare_recipe(proposed)
        # Timing and retained observations must be unchanged. Extension may add prior
        # observations, but cannot remove or reorder the old eligible subsequence.
        _check_retained_observations(history,after.outputs['history'],mapped,required)
        old_hist=_history_sets(history,set(original_members['target_ids']),mapped)
        new_hist=_history_sets(after.outputs['history'],set(original_members['target_ids']),{int(e):int(e) for e in after.outputs['history'].event_id},named=True)
        for key,old in old_hist.items():
            new=new_hist.get(key)
            if new is None or (new!=old if mode=='single_dataset' else tuple(e for e in new if e in set(old))!=old):
                raise ValueError(f'Historical eligibility changed for fixture/team {key}; explicit timing/data repair required')
        old_cut={mapped[int(r.event_id)]:pd.Timestamp(r.kickoff_at) for r in history.itertuples() if int(r.event_id) in mapped}
        for r in after.dataset.metadata.itertuples():
            if int(r.event_id) in old_cut and pd.Timestamp(r.kickoff_at)!=old_cut[int(r.event_id)]:
                raise ValueError(f'Prediction timing changed for fixture {r.event_id}')
        identity={int(e):int(e) for e in after.outputs['history'].event_id}
        new_members=_members(after,identity);new_members['history_ids']=selection['history_ids'];report['proposed_membership']=new_members
        report['proposed_summary']=_membership_summary(new_members)
        report['population_changes']=any(original_members[k]!=new_members[k] for k in ('target_ids','history_ids','folds'))
        if mode=='single_dataset' and report['population_changes']:raise ValueError('Exact membership/split preservation failed')
        if mode=='single_dataset':
            pd.testing.assert_frame_equal(before.dataset.X.reset_index(drop=True),after.dataset.X.reset_index(drop=True),check_dtype=False,check_exact=True)
            (pd.testing.assert_series_equal if isinstance(before.dataset.y,pd.Series) else pd.testing.assert_frame_equal)(before.dataset.y.reset_index(drop=True),after.dataset.y.reset_index(drop=True),check_dtype=False,check_exact=True)
        proposed=deepcopy(original);proposed['data_selection']=dict(root=root,selection=selection)
        protected={k:proposed.get(k) for k in ('split','split_options','fold_ids','features','labels','target','assembly','data_selection')}
        proposed['dataset_migration']=_sealed(dict(schema=SCHEMA,opaque_settings_hash=digest({k:original[k] for k in unknown}),protected_settings_hash=digest(protected)))
        proposed=json.loads(_json(proposed));validate_recipe(proposed);resolve_selection(root,proposed['data_selection']['selection'])
        report.update(status='ready',proposed_recipe=proposed,selection_fingerprint=selection['fingerprint'],retained_opaque_fields=sorted(unknown),
            diff=''.join(difflib.unified_diff(_json(original).splitlines(True),_json(proposed).splitlines(True),fromfile='original',tofile='proposed')))
    except (ValueError,KeyError,TypeError,FileNotFoundError,AssertionError) as error:
        report['blockers'].append(str(error));report['status']='blocked'
    return _sealed(report)


def apply_recipe_dataset_migration(plan,*,output_path,overwrite=False,approve_population_changes=False):
    """Revalidate dependencies and atomically write a new recipe. Originals stay intact."""
    _verify(plan)
    if plan['status']=='blocked':raise ValueError('Blocked migration: '+'; '.join(plan['blockers']))
    original=plan['source_path'] or plan['original_recipe']
    fresh=plan_recipe_dataset_migration(original,**plan['options'])
    if fresh!=plan:raise ValueError('Migration dependencies changed; create a new preview')
    if plan['status']=='unchanged':return dict(status='unchanged',path=plan['source_path'],fingerprint=plan['selection_fingerprint'])
    if plan['population_changes'] and not approve_population_changes:raise ValueError('Population changes require explicit approval of this preview')
    output=Path(output_path).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    content=_json(plan['proposed_recipe']).encode('utf-8');backup=None;previous=None
    if output.exists():
        if not overwrite:raise FileExistsError('Output exists; overwriting requires an explicit option and backup')
        previous=output.read_bytes();backup=output.with_name(output.name+'.bak-'+hashlib.sha256(previous).hexdigest()[:16])
        if backup.exists() and backup.read_bytes()!=previous:raise ValueError('Backup collision')
        if not backup.exists():
            with backup.open('xb') as stream:stream.write(previous);stream.flush();os.fsync(stream.fileno())
    fd,tmp=tempfile.mkstemp(dir=output.parent,prefix='.recipe-migration-')
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(content);stream.flush();os.fsync(stream.fileno())
        if overwrite:
            if output.exists() and (previous is None or output.read_bytes()!=previous):raise ValueError('Output changed during migration')
            os.replace(tmp,output)
        else:os.link(tmp,output)
    finally:Path(tmp).unlink(missing_ok=True)
    return dict(status='written',path=str(output),backup=None if backup is None else str(backup),fingerprint=plan['selection_fingerprint'])


def main(argv=None):
    p=argparse.ArgumentParser(description='Preview explicit named-dataset recipe migrations (no fitting)')
    p.add_argument('inputs',nargs='+');p.add_argument('--catalogue-root',required=True);p.add_argument('--snapshot',action='append',required=True,help='dataset=exact-version')
    p.add_argument('--mapping');p.add_argument('--mode',default='single_dataset',choices=['single_dataset','union','extend_history'])
    p.add_argument('--profile',default='retrospective_reported_goals',choices=['retrospective_reported_goals','strict_as_of'])
    p.add_argument('--output-dir');p.add_argument('--apply',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--overwrite',action='store_true');p.add_argument('--approve-population-changes',action='store_true');a=p.parse_args(argv)
    if a.apply and a.dry_run:p.error('Choose apply or dry-run')
    if a.apply and not a.output_dir:p.error('--apply requires --output-dir')
    snapshots=dict(value.split('=',1) for value in a.snapshot)
    mapping=None if not a.mapping else json.loads(Path(a.mapping).read_text(encoding='utf-8'))
    files=[]
    for item in a.inputs:
        path=Path(item).resolve()
        files.extend(sorted(path.glob('*.json')) if path.is_dir() else [path])
    if not files:p.error('No recipe files found')
    reports=[]
    for path in dict.fromkeys(files):
        plan=plan_recipe_dataset_migration(path,catalogue_root=a.catalogue_root,snapshots=snapshots,mapping=mapping,mode=a.mode,profile=a.profile)
        name=path.stem+'.'+hashlib.sha256(str(path).encode()).hexdigest()[:8]+'.named.json'
        output=None if not a.output_dir else str(Path(a.output_dir).resolve()/name)
        result=dict(source=str(path),status=plan['status'],output=output,plan=plan)
        if a.apply and plan['status']!='blocked':
            try:result.update(apply_recipe_dataset_migration(plan,output_path=output,overwrite=a.overwrite,approve_population_changes=a.approve_population_changes))
            except (ValueError,OSError) as error:result.update(status='blocked',error=str(error))
        reports.append(result)
    print(_json(dict(schema=SCHEMA,dry_run=not a.apply,recipes=reports)))
    return 2 if any(r['status']=='blocked' for r in reports) else 0

if __name__=='__main__':raise SystemExit(main())
