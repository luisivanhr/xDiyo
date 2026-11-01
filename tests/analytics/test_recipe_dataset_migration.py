"""Dry-run/apply migrations preserve original recipes, fixed populations and folds."""
from copy import deepcopy
import json
from pathlib import Path
import pandas as pd
import pytest
from test_ui_workflow import ui_recipe
from xdiyo_analytics.data.catalogue import *
from xdiyo_analytics.data.catalogue.native import seed_comprehensive
from xdiyo_analytics.ui.recipe import node,prepare_recipe,read_recipe


@pytest.fixture
def migration_case(tmp_path,ui_recipe):
    recipe=deepcopy(ui_recipe)
    recipe['features']={'goals':node('features.RollingMean',source=node('features.MatchScore'),window=3)}
    recipe['labels']={'goals':node('labels.Outcome',perspective='home')};recipe['target']='goals'
    r,tables,_=seed_comprehensive(recipe['data']['data_root'],reviewed_competitions=())
    for team in r.payload['entities']['team']:r.classify(int(team),'not_reserve',decision='reviewed synthetic senior')
    catalogue=tmp_path/'catalogue';version=publish_dataset(catalogue,'comprehensive_dataset',tables,r)
    path=tmp_path/'original.json';path.write_text(json.dumps(recipe,indent=2),encoding='utf-8')
    return path,catalogue,{'comprehensive_dataset':version},r,tables


def test_migration_preview_new_file_reload_and_idempotence(migration_case,tmp_path):
    path,root,snapshots,_,_=migration_case;raw=path.read_bytes()
    plan=plan_recipe_dataset_migration(path,catalogue_root=root,snapshots=snapshots)
    assert plan['status']=='ready',plan['blockers']
    assert plan['original_membership']==plan['proposed_membership']
    assert not plan['population_changes'] and plan['diff']
    destination=tmp_path/'new.json'
    assert not destination.exists() and path.read_bytes()==raw
    result=apply_recipe_dataset_migration(plan,output_path=destination)
    assert result['status']=='written' and path.read_bytes()==raw
    before=prepare_recipe(read_recipe(path));after=prepare_recipe(read_recipe(destination))
    pd.testing.assert_frame_equal(before.dataset.X,after.dataset.X,check_dtype=False)
    for a,b in zip(before.split_plan.folds,after.split_plan.folds):
        for role in ('train','test','score'):assert (getattr(a,role)==getattr(b,role)).all()
    again=plan_recipe_dataset_migration(destination,catalogue_root=root,snapshots=snapshots)
    assert again['status']=='unchanged',again['blockers']
    assert apply_recipe_dataset_migration(again,output_path=tmp_path/'unused.json')['status']=='unchanged'
    assert not (tmp_path/'unused.json').exists()


def test_opaque_settings_hyperparameters_and_backup(migration_case,tmp_path):
    path,root,snapshots,_,_=migration_case;r=json.loads(path.read_text());r['research_note']={'large_id':2**63+100,'text':'keep'};r['model']['params']['alpha']=0.47
    path.write_text(json.dumps(r),encoding='utf-8');raw=path.read_bytes()
    plan=plan_recipe_dataset_migration(path,catalogue_root=root,snapshots=snapshots)
    assert plan['status']=='ready',plan['blockers']
    assert plan['proposed_recipe']['research_note']==r['research_note'] and plan['proposed_recipe']['model']==r['model']
    with pytest.raises(FileExistsError):apply_recipe_dataset_migration(plan,output_path=path)
    result=apply_recipe_dataset_migration(plan,output_path=path,overwrite=True)
    assert Path(result['backup']).read_bytes()==raw
    loaded=read_recipe(path);assert loaded['research_note']==r['research_note']
    loaded['research_note']['text']='changed'
    from xdiyo_analytics.ui.recipe import validate_recipe
    with pytest.raises(ValueError,match='Preserved'):validate_recipe(loaded)


@pytest.mark.parametrize('setting,value', [('cutoff_hours',12),('history',{'stat_fields':['value']}),('feature_options',{'group_by':['team_id']})])
def test_incompatible_overrides_block_without_deletion(migration_case,setting,value):
    path,root,snapshots,_,_=migration_case;r=json.loads(path.read_text());r[setting]=value
    plan=plan_recipe_dataset_migration(r,catalogue_root=root,snapshots=snapshots)
    assert plan['status']=='blocked' and setting in plan['blockers'][0]
    assert plan['original_recipe'][setting]==value


def test_missing_catalogue_reserve_losses_and_plan_tampering(migration_case,tmp_path):
    path,root,snapshots,r,tables=migration_case
    bad=plan_recipe_dataset_migration(path,catalogue_root=tmp_path/'absent',snapshots=snapshots)
    assert bad['status']=='blocked'
    removed=tables['matches'].iloc[1:].copy();version=publish_dataset(root,'comprehensive_dataset',dict(tables,matches=removed),r) if not tables['statistics'].shape[0] else None
    assert version
    plan=plan_recipe_dataset_migration(path,catalogue_root=root,snapshots={'comprehensive_dataset':version})
    assert plan['status']=='blocked' and plan['unresolved_mappings']
    ready=plan_recipe_dataset_migration(path,catalogue_root=root,snapshots=snapshots)
    ready['proposed_recipe']['model']['params']['alpha']=9
    with pytest.raises(ValueError,match='plan changed'):apply_recipe_dataset_migration(ready,output_path=tmp_path/'bad.json')


def test_strict_unknown_availability_and_stale_source_block(migration_case,tmp_path):
    path,root,snapshots,_,_=migration_case
    strict=plan_recipe_dataset_migration(path,catalogue_root=root,snapshots=snapshots,profile='strict_as_of')
    assert strict['status']=='blocked'
    plan=plan_recipe_dataset_migration(path,catalogue_root=root,snapshots=snapshots)
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='dependencies changed'):apply_recipe_dataset_migration(plan,output_path=tmp_path/'new.json')


@pytest.mark.parametrize('mode',['union','extend_history'])
def test_explicit_population_preview_requires_approval(migration_case,tmp_path,mode):
    path,root,snapshots,_,_=migration_case
    recipe=json.loads(path.read_text());recipe['data']['seasons']=['23_24','24_25']
    recipe['split']['params']['train_size']=1
    path.write_text(json.dumps(recipe),encoding='utf-8')
    plan=plan_recipe_dataset_migration(path,catalogue_root=root,snapshots=snapshots,mode=mode,
        seasons=None if mode=='extend_history' else [202223,202324,202425])
    assert plan['status']=='ready',plan['blockers']
    assert plan['population_changes']
    if mode=='extend_history':
        assert plan['original_membership']['target_ids']==plan['proposed_membership']['target_ids']
        assert plan['original_membership']['folds']==plan['proposed_membership']['folds']
    with pytest.raises(ValueError,match='explicit approval'):
        apply_recipe_dataset_migration(plan,output_path=tmp_path/'expanded.json')
    result=apply_recipe_dataset_migration(plan,output_path=tmp_path/'expanded.json',approve_population_changes=True)
    assert result['status']=='written'


def test_paired_rows_and_multiple_folds_preserved(migration_case):
    path,root,snapshots,_,_=migration_case
    recipe=json.loads(path.read_text());recipe['assembly']={'layout':'team_match','drop_missing_targets':True};recipe['labels']['goals']=node('labels.Outcome',perspective='team')
    recipe['split']['params']['train_size']=1
    plan=plan_recipe_dataset_migration(recipe,catalogue_root=root,snapshots=snapshots)
    assert plan['status']=='ready',plan['blockers']
    assert len(plan['original_membership']['folds'])==2
    before=prepare_recipe(recipe);after=prepare_recipe(plan['proposed_recipe'])
    assert len(after.dataset.X)==46
    for a,b in zip(before.split_plan.folds,after.split_plan.folds):
        for role in ('train','test','score'):assert (getattr(a,role)==getattr(b,role)).all()
    assert recipe['split']==plan['proposed_recipe']['split']


def test_batch_cli_dry_run_reports_each_recipe(migration_case,tmp_path,capsys):
    from xdiyo_analytics.data.catalogue.migration import main
    path,root,snapshots,_,_=migration_case
    directory=tmp_path/'batch';directory.mkdir()
    original=path.read_bytes();(directory/'ready.json').write_bytes(original)
    plan=plan_recipe_dataset_migration(path,catalogue_root=root,snapshots=snapshots)
    (directory/'unchanged.json').write_text(json.dumps(plan['proposed_recipe']),encoding='utf-8')
    bad=json.loads(original);bad['history']={'unsupported':True}
    (directory/'blocked.json').write_text(json.dumps(bad),encoding='utf-8')
    assert main([str(directory),'--catalogue-root',str(root),'--snapshot','comprehensive_dataset='+snapshots['comprehensive_dataset']])==2
    report=json.loads(capsys.readouterr().out)
    assert report['dry_run'] and {r['status'] for r in report['recipes']}=={'ready','blocked','unchanged'}
    assert (directory/'ready.json').read_bytes()==original and len(list(directory.glob('*')))==3


def test_reserve_and_coverage_losses_and_root_change_block(migration_case,tmp_path):
    path,root,snapshots,r,tables=migration_case
    payload=deepcopy(r.payload)
    for d in payload['squads']:d['classification']='reserve_or_development_squad'
    version=publish_dataset(root,'comprehensive_dataset',tables,Registry(payload))
    plan=plan_recipe_dataset_migration(path,catalogue_root=root,snapshots={'comprehensive_dataset':version})
    assert plan['status']=='blocked' and plan['unresolved_mappings']
    recipe=json.loads(path.read_text());recipe['features']={'corners':node('features.RollingMean',source=node('features.Stat',period='ALL',group='Match overview',key='cornerKicks'),window=3)}
    plan=plan_recipe_dataset_migration(recipe,catalogue_root=root,snapshots=snapshots)
    assert plan['status']=='blocked' and 'ineligible' in plan['blockers'][0]
    ready=plan_recipe_dataset_migration(path,catalogue_root=root,snapshots=snapshots)
    relocated=plan_recipe_dataset_migration(ready['proposed_recipe'],catalogue_root=tmp_path/'other',snapshots=snapshots)
    assert relocated['status']=='blocked' and 'root relocation' in relocated['blockers'][0]



def test_expansion_cannot_silently_change_retained_scores(migration_case):
    path,root,snapshots,r,tables=migration_case
    altered=tables['matches'].copy();altered.loc[0,'home_score_current']=99
    version=publish_dataset(root,'comprehensive_dataset',dict(tables,matches=altered),r)
    plan=plan_recipe_dataset_migration(path,catalogue_root=root,snapshots={'comprehensive_dataset':version},mode='union')
    assert plan['status']=='blocked' and 'observations changed' in plan['blockers'][0]



def test_invalid_json_object_is_a_batch_blocker(migration_case,tmp_path):
    path,root,snapshots,_,_=migration_case
    path.write_text('null',encoding='utf-8')
    result=plan_recipe_dataset_migration(path,catalogue_root=root,snapshots=snapshots)
    assert result['status']=='blocked' and 'JSON object' in result['blockers'][0]
