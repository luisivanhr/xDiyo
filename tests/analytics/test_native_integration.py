from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from xdiyo_analytics.data import load_season, supervised_native_rows
from xdiyo_analytics.data.native_integration import POLICY, native_rule, validate_integration
from xdiyo_analytics.data.native_publication import publish_integration
from xdiyo_analytics.ui import default_recipe, refresh_recipe_data

STEM='Premier_League_24_25'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()


def setup_native(export):
    table=export.table('matches')
    for name, values in dict(home_name=['Source Home','Other'],away_name=['Source Away','Away'],
            tournament_id=[1,1],round=[1,2],is_awarded=[False,False],status=['finished','finished'],
            home_score_current=[0.,2.],away_score_current=[1.,1.]).items():
        table=table.append_column(name,pa.array(values))
    export.replace('matches',table)
    r=table.to_pylist()[0]
    metadata=dict(policy=POLICY,competition=dict(id=17,name='Premier League'),
        season=dict(id=61627,name='Premier League 24/25',year='24/25'),
        teams={str(r['home_id']):dict(id=r['home_id'],name='Verified Home')},tournaments={'1':dict(id=1,name='Premier League')},
        rules=dict(enforce=True,standings=dict(status='unsupported',reason='Unverified table'),
                   countdown=dict(status='supported',format='double',total_rounds=38)),
        previous_revision=dict(git_commit='test-fixture',manifest_sha256=sha(export.manifest_path),
                               tables=export.manifest()['tables']),
        rulings=[dict(event_id=r['event_id'],home_id=r['home_id'],away_id=r['away_id'],kickoff_utc=r['kickoff_utc'],
                      is_awarded=True,play_status='played_administrative_award',administrative_home_score=0,
                      administrative_away_score=3,source='reviewed test ruling')])
    return metadata


def publish(export, metadata):
    return publish_integration(export.root,STEM,metadata,expected_manifest=sha(export.manifest_path),
                               expected_publication=sha(export.publication_path))


def test_names_awards_and_children_from_same_pin_without_rewriting_tables(export,tmp_path):
    metadata=setup_native(export);original=export.table('matches')
    hashes={n:info['sha256'] for n,info in export.manifest()['tables'].items()}
    pin=tmp_path/'old.json';load_season(export.root,STEM,record_path=pin)
    before=export.manifest_path.read_bytes()
    publish(export,metadata)
    data=load_season(export.root,STEM,include_awarded=True,verify_hashes=True)
    r=data.matches.iloc[0]
    assert r.home_name=='Source Home' and r.home_display_name=='Verified Home'
    assert r.is_awarded and not r.provider_is_awarded
    assert r.away_score_current==1 and r.administrative_away_score==3
    assert data.matches.competition_name.eq('Premier League').all()
    assert export.table('matches').equals(original)
    assert hashes=={n:info['sha256'] for n,info in export.manifest()['tables'].items()}
    assert len(load_season(export.root,STEM).matches)==1
    assert load_season(export.root,STEM,tables='statistics').statistics.event_id.eq(r.event_id).sum()==0
    with pytest.raises(ValueError,match='manifest has changed'):load_season(export.root,STEM,record_path=pin)
    # Restoring the exact former metadata reproduces the old pin and values.
    export.manifest_path.write_bytes(before)
    assert len(load_season(export.root,STEM,record_path=pin).matches)==2


def test_publication_interruption_recovers_and_rejects_changed_dependencies(export,monkeypatch):
    metadata=setup_native(export);old=sha(export.manifest_path);pub=sha(export.publication_path)
    replace=Path.replace
    def interrupt(path,target):
        if Path(target)==export.publication_path:raise OSError('interrupted')
        return replace(path,target)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'replace',interrupt)
        with pytest.raises(OSError,match='interrupted'):
            publish_integration(export.root,STEM,metadata,expected_manifest=old,expected_publication=pub)
    publish_integration(export.root,STEM,metadata,expected_manifest=old,expected_publication=pub)
    assert json.loads(export.publication_path.read_bytes())['native_integration']==metadata
    wrong=deepcopy(metadata);wrong['season']['name']='Changed'
    with pytest.raises(ValueError,match='concurrently'):
        publish_integration(export.root,STEM,wrong,expected_manifest=old,expected_publication=pub)


@pytest.mark.parametrize('field',['home_id','away_id','kickoff_utc'])
def test_awarded_evidence_cannot_rebind_fixture(export,field):
    metadata=setup_native(export);metadata['rulings'][0][field]+=1;publish(export,metadata)
    with pytest.raises(ValueError,match='identity changed'):load_season(export.root,STEM)


def test_scope_and_unsupported_rules_fail_instead_of_inventing_tables(export):
    metadata=setup_native(export)
    wrong=deepcopy(metadata);wrong['season']['id']=123
    with pytest.raises(ValueError,match='scope mismatch'):validate_integration(wrong,export.manifest()['scope'])
    publish(export,metadata)
    frame=load_season(export.root,STEM,include_awarded=True).matches
    with pytest.raises(ValueError,match='Unverified table'):native_rule(frame,'standings')
    assert native_rule(frame,'countdown')['total_rounds']==38


def test_supervised_explicit_availability_future_and_unfinished():
    frame=pd.DataFrame(dict(kickoff_utc=[1704067200.]*5,status=['finished']*4+['inprogress'],
        home_score_current=[1,1,1,None,1],away_score_current=[0]*5,
        result_available_at=['2024-01-03T00:00:00Z',None,None,None,None]))
    releases=pd.Series(pd.to_datetime(['2024-01-02T00:00:00Z','2024-01-02T00:00:00Z',None,'2024-01-02T00:00:00Z','2024-01-02T00:00:00Z']))
    mask=supervised_native_rows(frame,cutoff='2024-01-02T00:00:00Z',available_at=releases)
    assert mask.tolist()==[False,True,False,False,False]
    changed=frame.copy();changed.loc[0,'home_score_current']=1000
    assert supervised_native_rows(changed,cutoff='2024-01-02T00:00:00Z',available_at=releases).equals(mask)
    with pytest.raises(ValueError,match='timezone-aware'):
        supervised_native_rows(frame,cutoff='2024-01-02',available_at=releases)


def test_refresh_metadata_only_persists_review_and_does_not_write_during_preview(export,tmp_path):
    metadata=setup_native(export);old_dir=tmp_path/'old'
    load_season(export.root,STEM,record_path=old_dir/(STEM+'.json'))
    old=(old_dir/(STEM+'.json')).read_bytes();publish(export,metadata)
    recipe=default_recipe(str(export.root));recipe['data'].update(seasons=['24_25'],record_dir=str(old_dir),tables=['matches'])
    path=tmp_path/'v2.json'
    preview=refresh_recipe_data(recipe,path,dry_run=True)
    assert not path.exists() and preview['changes'][0]['change_kind']=='manifest_only'
    result=refresh_recipe_data(recipe,path)
    report=json.loads(Path(result['report_path']).read_text())
    assert report['changes'][0]['previous_dataset_revision']=='test-fixture'
    assert report['changes'][0]['changed_tables']==[]
    assert (old_dir/(STEM+'.json')).read_bytes()==old
    assert len(load_season(export.root,STEM,record_path=Path(result['recipe']['data']['record_dir'])/(STEM+'.json')).matches)==1


def test_refresh_fails_on_missing_pin_and_invalid_current_inputs(export,tmp_path):
    recipe=default_recipe(str(export.root));recipe['data'].update(seasons=['24_25'],record_dir=str(tmp_path/'missing'))
    with pytest.raises(FileNotFoundError,match='previous selection'):refresh_recipe_data(recipe,tmp_path/'new.json')
    recipe['data']['record_dir']=None
    table=export.version/'matches.parquet';table.write_bytes(table.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='fingerprint'):refresh_recipe_data(recipe,tmp_path/'new.json')


def test_refresh_substantive_changes_and_ambiguous_pin(export,tmp_path):
    metadata=setup_native(export);records=tmp_path/'pins'
    load_season(export.root,STEM,record_path=records/(STEM+'.json'))
    table=export.table('matches').set_column(0,'event_id',pa.array([123,7]))
    export.replace('matches',table)
    publish(export,metadata)
    # The evidence binding correctly rejects a changed fixture during refresh.
    recipe=default_recipe(str(export.root));recipe['data'].update(seasons=['24_25'],record_dir=str(records),tables=['matches'])
    with pytest.raises(ValueError,match='missing or duplicated'):refresh_recipe_data(recipe,tmp_path/'new.json')
    export.change_manifest(lambda m:m['native_integration'].update(rulings=[]))
    result=refresh_recipe_data(recipe,tmp_path/'new.json')
    assert result['changes'][0]['change_kind']=='substantive'
    pin=records/(STEM+'.json');p=json.loads(pin.read_bytes());p['season_stem']='Other_24_25';pin.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='another season'):refresh_recipe_data(recipe,tmp_path/'other.json')


def test_physical_award_repair_is_recoverable_and_preserves_played_score(export,monkeypatch):
    import pyarrow.parquet as pq
    from xdiyo_analytics.data.native_publication import materialize_awarded_rulings
    metadata=setup_native(export);original=export.table('matches')
    nested=export.root/(STEM+'.parquet');pq.write_table(original,nested,compression='zstd')
    publish(export,metadata)
    replace=Path.replace
    def interrupt(path,target):
        if Path(target)==export.manifest_path:raise OSError('interrupted four-file publication')
        return replace(path,target)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'replace',interrupt)
        with pytest.raises(OSError,match='four-file'):
            materialize_awarded_rulings(export.root,STEM)
    result=materialize_awarded_rulings(export.root,STEM)
    assert result['changed']==[original['event_id'][0].as_py()]
    for table in (export.table('matches'),pq.ParquetFile(nested).read()):
        assert table['is_awarded'].to_pylist()==[True,False]
        assert table['provider_is_awarded'].to_pylist()==[False,False]
        columns=[n for n in original.column_names if n!='is_awarded']
        assert table.select(columns).equals(original.select(columns))
    assert len(load_season(export.root,STEM,verify_hashes=True).matches)==1
    assert materialize_awarded_rulings(export.root,STEM)['changed']==[]
    assert not list(export.root.rglob('*.awards-tmp'))


def test_ui_discovery_and_team_choices_keep_keys_but_show_sofascore_names(export,tmp_path):
    from xdiyo_analytics.ui.server import BuilderState
    from xdiyo_analytics.ui.recipe import catalog_for_ui
    metadata=setup_native(export);publish(export,metadata)
    state=BuilderState(tmp_path,catalog_for_ui())
    try:
        row=state.dispatch('discover',{'root':str(export.root)})['publications'][0]
        assert row['league']=='Premier_League' and row['league_label']=='Premier League'
        assert row['season_label']=='Premier League 24/25'
        assert row['experiment_season']=='2024/2025'
        choices=state.dispatch('choices',{'root':str(export.root),'seasons':['24_25']})
        assert 'Verified Home' in [t['label'] for t in choices['teams']]
    finally:state.executor.shutdown()
