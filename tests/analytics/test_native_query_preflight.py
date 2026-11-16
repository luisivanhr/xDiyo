from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from xdiyo_analytics.data import NativeLibrary, AmbiguousNativeName, load_seasons
from xdiyo_analytics.data.native_preflight import preflight_loaded, prepare_native_history
from xdiyo_analytics.features import SeasonCountdown, DerivedStandings, evaluate_features
from xdiyo_analytics.features.countdown import season_countdown
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.ui import default_recipe, prepare_recipe, refresh_recipe_data
from xdiyo_analytics.ui.recipe import node

STEM='Premier_League_24_25'


def publication(export, meetings=2, phases=(1,), partial=False):
    rows=[]; rounds=[[(1,4),(2,3)],[(1,3),(4,2)],[(1,2),(3,4)]]
    for phase_index,tid in enumerate(phases):
        for leg in range(meetings):
            for rnd,pairs in enumerate(rounds,1):
                for home,away in pairs:
                    if leg%2:home,away=away,home
                    n=len(rows);kick=pd.Timestamp('2024-08-01',tz='UTC')+pd.Timedelta(days=phase_index*90+(leg*3+rnd)*7)
                    rows.append(dict(event_id=100+n,competition_id=17,season_id=61627,tournament_id=tid,
                        home_id=home,away_id=away,home_name='Old '+str(home),away_name='Old '+str(away),round=leg*3+rnd,
                        kickoff_utc=kick.timestamp(),status='finished',is_awarded=False,
                        home_score_current=n%3,away_score_current=0,result_available_at=(kick+pd.Timedelta(hours=3)).isoformat()))
    if partial:rows=rows[2:]
    export.replace('matches',pa.Table.from_pylist(rows))
    roster=[dict(competition_id=17,season_id=61627,team_id=i,team_name='Team '+str(i),movement='retained',
                 got_promoted=False,got_demoted=False,previous_competition_id=None,previous_season_id=None,evidence='test roster') for i in range(1,5)]
    export.replace('team_seasons',pa.Table.from_pylist(roster))
    evidence=dict(observed_at=2,sha256='a'*64,event_id=100)
    meta=dict(policy='sofascore_native_integration_v1',competition=dict(id=17,name='Premier League'),
        season=dict(id=61627,name='Premier League 24/25',year='24/25'),
        evidence=dict(competition=evidence,season=evidence),
        teams={str(i):dict(id=i,name='Team '+str(i),aliases=['Historic '+str(i)],evidence=evidence) for i in range(1,5)},
        tournaments={str(t):dict(id=t,name='Apertura' if j==0 else 'Clausura') for j,t in enumerate(phases)},
        rules=dict(enforce=True,countdown=dict(status='unsupported',reason='old blanket unsupported'),standings=dict(status='unsupported',reason='old blanket unsupported')),
        coverage=dict(kind='completed_match_snapshot' if partial else 'published_scope'),availability={})
    calendar=dict(policy='native_annual_calendar_v1',competition_id=17,season_id=61627,annual_league='Annual_Test',annual_season='2024/2025',phase='regular',half=None,
        tournaments={str(t):dict(phase='Apertura' if j==0 else 'Clausura',half=j+1) for j,t in enumerate(phases)})
    export.change_manifest(lambda m:m.update(native_integration=meta,season_calendar=calendar))
    p=json.loads(export.publication_path.read_bytes());p.update(matches=len(rows),native_integration=meta,season_calendar=calendar)
    cohort=dict(schedule_verified=True,format_source='https://example.test/verified-rules',selection='all',regular_matches=6*meetings,snapshot=partial)
    if len(phases)>1:
        cohort.update(selection='reviewed_ordinary_phases',reviewed_phases=[dict(tournament_id=t,clubs=4,round_mode='strict',table_ids=[t+1000],pair_multiplicities={'1':6}) for t in phases])
    p['reconciliation']=dict(archive_cohort=cohort)
    export.publication_path.write_text(json.dumps(p),encoding='utf8')
    return NativeLibrary(export.root)


def recipe(export,query=None):
    r=default_recipe(str(export.root))
    r['data']=dict(data_root=str(export.root),query=query or dict(competitions=17,seasons=61627),tables=['matches','team_seasons'])
    r.update(features={'rank':node('features.DerivedStandings'),'clock':node('features.SeasonCountdown')},
             labels={'outcome':node('labels.Outcome',perspective='home')},target='outcome',
             split=node('splits.TemporalSplit',train_size=2,test_size=1,unit='rounds'))
    return r


def test_all_entities_resolve_ids_names_aliases_without_changing_keys(export):
    lib=publication(export,1,(1,2));before=export.snapshot()
    query=dict(competitions='Premier League',seasons='24/25',phases='Clausura',teams='Historic 1')
    selection=lib.select(**query)
    assert selection==lib.select(**query)
    assert selection['team_ids']==[1] and selection['phase_ids']==[[17,2]]
    assert lib.discover('competitions')[0]['id']==17
    assert lib.discover('seasons',competitions='Premier_League')[0]['id']==61627
    assert len(lib.discover('phases',competitions=17,seasons=61627))==2
    assert lib.discover('annual_groups')[0]['id']=='Annual_Test:2024/2025'
    d=load_seasons(export.root,query=query,tables=['matches','team_seasons'])
    assert len(d.matches)==6 and d.matches.native_target.sum()==3
    assert d.matches.tournament_id.eq(2).all() and d.matches.annual_phase.eq('Clausura').all()
    assert d.matches.source_league.eq('Premier_League').all()
    assert export.snapshot()==before
    numeric=lib.select(competitions=np.int64(17),seasons=(np.int64(61627),))
    assert numeric==json.loads(json.dumps(numeric))
    assert lib.validate_selection(json.loads(json.dumps(numeric)))==numeric


def test_duplicate_names_return_candidates_and_conflicting_evidence(export):
    publication(export)
    def change(m):
        m['native_integration']['teams']['2']['name']='Team 1'
    export.change_manifest(change)
    lib=NativeLibrary(export.root)
    with pytest.raises(AmbiguousNativeName) as exc:lib.select(teams='Team 1')
    assert {r['id'] for r in exc.value.candidates}=={1,2}
    assert lib.select(teams=1)['team_ids']==[1]
    lib._add('teams',1,'Renamed',[],dict(observed_at=2,sha256='b'*64),STEM,id=1)
    assert lib.discover('teams',teams=1)[0]['evidence_status']=='conflicting'
    with pytest.raises(AmbiguousNativeName):lib.select(teams='Renamed')


def test_name_changes_keep_old_aliases_and_missing_evidence_is_explicit(export):
    lib=publication(export)
    lib._add('teams',1,'New Team',[],dict(observed_at=3,sha256='b'*64),STEM,id=1)
    assert lib.discover('teams',teams='Team 1')[0]['name']=='New Team'
    loaded=lib.load()
    assert loaded.matches.loc[loaded.matches.home_id.eq(1),'home_display_name'].eq('New Team').all()
    assert loaded.matches.loc[loaded.matches.home_id.eq(1),'home_name'].eq('Old 1').all()
    export.change_manifest(lambda m:m['native_integration']['teams'].pop('4'))
    row=NativeLibrary(export.root).discover('teams',teams=4)[0]
    assert row['name'] is None and row['evidence_status']=='missing' and 'Old 4' in row['aliases']


def test_selection_rejects_manifest_change_tampering_and_stale_index(export,tmp_path):
    lib=publication(export);pin=lib.select(competitions=17);tampered=deepcopy(pin)
    tampered['team_ids']=[2]
    with pytest.raises(ValueError,match='tampered'):lib.load(selection=tampered)
    export.change_manifest(lambda m:m.update(note='metadata-only'))
    with pytest.raises(ValueError,match='catalogue changed'):lib.load(selection=pin)
    with pytest.raises(ValueError,match='pins changed'):NativeLibrary(export.root).load(selection=pin)
    r=recipe(export);r['data']['native_selection']=pin
    result=refresh_recipe_data(r,tmp_path/'new.json')
    assert result['recipe']['data']['native_selection']!=pin
    assert r['data']['native_selection']==pin
    assert json.loads((tmp_path/'new_refresh.json').read_bytes())['native_selection']['previous']==pin
    r['data']['native_selection']=tampered
    with pytest.raises(ValueError,match='tampered'):refresh_recipe_data(r,tmp_path/'tampered.json')


@pytest.mark.parametrize('meetings',[1,2,3,4])
@pytest.mark.parametrize('partial',[False,True])
def test_verified_regular_counts_partial_round_one_and_standings(export,meetings,partial):
    lib=publication(export,meetings,partial=partial)
    d=lib.load(tables=['matches','team_seasons']);h=build_team_history(d)
    result=season_countdown(d.matches,SeasonCountdown(),team_seasons=d['team_seasons'])
    expected=(meetings*3-d.matches['round'].to_numpy(dtype=float))/(meetings*3-1)
    np.testing.assert_allclose(result,expected)
    values=evaluate_features(h,{'rank':DerivedStandings(normalized_ranks=(1,7))},team_seasons=d['team_seasons'])
    assert np.isfinite(values.to_numpy(dtype=float)).all()
    assert preflight_loaded(d,required_features=['countdown','standings'])['cohorts'][0]['blockers']==[]


def test_independent_phase_reset_and_target_history_preservation(export):
    lib=publication(export,1,(1,2));d=lib.load(query=dict(teams=1),tables=['matches','team_seasons'])
    h=build_team_history(d)
    v=evaluate_features(h,{'rank':DerivedStandings(normalized_ranks=(5,))},team_seasons=d['team_seasons'])
    opening=h['round'].eq(1)
    assert np.allclose(v.loc[opening].to_numpy(dtype=float),1)
    clock=season_countdown(d.matches,SeasonCountdown(),team_seasons=d['team_seasons'])
    assert clock.loc[d.matches['round'].eq(1)].eq(1).all()
    prepared=prepare_recipe(recipe(export,dict(competitions=17,seasons=61627,phases='Apertura',teams=1)))
    assert len(prepared.dataset.X)==3 and len(prepared.outputs['history'])==12
    assert prepared.dataset.definitions['native_preprocessing']['policy']=='native_scoped_preprocessing_v1'


def test_future_administrative_and_delayed_results_preflight_tminus_one(export):
    publication(export)
    rows=export.table('matches').to_pylist();rows[-1]['status']='notstarted';rows[-2]['is_awarded']=True
    rows[-3]['result_available_at']='2030-01-01T00:00:00+00:00'
    export.replace('matches',pa.Table.from_pylist(rows))
    lib=NativeLibrary(export.root);d=lib.load(tables=['matches','team_seasons'],include_awarded=True)
    report=preflight_loaded(d,cutoff='2025-01-01T00:00:00Z',options={'availability':'explicit'})
    assert report['cohorts'][0]['counts']['eligible_targets']==9
    assert report['cohorts'][0]['standings']['status']=='unsupported'
    h=prepare_native_history(d,build_team_history(d),{'availability':'explicit'})
    assert h.loc[h.event_id.eq(rows[-2]['event_id']),'status'].eq('excluded_non_sporting_result').all()
    r=recipe(export);r['features']={'is_home':node('features.IsHome')};r['cutoff_hours']=24
    r['native_preprocessing']={'availability':'explicit'}
    prepared=prepare_recipe(r)
    assert not set(prepared.dataset.metadata.event_id)&{rows[-1]['event_id'],rows[-2]['event_id']}
    for fold in prepared.split_plan.folds:
        train=prepared.dataset.metadata.iloc[fold.train];test=prepared.dataset.metadata.iloc[fold.test]
        assert pd.to_datetime(train.result_available_at,utc=True).max()<=test.kickoff_at.min()-pd.Timedelta(days=1)


def test_missing_availability_and_strict_schedule_do_not_pass(export):
    lib=publication(export);d=lib.load(tables=['matches','team_seasons'])
    report=preflight_loaded(d,required_features=['countdown','movements'],options={'availability':'explicit','entry_information':'explicit'})
    assert any('schedule-at-issuance' in b for b in report['cohorts'][0]['blockers'])
    assert any('publication times' in b for b in report['cohorts'][0]['blockers'])
    d.matches['result_available_at']=pd.NaT
    with pytest.raises(ValueError,match='unknown result'):prepare_native_history(d,build_team_history(d),{'availability':'explicit'})


def test_legacy_loader_does_not_apply_query_policy(export):
    publication(export)
    legacy=load_seasons(export.root,'24_25',tables=['matches','team_seasons'])
    assert 'native_selection' not in legacy.provenance and 'standings_table_id' not in legacy.matches
    with pytest.raises(ValueError,match='not both'):load_seasons(export.root,'24_25',query={'competitions':17})


def test_ui_native_routes_share_resolver(export,tmp_path):
    publication(export)
    from xdiyo_analytics.ui.server import BuilderState
    from xdiyo_analytics.ui.recipe import catalog_for_ui
    state=BuilderState(tmp_path,catalog_for_ui())
    try:
        request=dict(root=str(export.root),query=dict(competitions='Premier League'))
        assert state.dispatch('native-select',request)==NativeLibrary(export.root).select(**request['query'])
        assert state.dispatch('native-discover',{**request,'kind':'teams'})['entities'][0]['name']=='Team 1'
        assert state.dispatch('native-preflight',request)['cohorts'][0]['competition_id']==17
    finally:state.executor.shutdown()


def test_unknown_phase_stays_discoverable_without_fabricated_id(export):
    publication(export)
    rows=export.table('matches').to_pylist();rows[0]['tournament_id']=None
    export.replace('matches',pa.Table.from_pylist(rows))
    export.change_manifest(lambda m:m['season_calendar'].pop('tournaments'))
    lib=NativeLibrary(export.root)
    assert lib.discover()[0]['unknown_phase_rows']==1
    report=lib.preflight(required_features=['countdown'])
    missing=next(r for r in report['cohorts'] if r['phase_id'] is None)
    assert missing['counts']['loaded']==1 and missing['readiness']=='blocked'
    assert all(r['readiness']=='blocked' for r in report['cohorts'])
    assert len(lib.load().matches)==12


def test_future_score_perturbation_truncation_and_row_order_preserve_earlier_features(export):
    from xdiyo_analytics.features import RollingMean, MatchScore
    publication(export)
    d=NativeLibrary(export.root).load(tables=['matches','team_seasons'])
    h=prepare_native_history(d,build_team_history(d),{})
    definition={'goals':RollingMean(MatchScore(),window=2)}
    cutoff=h.kickoff_at-pd.Timedelta(days=1)
    reference=evaluate_features(h,definition,cutoffs=cutoff,available_at='result_available_at')
    boundary=h.kickoff_at.sort_values().unique()[3]
    changed=h.copy();changed.loc[changed.kickoff_at>boundary,['goals_for','goals_against']]=999
    other=evaluate_features(changed,definition,cutoffs=cutoff,available_at='result_available_at')
    pd.testing.assert_frame_equal(reference.loc[h.kickoff_at<=boundary],other.loc[h.kickoff_at<=boundary])
    earlier=h.loc[h.kickoff_at<=boundary].copy()
    truncated=evaluate_features(earlier,definition,cutoffs=earlier.kickoff_at-pd.Timedelta(days=1),available_at='result_available_at')
    pd.testing.assert_frame_equal(reference.loc[earlier.index],truncated)
    shuffled=h.sample(frac=1,random_state=3)
    shuffled_values=evaluate_features(shuffled,definition,cutoffs=shuffled.kickoff_at-pd.Timedelta(days=1),available_at='result_available_at')
    pd.testing.assert_frame_equal(reference,shuffled_values.reindex(reference.index))


def test_timezone_and_entry_cutoff_validation(export):
    from xdiyo_analytics.data.native_preflight import validate_entry_cutoffs
    publication(export);d=NativeLibrary(export.root).load(tables=['matches','team_seasons']);h=build_team_history(d)
    d.matches['result_available_at']='2024-08-01 12:00:00'
    with pytest.raises(ValueError,match='timezone-aware'):
        preflight_loaded(d,options={'availability':'explicit'})
    roster=d['team_seasons'].copy();roster['entry_available_at']=pd.Timestamp('2024-01-01',tz='UTC')
    validate_entry_cutoffs(h,roster,h.kickoff_at-pd.Timedelta(days=1))
    roster.loc[roster.team_id.eq(1),'entry_available_at']=pd.Timestamp('2025-01-01',tz='UTC')
    with pytest.raises(ValueError,match='unavailable at prediction cutoff'):
        validate_entry_cutoffs(h,roster,h.kickoff_at-pd.Timedelta(days=1))


def test_calendar_season_aliases_are_separate_from_annual_group_and_phase(export):
    publication(export,1,(1,2))
    export.change_manifest(lambda m:m['native_integration']['season'].update(name='Premier League 2024',year='2024'))
    lib=NativeLibrary(export.root)
    assert lib.select(seasons='2024')['publications']==lib.select(seasons=61627)['publications']
    assert lib.select(seasons='24_25')['publications']==lib.select(seasons='Premier League 2024')['publications']
    assert lib.select(annual_groups='Annual_Test:2024/2025',phases='Clausura')['phase_ids']==[[17,2]]
    with pytest.raises(ValueError,match='No native seasons'):lib.select(seasons='2024/2025')


def test_explicit_transition_and_curtailed_scopes_do_not_invent_standings(export):
    from xdiyo_analytics.data.native_formats import scope_rules
    lib=publication(export);pub=deepcopy(lib.publications[0]);frame=export.table('matches').to_pandas();roster=export.table('team_seasons').to_pandas()
    pub['competition_id']=11653
    pub['manifest']['native_integration']['coverage']={'kind':'curtailed'}
    pub['manifest']['native_integration']['rules']['standings']={'status':'unsupported','reason':'Administrative draw publication is unknown'}
    result=scope_rules(pub,frame,roster)
    assert result['1']['standings']['status']=='unsupported'
    assert 'Administrative' in result['1']['standings']['reason']
    pub['competition_id']=196;pub['season_id']=87931
    pub['manifest']['native_integration']['coverage']={}
    frame['tournament_id']=171852
    result=scope_rules(pub,frame,roster)
    assert 'shootout' in result['171852']['standings']['reason']


def test_refresh_validated_query_recipe_export_roundtrip(export,tmp_path):
    publication(export);r=recipe(export)
    before=prepare_recipe(r)
    result=refresh_recipe_data(r,tmp_path/'pinned.json')
    after=prepare_recipe(json.loads((tmp_path/'pinned.json').read_bytes()))
    pd.testing.assert_frame_equal(before.dataset.X,after.dataset.X)
    pd.testing.assert_frame_equal(before.dataset.y,after.dataset.y)
    assert result['recipe']['data']['native_selection']['fingerprint']
    from xdiyo_analytics.ui.recipe import export_python
    namespace={};exec(export_python(result['recipe']).split('result =')[0],namespace)
    pd.testing.assert_frame_equal(namespace['prepared'].dataset.X,after.dataset.X)


def test_retrospective_recipe_uses_results_without_inventing_release_dates(export):
    publication(export)
    rows=export.table('matches').to_pylist()
    for row in rows:row['result_available_at']=None
    export.replace('matches',pa.Table.from_pylist(rows))
    r=recipe(export);r['features']={'rank':node('features.DerivedStandings',normalized_ranks=[5])}
    prepared=prepare_recipe(r)
    assert prepared.outputs['history'].result_available_at.isna().all()
    assert prepared.dataset.X.nunique().max()>1
    assert prepared.outputs['native_preflight']['options']['availability']=='retrospective_finished'


def test_phase_round_groups_are_explicit_for_splits_and_gat(export):
    from xdiyo_analytics.data.native_preflight import validate_round_groups
    publication(export,1,(1,2))
    prepared=prepare_recipe(recipe(export))
    h=prepared.outputs['history']
    with pytest.raises(ValueError,match='ambiguous native phase'):
        validate_round_groups(h,['competition_id','season_id','round'],'GAT issuance')
    validate_round_groups(h,['competition_id','season_id','countdown_calendar_id','round'],'GAT issuance')
    r=recipe(export);r['split']['params']['block_by']=['season_id','round']
    with pytest.raises(ValueError,match='ambiguous native phase'):prepare_recipe(r)


def test_missing_roster_members_and_statistics_publication_are_not_verified(export):
    lib=publication(export);d=lib.load(tables=['matches','team_seasons'])
    d.tables['team_seasons']=d['team_seasons'].iloc[:3].copy()
    d.tables['team_seasons']['entry_available_at']='2024-01-01T00:00:00Z'
    d.tables['statistics']=pd.DataFrame({'value':[1]})
    r=preflight_loaded(d,required_features=['movements'],options={'entry_information':'explicit','availability':'explicit'})['cohorts'][0]
    assert r['movement']['unknown_teams']==1
    assert any('publication times' in b for b in r['blockers'])
    assert any('Per-statistic' in b for b in r['blockers'])


@pytest.mark.parametrize('kind,id_fields,first,second',[
    ('competitions',{},17,18),
    ('seasons',{'competition_id':17},(17,61627),(17,61628)),
    ('phases',{'competition_id':17},(17,1),(17,2)),
    ('annual_groups',{'annual_league':'Annual_Test'},('Annual_Test','2024/2025'),('Other','2024/2025'))])
def test_duplicate_entity_names_require_an_id(export,kind,id_fields,first,second):
    lib=publication(export)
    lib._add(kind,first,'Shared Name',[],{'observed_at':10},STEM,id=first[-1] if isinstance(first,tuple) else first,**id_fields)
    lib._add(kind,second,'Shared Name',[],{'observed_at':10},STEM,id=second[-1] if isinstance(second,tuple) else second,**id_fields)
    with pytest.raises(AmbiguousNativeName) as exc:lib._resolve(kind,['Shared Name'],lib.publications)
    assert len(exc.value.candidates)==2


def test_native_gat_preparation_preserves_previous_day_and_phase_identity(export):
    publication(export,1,(1,2));r=recipe(export)
    r.update(features={'home':node('features.IsHome')},preprocessors=[],target_transformer=None,adapter=None,
        model=node('graphs.TemporalGraphModel',round_stage_column='countdown_calendar_id',
                   config=node('graphs.TemporalGraphConfig',timezone='UTC',round_keys=['competition_id','season_id','round_stage','round'])))
    prepared=prepare_recipe(r)
    assert prepared.outputs['graph_inputs'] is not None
    cutoff=pd.to_datetime(prepared.dataset.metadata.feature_cutoff_at,utc=True)
    # GAT's existing t-1 convention is exclusive local midnight, not a newly
    # introduced 24-hour subtraction from that midnight.
    pd.testing.assert_series_equal(cutoff,prepared.dataset.metadata.kickoff_at.dt.normalize(),check_names=False)
    h=prepared.outputs['history']
    assert h.round_stage.equals(h.countdown_calendar_id)
    r['model']['params']['round_stage_column']=None
    r['model']['params']['config']['params']['round_keys']=['competition_id','season_id','round']
    with pytest.raises(ValueError,match='GAT issuance: ambiguous native phase'):prepare_recipe(r)


def test_reviewed_resumption_enters_standings_only_at_completion(export):
    from xdiyo_analytics.ratings import build_standings
    publication(export)
    rows=export.table('matches').to_pylist();row=rows[0]
    release='2024-08-30T23:00:00Z'
    row.update(play_status='resumed',result_available_at=release)
    row['result_review_json']=json.dumps(dict(classification='resumed_completed',event_id=row['event_id'],home_id=row['home_id'],away_id=row['away_id'],
        is_awarded=False,play_status='resumed',played_score=[row['home_score_current'],row['away_score_current']],result_available_at=release,sources=['https://example.test/resumption']))
    export.replace('matches',pa.Table.from_pylist(rows))
    d=NativeLibrary(export.root).load(tables=['matches','team_seasons']);h=prepare_native_history(d,build_team_history(d),{})
    assert preflight_loaded(d,required_features=['standings'])['cohorts'][0]['standings']['status']=='supported'
    q=h.loc[h.side.eq('home') & h.kickoff_at.gt(pd.Timestamp(release))].iloc[:1]
    before=build_standings(h,team_seasons=d['team_seasons'],queries=q,cutoffs=pd.Timestamp(release)-pd.Timedelta(seconds=1),available_at='kickoff_at')
    after=build_standings(h,team_seasons=d['team_seasons'],queries=q,cutoffs=pd.Timestamp(release),available_at='kickoff_at')
    assert row['event_id'] not in before.publications[0]['contributing_events']
    assert row['event_id'] in after.publications[0]['contributing_events']
    assert d.matches.home_score_current.tolist()==[r['home_score_current'] for r in rows]
    review=json.loads(row['result_review_json']);review['played_score']=[99,99]
    row['result_review_json']=json.dumps(review)
    export.replace('matches',pa.Table.from_pylist(rows))
    blocked=NativeLibrary(export.root).preflight(required_features=['standings'])['cohorts'][0]
    assert blocked['readiness']=='blocked'


@pytest.mark.parametrize('explicit_table',[False,True])
def test_legacy_rule_annotations_do_not_change_existing_standings_cache_identity(export,explicit_table):
    from xdiyo_analytics.ratings import build_standings
    publication(export)
    d=load_seasons(export.root,'24_25',tables=['matches','team_seasons']);h=build_team_history(d)
    h['native_standings_rule']=json.dumps({'status':'legacy_verified'})
    if explicit_table:
        h['standings_table_id']='regular';h['standings_rule']='independent_3_1_0'
        d.tables['team_seasons']['standings_table_id']='regular'
    kwargs=dict(team_seasons=d['team_seasons'],available_at='kickoff_at',normalized_ranks=[1,5])
    annotated=build_standings(h,**kwargs)
    without=build_standings(h.drop(columns='native_standings_rule'),**kwargs)
    pd.testing.assert_frame_equal(annotated.snapshots,without.snapshots)
    assert annotated.metadata==without.metadata and annotated.publications==without.publications
