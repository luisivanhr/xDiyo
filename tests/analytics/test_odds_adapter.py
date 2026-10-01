from dataclasses import replace
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from test_footiqo import extract
from xdiyo_analytics.odds import (OddsSeries, paired_quotes, build_fixture_crosswalk,
                                  save_crosswalk, read_crosswalk)
from xdiyo_analytics.odds.crosswalk import MATCH_KEYS
from xdiyo_analytics.features import Stat
from xdiyo_analytics.labels import MatchTotal, Outcome, BetOption
from xdiyo_analytics.evaluation import BetOffer, TightestLine, prepare_bets
from xdiyo_analytics.ui import catalog_for_ui
from post_training_samples import context


def native():
    return pd.DataFrame([dict(source_league='Premier_League', source_season=f'{y}_{y+1}',
        competition_id=17, season_id=2000+y, event_id=2**63+y,
        home_id=101, away_id=202, home_name='A', away_name='B',
        kickoff_at=f'20{y}-09-01T12:00:00Z') for y in [25,26]])


def source(tmp_path, *, main=False):
    snapshot,_,_=extract(tmp_path,sheet='Odds' if main else 'Corners_Closing_Odds')
    walk=build_fixture_crosswalk(snapshot,native(),seasons=['25_26','26_27'])
    saved=save_crosswalk(walk,snapshot=snapshot,output_root=tmp_path/'maps')
    return OddsSeries(str(snapshot),str(saved),('25_26','26_27'),
                      '1x2' if main else 'corners','home' if main else 'over',
                      line=None if main else 9.5,settlement_confirmed=True)


def test_crosswalk_and_alignment_ignore_order_and_provider_id(tmp_path):
    s=source(tmp_path)
    walk=read_crosswalk(s.crosswalk,json.loads((__import__('pathlib').Path(s.snapshot)/'manifest.json').read_text())['snapshot'])
    assert walk.status.eq('matched').all()
    assert walk.vendor_match_id.tolist()!=walk.event_id.tolist()
    metadata=native().iloc[[1,0]].copy()
    ctx=SimpleNamespace(metadata=metadata,y=pd.DataFrame(index=[11,17]))
    resolved=s.resolve(ctx)
    assert resolved.index.tolist()==[11,17]
    assert resolved.quote_status.tolist()==['valid','invalid_decimal_odds']
    assert resolved.decimal_odds.iloc[0]==2.1 and pd.isna(resolved.decimal_odds.iloc[1])
    assert resolved.quote_type.eq('closing').all()
    assert len(resolved.quote_snapshot.unique())==1


def test_unknown_alias_reversal_and_ambiguity_are_quarantined(tmp_path):
    s=source(tmp_path)
    data=native();data['home_name']='New name'
    missing=build_fixture_crosswalk(s.snapshot,data,seasons=['26_27'])
    assert missing.status.eq('unmatched').all()
    fixed=build_fixture_crosswalk(s.snapshot,data,seasons=['26_27'],team_aliases={('Premier_League','A'):101})
    assert fixed.status.eq('matched').all()
    ambiguous=pd.concat([native(),native().assign(event_id=lambda f:f.event_id+100)],ignore_index=True)
    result=build_fixture_crosswalk(s.snapshot,ambiguous,seasons=['26_27'])
    assert result.status.eq('ambiguous').all()
    reversed_data=native().rename(columns={'home_id':'away_id','away_id':'home_id','home_name':'away_name','away_name':'home_name'})
    assert build_fixture_crosswalk(s.snapshot,reversed_data,seasons=['26_27']).status.eq('unmatched').all()


def test_bounded_date_tolerance_and_missing_native_fixture(tmp_path):
    s=source(tmp_path)
    data=native();data['kickoff_at']=pd.to_datetime(data.kickoff_at)+pd.Timedelta(days=1)
    assert build_fixture_crosswalk(s.snapshot,data,seasons=['26_27']).status.eq('unmatched').all()
    assert build_fixture_crosswalk(s.snapshot,data,seasons=['26_27'],date_tolerance_days=1).status.eq('matched').all()
    with pytest.raises(ValueError,match='0 to 7'):
        build_fixture_crosswalk(s.snapshot,data,seasons=['26_27'],date_tolerance_days=20)
    ctx=SimpleNamespace(metadata=native().assign(event_id=123),y=pd.DataFrame(index=[0,1]))
    assert s.resolve(ctx).quote_status.eq('unmatched_fixture').all()


def test_sheet_name_variants_resolve_only_with_agreed_aliases(tmp_path, monkeypatch):
    from xdiyo_analytics.odds import crosswalk
    s=source(tmp_path)
    original=crosswalk.load_odds
    def variants(*args, **kwargs):
        frame=original(*args, **kwargs)
        return pd.concat([frame,frame.assign(home_team='Alternative A')],ignore_index=True)
    monkeypatch.setattr(crosswalk,'load_odds',variants)
    missing=build_fixture_crosswalk(s.snapshot,native(),seasons=['26_27'])
    assert missing.status.eq('conflict').all()
    aliases={('Premier_League','A'):101,('Premier_League','Alternative A'):101}
    matched=build_fixture_crosswalk(s.snapshot,native(),seasons=['26_27'],team_aliases=aliases)
    assert matched.status.eq('matched').all()
    assert 'Alternative A' in matched.source_evidence.iloc[0]
    aliases[('Premier_League','Alternative A')]=999
    conflict=build_fixture_crosswalk(s.snapshot,native(),seasons=['26_27'],team_aliases=aliases)
    assert conflict.status.eq('conflict').all()


def test_reviewed_resumption_does_not_match_other_meetings(tmp_path):
    s=source(tmp_path)
    data=native()
    data.loc[data.source_season.eq('26_27'),'kickoff_at']='2026-08-01T12:00:00Z'
    item=dict(vendor_match_id='9007199254740993',source_league='Premier_League',source_season='26_27',
        event_id=str(2**63+26),vendor_scheduled_at='2026-09-01 12:00:00',native_kickoff_at='2026-08-01T12:00:00Z',
        reason='Reviewed resumed fixture',evidence='Local fixture identity review')
    assert build_fixture_crosswalk(s.snapshot,data,seasons=['26_27']).status.eq('unmatched').all()
    mapped=build_fixture_crosswalk(s.snapshot,data,seasons=['26_27'],fixture_overrides=[item])
    assert mapped.status.eq('matched').all()
    assert mapped.schedule_evidence.notna().all()
    with pytest.raises(ValueError,match='teams or dates'):
        build_fixture_crosswalk(s.snapshot,data,seasons=['26_27'],fixture_overrides=[{**item,'native_kickoff_at':'2026-08-02T12:00:00Z'}])
    with pytest.raises(ValueError,match='teams or dates'):
        build_fixture_crosswalk(s.snapshot,data.assign(home_id=999),seasons=['26_27'],
            team_aliases={('Premier_League','A'):101},fixture_overrides=[item])


def test_saved_alias_rules_are_reused_by_python_builder(tmp_path):
    from pathlib import Path
    s=source(tmp_path)
    data=native().assign(home_name='Native A')
    assert build_fixture_crosswalk(s.snapshot,data,seasons=['26_27']).status.eq('unmatched').all()
    pd.DataFrame([dict(source_league='Premier_League',vendor_team='A',native_team_id='101')]).to_csv(
        Path(s.snapshot).parent/'team_aliases.csv',index=False)
    assert build_fixture_crosswalk(s.snapshot,data,seasons=['26_27']).status.eq('matched').all()


def test_pairing_pins_recipe_roundtrip_and_fixed_compatibility(tmp_path):
    s=source(tmp_path,main=True)
    paired=paired_quotes(s)
    assert paired.paired.sum()==1
    assert paired.missing_opening.sum()==1 and paired.missing_closing.sum()==1
    catalog=catalog_for_ui()
    rebuilt=catalog.build(json.loads(json.dumps(catalog.encode(s))))
    assert rebuilt==s
    assert 'input.OddsSeries' not in catalog.cache_key()
    with pytest.raises(ValueError,match='pinned'):
        replace(s,snapshot_hash='wrong')
    assert catalog.build({'component':'evaluation.BetOffer','params':{
        'option':{'component':'labels.BetOption','params':{'source':{'component':'labels.Outcome','params':{'perspective':'home'}},'selection':'win'}},'odds':2.1}}).odds==2.1


def test_one_sided_opening_missingness_has_no_fallback(tmp_path):
    snapshot,_,_=extract(tmp_path,sheet='Odds',missing_opening=True)
    saved=save_crosswalk(build_fixture_crosswalk(snapshot,native(),seasons=['26_27']),snapshot=snapshot,output_root=tmp_path/'maps')
    s=OddsSeries(str(snapshot),str(saved),('26_27',),'1x2','home',quote_type='opening')
    paired=paired_quotes(s)
    assert paired.missing_opening.all() and not paired.missing_closing.any()
    assert not paired.paired.any()
    assert s.quotes().decimal_odds.isna().all()


def test_market_guards(tmp_path):
    s=source(tmp_path)
    option=BetOption(MatchTotal(Stat('ALL','Match overview','cornerKicks')),selection='over',line=9.5)
    s.validate_option(option)
    with pytest.raises(ValueError,match='no opening'):replace(s,quote_type='opening')
    with pytest.raises(ValueError,match='quarter'):replace(s,line=9.25)
    with pytest.raises(ValueError,match='Confirm'):replace(s,settlement_confirmed=False).validate_option(option)
    with pytest.raises(ValueError,match='MatchTotal'):replace(s,market='yellow_cards').validate_option(option)
    with pytest.raises(ValueError,match='unsupported'):replace(s,market='btts',line=None,selection='yes').validate_option(option)
    one=replace(s,market='1x2',line=None,selection='away')
    one.validate_option(BetOption(Outcome(perspective='home'),selection='loss'))
    with pytest.raises(ValueError,match='perspective'):one.validate_option(BetOption(Outcome(perspective='home'),selection='win'))


def test_missing_imported_quotes_never_fall_back_or_place(tmp_path):
    s=source(tmp_path)
    ctx=context(pd.DataFrame({'count':[1,1]}),{'predict_proba':pd.DataFrame([[0.,1.],[0.,1.]],
        columns=pd.MultiIndex.from_product([['count'],[7,12]]))},metadata=native(),identity_columns=MATCH_KEYS)
    option=BetOption(MatchTotal(Stat('ALL','Match overview','cornerKicks')),selection='over',line=9.5)
    specs,table=prepare_bets(ctx,{'corners':BetOffer(option,s)},TightestLine(min_probability=.5),default_odds=9.)
    assert table['take'].tolist()==[False,True]
    assert table.reason.iloc[0]=='invalid_decimal_odds'
    assert table.quote_type.eq('closing').all()
    assert isinstance(specs['corners'].odds,OddsSeries)


def test_structured_ui_inventory():
    schemas={s['id']:s for s in catalog_for_ui().schema()}
    for key in ('evaluation.BetOffer','evaluation.BetSpec'):
        assert next(f for f in schemas[key]['fields'] if f['name']=='odds')['kind']=='odds'
    fields={f['name']:f for f in schemas['input.OddsSeries']['fields']}
    assert fields['quote_type']['choices_by']['market']['corners']==['closing']
    assert fields['snapshot_hash']['hidden']
    assert fields['snapshot']['hidden'] and fields['snapshot']['initial']=='data/odds'
    assert fields['crosswalk']['title']=='Fixture mapping'
    assert 'footiqo' not in json.dumps(schemas['input.OddsSeries']).lower()
    assert 'source_season' in next(f for f in schemas['input.Table']['fields'] if f['name']=='index')['choices']


def test_current_database_and_mapping_discovery(tmp_path):
    import os
    import shutil
    from pathlib import Path
    from xdiyo_analytics.odds.database import current_database
    from xdiyo_analytics.ui.server import BuilderState
    s=source(tmp_path)
    root=tmp_path/'data/odds'
    old=root/'database/old'
    latest=root/'database/latest'
    shutil.copytree(s.snapshot,old)
    shutil.copytree(s.snapshot,latest)
    os.utime(old/'manifest.json',ns=(1000000000,1000000000))
    os.utime(latest/'manifest.json',ns=(2000000000,2000000000))
    assert current_database(root)==latest
    assert replace(s,snapshot=str(root)).snapshot==str(latest)
    mapping=root/'mappings/reviewed'
    shutil.copytree(s.crosswalk,mapping)
    stale=root/'mappings/stale'
    shutil.copytree(s.crosswalk,stale)
    path=stale/'manifest.json'
    record=json.loads(path.read_text());record['snapshot']='old-unrelated-version'
    path.write_text(json.dumps(record))
    state=BuilderState(tmp_path,catalog_for_ui())
    try:
        discovered=state.dispatch('odds-discover',{})
        assert 'odds_snapshots' not in discovered
        assert [x['value'] for x in discovered['odds_crosswalks']]==[str(mapping)]
    finally:
        state.executor.shutdown()
    with pytest.raises(ValueError,match='No imported odds database'):
        current_database(tmp_path/'empty')


def test_quote_change_refreshes_persisted_reports_without_fit_or_predict(tmp_path, monkeypatch):
    import numpy as np
    from football_experiment_samples import ridge
    from xdiyo_analytics.analysis import PostTrainingAnalysis
    from xdiyo_analytics.datasets import ModelDataset
    from xdiyo_analytics.experiments import FootballExperiment, PreparedExperiment
    from xdiyo_analytics.labels import LabelData
    from xdiyo_analytics.evaluation import BetSpec
    from xdiyo_analytics.reporting import BetPerformanceReporter
    from xdiyo_analytics.splits import Fold, SplitPlan
    from xdiyo_analytics.training import EstimatorAdapter
    s=source(tmp_path,main=True)
    option=BetOption(Outcome(perspective='home'),selection='win')
    meta=native()
    labels={'win':LabelData(pd.DataFrame({'win':[1.,1.]}),meta,'match','home',MATCH_KEYS,option,
                           pd.DataFrame({'win':['win','win']}))}
    dataset=ModelDataset(pd.DataFrame({'x':[1.,2.]}),pd.DataFrame({'result':[1.,1.]}),meta,
                         'match',MATCH_KEYS,MATCH_KEYS,'home',{'label':option.source})
    plan=SplitPlan([Fold(np.array([0]),np.array([1]),np.array([1]))],2,np.arange(2))
    prepared=PreparedExperiment(dataset,plan)
    experiment=FootballExperiment('Synthetic odds reuse',output_dir=tmp_path/'experiments')
    def reports(odds):
        return PostTrainingAnalysis({'Bets':BetPerformanceReporter(type='overall',partition='test',
            bets={'win':BetSpec(option,odds,True)},labels=labels)})
    first=experiment.run(prepared,model=ridge(),post_analysis=reports(replace(s,quote_type='opening')))
    def forbidden(*args,**kwargs):raise AssertionError('Odds change must not fit or predict')
    monkeypatch.setattr(EstimatorAdapter,'fit',forbidden)
    monkeypatch.setattr(EstimatorAdapter,'predict',forbidden)
    second=experiment.run(prepared,model=ridge(),post_analysis=reports(s))
    assert second.reused and second.record['run_id']==first.record['run_id']
    reopened=experiment.load(first.record['run_id'])
    provenance=reopened.post_report.studies[0].result.tables['odds_provenance']
    assert provenance.quote_type.eq('closing').all()
    assert provenance.quote_snapshot_hash.eq(s.snapshot_hash).all()
    assert reopened.training.folds[0].model is not None


def test_composed_reporter_preserves_imported_leg_provenance(tmp_path):
    from test_post_training_betting import as_training
    from xdiyo_analytics.analysis import PostTrainingAnalysis
    from xdiyo_analytics.evaluation import Parlay
    from xdiyo_analytics.labels import LabelData
    from xdiyo_analytics.reporting import BetOutcomeReporter, BetPerformanceReporter
    s=source(tmp_path)
    option=BetOption(MatchTotal(Stat('ALL','Match overview','cornerKicks')),selection='over',line=9.5)
    meta=native().assign(round=1)
    ctx=context(pd.DataFrame({'count':[12.,12.]}),{'predict_proba':pd.DataFrame([[0.,1.],[0.,1.]],
        columns=pd.MultiIndex.from_product([['count'],[7,12]]))},metadata=meta,identity_columns=MATCH_KEYS)
    labels={'over':LabelData(pd.DataFrame({'over':[1.,1.]}),meta,'match','total',MATCH_KEYS,option,
                            pd.DataFrame({'over':['win','win']}))}
    report=PostTrainingAnalysis({
        'Decisions':BetOutcomeReporter(type='overall',partition='test',offers={'over':BetOffer(option,s)},
            policy=TightestLine(min_probability=.5),labels=labels,composition=Parlay(size=1)),
        'Performance':BetPerformanceReporter(type='overall',partition='test',source='Decisions'),
    }).run(as_training(ctx))
    performance=report.studies[1].result
    assert 'odds_provenance' in performance.tables
    assert 'quote_snapshot' in performance.tables['leg_ledger']
    assert any('common bookmaker' in note for note in performance.notes)
