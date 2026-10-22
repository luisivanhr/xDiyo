"""GAP library feature contracts and explicitly bounded synthetic tuning."""
import pandas as pd
import pytest
from xdiyo_analytics.features import GAPFixture, Rating, IsHome, evaluate_features, gap_feature_set
from xdiyo_analytics.ratings import GAPParameters, GAPRatingRun, gap_mae, tune_gap, gap_season_from_context
from gap_fixtures import fixture, statistic, season_plan, build, PARAMS


def test_named_run_fixture_frame_scope_order_keys_and_selected_latents(tmp_path):
    h = pd.DataFrame(fixture(1,'2020-01-02')+fixture(2,'2020-01-04')).iloc[[3,0,2,1]]
    h.index = [40,10,30,20]
    run = build(h,stats=('goals','corners'))
    defs = {**gap_feature_set('gap', ('goals','corners')), 'latent':Rating('gap',fields=('HA','HD','AA','AD'))}
    kept = {}
    x = evaluate_features(h,defs,ratings={'gap':run},rating_runs=kept)
    assert x.index.equals(h.index)
    assert list(x.columns[:2]) == ['gap_goals_diff','gap_corners_diff']
    assert kept['gap'] is run
    assert x.attrs['feature_scopes']['gap_goals_diff']=='fixture'
    assert x.attrs['feature_scopes']['latent::goals::team::HA']=='team'
    assert len([c for c in x if c.startswith('latent::')]) == 16
    assert not any('games_seen' in c or 'initialized' in c for c in x)
    assert x.iloc[0].gap_goals_diff == pytest.approx(.27)
    assert x.iloc[0].gap_goals_diff == x.iloc[2].gap_goals_diff
    keyed = evaluate_features(h,defs,ratings={'gap':run},keyed=True)
    assert keyed.index.get_level_values('event_id').tolist() == h.event_id.tolist()
    assert keyed.index.get_level_values('team_id').tolist() == h.team_id.tolist()
    assert keyed.index.get_level_values('side').tolist() == h.side.tolist()
    assert keyed.attrs['prediction_cutoffs']['records'][0]['team_id'] == h.iloc[0].team_id
    assert keyed.attrs['prediction_cutoffs']['records'][0]['cutoff'] == h.iloc[0].kickoff_at.isoformat()
    pd.testing.assert_frame_equal(keyed.reset_index(drop=True), x.reset_index(drop=True))
    inspected = evaluate_features(h,{'pred':GAPFixture('gap',('goals',),('home','away'))},ratings={'gap':run})
    assert inspected.iloc[0].tolist() == pytest.approx([.54,.27])
    assert inspected.iloc[0].tolist() == inspected.iloc[2].tolist()
    run.save(tmp_path/'native')
    loaded = GAPRatingRun.load(tmp_path/'native')
    pd.testing.assert_frame_equal(evaluate_features(h,defs,ratings={'gap':loaded}),x)
    pd.testing.assert_frame_equal(loaded.diagnostics(h),run.diagnostics(h))


def test_opt_in_disabled_keeps_existing_outputs_and_never_fits(monkeypatch):
    import xdiyo_analytics.ratings as ratings
    import xdiyo_analytics.ratings.gap as gap
    from xdiyo_analytics.features import FeatureBankPreset
    baseline = FeatureBankPreset().build([('ALL','Match overview','cornerKicks')])
    assert not any('gap' in name.lower() or isinstance(expr,GAPFixture) for name,expr in baseline.items())
    h = pd.DataFrame(fixture(1,'2020-01-02'))
    before = evaluate_features(h,{'home':IsHome()})
    def forbidden(*a,**k):
        pytest.fail('Imports/configuration/disabled GAP must not trigger fitting or replay')
    monkeypatch.setattr(ratings,'tune_gap',forbidden)
    monkeypatch.setattr(gap,'build_gap_ratings',forbidden)
    gap_feature_set('unused',('goals',))
    pd.testing.assert_frame_equal(evaluate_features(h,{'home':IsHome()},ratings={'unused':object()}),before)
    assert FeatureBankPreset().build([('ALL','Match overview','cornerKicks')]) == baseline
    with pytest.raises(KeyError,match='missing'):
        evaluate_features(h,{'gap':GAPFixture('missing')})
    with pytest.raises(TypeError,match='GAPRatingRun'):
        evaluate_features(h,{'gap':GAPFixture('bad')},ratings={'bad':object()})


def test_scope_metadata_allows_existing_match_assembly():
    from xdiyo_analytics.datasets import assemble_dataset
    from xdiyo_analytics.labels import LabelData
    h = pd.DataFrame(fixture(1,'2020-01-02')+fixture(2,'2020-01-04'))
    x = evaluate_features(h,gap_feature_set('g',('goals',)),ratings={'g':build(h)},keyed=True)
    assert x.attrs['feature_scopes'] == {'gap_goals_diff':'fixture'}
    metadata = h.loc[h.side.eq('home'),['competition_id','season_id','event_id','team_id','opponent_id']].rename(columns={'team_id':'home_id','opponent_id':'away_id'}).reset_index(drop=True)
    label = LabelData(y=pd.DataFrame({'target':[1.,1.]}), metadata=metadata, unit='match', perspective='home', identity_columns=('competition_id','season_id','event_id'), definition=None)
    dataset = assemble_dataset(x,label,layout='match')
    assert list(dataset.X) == ['fixture::gap_goals_diff']
    assert dataset.X.iloc[:,0].tolist() == pytest.approx([0.,.27])


def test_exact_mae_scaling_and_explicit_mask_independent_of_state_updates():
    h = pd.DataFrame(fixture(1,'2020-01-02',yh=4,ya=2)+fixture(2,'2020-01-04',yh=3,ya=1))
    options = dict(statistic=statistic(),seasons=[season_plan()],parameters=PARAMS,cutoffs=h.kickoff_at)
    # First prediction 0,0: error 6. Next prediction .54,.27: error 3.19.
    assert gap_mae(h,objective_mask=[True]*4,**options) == pytest.approx((6+3.19)/2)
    assert gap_mae(h,objective_mask=[False,False,True,True],**options) == pytest.approx(3.19)
    with pytest.raises(ValueError,match='explicit objective'):
        gap_mae(h,objective_mask=None,**options)
    with pytest.raises(ValueError,match='No eligible'):
        gap_mae(h,objective_mask=[False]*4,**options)
    with pytest.raises(ValueError,match='same objective'):
        gap_mae(h,objective_mask=[True,False,True,True],**options)


def test_pooling_stat_parameters_preserves_league_states():
    h = pd.DataFrame(fixture(1,'2020-01-02',league=10,yh=4,ya=2)+fixture(2,'2020-01-02',league=20,yh=8,ya=4))
    loss = gap_mae(h,statistic=statistic(),seasons=[season_plan(league=10),season_plan(league=20)],
                   parameters=PARAMS,cutoffs=h.kickoff_at,objective_mask=[True]*4)
    assert loss == 9.  # (6+12)/2; no pooling of league states


def test_tiny_optimizer_smoke_explicit_limit_and_fit_boundary():
    h = pd.DataFrame(fixture(1,'2020-01-02')+fixture(2,'2020-01-04',yh=3,ya=1))
    options = dict(statistic=statistic(),seasons=[season_plan()],fit_cutoff='2020-12-31',target_start='2021-01-01',
                   initial=PARAMS,cutoffs=h.kickoff_at,objective_mask=[True]*4,maxiter=3,maxfev=8,xatol=1e-3,fatol=1e-3)
    result = tune_gap(h,**options)
    assert result.iterations <= 3 and result.evaluations <= 8
    assert result.parameters.learning_rate > 0 and 0 < result.parameters.phi1 < 1 and 0 < result.parameters.phi2 < 1
    assert result.provenance['eligible_matches'] == 2
    assert {k:result.provenance['options'][k] for k in ('maxiter','maxfev','xatol','fatol')} == dict(maxiter=3,maxfev=8,xatol=1e-3,fatol=1e-3)
    assert result.provenance['options']['adaptive'] is False
    assert len(result.provenance['options']['initial_simplex']) == 4
    assert result.provenance['scipy_version']
    assert result.provenance['initial'] == dict(learning_rate=.2,phi1=.75,phi2=.6)
    assert result.provenance['state_policy'] == 'replay'
    assert result.loss <= result.provenance['initial_loss']
    assert isinstance(result.success,bool) and result.message
    with pytest.raises(ValueError,match='simplex'):
        tune_gap(h,**options,initial_simplex=[[0,0,0]]*4)
    with pytest.raises(ValueError,match='after the parameter fit cutoff'):
        tune_gap(h,**{**options,'fit_cutoff':'2020-01-03'})
    with pytest.raises(ValueError,match='preceding seasons'):
        tune_gap(h,**{**options,'target_start':'2020-01-01','fit_cutoff':'2019-12-31'})


def test_transition_context_reuse_and_unknown_entrant_rejected():
    from xdiyo_analytics.features import TransitionContext
    h = pd.DataFrame(fixture(1,'2020-01-02')+fixture(2,'2021-01-02',season=2021))
    h['season_year'] = h.season_id.astype(str)
    context = TransitionContext(h,season_starts={(10,2020):'2020-01-01',(10,2021):'2021-01-01'})
    first = gap_season_from_context(context,competition_id=10,season_id=2020,statistic='goals',known_at='2019-12-31',
                                    evidence='Synthetic complete membership',first_stat_season=True)
    second = gap_season_from_context(context,competition_id=10,season_id=2021,statistic='goals',known_at='2020-12-31',
                                     evidence='Synthetic complete membership')
    assert first.previous_season_id is None and second.previous_season_id==2020
    assert second.members == ((1,'retained'),(2,'retained'))
    build(h,plans=[first,second])
    h.loc[h.season_id.eq(2021),'team_id'] = h.loc[h.season_id.eq(2021),'team_id'].replace({1:3})
    h.loc[h.season_id.eq(2021),'opponent_id'] = h.loc[h.season_id.eq(2021),'opponent_id'].replace({1:3})
    context = TransitionContext(h,season_starts={(10,2020):'2020-01-01',(10,2021):'2021-01-01'})
    with pytest.raises(ValueError,match='movement'):
        gap_season_from_context(context,competition_id=10,season_id=2021,statistic='goals',known_at='2020-12-31',evidence='Test')


def test_four_family_schema_and_existing_stat_selection():
    from xdiyo_analytics.features import Stat
    from xdiyo_analytics.ratings import GAPStatistic
    families = ('goals','shots_on_target','shots_off_target','corners')
    h = pd.DataFrame(fixture(1,'2020-01-02')+fixture(2,'2020-01-04'))
    for name, for_values, against_values in [('shots_on_target',[8,3,8,3],[3,8,3,8]),
                                            ('shots_off_target',[10,4,10,4],[4,10,4,10])]:
        h[f'{name}_for'] = for_values
        h[f'{name}_against'] = against_values
    source = Stat('ALL','Shots','shotsOffGoal')
    h.attrs['stat_columns'] = {
        'shots_off_target_for':dict(role='team',period='ALL',group_name='Shots',key='shotsOffGoal',field=source.field),
        'shots_off_target_against':dict(role='opponent',period='ALL',group_name='Shots',key='shotsOffGoal',field=source.field),
    }
    selected = GAPStatistic.from_stat(h,'shots_off_target',source,definition='Synthetic directly observed off-target count')
    assert selected.for_column == 'shots_off_target_for'
    with pytest.raises(ValueError,match='period'):
        GAPStatistic.from_stat(h,'shots_off_target',Stat(None,'Shots','shotsOffGoal'),definition='x')
    run = build(h,stats=families)
    x = evaluate_features(h,gap_feature_set('g',families),ratings={'g':run})
    assert list(x) == [f'gap_{name}_diff' for name in families]
    assert x.iloc[2].tolist() == pytest.approx([.27,.675,.81,.405])
    assert len(run.features(h).columns) == 32  # explicit state inspection only
    assert run.fixture_features(h).shape[1] == 4
