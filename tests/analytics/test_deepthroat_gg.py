"""Research-specific contracts, independent replay, and native persistence."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.data import SeasonData
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.research.deepthroat_features import FEATURES, normalized_ranks, prepare_bundles
from xdiyo_analytics.research.deepthroat_gg import DeepthroatGGRecipe, joint_score_rmse, rps, select_neighbors, save_research
from xdiyo_analytics.training import load_model


def sample_history(rounds=48):
    rng = np.random.default_rng(29)
    rows = []
    for round_ in range(rounds):
        teams = np.roll(np.arange(6), round_ % 5)
        for i in range(3):
            home, away = int(teams[i]), int(teams[5-i])
            if round_ % 2:
                home, away = away, home
            date = pd.Timestamp('2020-01-01', tz='UTC') + pd.Timedelta(days=round_*3, hours=i)
            rows.append(dict(event_id=len(rows)+1, competition_id=17, tournament_id=50,
                season_id=1+round_//24, round=1+round_%24, home_id=home, away_id=away,
                kickoff_utc=date.timestamp(), status='finished',
                home_score_current=int(rng.poisson(1.6)), away_score_current=int(rng.poisson(1.1))))
    history = build_team_history(SeasonData({'matches':pd.DataFrame(rows)}, {}))
    roster = history[['competition_id','season_id','team_id']].drop_duplicates().reset_index(drop=True)
    roster['known_at'] = pd.Timestamp('2019-12-01', tz='UTC')
    history['released_at'] = history.kickoff_at + pd.Timedelta(hours=3)
    return history, roster


def test_metrics_and_paper_totals():
    assert joint_score_rmse([[1,2]], [[2,4]]) == pytest.approx(np.sqrt(5))
    assert rps([0], [[.6,.3,.1]]) == pytest.approx(.185)
    assert rps([1,0,-1], np.eye(3)) == 0
    city = np.array([[3,1],[3,1],[1,1],[4,1],[2,0],[1,0]])
    liverpool = np.array([[2,0],[2,0],[0,0],[2,0],[7,0],[0,1]])
    totals = []
    for scores in [city,liverpool]:
        p = np.where(scores[:,0]>scores[:,1],3,np.where(scores[:,0]==scores[:,1],1,0)).sum()
        totals.append([p,*scores.sum(0)])
    assert totals == [[16,14,4],[13,13,1]]
    np.testing.assert_array_equal(normalized_ranks(totals), [1,0])
    np.testing.assert_array_equal(normalized_ranks([[0,0,0]]*4), [.5]*4)


def test_order_and_independent_replay_all_venues_across_seasons():
    h, roster = sample_history()
    bundles = prepare_bundles(h, roster, n_values=[6,9,20], available_at='released_at')
    for n, (data, outcome) in bundles.items():
        assert tuple(data.X) == FEATURES
        for pos in [30,50,80,120]:
            md = data.metadata.iloc[pos]
            time = md.prediction_cutoff
            totals, means = {}, {}
            active = roster[(roster.competition_id == 17) & (roster.season_id == md.season_id)].team_id
            for team in active:
                past = h[(h.team_id == team) & (h.competition_id == 17) & (h.kickoff_at < time) & (h.released_at <= time)].sort_values('kickoff_at').tail(n)
                gf, ga = past.goals_for.to_numpy(), past.goals_against.to_numpy()
                totals[team] = [np.where(gf>ga,3,np.where(gf==ga,1,0)).sum(),gf.sum(),ga.sum()]
                means[team] = [gf.mean(),ga.mean()]
            ranks = dict(zip(totals, normalized_ranks(list(totals.values()))))
            expected = [*means[md.home_id], ranks[md.home_id], *means[md.away_id], ranks[md.away_id]]
            np.testing.assert_allclose(data.X.iloc[pos], expected, atol=0, rtol=0)
            goals = data.y.iloc[pos].to_numpy()
            assert outcome.y.iloc[pos,0] == np.sign(goals[0]-goals[1])


def test_future_unreleased_missing_and_short_histories():
    h, roster = sample_history()
    base = prepare_bundles(h, roster, n_values=[9], available_at='released_at')[9][0]
    changed = h.copy()
    future = changed.kickoff_at.ge(pd.Timestamp('2020-04-01', tz='UTC'))
    changed.loc[future, ['goals_for','goals_against']] = 1
    changed.loc[future, 'result'] = 'D'
    actual = prepare_bundles(changed, roster, n_values=[9], available_at='released_at')[9][0]
    prior = base.metadata.prediction_cutoff.lt(pd.Timestamp('2020-04-01', tz='UTC'))
    pd.testing.assert_frame_equal(base.X[prior], actual.X[prior])
    assert not base.metadata.research_eligible.iloc[:18].any()
    assert base.metadata.research_eligible.iloc[18:].all()
    # Whole result has incomplete paired scores; neither team's history consumes it.
    missing = h.copy()
    missing.loc[missing.event_id.eq(1), ['goals_for','goals_against']] = np.nan
    m = prepare_bundles(missing, roster, n_values=[9], available_at='released_at')[9][0]
    assert not m.metadata.research_eligible.iloc[18]
    # One side released late: neither side can consume it early.
    delayed = h.copy()
    delayed.loc[delayed.event_id.eq(1) & delayed.side.eq('away'), 'released_at'] = pd.Timestamp('2021-01-01', tz='UTC')
    d = prepare_bundles(delayed, roster, n_values=[9], available_at='released_at')[9][0]
    pd.testing.assert_frame_equal(m.X, d.X)


def test_neighbors_self_exclusion_metrics_and_infeasible():
    X = np.zeros((8,6))  # Duplicate points must still exclude the exact query.
    goals = np.arange(16).reshape(8,2)
    y = np.array([1,0,-1,1,0,-1,1,0])
    best, ties, table, indices = select_neighbors(X, goals, y, [3,7,8,350])
    assert not (indices == np.arange(8)[:,None]).any()
    assert table.status.tolist() == ['evaluated','evaluated','infeasible','infeasible']
    for k in [3,7]:
        predictions, p = independent_loo(X, goals, y, k)
        row = table[table.k.eq(k)].iloc[0]
        assert row.rps == rps(y, p)
        assert row.joint_rmse == joint_score_rmse(goals, predictions)
    with pytest.raises(ValueError, match='No feasible'):
        select_neighbors(X[:3], goals[:3], y[:3])


def test_recipe_native_models_exports_and_persistence(tmp_path):
    h, roster = sample_history()
    recipe = DeepthroatGGRecipe('2020-04-15', '2020-06-01', n_values=(9,10), k_values=(3,5,350), available_at='released_at')
    restored = DeepthroatGGRecipe.from_dict(json.loads(json.dumps(recipe.to_dict())))
    scope = {}; exec(recipe.export_python(), scope)
    assert restored == scope['recipe'] == recipe
    results = recipe.run(h, roster)
    assert len(results) == 1
    result = results[0]
    manifest = save_research(results, recipe, h, roster, tmp_path/'research')
    assert manifest['classes'] == (1,0,-1)
    assert manifest['neighbor_policy'] == 'native_sklearn_selected_k_leave_one_out_v2'
    assert result.k_table.iloc[-1].status == 'infeasible'
    for name, run in [('goals',result.goals),('outcomes',result.outcomes)]:
        fold = run.folds[0]
        loaded = load_model(tmp_path/'research'/'competition-0'/name)
        data = result.dataset
        subset = replace(data, **{key:getattr(data,key).iloc[fold.test_positions].copy() for key in ('X','y','metadata')})
        for method, pred in loaded.predict(subset).items():
            pd.testing.assert_frame_equal(pred.reset_index(drop=True), fold.predictions[method].reset_index(drop=True))


def test_incremental_own_windows_equal_replay_with_postponement_and_late_release():
    from collections import deque
    h, roster = sample_history()
    h.loc[h.event_id.eq(4), 'kickoff_at'] += pd.Timedelta(days=60)
    h['released_at'] = h.kickoff_at + pd.Timedelta(hours=3)
    h.loc[h.event_id.eq(5), 'released_at'] += pd.Timedelta(days=31)
    data = prepare_bundles(h, roster, n_values=[9], available_at='released_at')[9][0]
    prior = {team:[] for team in roster.team_id.unique()}
    records = h.sort_values('released_at').to_dict('records')
    cursor = 0
    for pos in np.argsort(data.metadata.prediction_cutoff, kind='stable'):
        row = data.metadata.iloc[pos]
        time = row.prediction_cutoff
        while cursor < len(records) and records[cursor]['released_at'] <= time:
            record = records[cursor]
            prior[record['team_id']].append(record)
            cursor += 1
        totals, means = [], {}
        for team in prior:
            # Reinsert late results in kickoff order, and evict expired own results.
            queue = deque(maxlen=9)
            total = np.zeros(3)
            eligible = sorted([r for r in prior[team] if r['kickoff_at'] < time],key=lambda r:r['kickoff_at'])
            for r in eligible:
                gf,ga = r['goals_for'],r['goals_against']
                value = np.array([3 if gf>ga else 1 if gf==ga else 0,gf,ga])
                if len(queue) == 9:
                    total -= queue[0]
                queue.append(value); total += value
            replay = np.sum(list(queue), axis=0) if queue else np.zeros(3)
            np.testing.assert_array_equal(total,replay)
            totals.append(total)
            means[team] = total[1:]/len(queue) if len(queue)>=6 else [np.nan,np.nan]
        ranks = dict(zip(prior,normalized_ranks(totals)))
        np.testing.assert_allclose(data.X.iloc[pos], [*means[row.home_id],ranks[row.home_id],*means[row.away_id],ranks[row.away_id]], equal_nan=True, atol=0, rtol=0)


def test_membership_relegation_return_and_competition_isolation():
    h, roster = sample_history(rounds=72)
    # Team 0 is absent in season 2 and returns in season 3; new team 6 replaces it.
    absent = h.season_id.eq(2)
    h.loc[absent & h.team_id.eq(0),'team_id'] = 6
    h.loc[absent & h.opponent_id.eq(0),'opponent_id'] = 6
    roster.loc[roster.season_id.eq(2) & roster.team_id.eq(0),'team_id'] = 6
    data = prepare_bundles(h,roster,n_values=[20],available_at='released_at')[20][0]
    first = data.metadata[(data.metadata.season_id.eq(3)) & (data.metadata.home_id.eq(0)|data.metadata.away_id.eq(0))].index[0]
    md = data.metadata.loc[first]
    source = h[h.season_id.eq(1) & h.team_id.eq(0)].sort_values('kickoff_at').tail(20)
    prefix = 'home' if md.home_id==0 else 'away'
    assert data.X.loc[first,prefix+'::mean_scored'] == source.goals_for.mean()
    assert data.metadata.loc[first,'research_eligible']
    # Division-2 matches with extreme scores cannot affect this competition.
    other = h.copy(); other.competition_id = 18
    other.goals_for = 99; other.goals_against = 99; other.result = 'D'
    other_roster = roster.copy(); other_roster.competition_id = 18
    mixed = prepare_bundles(pd.concat([h,other],ignore_index=True),pd.concat([roster,other_roster]),n_values=[20],available_at='released_at')[20][0]
    pd.testing.assert_frame_equal(data.X,mixed.X.iloc[:len(data.X)])
    # Same totals but different roster size changes normalized rank denominator.
    extra = roster.iloc[[0]].copy(); extra.team_id = 99
    extra.known_at = pd.Timestamp('2020-01-20',tz='UTC')
    extended = prepare_bundles(h,pd.concat([roster,extra]),n_values=[20],available_at='released_at')[20][0]
    before = data.metadata.prediction_cutoff.lt(extra.known_at.iloc[0])
    pd.testing.assert_frame_equal(data.X[before],extended.X[before])
    after = ~before & data.metadata.season_id.eq(1) & data.metadata.research_eligible
    assert not data.X.loc[after,'home::form_rank'].equals(extended.X.loc[after,'home::form_rank'])


def test_frozen_round_cutoffs_and_holdout_outcomes_do_not_select_parameters():
    h, roster = sample_history()
    h['prediction_time'] = h.kickoff_at.dt.floor('D') - pd.Timedelta(hours=12)
    b = prepare_bundles(h, roster, n_values=[9], cutoffs='prediction_time', available_at='released_at')[9][0]
    assert b.metadata.groupby(b.metadata.kickoff_at.dt.floor('D')).prediction_cutoff.nunique().eq(1).all()
    recipe = DeepthroatGGRecipe('2020-04-15','2020-06-01',n_values=(9,10,11), k_values=(3,5), available_at='released_at')
    baseline = recipe.run(h,roster)[0]
    later = h.kickoff_at.gt(pd.Timestamp(recipe.development_end,tz='UTC'))
    h.loc[later, ['goals_for','goals_against']] = 0
    h.loc[later,'result'] = 'D'
    changed = recipe.run(h,roster)[0]
    pd.testing.assert_frame_equal(baseline.n_table, changed.n_table)
    pd.testing.assert_frame_equal(baseline.k_table, changed.k_table)
    assert (baseline.n,baseline.k_goals,baseline.k_outcomes) == (changed.n,changed.k_goals,changed.k_outcomes)


def test_roster_validation_and_missing_pair_release():
    h, roster = sample_history()
    with pytest.raises(ValueError,match='explicit known_at'):
        prepare_bundles(h,roster.drop(columns='known_at'),n_values=[9])
    with pytest.raises(ValueError,match='unique'):
        prepare_bundles(h,pd.concat([roster,roster.iloc[[0]]]),n_values=[9])
    h.loc[h.event_id.eq(1) & h.side.eq('away'),'released_at'] = pd.NaT
    data = prepare_bundles(h,roster,n_values=[9],available_at='released_at')[9][0]
    assert pd.isna(data.metadata.iloc[0].label_available_at)
    assert not data.metadata.iloc[18].research_eligible


def test_signed_pearson_and_exact_tie_policy():
    from xdiyo_analytics.research.deepthroat_gg import fit_bundles
    h, roster = sample_history()
    bundles = prepare_bundles(h,roster,n_values=[9,10,11],available_at='released_at')
    end = pd.Timestamp('2020-04-15',tz='UTC')
    data = bundles[9][0]
    dev = np.flatnonzero(data.metadata.research_eligible & data.metadata.prediction_cutoff.le(end) & data.metadata.label_available_at.le(end))
    d = data.y.iloc[dev,0].to_numpy()-data.y.iloc[dev,1].to_numpy()
    d -= d.mean()
    residual = np.random.default_rng(82).normal(size=len(dev)); residual -= residual.mean()
    residual -= np.dot(residual,d)/np.dot(d,d)*d
    residual *= np.linalg.norm(d)/np.linalg.norm(residual)
    for n,(goals,outcomes) in bundles.items():
        goals.X = goals.X.copy()
        z = -d if n==9 else -.5*d+np.sqrt(.75)*residual
        goals.X.iloc[dev,0] = z/2
        goals.X.iloc[dev,1] = 0
        goals.X.iloc[dev,3:5] = 0
        outcomes.X = goals.X.copy()
    result = fit_bundles(bundles,development_end=end,holdout_end='2020-06-01',k_values=[3])[0]
    assert result.n == 10  # -0.5 beats -1, and the smaller n wins the exact tie.
    assert result.ties['n'] == [10,11]
    np.testing.assert_allclose(result.n_table.pearson,[-1,-.5,-.5],atol=1e-14)


@pytest.mark.parametrize('metric,args',[
    (joint_score_rmse,([[1]],[[1]])),
    (joint_score_rmse,([[1,2]],[[np.nan,2]])),
    (rps,([0],[[.5,.5,.5]])),
    (rps,([2],[[0,1,0]])),
])
def test_invalid_scoring_inputs_fail(metric,args):
    with pytest.raises(ValueError):
        metric(*args)


def independent_loo(X, goals, outcomes, k):
    """Refit both actual estimators independently for every candidate and row."""
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    predictions, probabilities = [], []
    for row in range(len(X)):
        keep = np.arange(len(X)) != row
        params = dict(n_neighbors=k, weights='uniform', metric='euclidean',
                      algorithm='kd_tree', n_jobs=1)
        reg = KNeighborsRegressor(**params).fit(X[keep], goals[keep])
        clf = KNeighborsClassifier(**params).fit(X[keep], outcomes[keep])
        query = X[row:row+1]
        predictions.append(reg.predict(query)[0])
        native = dict(zip(clf.classes_, clf.predict_proba(query)[0]))
        probabilities.append([native.get(c, 0.) for c in (1, 0, -1)])
    return np.array(predictions), np.array(probabilities)


@pytest.mark.parametrize('kind', ['duplicate', 'equidistant', 'tree_splits', 'untied', 'missing_class'])
@pytest.mark.parametrize('reverse', [False, True])
def test_selected_k_matches_independent_sklearn_loo(kind, reverse):
    rng = np.random.default_rng(13)
    size = 68 if kind == 'tree_splits' else 8
    X = rng.normal(size=(size, 6))
    if kind in ('duplicate', 'missing_class'):
        X[:] = 0
    elif kind == 'equidistant':
        X = np.vstack([np.eye(6), -np.eye(6)[:2]])
    elif kind == 'tree_splits':
        X = rng.integers(-1, 2, size=(size, 6)).astype(float)
        X[::3] = X[1]
    goals = rng.integers(0, 7, size=(size, 2))
    outcomes = np.resize([1, 0, -1], size)
    if kind == 'missing_class':
        outcomes[:] = 1
        outcomes[-1] = 0  # A held-out class may be absent from fitting entirely.
    if reverse:
        X, goals, outcomes = X[::-1].copy(), goals[::-1].copy(), outcomes[::-1].copy()
    ks = list(range(3, 8)) if size == 8 else [3, 7, 31, 67]
    best, ties, table, indices = select_neighbors(X, goals, outcomes, ks)
    expected = []
    for k in ks:
        predictions, probabilities = independent_loo(X, goals, outcomes, k)
        # Independent equations as well as independently fitted estimators.
        rmse = np.sqrt(((goals-predictions)**2).sum(axis=1).mean())
        actual = outcomes[:, None] == [1, 0, -1]
        score = (((probabilities.cumsum(axis=1)[:, :2] - actual.cumsum(axis=1)[:, :2])**2).sum(axis=1)/2).mean()
        expected.append([rmse, score])
    np.testing.assert_array_equal(table[['joint_rmse', 'rps']].to_numpy(), expected)
    for j, metric in enumerate(('joint_rmse', 'rps')):
        values = np.array(expected)[:, j]
        assert best[metric] == ks[int(values.argmin())]
        assert ties[metric] == [k for k, value in zip(ks, values) if value == values.min()]
    assert not (indices == np.arange(size)[:, None]).any()


@pytest.mark.parametrize('dtype', ['Int64', 'Float64'])
@pytest.mark.parametrize('partial', [False, True])
def test_nullable_missing_scores_equal_float_nan_and_do_not_mutate(dtype, partial):
    h, roster = sample_history(rounds=12)
    h[['goals_for', 'goals_against']] = h[['goals_for', 'goals_against']].astype(dtype)
    missing = h.event_id.eq(1)
    if partial:
        h.loc[missing & h.side.eq('home'), 'goals_for'] = pd.NA
        h.loc[missing & h.side.eq('away'), 'goals_against'] = pd.NA
    else:
        h.loc[missing, ['goals_for', 'goals_against']] = pd.NA
    original = h.copy(deep=True)
    baseline = h.copy()
    baseline[['goals_for', 'goals_against']] = baseline[['goals_for', 'goals_against']].astype(float)
    expected = prepare_bundles(baseline, roster, n_values=[6, 9], available_at='released_at')
    actual = prepare_bundles(h, roster, n_values=[6, 9], available_at='released_at')
    for n in actual:
        for data, reference in zip(actual[n], expected[n]):
            for attribute in ('X', 'y', 'metadata'):
                pd.testing.assert_frame_equal(getattr(data, attribute), getattr(reference, attribute))
        assert not actual[n][0].metadata.research_eligible.iloc[18]
    pd.testing.assert_frame_equal(h, original)


@pytest.mark.parametrize('delta', [pd.Timedelta(days=1), pd.Timedelta(nanoseconds=1)])
def test_mismatched_fixture_cutoffs_rejected_before_features(monkeypatch, delta):
    import xdiyo_analytics.research.deepthroat_features as module
    h, roster = sample_history(rounds=12)
    h['prediction_time'] = h.kickoff_at.astype('datetime64[ns, UTC]')
    # An otherwise eligible target must not consume away evidence after home cutoff.
    target = h.event_id.eq(25) & h.side.eq('away')
    h.loc[target, 'prediction_time'] += delta
    def forbidden(*args, **kwargs):
        pytest.fail('Features were evaluated before fixture cutoffs were validated.')
    monkeypatch.setattr(module, 'evaluate_features', forbidden)
    with pytest.raises(ValueError, match='identical prediction cutoffs.*inconsistent fixture'):
        prepare_bundles(h, roster, n_values=[9], cutoffs='prediction_time', available_at='released_at')


def test_identical_cutoffs_accept_equivalent_timezone_instants():
    h, roster = sample_history(rounds=12)
    expected = prepare_bundles(h, roster, n_values=[9], available_at='released_at')[9]
    cutoffs = h.kickoff_at.astype(object)
    away = h.side.eq('away')
    cutoffs.loc[away] = h.loc[away, 'kickoff_at'].dt.tz_convert('Asia/Tokyo')
    actual = prepare_bundles(h, roster, n_values=[9], cutoffs=cutoffs, available_at='released_at')[9]
    for data, reference in zip(actual, expected):
        pd.testing.assert_frame_equal(data.X, reference.X)
        pd.testing.assert_frame_equal(data.metadata, reference.metadata)


def test_january_seventh_result_cannot_enter_only_away_perspective():
    matches = pd.DataFrame([
        dict(event_id=day, competition_id=17, tournament_id=50, season_id=1,
             round=day, home_id=1, away_id=2,
             kickoff_utc=pd.Timestamp(f'2020-01-{day:02d}T09:00:00Z').timestamp(),
             status='finished', home_score_current=1, away_score_current=day-1)
        for day in range(1, 9)])
    h = build_team_history(SeasonData({'matches': matches}, {}))
    h['released_at'] = h.kickoff_at + pd.Timedelta(hours=3)
    roster = h[['competition_id', 'season_id', 'team_id']].drop_duplicates()
    roster['known_at'] = pd.Timestamp('2019-12-01', tz='UTC')
    target = h.event_id.eq(8)
    early, later = h.kickoff_at.copy(), h.kickoff_at.copy()
    early.loc[target] = pd.Timestamp('2020-01-07', tz='UTC')
    later.loc[target] = pd.Timestamp('2020-01-08', tz='UTC')
    for cutoffs, expected_mean in [(early, 2.5), (later, 3.5)]:
        data = prepare_bundles(h, roster, n_values=[6], cutoffs=cutoffs, available_at='released_at')[6][0]
        assert data.metadata.research_eligible.iloc[-1]
        assert data.X['away::mean_scored'].iloc[-1] == expected_mean
    mixed = early.copy()
    mixed.loc[target & h.side.eq('away')] = later.loc[target & h.side.eq('away')]
    with pytest.raises(ValueError, match='identical prediction cutoffs'):
        prepare_bundles(h, roster, n_values=[6], cutoffs=mixed, available_at='released_at')
