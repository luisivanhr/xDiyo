"""Small synthetic target population with large disconnected graph history."""
from dataclasses import replace

import numpy as np
import pandas as pd

from xdiyo_analytics.graphs import GraphExecution, TemporalEventStore, TemporalGraphAdapter
from test_temporal_graph import config, context, history


def execution_fixture(mode='dependency_packed', statistics=True, layers=2, ratings=True):
    """Return adapter/train/query; every optimized issuance has sparse cones.

    Unrelated league histories are actual paired synthetic fixtures, never
    padding of target rows. The 74 target schema is the checked full-statistics
    schema with synthetic observations; fixture ratings use the real Bayesian
    producer at the same previous-day graph cutoff.
    """
    events = []
    members = []
    anchor = pd.Timestamp('2019-12-01')
    for league in range(20):
        teams = tuple(100+4*league+i for i in range(4))
        competition, season = 1000+league, 2000+league
        for team in teams:
            members.append(dict(team_id=team, competition_id=competition, season_id=season,
                                entry_at=pd.Timestamp('2019-07-01', tz='UTC'), movement='retained'))
        for i in range(12):
            events.append((1000+league*12+i, (anchor+pd.Timedelta(days=2*i)).isoformat(),
                teams[i%4], teams[(i+1)%4], i%4, (i+2)%3, competition, season, i+1))
    for team in (1, 2, 3, 4):
        members.append(dict(team_id=team, competition_id=10, season_id=900,
                            entry_at=pd.Timestamp('2019-07-01', tz='UTC'), movement='retained'))
    for i in range(14):
        events.append((i+1, (pd.Timestamp('2020-01-01')+pd.Timedelta(days=3*i)).isoformat(),
                       1+i%4, 1+(i+1)%4, i%3, (i+1)%3, 10, 900, i+1))
    h = history(events)
    forecast = None
    if statistics:
        from test_graph_numeric_targets import full_config
        forecast = replace(full_config(), route='combined')
        values = np.arange(len(h), dtype=float)
        columns = {}
        for i, target in enumerate(forecast.targets):
            if target.column in h or target.column in columns:
                continue
            if target.support == 'signed_continuous':
                observed = (values%17-8)/10
            elif target.support == 'nonnegative_continuous':
                observed = (values%13)/10
            elif target.support == 'bounded_continuous':
                observed = 30+(values%11)/2
            else:
                observed = values%7
            # Missing target observations must retain their native per-target
            # weight denominator across packing and resume.
            observed = observed.copy()
            observed[(np.arange(len(h))+i)%11 == 0] = np.nan
            columns[target.column] = observed
        h = pd.concat((h, pd.DataFrame(columns, index=h.index)), axis=1)
    store = TemporalEventStore(h, pd.DataFrame(members))
    # Small bounded fixtures need a modest explicit reserve on shared CI hosts;
    # production policy defaults and admission budgets are tested separately.
    cfg = config(layers=layers, deterministic=True, execution=GraphExecution(mode=mode, headroom_bytes=64*1024**2),
                 statistics_forecast=forecast, fixture_features=('bayesian_expected_home_goals',) if ratings else ())
    primary = np.flatnonzero(store.schedule.competition_id.to_numpy() == 10)
    train = context(store, primary[2:10].tolist())
    train.validation = context(store, primary[10:12].tolist())
    query = context(store, primary[12:14].tolist(), fit=False)
    train.sample_weight = pd.Series([1., 0., .5, 3., 2., 1., .25, 4.], index=train.X.index)
    train.validation.sample_weight = pd.Series([.5, 2.], index=train.validation.X.index)
    if ratings:
        from xdiyo_analytics.ratings import BayesianConfig, BayesianModel, build_bayesian_ratings
        directed = store.history
        directed = directed.loc[directed.competition_id.eq(10)].copy()
        run = build_bayesian_ratings(directed,
            model=BayesianModel(config=BayesianConfig(method='one_step', bivariate=False)),
            available_at=cfg.labels_known_at(directed.kickoff_at))
        for item in (train, train.validation, query):
            cutoff = pd.Series([cfg.history_cutoff(t) for t in store.issue_times(item.metadata, cfg)],
                               index=item.metadata.index)
            # Fixture producer consumes the paired directed history view.
            keys = pd.MultiIndex.from_frame(item.metadata[['competition_id', 'season_id', 'event_id']])
            home = directed.loc[directed.side.eq('home')].set_index(['competition_id', 'season_id', 'event_id'])
            selected = home.reindex(keys).reset_index()
            selected.index = item.metadata.index
            computed = run.fixture_features(selected, cutoffs=cutoff, fields=('expected_home_goals',))
            item.X['bayesian_expected_home_goals'] = computed['expected_home_goals'].to_numpy()
            item.metadata['feature_cutoff_at'] = cutoff
            assert np.isfinite(item.X['bayesian_expected_home_goals']).all()
    return TemporalGraphAdapter(cfg, store), train, query
