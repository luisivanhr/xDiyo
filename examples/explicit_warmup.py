"""Preparation-only warm-start definitions; callers choose production parameters.

Pass a full historical context (including predecessor leagues/seasons) and
explicit season_starts when the supplied history does not establish boundaries.
These functions do not load production data or fit an estimator.
"""
from xdiyo_analytics.features import (
    WarmStart, SeededEMA, LinearFade, RollingMean, RollingStd, RollingZScore,
    Stat, MatchScore, ForAgainst, TeamMovement, Heatmap, RegionMass, Difference,
    evaluate_features,
)


def feature_definitions(*, mode, alpha, fade_start, fade_rounds, bottom, top,
                        variance_estimator, prior_strength):
    policy = SeededEMA(mode=mode, alpha=alpha,
        handoff=LinearFade(start=fade_start, rounds=fade_rounds), bottom=bottom, top=top,
        variance_prior='within_team', variance_estimator=variance_estimator,
        prior_strength=prior_strength, variance_fade='estimate_interpolation')
    ddof = 0 if variance_estimator == 'population' else 1
    shots = Stat('ALL', 'Shots', 'totalShotsOnGoal')
    target = Stat('ALL', 'Shots', 'shotsOnGoal')
    goals = MatchScore(score_field='current')
    features = {
        'was_promoted': TeamMovement('promoted'),
        'was_relegated': TeamMovement('relegated'),
    }
    for name, source in [('shots', shots), ('shots_on_target', target), ('goals', goals)]:
        for side in ('for', 'against'):
            features[f'{name}_{side}'] = WarmStart(
                RollingMean(ForAgainst(source, side), window=20, min_periods=1, venue='all'), policy)
    corners = Stat(None, 'Match overview', 'cornerKicks')  # Multi-period expansion.
    features['corners_sd'] = WarmStart(RollingStd(corners,20,ddof=ddof),policy)
    features['corners_z'] = WarmStart(RollingZScore(corners,20,ddof=ddof),policy)
    features['goal_difference_sd'] = WarmStart(RollingStd(
        Difference(ForAgainst(goals,'for'),ForAgainst(goals,'against')),20,ddof=ddof),policy)
    return features


def spatial_definitions(*, mode, alpha, fade_start, fade_rounds, bottom, top):
    policy = SeededEMA(mode=mode,alpha=alpha,handoff=LinearFade(fade_start,fade_rounds),
                       bottom=bottom,top=top,variance_prior='within_team',variance_estimator='population')
    source = Heatmap(grid_size=5,normalization='mass',orientation='home')
    return {
        'heatmap_mean': WarmStart(RollingMean(source,20,venue='same'),policy),
        'own_half_sd': WarmStart(RollingStd(RegionMass(source,'own_half'),20,ddof=0),policy),
    }


def native_recipe(data_root, *, mode, alpha, fade_start, fade_rounds, bottom, top,
                  variance_estimator, prior_strength):
    from xdiyo_analytics.ui import default_recipe
    from xdiyo_analytics.ui.catalog import default_catalog
    recipe = default_recipe(data_root)
    # Native movement table carries origin identities and evidence in addition
    # to the nullable flags exposed in ordinary team history.
    recipe['data']['tables'] = ['matches','statistics','pregame','team_seasons']
    features = feature_definitions(mode=mode,alpha=alpha,fade_start=fade_start,
        fade_rounds=fade_rounds,bottom=bottom,top=top,
        variance_estimator=variance_estimator,prior_strength=prior_strength)
    recipe['features'] = default_catalog().encode(features)
    return recipe


def prepare_features(history, *, features, team_seasons, season_starts, available_at, heatmaps=None):
    return evaluate_features(history,features,team_seasons=team_seasons,
        season_starts=season_starts,available_at=available_at,heatmaps=heatmaps,keyed=True)
