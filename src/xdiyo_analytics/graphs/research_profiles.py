"""Versioned retained architectures; folds, seeds and fitted state are separate."""
from .config import TemporalGraphConfig
from .extensions import GoalAuxiliary
from .phase_context import PhaseRankFeatures, CORRECTED_POLICY

RETAINED_STATISTICS = (
    'team::ALL::Match overview::totalShotsOnGoal::value',
    'team::ALL::Shots::shotsOffGoal::value',
    'team::ALL::Match overview::cornerKicks::value',
    'opponent::ALL::Match overview::cornerKicks::value',
)


def _config(beta, phase):
    # Explicitly fix every constructor field: future defaults cannot change V1.
    return TemporalGraphConfig(
        timezone='Europe/Berlin', forecast_policy='by_round',
        round_keys=('competition_id', 'season_id', 'round_stage', 'round'),
        embedding_dim=32, layers=2, heads=2, half_life_days=180.,
        seasonal_retention=.5, support_kappa=5., hidden_width=64,
        statistics=RETAINED_STATISTICS, fixture_features=(), learning_rate=.001,
        weight_decay=.001, prior_penalty=.001, heatmaps=None,
        include_goal_features=True, deterministic=True, boundary_memory=None,
        statistics_forecast=None, temporal_mode='time_features',
        time_scales_days=(7., 30., 180.), time_encoding='rational_decay_v1',
        age_scale_days=180., gap_scale_days=7.,
        goal_auxiliary=GoalAuxiliary(beta=beta, rate_floor=1e-6, detach_encoder=False,
            architecture='linear', likelihood='poisson', competitions=(),
            initial_alpha=.1, dispersion_learning_rate=.05),
        fixture_gap_encoding='bounded_observed_gap_v1',
        observed_gap_scale_days=7., observed_gap_recent_days=30.,
        sender_state_mode='updated', phase_rank_features=phase,
        statistic_availability_policy='per_stat_release_v1')


def GATResearchBNV1():
    """B/N: retained temporal attention, shots/corners, materialized beta-zero head."""
    return _config(0., None)


def GATResearchCE1V1(source_id: str):
    """C/E1: historical opponent ranks and attached beta=0.1 goal supervision."""
    return _config(.1, PhaseRankFeatures('E1', source_id, None, CORRECTED_POLICY))


def GATResearchBCDDWV1(source_id: str, window: int):
    """B/CD_DW: countdown and an explicitly selected retained rank window."""
    if type(window) is not int or window not in (10, 20):
        raise ValueError('Retained CD_DW V1 requires an explicit window of 10 or 20.')
    return _config(0., PhaseRankFeatures('CD_DW', source_id, window, CORRECTED_POLICY))


def profile_key(config):
    """Descriptive expanded-profile provenance, never an identity authority."""
    phase = config.phase_rank_features
    if phase is None:
        return 'graphs.GATResearchBNV1' if config == GATResearchBNV1() else None
    if phase.policy == CORRECTED_POLICY:
        if phase.route == 'E1' and config == GATResearchCE1V1(phase.source_id):
            return 'graphs.GATResearchCE1V1'
        if phase.route == 'CD_DW' and phase.window in (10,20) and config == GATResearchBCDDWV1(phase.source_id,phase.window):
            return 'graphs.GATResearchBCDDWV1'
    return None
