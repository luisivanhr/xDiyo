"""Explicit input semantics and evidence-backed import identity projections."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from .compatibility_records import RETAINED_EXPORTS

LEGACY = "legacy_unmasked_v1"
RELEASE = "per_stat_release_v1"
MAIN_IDENTITY = "legacy_main_v1"
RESEARCH_IDENTITY = "research_original_v1"
EXPLICIT_IDENTITY = "explicit_statistic_policy_v1"


def main_representable(config):
    return (config.temporal_mode == "snapshot"
            and config.fixture_gap_encoding == "legacy_standardized_gap_v1"
            and config.sender_state_mode == "updated"
            and config.goal_auxiliary is None and config.phase_rank_features is None
            and config.statistic_availability_policy == LEGACY)


def default_identity(config):
    return MAIN_IDENTITY if main_representable(config) else EXPLICIT_IDENTITY


def identity_config(config, schema=None):
    schema = schema or default_identity(config)
    if schema not in (MAIN_IDENTITY, RESEARCH_IDENTITY, EXPLICIT_IDENTITY):
        raise ValueError("Unsupported graph identity schema.")
    if schema == MAIN_IDENTITY and not main_representable(config):
        raise ValueError("Active research semantics cannot use the main identity projection.")
    config_state = asdict(config)
    if config_state['phase_rank_features'] is None:
        del config_state['phase_rank_features']  # Preserve all pre-context model versions.
    from .observed_gap import LEGACY_GAP_ENCODING, GAP_CONFIG_FIELDS
    if config.fixture_gap_encoding == LEGACY_GAP_ENCODING:
        for name in GAP_CONFIG_FIELDS:
            del config_state[name]
    if config.sender_state_mode == 'updated':
        del config_state['sender_state_mode']  # Original propagation fingerprint.
    if config_state['goal_auxiliary'] is None:
        del config_state['goal_auxiliary']  # Preserve every pre-goal model fingerprint.
    elif config.goal_auxiliary.architecture == 'linear':
        # New fields are inactive defaults for the original flexible head.
        for name in ('likelihood', 'competitions', 'initial_alpha', 'dispersion_learning_rate'):
            config_state['goal_auxiliary'].pop(name)
    if config.temporal_mode == 'snapshot' or config.time_encoding == 'log1p_v1':
        # Logarithmic encoding was implicit in legacy temporal fingerprints.
        # Its rational scales are inactive. Rational models include all three.
        for name in ('time_encoding', 'age_scale_days', 'gap_scale_days'):
            del config_state[name]
    if config_state['temporal_mode'] == 'snapshot':
        del config_state['temporal_mode']
        del config_state['time_scales_days']  # Preserve existing baseline artifact fingerprints.
    if config.statistics_forecast is not None:
        from .extensions import target_schema
        config_state["statistics_forecast"]["targets"] = [target_schema(t) for t in config.statistics_forecast.targets]
    for name in ('boundary_memory', 'statistics_forecast'):
        if config_state[name] is None:
            del config_state[name]
    if not config_state['deterministic']:
        del config_state['deterministic']  # Preserve pre-option artifact fingerprints.
    if config_state['include_goal_features']:
        del config_state['include_goal_features']  # Preserve pre-option artifact fingerprints.
    if config_state['heatmaps'] is None:
        del config_state['heatmaps']  # Preserve existing scalar artifact fingerprints.
    else:
        reconstruction = config_state['heatmaps']['encoder']['reconstruction']
        if reconstruction is not None and reconstruction.get('loss') == 'mse':
            del reconstruction['loss']  # MSE was the original auxiliary default.
    if schema != EXPLICIT_IDENTITY:
        config_state.pop("statistic_availability_policy")
    return config_state


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def _json_digest(value):
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class GraphCompatibility:
    statistic_policy: str
    identity_schema: str
    evidence: str
    origin_graph_sha256: str | None = None

    def metadata(self):
        return asdict(self)


def resolve_graph_compatibility(meta, directory, store):
    """Resolve before constructing tensors; never infer policy from a filename."""
    graph_hash, weight_hash = _digest(directory / "graph.json"), _digest(directory / "weights.pt")
    config = meta["config"]
    explicit = config.get("statistic_availability_policy")
    declaration = meta.get("graph_compatibility")
    known = next((r for r in RETAINED_EXPORTS
                  if r.graph_sha256 == graph_hash or r.model_version == meta["model_version"]), None)
    if known is not None and declaration is None:
        if (known.graph_sha256 != graph_hash or known.weights_sha256 != weight_hash
                or known.model_version != meta["model_version"]
                or known.event_store_version != meta["event_store_version"]):
            raise ValueError("Known retained export hash or identity changed; original evidence is required.")
        return GraphCompatibility(RELEASE, RESEARCH_IDENTITY, "verified_retained_manifest", graph_hash)
    if explicit is not None or declaration is not None:
        if explicit not in (LEGACY, RELEASE) or declaration is None:
            raise ValueError("Explicit statistic policy and graph identity metadata must both be present.")
        resolved = GraphCompatibility(**declaration)
        if resolved.statistic_policy != explicit:
            raise ValueError("Saved statistic policy disagrees with its identity metadata.")
        schema = resolved.identity_schema
        if schema not in (MAIN_IDENTITY, RESEARCH_IDENTITY, EXPLICIT_IDENTITY):
            raise ValueError("Unsupported graph identity schema.")
        if schema == RESEARCH_IDENTITY:
            if resolved.origin_graph_sha256 is not None:
                record = next((r for r in RETAINED_EXPORTS if r.graph_sha256 == resolved.origin_graph_sha256), None)
                original = dict(config); original.pop("statistic_availability_policy")
                if (record is None or explicit != RELEASE or record.weights_sha256 != weight_hash
                        or record.model_version != meta["model_version"]
                        or record.config_sha256 != _json_digest(original)
                        or record.preprocessing_sha256 != _json_digest(meta["preprocessing"])):
                    raise ValueError("Imported research identity requires unchanged evidence-backed weights and semantics.")
            elif config.get("statistics"):
                has_releases = any('available_at::'+col in store._history for col in config['statistics'])
                if explicit != LEGACY or has_releases:
                    raise ValueError("Research statistic identity requires a validated frozen-source compatibility record.")
        return resolved
    research_fields = {"temporal_mode", "fixture_gap_encoding", "goal_auxiliary",
                       "phase_rank_features", "sender_state_mode", "time_encoding"}
    research = bool(research_fields.intersection(config))
    statistics = config.get("statistics", ())
    releases = any("available_at::" + col in store._history for col in statistics)
    if research and statistics and releases:
        raise ValueError("Research export with released statistics lacks a validated frozen-source compatibility record.")
    return GraphCompatibility(LEGACY, RESEARCH_IDENTITY if research else MAIN_IDENTITY,
                              "inactive_statistics" if research and not statistics else "legacy_metadata")
