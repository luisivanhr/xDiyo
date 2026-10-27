"""Native prediction artifacts: JSON metadata, tensor weights, explicit history."""
from dataclasses import asdict
import json
from pathlib import Path
from .config import TemporalGraphConfig, CLASS_ORDER
from .adapter import TemporalGraphAdapter, _Backend, _torch
from .preprocessing import GraphPreprocessor
from .profiling import timed
from .execution import prepare_determinism


class TemporalGraphSerializer:
    """Use with save_model/load_model and immutable fingerprinted graph inputs.

    Native automatic saving embeds the store as ZSTD Parquet for portable UI
    loading. Explicit callers may omit it and supply the exact store at load.
    Its fingerprint is checked before inference. Load against the saved store first,
    then call attach_event_store to add later immutable evidence explicitly.
    """
    format_id = "xdiyo.temporal-team-gat.v2"

    def __init__(self, event_store=None, *, device="cpu", embed_event_store=False):
        self.event_store, self.device = event_store, device
        self.embed_event_store = embed_event_store

    @timed('checkpoint_export')
    def save(self, adapter, directory):
        torch = _torch()
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        backend = adapter.backend_
        metadata = dict(config=asdict(adapter.config), preprocessing=backend.preprocessor.state(),
            target_name=backend.target_name, class_order=CLASS_ORDER, output_classes=backend.output_classes,
            control=asdict(adapter.control),
            edge_feature_order=backend.preprocessor.edge_columns,
            fixture_feature_order=backend.preprocessor.fixture_columns,
            fixture_encoded_feature_order=backend.preprocessor.fixture_encoded_columns,
            fixture_encoded_dimension=backend.preprocessor.fixture_dim,
            fixture_gap_schema=backend.preprocessor.fixture_schema,
            fitting_boundary=backend.fitting_boundary.isoformat(),
            trained_through=backend.trained_through.isoformat(),
            model_version=backend.model_version, event_store_version=adapter.event_store.version,
            event_store_provenance=adapter.event_store.provenance,
            training_summary=adapter.training_summary_, inner_threads=adapter.inner_threads)
        if getattr(adapter, "input_cache_policy", None) is not None:
            metadata["input_cache_policy"] = asdict(adapter.input_cache_policy)
        if adapter.config.temporal_mode != 'snapshot':
            metadata['temporal_feature_order'] = backend.preprocessor.temporal_columns
        if adapter.config.statistics_forecast is not None:
            metadata['statistics_scale'] = backend.statistics_scale.state()
        (path/"graph.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        adapter.training_history_.to_parquet(path/"training_history.parquet", compression="zstd", index=False)
        if self.embed_event_store:
            from .store import TemporalEventStore
            if type(adapter.event_store) is TemporalEventStore:
                adapter.event_store._export_tables(path)
            else:
                for name in ('history', 'team_seasons', 'schedule'):
                    getattr(adapter.event_store, name).to_parquet(path/(name+'.parquet'), compression='zstd', index=False)
        torch.save({k: v.detach().cpu() for k, v in backend.model.state_dict().items()}, path/"weights.pt")

    def load(self, directory):
        import pandas as pd
        torch = _torch()
        from .network import TemporalNetwork
        path = Path(directory)
        meta = json.loads((path/"graph.json").read_text(encoding="utf-8"))
        store = self.event_store
        if store is None and (path/'history.parquet').exists():
            from .store import TemporalEventStore
            store = TemporalEventStore(pd.read_parquet(path/'history.parquet'),
                pd.read_parquet(path/'team_seasons.parquet'), schedule=pd.read_parquet(path/'schedule.parquet'),
                provenance=meta['event_store_provenance'])
        if store is None or store.version != meta["event_store_version"]:
            raise ValueError("Load requires the exact event_store matching the saved fingerprint.")
        if tuple(meta["class_order"]) != CLASS_ORDER:
            raise ValueError("Unsupported saved class order.")
        from .observed_gap import migrate_gap_config, BOUNDED_GAP_ENCODING
        config_state = migrate_gap_config(meta['config'])
        encoding_fields = {'time_encoding', 'age_scale_days', 'gap_scale_days'}
        present = encoding_fields.intersection(config_state)
        if present and present != encoding_fields:
            raise ValueError('Saved temporal encoding metadata is incomplete.')
        if not present:
            # Absence is a legacy migration, never the new constructor default.
            config_state['time_encoding'] = 'log1p_v1'
        if config_state.get('heatmaps') is not None:
            from .spatial import GraphHeatmaps
            config_state['heatmaps'] = GraphHeatmaps.from_dict(config_state['heatmaps'])
        from .extensions import BoundaryMemory, StatisticsForecast, GoalAuxiliary
        if config_state.get('goal_auxiliary') is not None:
            config_state['goal_auxiliary'] = GoalAuxiliary(**config_state['goal_auxiliary'])
        if config_state.get('boundary_memory') is not None:
            config_state['boundary_memory'] = BoundaryMemory(**config_state['boundary_memory'])
        if config_state.get('statistics_forecast') is not None:
            config_state['statistics_forecast'] = StatisticsForecast.from_dict(config_state['statistics_forecast'])
        config = TemporalGraphConfig(**config_state)
        from ..training.control import TrainingControl, EarlyStopping, ReduceOnPlateau
        control = dict(meta.get('control', {}))
        for key, cls in [('early_stopping', EarlyStopping), ('scheduler', ReduceOnPlateau)]:
            if control.get(key) is not None:
                control[key] = cls(**control[key])
        adapter = TemporalGraphAdapter(config, store, control=TrainingControl(**control), inner_threads=meta["inner_threads"])
        adapter.configure_device(self.device)
        if meta.get("input_cache_policy") is not None:
            adapter.configure_input_cache(**meta["input_cache_policy"])
        prepare_determinism(config, adapter.device)
        backend = _Backend(config, store, 0, adapter.device, getattr(adapter, "input_cache_policy", None))
        backend.preprocessor = GraphPreprocessor.from_state(config, meta["preprocessing"])
        p = backend.preprocessor
        if tuple(meta["edge_feature_order"]) != p.edge_columns or tuple(meta["fixture_feature_order"]) != p.fixture_columns:
            raise ValueError("Saved feature order disagrees with the architecture configuration.")
        fields = {'fixture_encoded_feature_order', 'fixture_encoded_dimension', 'fixture_gap_schema'}
        present = fields.intersection(meta)
        if present and present != fields:
            raise ValueError('Saved encoded fixture metadata is incomplete.')
        if config.fixture_gap_encoding == BOUNDED_GAP_ENCODING and not present:
            raise ValueError('Corrected artifacts require explicit encoded fixture metadata.')
        if present and (tuple(meta['fixture_encoded_feature_order']) != p.fixture_encoded_columns or
                        meta['fixture_encoded_dimension'] != p.fixture_dim or meta['fixture_gap_schema'] != p.fixture_schema):
            raise ValueError('Saved encoded fixture metadata disagrees with preprocessing.')
        if tuple(meta.get('temporal_feature_order', ())) != p.temporal_columns:
            raise ValueError('Saved temporal feature order disagrees with the architecture configuration; '
                             'legacy temporal artifacts require the logarithmic feature order.')
        if config.statistics_forecast is not None:
            from .statistics import StatisticsScale
            backend.statistics_scale = StatisticsScale.from_state(config.statistics_forecast, meta['statistics_scale'])
        # Loading weights must not advance the caller's initialization RNG.
        with torch.random.fork_rng(devices=[]):
            backend.model = TemporalNetwork(config, len(p.leagues),
                p.edge_dim,
                p.fixture_dim).to(adapter.device)
        backend.model.load_state_dict(torch.load(path/"weights.pt", map_location=adapter.device, weights_only=True))
        backend.target_name = meta["target_name"]
        backend.output_classes = tuple(meta.get("output_classes", CLASS_ORDER))
        if backend.output_classes != CLASS_ORDER:
            raise ValueError('Saved output classes must use native home-perspective 1/0/-1.')
        backend.fitting_boundary = pd.Timestamp(meta["fitting_boundary"]).as_unit('ns')
        backend.trained_through = pd.Timestamp(meta["trained_through"]).as_unit('ns')
        backend.refresh_version()
        if backend.model_version != meta["model_version"]:
            raise ValueError("Model version does not match saved metadata.")
        adapter.backend_ = backend
        adapter.training_summary_ = meta["training_summary"]
        adapter.training_history_ = pd.read_parquet(path/"training_history.parquet") if (path/"training_history.parquet").exists() else pd.DataFrame()
        return adapter
