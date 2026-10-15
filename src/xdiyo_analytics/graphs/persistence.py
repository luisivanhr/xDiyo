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
            fitting_boundary=backend.fitting_boundary.isoformat(),
            trained_through=backend.trained_through.isoformat(),
            model_version=backend.model_version, event_store_version=adapter.event_store.version,
            event_store_provenance=adapter.event_store.provenance,
            training_summary=adapter.training_summary_, inner_threads=adapter.inner_threads)
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
        config_state = dict(meta['config'])
        if config_state.get('heatmaps') is not None:
            from .spatial import GraphHeatmaps
            config_state['heatmaps'] = GraphHeatmaps.from_dict(config_state['heatmaps'])
        config = TemporalGraphConfig(**config_state)
        from ..training.control import TrainingControl, EarlyStopping, ReduceOnPlateau
        control = dict(meta.get('control', {}))
        for key, cls in [('early_stopping', EarlyStopping), ('scheduler', ReduceOnPlateau)]:
            if control.get(key) is not None:
                control[key] = cls(**control[key])
        adapter = TemporalGraphAdapter(config, store, control=TrainingControl(**control), inner_threads=meta["inner_threads"])
        adapter.configure_device(self.device)
        prepare_determinism(config, adapter.device)
        backend = _Backend(config, store, 0, adapter.device)
        backend.preprocessor = GraphPreprocessor.from_state(config, meta["preprocessing"])
        p = backend.preprocessor
        if tuple(meta["edge_feature_order"]) != p.edge_columns or tuple(meta["fixture_feature_order"]) != p.fixture_columns:
            raise ValueError("Saved feature order disagrees with the architecture configuration.")
        # Loading weights must not advance the caller's initialization RNG.
        with torch.random.fork_rng(devices=[]):
            backend.model = TemporalNetwork(config, len(p.leagues),
                2*len(p.edge_columns)+1+len(p.leagues)+1,
                2*len(p.fixture_columns)+len(p.leagues)+1).to(adapter.device)
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
