"""Native prediction artifacts: JSON metadata, tensor weights, explicit history."""
from dataclasses import asdict
import json
from pathlib import Path
from .config import TemporalGraphConfig, CLASS_ORDER
from .adapter import TemporalGraphAdapter, _Backend, _torch
from .preprocessing import GraphPreprocessor


class TemporalGraphSerializer:
    """Use with save_model/load_model; provide the exact event store at load.

    The artifact does not duplicate source datasets. Its fingerprint is checked
    against event_store before inference. Load against the saved store first,
    then call attach_event_store to add later immutable evidence explicitly.
    """
    format_id = "xdiyo.temporal-team-gat.v1"

    def __init__(self, event_store=None, *, device="cpu"):
        self.event_store, self.device = event_store, device

    def save(self, adapter, directory):
        torch = _torch()
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        backend = adapter.backend_
        metadata = dict(config=asdict(adapter.config), preprocessing=backend.preprocessor.state(),
            target_name=backend.target_name, class_order=CLASS_ORDER,
            edge_feature_order=backend.preprocessor.edge_columns,
            fixture_feature_order=backend.preprocessor.fixture_columns,
            fitting_boundary=backend.fitting_boundary.isoformat(),
            trained_through=backend.trained_through.isoformat(),
            model_version=backend.model_version, event_store_version=adapter.event_store.version,
            event_store_provenance=adapter.event_store.provenance,
            training_summary=adapter.training_summary_, inner_threads=adapter.inner_threads)
        (path/"graph.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        torch.save({k: v.detach().cpu() for k, v in backend.model.state_dict().items()}, path/"weights.pt")

    def load(self, directory):
        import pandas as pd
        torch = _torch()
        from .network import TemporalNetwork
        path = Path(directory)
        meta = json.loads((path/"graph.json").read_text(encoding="utf-8"))
        if self.event_store is None or self.event_store.version != meta["event_store_version"]:
            raise ValueError("Load requires the exact event_store matching the saved fingerprint.")
        if tuple(meta["class_order"]) != CLASS_ORDER:
            raise ValueError("Unsupported saved class order.")
        config = TemporalGraphConfig(**meta["config"])
        adapter = TemporalGraphAdapter(config, self.event_store, inner_threads=meta["inner_threads"])
        adapter.configure_device(self.device)
        backend = _Backend(config, self.event_store, 0, adapter.device)
        backend.preprocessor = GraphPreprocessor.from_state(config, meta["preprocessing"])
        p = backend.preprocessor
        if tuple(meta["edge_feature_order"]) != p.edge_columns or tuple(meta["fixture_feature_order"]) != p.fixture_columns:
            raise ValueError("Saved feature order disagrees with the architecture configuration.")
        backend.model = TemporalNetwork(config, len(p.leagues),
            2*len(p.edge_columns)+1+len(p.leagues)+1,
            2*len(p.fixture_columns)+len(p.leagues)+1).to(adapter.device)
        backend.model.load_state_dict(torch.load(path/"weights.pt", map_location=adapter.device, weights_only=True))
        backend.target_name = meta["target_name"]
        backend.fitting_boundary = pd.Timestamp(meta["fitting_boundary"])
        backend.trained_through = pd.Timestamp(meta["trained_through"])
        backend.refresh_version()
        if backend.model_version != meta["model_version"]:
            raise ValueError("Model version does not match saved metadata.")
        adapter.backend_ = backend
        adapter.training_summary_ = meta["training_summary"]
        adapter.training_history_ = pd.DataFrame()
        return adapter
