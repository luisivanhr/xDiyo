"""Optional, inspectable CSV exports and evaluated-model round-trip checks."""

from dataclasses import dataclass
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd


@dataclass(frozen=True)
class ArtifactExport:
    """Export retained results without fitting. Each invocation gets a new folder.

    Tables retain row indexes and all report scopes. Model verification predicts
    the retained test rows again and compares them with their saved predictions.
    It requires the prepared dataset and loaded models; no training is performed.
    """
    report_tables: bool = True
    predictions: bool = True
    save_models: bool = True
    verify_reload: bool = True
    serializer: object = None

    def run(self, result):
        from ..training import save_model, load_model
        if result.path is None:
            raise ValueError("Artifact export requires a saved experiment run.")
        if self.verify_reload and not self.save_models:
            raise ValueError("Reload verification requires save_models=True.")
        if self.verify_reload and result.dataset is None:
            raise ValueError("Reload verification requires the prepared dataset.")
        destination = Path(result.path) / "exports" / str(uuid4())
        destination.mkdir(parents=True)
        manifest = {"run_id": result.record.get("run_id"), "tables": [], "models": []}

        def table(frame, label, **scope):
            name = f"table_{len(manifest['tables']):03d}.csv"
            frame.to_csv(destination / name, index=True)
            manifest["tables"].append({"name": label, "file": name, **scope})

        if result.dataset is not None:
            table(pd.DataFrame({"feature": result.dataset.X.columns,
                                "dtype": result.dataset.X.dtypes.astype(str).to_numpy()}), "feature_manifest")
        if self.report_tables:
            for study in result.report.studies:
                for key, frame in study.result.tables.items():
                    table(frame, key, study=study.name, fold_id=study.fold_id,
                          type=study.type, partition=study.partition)
        for fold in result.training.folds:
            if self.predictions:
                table(fold.y_true, "observed", fold_id=fold.fold_id)
                table(fold.metadata, "match_metadata", fold_id=fold.fold_id)
                for key, frame in fold.predictions.items():
                    table(frame, key, fold_id=fold.fold_id)
            table(pd.DataFrame({"feature": fold.feature_columns}), "selected_features", fold_id=fold.fold_id)
            if self.save_models:
                path = destination / f"model_{len(manifest['models']):03d}"
                save_model(result.training, path, fold_id=fold.fold_id, serializer=self.serializer)
                if self.verify_reload:
                    outputs = load_model(path, serializer=self.serializer).predict(result.dataset, positions=fold.test_positions)
                    if set(outputs) != set(fold.predictions):
                        raise AssertionError("Reloaded model prediction outputs differ.")
                    for key, frame in outputs.items():
                        pd.testing.assert_frame_equal(frame, fold.predictions[key], check_exact=False,
                                                      rtol=1e-10, atol=1e-12)
                manifest["models"].append({"fold_id": fold.fold_id, "path": path.name,
                                           "reload_verified": self.verify_reload})
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        return destination
