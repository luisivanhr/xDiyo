"""Materialize round annotations in a new local dataset; never build features."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd
from xdiyo_analytics.data.fixture_rounds import file_sha256, load_round_enriched_matches, SELECTED


def integrate(matches, mapping_root, output):
    matches, mapping_root, output = map(Path, (matches, mapping_root, output))
    if output.exists():
        raise FileExistsError("Dataset output already exists; use a new version directory")
    original = pd.read_parquet(matches)
    enriched = load_round_enriched_matches(matches, mapping_root)
    # Audit the whole source population, including the 84,512 untargeted rows.
    unchanged = [col for col in original if col not in SELECTED]
    pd.testing.assert_frame_equal(original[unchanged], enriched[unchanged])
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".pending-rounds-", dir=output.parent))
    try:
        path = stage / "Matches_Results.parquet"
        enriched.to_parquet(path, index=False, compression="zstd")
        restored = pd.read_parquet(path)
        pd.testing.assert_frame_equal(enriched, restored)
        if restored.attrs != enriched.attrs:
            raise ValueError("Round provenance lost during serialization")
        receipt = dict(schema=1, rows=len(enriched), input_sha256=file_sha256(matches),
            output_sha256=file_sha256(path), round_provenance=enriched.attrs["fixture_rounds"],
            changed_columns=list(SELECTED), original_columns_unchanged=True,
            serialized_values_types_and_provenance_verified=True,
            experiment_prepared=False, experiment_run=False, forecast_timing_changed=False)
        (stage / "integration.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        os.rename(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", required=True, type=Path)
    parser.add_argument("--mapping-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    integrate(args.matches, args.mapping_root, args.output)
