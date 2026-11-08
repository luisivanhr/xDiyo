"""Package the audited V59 data and provenance without preparing an experiment."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd
from xdiyo_analytics.data.fixture_rounds import (
    COMPACT, LEDGER, RELEASE, SELECTED, file_sha256, load_fixture_rounds,
    project_selections, validate_audit,
)

PINNED = {
    COMPACT: "3840a0b53928e89dd67eb1dfffe76826505b258c3e9650f8cc7a52c393ff8f27",
    LEDGER: "a89c4ceca4cdbcb5d322d582edaaeb36680705cb155c0186e6532495976aa8c6",
    "v59_full_mapping_REVIEW_NOT_APPROVED.parquet": "75919810aaf2375af77c3e5b3c87eec64fc10ea75fdddc67e6dcfa44dc0a7f3c",
    "v59_held_794_queue_REVIEW.csv": "7106e1ae32fd6cdef59b0248f2d1374987d40af9ef98549af0ca556eead2056d",
}
SUPPORTING = (
    "v54_australia_source_specific_claims_REVIEW.csv",
    "v56_australia_source_table_rows_REVIEW.csv",
    "v56_australia_distinct_league_draw_claims_REVIEW.csv",
    "v56_australia_161_target_round_comparison_REVIEW.csv",
    "v57_australia_161_identity_and_round_comparison_REVIEW.csv",
    "v58_four_sofascore_event_observations_REVIEW.csv",
    "v45_mls_series_game_lineage_REVIEW.csv",
    "v59_validation.json",
    "v59_user_selected_sofascore_fallback_REVIEW_20261108.py",
    "INDEPENDENT_V59_FINAL_TERRA_AUDIT_20261108.md",
    "INDEPENDENT_V59_FOUR_SOFASCORE_FALLBACK_AUDIT_20261108.md",
    "INDEPENDENT_V56_TERRA_AUDIT_20261108.md",
    "V59_USER_SELECTED_SOFASCORE_FALLBACK_AUDIT_HANDOFF_20261108.md",
    "V58_FOUR_SOFASCORE_EVENT_OBSERVATIONS_REVIEW_20261108.md",
    "V47_REAUDIT_HANDOFF_20261108.md",
)


def publish(review, inventory, output):
    review, inventory, output = map(Path, (review, inventory, output))
    if output.exists():
        raise FileExistsError("Immutable release already exists; inspect it instead of overwriting")
    for name, checksum in PINNED.items():
        if file_sha256(review / name) != checksum:
            raise ValueError(f"Audited input checksum changed: {name}")
    compact = pd.read_parquet(review / COMPACT)
    ledger = pd.read_parquet(review / LEDGER)
    queue = pd.read_csv(review / "v59_held_794_queue_REVIEW.csv")
    counts = validate_audit(compact, ledger, queue)
    old = pd.read_parquet(review / "v58_field_lineage_REVIEW_ONLY.parquet")
    keys = ["match_id", "field"]
    a = ledger.set_index(keys).sort_index().base_field_conflict_gate_status
    b = old.set_index(keys).sort_index().base_field_conflict_gate_status
    pd.testing.assert_series_equal(a, b)
    fixtures = project_selections(compact, pd.read_parquet(inventory / "target_fixtures.parquet"))
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".pending-v59-", dir=output.parent))
    try:
        provenance = stage / "provenance"
        provenance.mkdir()
        for name in (*PINNED, *SUPPORTING):
            shutil.copyfile(review / name, provenance / name)
        # Preserve every original candidate claim byte, including historical
        # outcome-bearing research text, solely in the provenance directory.
        candidates = inventory / "all_source_candidates_NOT_FOR_INTEGRATION.csv"
        with candidates.open("rb") as source, (provenance / (candidates.name + ".gz")).open("wb") as dest:
            with gzip.GzipFile(filename="", mode="wb", fileobj=dest, mtime=0) as zipped:
                shutil.copyfileobj(source, zipped)
        fixtures.to_parquet(stage / "fixtures.parquet", index=False, compression="zstd")
        manifest = dict(schema=1, release=RELEASE, identity_namespace="workbook_match_id",
            selected_columns=SELECTED, counts=counts,
            integration_status="AUDITED_STATUS_AWARE_DATASET_INTEGRATION",
            selection_semantics="MIXED_ORIGINAL_AND_PROVIDER_CONVENTIONS",
            model_input_approved=False, feature_asof_approved=False, experiment_prepared=False,
            authoritative_null_policy="REPLACE_WITH_SELECTED_NULL; NEVER_COALESCE_HISTORICAL_VALUES",
            audit="provenance/INDEPENDENT_V59_FINAL_TERRA_AUDIT_20261108.md",
            historical_approval_fields="Preserved unchanged; not current integration permission",
            source_comparisons="V56/V57 are historical snapshots, not current computations",
            lineage_pointers="Original research filenames and URLs retained verbatim; not availability timestamps",
            original_candidate_claims_sha256=file_sha256(candidates),
            original_target_inventory_sha256=file_sha256(inventory / "target_fixtures.parquet"),
            base_conflict_gates_equal_v58=True,
            v58_ledger_sha256=file_sha256(review / "v58_field_lineage_REVIEW_ONLY.parquet"),
            files={str(p.relative_to(stage)).replace(os.sep, "/"): dict(sha256=file_sha256(p), bytes=p.stat().st_size)
                   for p in sorted(stage.rglob("*")) if p.is_file()})
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        load_fixture_rounds(stage)
        os.rename(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(json.dumps(dict(release=str(output), **counts), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", required=True, type=Path)
    parser.add_argument("--inventory-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    publish(args.review_dir, args.inventory_dir, args.output)
