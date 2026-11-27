"""Print a pinned recipe section for the November 2026 published statistics release.

Defaults to the Git checkout. An explicit pilot root can reproduce the original local selection.
"""
import argparse
import hashlib
import json
from pathlib import Path


BASE_MANIFESTS = {
    "release/footiqo-only/manifest.json": "9b9ac09dc9f4dd12267fd2a0e981b6597d309220f1ac1b7ef5641eb23533c9c2",
    "api-football/releases/e8f31994f10e38fb/manifest.json": "c114fb803e465ada76ef6710b7cf9e78f39cb20312ec802851348687b0e06a89",
    "selected/f18695f5136b0798/manifest.json": "6da7911bd541ffe5686a90f055e702a3d276ca7946bdf8ba69e1065e89e149c7",
}
FULL_TIME_KEYS = ["cornerKicks", "totalShotsOnGoal", "shotsOnGoal", "shotsOffGoal", "ballPossession"]
SNAPSHOT = "multi-site/campaign-20261126/verified-snapshots/20261127T082851Z"
OMIYA = "multi-site/campaign-20261126-highest-shots-64ed978bffed3c57"
CAMPAIGN_PARTITIONS = (
    (SNAPSHOT, FULL_TIME_KEYS, "5275a57f41fea9b30dc2783e19b359c63393ebc10f35cd222d1a06f594af380b"),
    (OMIYA, ["shotsOffGoal"], "64ed978bffed3c5756e7a1787693b20d470bc11073600e6d1f0aa78e93845eb6"),
)


def checked(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Statistics release checksum mismatch: {path}")
    return raw


PUBLISHED_ROOT = Path(__file__).resolve().parents[1] / "data/statistics_backfill/2026-11-27"
PUBLISHED_SHA256 = "82871f845767583a564b974a506a645644f5677cd87340020a387cf16cb199d5"


def recipe_section(root=PUBLISHED_ROOT):
    root = Path(root).resolve()
    if (root / "manifest.json").is_file() or root == PUBLISHED_ROOT:
        path = root / "manifest.json"
        manifest = json.loads(checked(path, PUBLISHED_SHA256))
        for part in manifest["partitions"]:
            checked(root / part["path"], part["sha256"])
        return {"statistics_overlay": {"manifest_path": str(path), "manifest_sha256": PUBLISHED_SHA256,
                                       "keys": FULL_TIME_KEYS.copy()}}
    partitions = []
    for name, pin in BASE_MANIFESTS.items():
        manifest = json.loads(checked(root / name, pin))
        for part in manifest["partitions"]:
            path = Path(name).parent / part["path"]
            partitions.append({**part, "path": path.as_posix()})
    for folder, keys, pin in CAMPAIGN_PARTITIONS:
        if (root / folder / "superseded.json").exists():
            raise ValueError(f"Superseded collection evidence: {folder}")
        partitions.append({"path": f"{folder}/candidates.jsonl", "keys": keys, "sha256": pin})
    for part in partitions:
        checked(root / part["path"], part["sha256"])
    version = "statistics-20261127-omiya"
    manifest = {"version": version, "partitions": partitions}
    raw = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    pin = hashlib.sha256(raw).hexdigest()
    path = root / f"{version}-{pin[:16]}.json"
    try:
        with path.open("xb") as output:
            output.write(raw)
    except FileExistsError:
        checked(path, pin)
    return {"statistics_overlay": {"manifest_path": str(path), "manifest_sha256": pin,
                                   "keys": FULL_TIME_KEYS.copy()}}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=PUBLISHED_ROOT)
    print(json.dumps(recipe_section(cli.parse_args().root), indent=2))
