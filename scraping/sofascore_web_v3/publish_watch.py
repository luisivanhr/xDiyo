"""Publish newly completed Chrome queue cohorts through the verified offline exporter."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from scraping.sofascore.store import atomic_write, json_bytes
from .campaign import collector_lock
from .publish import ROOT, completed_season_stems, publish_captured_season
from .status_write import atomic_write_status


def queue_counts(directory):
    with closing(sqlite3.connect((directory / "queue.sqlite").as_uri() + "?mode=ro", uri=True)) as connection:
        return dict(connection.execute("SELECT state,COUNT(*) FROM queue GROUP BY state"))


def watch_publications(campaign_directory, *, output=ROOT / "data/xDiyo_data", watch=False,
                       interval_seconds=60, sleep=time.sleep):
    """Verify each ready cohort once per invocation; collection always stays read-only.

    A fresh invocation verifies existing publications again. Successful publications
    are not continually rehashed while waiting for another season. Errors stop this
    watcher and preserve the independent collector and all conflicting artifacts.
    """
    if not math.isfinite(interval_seconds) or interval_seconds < 1:
        raise ValueError("Publication polling interval must be at least one second")
    directory, output = Path(campaign_directory).resolve(), Path(output).resolve()
    config_path = directory / "config.json"
    config_stamp = (config_path.stat().st_size, config_path.stat().st_mtime_ns)
    queue_counts(directory)  # Fail before creating output if this is not an existing queue.
    state_directory = output / "_collection"
    run_id = uuid4().hex
    verified = {}
    counts = {}
    active_stem = None

    def record(state, *, event=False, publication=None, error=None):
        result = {"recorded_at": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
                  "pid": os.getpid(), "state": state, "campaign_directory": str(directory),
                  "counts": counts, "active_stem": active_stem,
                  "verified_in_this_run": verified, "error": error}
        atomic_write_status(state_directory / "chrome_publication_watch.json", json_bytes(result))
        if event:
            message = {key: value for key, value in result.items() if key != "verified_in_this_run"}
            message["verified_seasons"] = len(verified)
            if publication is not None:
                message["publication"] = publication
            with (state_directory / "chrome_publication_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(message, ensure_ascii=True) + "\n")
            print(json.dumps(message, ensure_ascii=True), flush=True)
        return result

    with collector_lock(state_directory / "chrome_publication_watch.lock"):
        record("starting", event=True)
        try:
            while True:
                stamp = (config_path.stat().st_size, config_path.stat().st_mtime_ns)
                if stamp != config_stamp:
                    raise ValueError("Campaign configuration changed; review it before restarting the export watcher")
                counts = queue_counts(directory)
                ready = completed_season_stems(directory)
                for stem in ready:
                    if stem in verified:
                        continue
                    active_stem = stem
                    record("publishing", event=True)
                    report = publish_captured_season(directory, stem, output=output)
                    verified[stem] = {"verified_at": datetime.now(timezone.utc).isoformat(),
                                      "publication_state": report["publication_state"],
                                      "file": str(output / report["season_file"]),
                                      "sha256": report["sha256"], "matches": report["matches"],
                                      "source_lineage": report["reconciliation"]["source_lineage"]}
                    record("published", event=True, publication=verified[stem])
                    active_stem = None
                counts = queue_counts(directory)
                if counts and set(counts) <= {"complete", "deferred"}:
                    # The final fixture can finish after the earlier cohort scan.
                    if any(stem not in verified for stem in completed_season_stems(directory)):
                        continue
                    return record("finished_with_gaps" if counts.get("deferred") else "finished", event=True)
                if not watch:
                    return record("pass_complete", event=True)
                record("waiting")
                sleep(interval_seconds)
        except KeyboardInterrupt:
            return record("stopped", event=True)
        except Exception as error:
            record("failed", event=True, error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/xDiyo_data")
    parser.add_argument("--watch", action="store_true", help="Keep waiting for further completed seasons")
    parser.add_argument("--interval-seconds", type=float, default=60)
    args = parser.parse_args()
    watch_publications(args.campaign, output=args.output, watch=args.watch, interval_seconds=args.interval_seconds)


if __name__ == "__main__":
    main()
