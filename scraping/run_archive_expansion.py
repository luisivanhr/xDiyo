"""Prepare the three archive selections; watch one only when explicitly requested.

Default: local preparation/ID listing, no browser or provider traffic.
--watch GROUP waits for discovery/collection and uses the existing converter.
--status prints saved progress without starting a converter.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scraping.sofascore_web_v3.archive_discovery import prepare_archive
from scraping.sofascore_web_v3.archive_formats import requested_archives
from scraping.sofascore_web_v3.discovery import open_discovery
from scraping.sofascore_web_v3.publish_watch import watch_publications

INCLUDE_LINEUPS = True
RANDOM_DELAY_MAX_SECONDS = None
# Settled in archive_expansion_selection.json before preparing Japan.
SELECTION = ROOT / "scraping/archive_expansion_selection.json"


def prepare_selected():
    settings = json.loads(SELECTION.read_text(encoding="utf-8"))
    selection = requested_archives(japan_transition=settings["japan_2026_j2_j3"])
    batches = {}
    for name, cohorts in selection.items():
        # Save the complete proposed Japan selection in the selection file's
        # documented rules, but don't mint an unusable ID before the scope reply.
        if any(s.get("scope_pending") for s in cohorts):
            print(f"{name}: pending the spring-2026 J2/J3 scope decision")
            continue
        batch = prepare_archive(name, cohorts, include_lineups=INCLUDE_LINEUPS,
                                random_delay_max_seconds=RANDOM_DELAY_MAX_SECONDS)
        batches[name] = batch
        print(f"{name}: {batch['discovery_id']} | {len(cohorts)} league-seasons/tournaments")
        print(f"  Saved: {batch['path']}")
    return batches


def watch_archive(batch, *, sleep=time.sleep):
    previous = None
    while True:
        with open_discovery(batch["directory"], read_only=True) as discovery:
            status = discovery.status()
        state = (status["jobs"], status["ready_to_collect"], status.get("campaign"))
        if state != previous:
            print(f"Discovery {status['discovery_id']}: {status['jobs']}; ready={status['ready_to_collect']}", flush=True)
            previous = state
            if status["discovery_complete"] and not status["ready_to_collect"]:
                for cohort in status["seasons"]:
                    if cohort["validation_issues"]:
                        print(f"Review {cohort['league']} {cohort['season']}: {cohort['validation_issues']}", flush=True)
            elif status["ready_to_collect"] and not status["campaign"]:
                print("Use Create collection campaign, then Start / resume in xDiyo.", flush=True)
        if status["campaign"]:
            campaign = status["campaign"]
            print(f"Campaign ID: {campaign['campaign_id']}. Use Start / resume in xDiyo.", flush=True)
            result = watch_publications(campaign["directory"], output=batch["settings"]["output_root"],
                                        watch=True, interval_seconds=5)
            if result["state"] == "finished":
                expected = {s["stem"] for s in status["seasons"]}
                if set(result["verified_in_this_run"]) != expected:
                    raise ValueError("Published cohort list does not match this archive campaign")
                print(f"All {len(expected)} selected outputs verified in data/xDiyo_data.", flush=True)
            return result
        sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", choices=list(requested_archives()))
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    batches = prepare_selected()
    if args.status:
        for name, batch in batches.items():
            with open_discovery(batch["directory"], read_only=True) as discovery:
                result = discovery.status()
            print(json.dumps({"name": name, **result}, indent=2))
    if args.watch:
        if args.watch not in batches:
            raise SystemExit("Resolve this group's pending scope in archive_expansion_selection.json first.")
        try:
            watch_archive(batches[args.watch])
        except KeyboardInterrupt:
            print("Converter watcher stopped. Chrome and the saved queue are unchanged.")


if __name__ == "__main__":
    main()
