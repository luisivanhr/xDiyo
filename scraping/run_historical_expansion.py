"""Prepare/resume the requested historical seasons and keep their converter open.

Run from any directory. Chrome owns all Sofascore discovery and capture.
Use --prepare-only to print IDs, --once to check/publish once, or --season 16/17
to watch one saved year independently. Ctrl+C stops this watcher, not Chrome.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scraping.sofascore_web_v3.campaign import read_json
from scraping.sofascore_web_v3.historical_batch import (
    prepare_historical_batch, print_batch, watch_historical_batch,
)

REQUESTS = ROOT / "scraping/historical_expansion_2015_2020.json"
INCLUDE_LINEUPS = True
RANDOM_DELAY_MAX_SECONDS = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--season", choices=["15/16", "16/17", "17/18", "18/19", "19/20"])
    args = parser.parse_args()
    batch = prepare_historical_batch(read_json(REQUESTS), name="historical-expansion-2015-2020",
        include_lineups=INCLUDE_LINEUPS, random_delay_max_seconds=RANDOM_DELAY_MAX_SECONDS)
    print_batch(batch)
    if not args.prepare_only:
        try:
            watch_historical_batch(batch, season=args.season, once=args.once)
        except KeyboardInterrupt:
            print("Watcher stopped. Re-run this script to resume using the same saved IDs.")


if __name__ == "__main__":
    main()
