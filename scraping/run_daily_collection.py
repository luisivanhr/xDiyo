"""Edit the settings, run this script, then follow the printed Chrome steps.

Resume a saved run: python scraping/run_daily_collection.py --resume discover-...
See scraping/DAILY_COLLECTION_README.md for operation and output loading.
"""
from pathlib import Path
import argparse
import sys

# Allows running this small script directly from any working directory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scraping.sofascore_web_v3.daily import prepare_daily_run, watch_daily_run

SEASON = "26/27"                 # Explicit split-year cycle.
CALENDAR_SEASON = "2026"         # Calendar-year leagues; change independently at New Year.
LEAGUES = None                   # Original 13 + all 32 new leagues. Or a list of catalog names.
SEASON_OVERRIDES = {}            # Optional league-name -> explicit year/cycle.
FIXTURE_WINDOW_DAYS = 14         # Future regular fixtures; lineups still use the shorter window.
INCLUDE_LINEUPS = True
RANDOM_DELAY_MAX_SECONDS = None  # Or T: an extra independent uniform wait from 0 to T.
LINEUP_WINDOW_HOURS = 24          # Upcoming pregame/lineup requests inside this window.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", help="Saved Discovery ID; uses that run's original settings")
    parser.add_argument("--prepare-only", action="store_true", help="Print the ID without waiting")
    parser.add_argument("--once", action="store_true", help="Check once and publish if collection has finished")
    args = parser.parse_args()
    discovery_id = args.resume or prepare_daily_run(season=SEASON, leagues=LEAGUES,
        calendar_season=CALENDAR_SEASON, season_overrides=SEASON_OVERRIDES, fixture_window_days=FIXTURE_WINDOW_DAYS,
        include_lineups=INCLUDE_LINEUPS, random_delay_max_seconds=RANDOM_DELAY_MAX_SECONDS,
        lineup_window_hours=LINEUP_WINDOW_HOURS)["discovery_id"]
    if args.prepare_only:
        print(f"Discovery ID: {discovery_id}")
        print(f"To resume watching: python scraping/run_daily_collection.py --resume {discovery_id}")
    else:
        try:
            watch_daily_run(discovery_id, once=args.once)
        except KeyboardInterrupt:
            print(f"Watcher stopped. Captures remain saved. Resume with --resume {discovery_id}.")


if __name__ == "__main__":
    main()
