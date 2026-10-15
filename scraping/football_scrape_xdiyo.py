"""Notebook compatibility entry points. Original implementation: scraping/legacy/."""
from pathlib import Path
import sys

_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.insert(0, _root)

from scraping.sofascore.notebook_api import scrape_league_data, scrape_league_recent
from scraping.sofascore.compatibility import merge_recent_leagues_columns

__all__ = ["scrape_league_data", "scrape_league_recent", "merge_recent_leagues_columns"]
