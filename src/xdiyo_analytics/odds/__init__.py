"""Local provider odds imports, separate from native match identities."""

from .workbook import extract_odds, load_odds
# Compatibility aliases for existing scripts.
extract_footiqo = extract_odds
load_footiqo = load_odds
from .crosswalk import build_fixture_crosswalk, save_crosswalk, read_crosswalk, vendor_fixtures, load_mapping_rules
from .selection import OddsSeries, paired_quotes

__all__ = ["extract_odds", "load_odds", "build_fixture_crosswalk", "save_crosswalk",
           "read_crosswalk", "vendor_fixtures", "load_mapping_rules", "OddsSeries", "paired_quotes"]
