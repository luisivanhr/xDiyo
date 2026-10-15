"""Compatibility entry point for the ID-based season-entry derivation.

The original slug-based implementation is preserved in scraping/legacy/.
"""
from pathlib import Path
import json
import uuid

from scraping.sofascore.movements import build_movements, read_flags
from scraping.sofascore.store import atomic_write, json_bytes
from utils.team_seasons import join_team_season_flags


def get_promotions_and_demotions(
    top_league, bottom_league=None, top_promoted=None, bottom_promoted=None,
    bottom_demoted=None, data_dir="data/xDiyo_data_heatmaps",
    output_dir="data/prepared_v2/team_seasons", *, top_seed_years=None,
    bottom_seed_years=None, evidence_path=None, export_legacy_csv=False,
):
    """Write a small versioned flag table; return its manifest and audit.

    Supplied slug seeds require explicit (full_start_year, full_end_year) pairs.
    Membership confirmations and ID overrides can be supplied in evidence_path.
    Full enriched CSVs are optional and written inside the new output version.
    """
    evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8")) if evidence_path else {}
    seeds = evidence.setdefault("slug_seeds", [])
    for league, years, movement, slugs in (
        (top_league, top_seed_years, "promoted", top_promoted),
        (bottom_league, bottom_seed_years, "promoted", bottom_promoted),
        (bottom_league, bottom_seed_years, "relegated", bottom_demoted),
    ):
        if slugs and (not years or len(years) != 2 or min(years) < 1800 or not league):
            raise ValueError("Slug seeds require their exact league and full seed years; see promotions_demotions.ipynb")
        for slug in slugs or []:
            seeds.append({"league": league, "start_year": years[0], "end_year": years[1],
                          "slug": slug, "movement": movement,
                          "evidence": "Explicit caller seed pinned to its supplied season years"})
    output = Path(output_dir)
    evidence_file = output/"inputs"/("evidence-"+uuid.uuid4().hex+".json")
    atomic_write(evidence_file, json_bytes(evidence))
    legacy_dir = Path(data_dir).parent/"xDiyo_data_promotions"
    manifest, report = build_movements(data_dir, output, evidence_path=evidence_file,
                                       legacy_dir=legacy_dir if legacy_dir.is_dir() else None,
                                       leagues=[top_league]+([bottom_league] if bottom_league else []))
    if export_legacy_csv:
        import pandas as pd
        flags = read_flags(manifest)
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        exports = []
        for source in metadata["sources"]:
            frame = pd.read_csv(source["file"])
            frame["competition_id"] = source["competition_id"]
            joined = join_team_season_flags(frame, flags)
            destination = manifest.parent/"compatibility"/Path(source["file"]).name
            atomic_write(destination, joined.to_csv(index=False).encode("utf-8"))
            exports.append(str(destination))
        atomic_write(manifest.parent/"compatibility_manifest.json", json_bytes({"flag_manifest": str(manifest), "files": exports}))
        report["compatibility_exports"] = exports
    return manifest, report
