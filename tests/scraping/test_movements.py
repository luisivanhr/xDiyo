import json

import pandas as pd
import pytest

from scraping.sofascore.movements import build_movements, read_flags, resolve_slug_seeds
from utils.team_seasons import join_team_season_flags


def test_offline_flags_pin_evidence_quarantine_wrong_year_and_keep_cardinality(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    for name, sid, year, teams in [("Top_23_24", 10, "23/24", (1, 2)),
                                    ("Top_24_25", 20, "24/25", (1, 3)),
                                    ("Low_23_24", 30, "23/24", (3, 4)),
                                    ("Top_25_26", 20, "24/25", (1, 3))]:
        pd.DataFrame([{"match_id": sid, "season_id": sid, "season_year": year,
                       "home_id": teams[0], "away_id": teams[1], "home_team": str(teams[0]),
                       "away_team": str(teams[1])}]).to_csv(source/(name+".csv"), index=False)
    registry = tmp_path/"registry.json"
    registry.write_text(json.dumps({"Top": {"competition_id": 1, "system": "X", "tier": 1},
                                    "Low": {"competition_id": 2, "system": "X", "tier": 2}}))
    evidence = tmp_path/"evidence.json"
    evidence.write_text(json.dumps({"complete_memberships": [{"competition_id": 1, "season_id": 10,
                                                             "team_ids": [1, 2], "evidence": "fixture official membership"}]}))
    manifest, report = build_movements(source, tmp_path/"out", registry_path=registry, evidence_path=evidence)
    assert report["api_requests"] == 0
    assert report["identity_issues"][0]["status"] == "quarantined_season_identity_conflict"
    flags = read_flags(manifest)
    flag = next(r for r in flags if r["season_id"] == 20 and r["team_id"] == 3)
    assert flag["got_promoted"] is True and flag["previous_competition_id"] == 2
    matches = pd.DataFrame([{"match_id": 12, "competition_id": 1, "season_id": 20, "home_id": 3, "away_id": 1}]*2)
    joined = join_team_season_flags(matches, flags)
    assert len(joined) == len(matches) and joined.home_got_promoted.all()


def test_slug_seed_cannot_drift_to_another_year():
    with pytest.raises(ValueError, match="exact season"):
        resolve_slug_seeds([{"league": "Top", "start_year": 2015, "end_year": 2016}],
                           [{"league": "Top", "start_year": 2014, "end_year": 2015}], {})
