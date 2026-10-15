from pathlib import Path

import pandas as pd
import pytest

from scraping.sofascore.compatibility import merge_recent_leagues_columns
from scraping.sofascore.notebook_api import scrape_league_data
from scraping.sofascore.legacy import load_season


def test_id_update_keeps_old_good_values_union_and_inputs(tmp_path):
    first = tmp_path/"old.csv"
    recent = tmp_path/"recent.csv"
    pd.DataFrame([{"match_id": 1, "season_id": 20, "home_id": 3, "away_id": 4, "value": 9},
                  {"match_id": 2, "season_id": 20, "home_id": 4, "away_id": 3, "value": 7}]).to_csv(first,index=False)
    pd.DataFrame([{"match_id": 2, "season_id": 20, "home_id": 4, "away_id": 3, "value": None, "new_stat": 2},
                  {"match_id": 3, "season_id": 20, "home_id": 3, "away_id": 4, "value": 0, "new_stat": 4}]).to_csv(recent,index=False)
    original = [p.read_bytes() for p in (first, recent)]
    output = merge_recent_leagues_columns([first, recent], ["match_id", "value"], output_dir=tmp_path/"outputs")
    result = pd.read_csv(output)
    assert result.match_id.tolist() == [1, 2, 3]
    assert result.value.tolist() == [9, 7, 0]
    assert pd.isna(result.new_stat.iloc[0])
    assert [p.read_bytes() for p in (first, recent)] == original


def test_notebook_adapter_shares_new_profile_and_preserves_requested_flags(monkeypatch):
    from scraping.sofascore import notebook_api
    monkeypatch.setattr(notebook_api, "_collect", lambda *args: args)
    scope, profile, path, timeout, *_ = scrape_league_data(17, 500, 38, 20, "Premier_League", 2024, 2025,
        with_lineups=False, with_xG=True, with_heatmaps=True, with_injuries=True, with_goal_actions=True)
    assert scope.rounds == tuple(range(1,39))
    assert profile.include_injuries and profile.include_goal_actions and not profile.include_lineups
    assert not profile.include_comments


def test_model_loader_rejects_conflicting_filename_year(tmp_path):
    path = tmp_path/"Example_16_17.csv"
    pd.DataFrame([{"match_id": 1, "season_id": 20, "season_year": "15/16"}]).to_csv(path,index=False)
    with pytest.raises(ValueError, match="season identity conflict"):
        load_season(path, include_lineups=False)
