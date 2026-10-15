import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from utils.football_dataset_xDiyoFire import FootballDataset, collate_fn


def test_real_batch_optional_data_zero_xg_missing_xg_and_padding():
    rows = []
    for eid in (100, 50):  # Provider ID order must not reorder the sequence.
        rows.append({"match_id": eid, "home_id": 1, "away_id": 2, "home_team": "a", "away_team": "b",
                     "home_score": 1, "away_score": 0, "round": len(rows)+1, "league_index": 0,
                     "home_shots": [{"xg": None, "shotType": "miss", "time": 10},
                                    {"xg": 0., "shotType": "goal", "time": 20},
                                    {"xg": 1., "xg_quality": "legacy_unverified_one", "shotType": "goal", "goalType": "own", "time": 30}],
                     "away_shots": [], "away_heatmap": {"playerPoints": [{"x": 50, "y": 50}], "goalkeeperPoints": []}})
    data = FootballDataset(pd.DataFrame(rows, index=[88, 99]), 2, {"<PAD>": 0, "a": 1, "b": 2}, 4, 3)
    batch = collate_fn([data[0], data[1]])
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key
    masks = batch["availability"]
    assert not masks["home_team_formations"].any()
    assert not masks["home_team_standings"].any()
    assert not masks["home_team_player_heatmap_sequence"].any()
    assert masks["home_enemy_player_heatmap_sequence"][0, -1].all()
    assert not masks["home_enemy_player_heatmap_sequence"][0, 0].any()
    assert not masks["home_team_shots"][0, -1, 0, 0]
    assert masks["home_team_shots"][0, -1, 1, 0]  # True zero xG.
    assert not masks["home_team_owngoals"][0, -1, 0, 0]
    assert batch["home_team_owngoals"][0, -1, 0, 1] == 1  # Event survives.
    assert batch["home_enemy_shots"].shape == (2, 3, 0, 3)
    assert data.team_history[1][1]["provider_event_id"] == 50
    for key, mask in masks.items():
        assert mask.shape == batch[key].shape


def test_missing_score_is_not_an_away_win():
    frame = pd.DataFrame([{"home_id": 1, "away_id": 2, "home_team": "a", "away_team": "b",
                           "home_score": None, "away_score": 0, "round": 1, "league_index": 0}])
    data = FootballDataset(frame, 2, {"a": 1, "b": 2}, 2, 2)
    with pytest.raises(ValueError, match="observed final scores"):
        data[0]
