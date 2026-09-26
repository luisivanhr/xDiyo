"""Portable example inputs stay runnable without publishing historical runs."""

import json
from pathlib import Path, PureWindowsPath

import pandas as pd

from xdiyo_analytics.data.loading import SeasonData
from xdiyo_analytics.ui import prepare_recipe, read_recipe


ROOT = Path(__file__).resolve().parents[2]
BUNDLES = ROOT / "examples/bundles"


def _nodes(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _nodes(item)


def test_example_bundle_contains_only_portable_configuration():
    assert {p.relative_to(BUNDLES).as_posix() for p in BUNDLES.rglob("*")
            if p.is_file()} == {"README.md", "corners_lasso.json"}
    recipe = read_recipe(BUNDLES / "corners_lasso.json")
    assert recipe["prediction"]["model_path"] == ""
    assert recipe["search"] is None
    assert recipe["checkpoint"] is None
    assert recipe["refit"] is None
    assert recipe["output_dir"].startswith("experiments/")
    for node in _nodes(recipe):
        assert not str(node.get("component", "")).startswith("input.")
        for value in node.values():
            if isinstance(value, str):
                assert not PureWindowsPath(value).is_absolute()
                assert not value.startswith("/")
                assert "_collection" not in value


def test_example_publication_inputs_resolve_inside_published_tables():
    recipe = read_recipe(BUNDLES / "corners_lasso.json")
    root = ROOT / recipe["data"]["data_root"]
    selected = []
    for path in root.glob("*.manifest.json"):
        stem = path.name.removesuffix(".manifest.json")
        league, start, end = stem.rsplit("_", 2)
        season = f"{start}_{end}"
        if season not in recipe["data"]["seasons"]:
            continue
        selected.append((league, season))
        publication = json.loads(path.read_text(encoding="utf-8"))
        manifest_path = (root / publication["manifest"]).resolve()
        assert manifest_path.is_relative_to((root / "_tables").resolve())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name in recipe["data"]["tables"]:
            item = manifest["tables"][name]
            table = (manifest_path.parent / item["file"]).resolve()
            assert table.is_relative_to(manifest_path.parent)
            assert table.stat().st_size == item["bytes"]
    assert {season for _, season in selected} == set(recipe["data"]["seasons"])


def test_bundled_recipe_prepares_fresh_inputs_without_prior_runs(monkeypatch):
    recipe = read_recipe(BUNDLES / "corners_lasso.json")
    definitions = {tuple(node["params"][key] for key in ("period", "group", "key"))
                   for node in _nodes(recipe)
                   if node.get("component") == "features.Stat"}
    matches, statistics, pregame = [], [], []
    for year in range(2020, 2025):
        for day in range(8):
            event = year * 100 + day
            origin = {"source_league": "Synthetic", "source_season": f"{year % 100:02}_{(year+1) % 100:02}"}
            matches.append({**origin, "event_id": event, "competition_id": 17,
                            "season_id": year, "home_id": 1, "away_id": 2,
                            "home_score_current": 1 + day % 3, "away_score_current": day % 2,
                            "kickoff_utc": pd.Timestamp(f"{year}-09-{day+1:02}", tz="UTC").timestamp(),
                            "round": day + 1, "status": "finished", "is_awarded": False})
            for side, team in (("home", 1), ("away", 2)):
                pregame.append({**origin, "event_id": event, "side": side,
                                "team_id": team, "position": team})
                for period, group, key in sorted(definitions):
                    statistics.append({**origin, "event_id": event, "side": side,
                                       "team_id": team, "period": period,
                                       "group_name": group, "key": key,
                                       "value": float(2 + day % 4 + team)})
    data = SeasonData({"matches": pd.DataFrame(matches), "statistics": pd.DataFrame(statistics),
                       "pregame": pd.DataFrame(pregame)}, {})
    monkeypatch.setattr("xdiyo_analytics.data.load_seasons", lambda **kwargs: data)
    prepared = prepare_recipe(recipe)
    assert len(prepared.dataset.X) == 40
    assert len(prepared.split_plan.folds) == 1
    fold = prepared.split_plan.folds[0]
    assert len(fold.train) == 32
    assert len(fold.test) == len(fold.score) == 8
    assert set(prepared.dataset.metadata.iloc[fold.test].source_season) == {"24_25"}
    assert prepared.dataset.X.shape[1] > len(recipe["features"])
