from copy import deepcopy
from datetime import datetime, timezone

import pytest

from scraping.sofascore.statistics_backfill import (
    besoccer_statistics, fill_missing_statistics, pool_player_statistics,
)


def fixture():
    match = {"event_id": 1, "home_id": 11, "away_id": 12, "status": "finished", "round": 28,
             "kickoff_utc": datetime(2024, 3, 13, 18, tzinfo=timezone.utc).timestamp(),
             "home_score_current": 0, "away_score_current": 0,
             "home_score_display": 0, "away_score_display": 0}
    roster, stats = [], []
    for side, team in (("home", 11), ("away", 12)):
        for player in range(12):
            context = {"event_id": 1, "side": side, "team_id": team, "player_id": team * 100 + player,
                       "phase": "postgame", "observed_at": 2, "raw_hash": "abc"}
            roster.append({**context, "substitute": player == 11})
            # A participating substitute replaces the first starter at halftime.
            for key, value in {"minutesPlayed": 45 if player in (0, 11) else 90,
                               "touches": 30, "totalPass": 20, "accuratePass": 15,
                               "totalShots": 1, "onTargetScoringAttempt": 1,
                               "totalLongBalls": 3 if player == 0 else 1,
                               "accurateLongBalls": 1}.items():
                stats.append({**context, "key": key, "value": value})
    return match, roster, stats


def fallback():
    return {"url": "https://www.besoccer.com/match/home/away/2024123", "observed_at": 3,
            "date": "2024-03-13", "round": 28, "home": "Home", "away": "Away", "score": [0, 0],
            "rows": [{"label": "Passes", "home": "999", "away": "998"},
                     {"label": "Corner kicks", "home": "7", "away": "4"}]}


ALIASES = {11: {"Home"}, 12: {"Away"}}


def test_substitutes_counted_snapshot_duplicates_not_added_and_ratios_weighted():
    match, roster, stats = fixture()
    candidates = pool_player_statistics(match, roster, stats + stats)
    home = {r["key"]: r for r in candidates if r["side"] == "home"}
    assert home["passes"]["value"] == 240
    assert home["totalShotsOnGoal"]["value"] == 12
    assert (home["accurateLongBalls"]["value"], home["accurateLongBalls"]["total"]) == (12, 14)
    assert home["accurateLongBalls"]["display"] == "12/14 (86%)"
    assert "ballPossession" not in home and "cornerKicks" not in home and "bigChanceCreated" not in home
    assert all(r["period"] == "ALL" for r in candidates)


@pytest.mark.parametrize("problem", ["placeholder", "missing_player", "unknown_value", "conflict", "incomplete_passes", "different_team"])
def test_incomplete_or_ambiguous_player_statistics_do_not_become_zeros(problem):
    match, roster, stats = fixture()
    if problem == "placeholder":
        for row in stats:
            if row["key"] == "minutesPlayed":
                row["value"] = 0
    elif problem == "missing_player":
        stats = [r for r in stats if r["player_id"] != 1100]
    elif problem == "unknown_value":
        next(r for r in stats if r["key"] == "totalShots")["value"] = None
    elif problem == "conflict":
        stats.append({**stats[0], "value": 89})
    elif problem == "incomplete_passes":
        stats = [r for r in stats if not (r["player_id"] == 1100 and r["key"] == "totalPass")]
    else:
        stats[0]["team_id"] = 99
    assert pool_player_statistics(match, roster, stats) == []


def test_latest_snapshot_selected_without_mixing_older_player_values():
    match, roster, stats = fixture()
    old_stats = [{**r, "observed_at": 1, "raw_hash": "old"} for r in stats]
    old_roster = [{**r, "observed_at": 1, "raw_hash": "old"} for r in roster]
    assert pool_player_statistics(match, roster + old_roster, stats + old_stats) == pool_player_statistics(match, roster, stats)


def test_ground_duels_and_duel_percent_use_summed_numerators_and_denominators():
    match, roster, stats = fixture()
    for row in roster:
        for key, value in {"duelWon": 4, "duelLost": 6, "aerialWon": 1, "aerialLost": 2}.items():
            stats.append({**row, "key": key, "value": value})
    home = {r["key"]: r for r in pool_player_statistics(match, roster, stats) if r["side"] == "home"}
    assert home["groundDuelsPercentage"]["value"] == 36
    assert home["groundDuelsPercentage"]["total"] == 84
    assert home["duelWonPercent"]["value"] == 40


def test_saved_values_then_our_players_then_fallback_idempotent_and_no_mutation():
    match, roster, stats = fixture()
    pooled = pool_player_statistics(match, roster, stats)
    external = besoccer_statistics(match, fallback(), team_aliases=ALIASES)
    saved = [{**pooled[0], "value": 123, "display": "123", "source_key": "original"}]
    prior = deepcopy(saved)
    result = fill_missing_statistics(saved, pooled, external)
    assert saved == prior
    assert result[0] == saved[0]
    home = {r["key"]: r for r in result if r["side"] == "home"}
    assert home["passes"]["value"] == 240
    assert home["passes"]["source_key"].startswith("derived|")
    assert home["cornerKicks"]["value"] == 7
    assert home["cornerKicks"]["source_key"].startswith("https://www.besoccer.com/")
    assert fill_missing_statistics(result, pooled, external) == result


@pytest.mark.parametrize("field,value", [("date", "2023-10-28"), ("round", 13), ("home", "Away"), ("score", [2, 0]),
                                        ("url", "https://www.besoccer.com/match/home/away/2024123/analysis")])
def test_wrong_fixture_or_prediction_page_rejected(field, value):
    match, _, _ = fixture()
    observation = {**fallback(), field: value}
    with pytest.raises(ValueError):
        besoccer_statistics(match, observation, team_aliases=ALIASES)


def test_legitimate_zero_preserved_and_null_row_filled_without_duplicate():
    match, _, _ = fixture()
    rows = besoccer_statistics(match, fallback(), team_aliases=ALIASES)
    saved = [{**rows[0], "value": 0, "display": "0"},
             {**rows[1], "value": None, "display": None, "group_name": "Existing", "stat_order": 7}]
    result = fill_missing_statistics(saved, rows)
    assert result[0]["value"] == 0
    assert result[1]["value"] == 998
    assert result[1]["group_name"] == "Existing" and result[1]["stat_order"] == 7
    assert len(result) == 4


def test_observed_half_statistics_remain_separate_from_match_totals():
    match, _, _ = fixture()
    observation = fallback()
    observation["rows"] = [
        {"label": "Corner kicks", "home": "7", "away": "4"},
        {"period": "1ST", "label": "Corner kicks", "home": "2", "away": "3"},
        {"period": "2ND", "label": "Corner kicks", "home": "5", "away": "1"},
    ]
    rows = besoccer_statistics(match, observation, team_aliases=ALIASES)
    assert {(r["period"], r["side"]): r["value"] for r in rows} == {
        ("ALL", "home"): 7, ("ALL", "away"): 4,
        ("1ST", "home"): 2, ("1ST", "away"): 3,
        ("2ND", "home"): 5, ("2ND", "away"): 1,
    }
    saved = [{**rows[0], "value": 8, "display": "8"}]
    merged = fill_missing_statistics(saved, rows)
    assert len(merged) == 6 and merged[0]["value"] == 8
    assert fill_missing_statistics(merged, rows) == merged
    observation["rows"][0]["period"] = "estimated"
    with pytest.raises(ValueError, match="observed"):
        besoccer_statistics(match, observation, team_aliases=ALIASES)


def publication_fixture(root):
    from scraping.sofascore.campaign import publish_season
    from scraping.sofascore.config import Profile, Scope
    from scraping.sofascore.schema import to_arrow

    match, _, _ = fixture()
    match.update(competition_id=54, season_id=52563)
    tables = {"matches": to_arrow("matches", [match]).to_pylist(),
              "coverage": to_arrow("coverage", [{"event_id": 1, "component": "statistics",
                                                  "status": "unavailable", "latest_attempt_status": "unavailable"}]).to_pylist(),
              "statistics": [], "player_statistics": [], "lineup_players": []}
    return publish_season(tables, root, Scope(54, 52563, (28,)), Profile(), "test_season", {})


def test_publication_keeps_prior_version_schema_and_second_run_unchanged(tmp_path):
    import hashlib
    import pyarrow.parquet as pq
    from scraping.sofascore.export import read_tables
    from scraping.sofascore.statistics_backfill import publish_backfill

    prior = publication_fixture(tmp_path)
    old_version = tmp_path / prior["manifest"]
    preserved = {p: hashlib.sha256(p.read_bytes()).digest() for p in old_version.parent.iterdir()}
    schema = pq.ParquetFile(tmp_path / "test_season.parquet").schema_arrow
    result = publish_backfill(tmp_path, "test_season", {1: fallback()}, team_aliases=ALIASES)
    assert result["matches"] == {1: {"besoccer": 4}}
    assert {p: hashlib.sha256(p.read_bytes()).digest() for p in preserved} == preserved
    assert pq.ParquetFile(tmp_path / "test_season.parquet").schema_arrow.equals(schema, check_metadata=False)
    new = read_tables(result["manifest"])
    assert len(new["statistics"]) == 4
    assert new["coverage"][0]["status"] == "present"
    assert new["coverage"][0]["latest_attempt_status"] == "unavailable"
    all_paths = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert publish_backfill(tmp_path, "test_season", {1: fallback()}, team_aliases=ALIASES)["state"] == "unchanged"
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()) == all_paths
    # No extra receipts, reports, or source metadata sidecars were created.
    assert all(p.endswith((".parquet", "/manifest.json", "/CURRENT.json", ".manifest.json", ".lock")) for p in all_paths)


def test_publication_failure_restores_all_public_entry_files(tmp_path, monkeypatch):
    from scraping.sofascore.statistics_backfill import publish_backfill
    from scraping.sofascore_web_v3 import status_write

    publication_fixture(tmp_path)
    entry_paths = [tmp_path / "test_season.parquet", tmp_path / "test_season.manifest.json",
                   tmp_path / "_tables/competition=54/season=52563/CURRENT.json"]
    before = {p: p.read_bytes() for p in entry_paths}
    real_write = status_write.atomic_write_status
    attempts = 0

    def fail_second(path, data):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise OSError("simulated interrupted publication")
        return real_write(path, data)

    monkeypatch.setattr(status_write, "atomic_write_status", fail_second)
    with pytest.raises(OSError, match="simulated interrupted"):
        publish_backfill(tmp_path, "test_season", {1: fallback()}, team_aliases=ALIASES)
    assert {p: p.read_bytes() for p in entry_paths} == before


def test_backfill_preserves_columns_appended_by_awarded_migration(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    import pyarrow.parquet as pq
    from scraping.sofascore.statistics_backfill import publish_backfill

    report = publication_fixture(tmp_path)
    canonical = tmp_path / report["manifest"]
    info = json.loads(canonical.read_text())
    match_path = canonical.parent / info["tables"]["matches"]["file"]
    flat_path = tmp_path / report["season_file"]
    layouts = {}
    for path in (match_path, flat_path):
        table = pq.ParquetFile(path).read()
        names = [name for name in table.column_names if name != "is_awarded"] + ["is_awarded"]
        table = table.select(names)
        pq.write_table(table, path, compression="zstd")
        layouts[path] = table
    info["tables"]["matches"].update(bytes=match_path.stat().st_size,
        sha256=hashlib.sha256(match_path.read_bytes()).hexdigest())
    canonical.write_text(json.dumps(info))
    report.update(bytes=flat_path.stat().st_size,
                  sha256=hashlib.sha256(flat_path.read_bytes()).hexdigest())
    (tmp_path / "test_season.manifest.json").write_text(json.dumps(report))

    result = publish_backfill(tmp_path, "test_season", {1: fallback()}, team_aliases=ALIASES)
    new = json.loads(Path(result["manifest"]).read_text())
    new_match = Path(result["manifest"]).parent / new["tables"]["matches"]["file"]
    assert pq.ParquetFile(new_match).read().equals(layouts[match_path], check_metadata=True)
    flat = pq.ParquetFile(flat_path).read()
    assert flat.column_names == layouts[flat_path].column_names
    assert flat["is_awarded"].equals(layouts[flat_path]["is_awarded"])
    assert publish_backfill(tmp_path, "test_season", {1: fallback()}, team_aliases=ALIASES)["state"] == "unchanged"
