"""Conservative team-statistic recovery using the existing schema.

Priority: saved team values, validated saved player totals, observed fallback
team values. No player fetching, guessed zeros, half-time imputation, or new
metadata columns. Publication remains the caller's responsibility.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from urllib.parse import urlparse


# These definitions were checked against complete La Liga 2 2023/24 captures.
# Player bigChanceCreated is deliberately absent: it is not team Big chances.
COUNTS = {
    "totalShotsOnGoal": ("Total shots", "Shots", "totalShots"),
    "shotsOnGoal": ("Shots on target", "Shots", "onTargetScoringAttempt"),
    "shotsOffGoal": ("Shots off target", "Shots", "shotOffTarget"),
    "blockedScoringAttempt": ("Blocked shots", "Shots", "blockedScoringAttempt"),
    "hitWoodwork": ("Hit woodwork", "Shots", "hitWoodwork"),
    "offsides": ("Offsides", "Attack", "totalOffside"),
    "bigChanceMissed": ("Big chances missed", "Attack", "bigChanceMissed"),
    "fouls": ("Fouls", "Match overview", "fouls"),
    "passes": ("Passes", "Passes", "totalPass"),
    "accuratePasses": ("Accurate passes", "Passes", "accuratePass"),
    "goalkeeperSaves": ("Total saves", "Goalkeeping", "saves"),
    "totalTackle": ("Total tackles", "Defending", "totalTackle"),
    "interceptionWon": ("Interceptions", "Defending", "interceptionWon"),
    "ballRecovery": ("Recoveries", "Defending", "ballRecovery"),
    "totalClearance": ("Clearances", "Defending", "totalClearance"),
    "dispossessed": ("Dispossessed", "Duels", "dispossessed"),
    "highClaims": ("High claims", "Goalkeeping", "goodHighClaim"),
    "punches": ("Punches", "Goalkeeping", "punches"),
}
RATIOS = {
    "accurateLongBalls": ("Long balls", "Passes", "accurateLongBalls", "totalLongBalls"),
    "accurateCross": ("Crosses", "Passes", "accurateCross", "totalCross"),
    "dribblesPercentage": ("Dribbles", "Duels", "wonContest", "totalContest"),
    "aerialDuelsPercentage": ("Aerial duels", "Duels", "aerialWon", "aerialLost"),
    "wonTacklePercent": ("Tackles won", "Defending", "wonTackle", "totalTackle"),
}
FALLBACK = {
    "Ball possession": ("ballPossession", "Ball possession", "Match overview"),
    "Corner kicks": ("cornerKicks", "Corner kicks", "Match overview"),
    "Yellow cards": ("yellowCards", "Yellow cards", "Match overview"),
    "Red cards": ("redCards", "Red cards", "Match overview"),
    "Offsides": ("offsides", "Offsides", "Attack"),
    "Total shots": ("totalShotsOnGoal", "Total shots", "Shots"),
    "Shots on target": ("shotsOnGoal", "Shots on target", "Shots"),
    "Shots off target": ("shotsOffGoal", "Shots off target", "Shots"),
    "Shots blocked": ("blockedScoringAttempt", "Blocked shots", "Shots"),
    "Goalkeeper saves": ("goalkeeperSaves", "Total saves", "Goalkeeping"),
    "Tackles": ("totalTackle", "Total tackles", "Defending"),
    "Passes": ("passes", "Passes", "Passes"),
    "Completed passes": ("accuratePasses", "Accurate passes", "Passes"),
    "Fouls": ("fouls", "Fouls", "Match overview"),
}
FRENCH = dict(zip(
    ("Possession de balle", "Corners", "Cartons jaunes", "Cartons rouges", "Hors-jeu",
     "Nombre total de tirs", "Arrêts du gardien", "Tacles", "Passes totales", "Passes réussies", "Fautes"),
    ("Ball possession", "Corner kicks", "Yellow cards", "Red cards", "Offsides",
     "Total shots", "Goalkeeper saves", "Tackles", "Passes", "Completed passes", "Fouls")))
GROUPS = ("Match overview", "Shots", "Attack", "Passes", "Duels", "Defending", "Goalkeeping")
NEGATIVE = {"fouls", "offsides", "shotsOffGoal", "bigChanceMissed", "dispossessed", "yellowCards", "redCards"}
PLAYER_KEYS = ({v[2] for v in COUNTS.values()} | {k for v in RATIOS.values() for k in v[2:]}
               | {"minutesPlayed", "touches", "duelWon", "duelLost"})


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _row(match, side, key, name, group, value, context, *, total=None, display=None, source=None):
    return {"event_id": match["event_id"], **context, "period": "ALL", "group_name": group,
            "key": key, "name": name, "side": side, "team_id": match[side + "_id"],
            "value": float(value), "total": float(total) if total is not None else None,
            "display": display if display is not None else f"{value:g}",
            "value_type": "team" if total is not None else "event",
            "statistics_type": "negative" if key in NEGATIVE else "positive",
            "source_item_json": _json(source) if source is not None else None,
            "group_order": GROUPS.index(group), "stat_order": 0}


def pool_player_statistics(match, lineup_players, player_statistics):
    """Sum one complete postgame snapshot, including participating substitutes.

    The conservative gate currently supports a full ordinary 90-minute match
    (990 player-minutes per team). Incomplete, dismissed-player, extra-time and
    placeholder snapshots return no candidates rather than fabricated totals.
    Missing sparse count keys are treated as zero only inside this rich snapshot
    and only if that metric is actually supplied for at least one player.
    """
    eid = match["event_id"]
    rows = [r for r in player_statistics if r["event_id"] == eid and r.get("phase") == "postgame"]
    if not rows or match.get("status") != "finished":
        return []
    snapshots = {(r.get("observed_at"), r.get("raw_hash")) for r in rows}
    if any(at is None or not digest for at, digest in snapshots):
        return []
    latest_time = max(at for at, _ in snapshots)
    latest = [s for s in snapshots if s[0] == latest_time]
    if len(latest) != 1:
        return []
    snapshot = latest[0]
    rows = [r for r in rows if (r["observed_at"], r["raw_hash"]) == snapshot]
    rosters = [r for r in lineup_players if r["event_id"] == eid and r.get("phase") == "postgame"
               and (r.get("observed_at"), r.get("raw_hash")) == snapshot]
    unique = {}
    for r in rows:
        ident = (r["side"], r["player_id"], r["key"])
        if ident in unique and unique[ident] != r:
            return []
        unique[ident] = r
    sums, keys = {}, set()
    for side in ("home", "away"):
        side_roster = [r for r in rosters if r["side"] == side]
        roster = {r["player_id"]: r for r in side_roster}
        if len(roster) != len(side_roster):
            return []
        if any(r.get("team_id") != match[side + "_id"] for r in roster.values()):
            return []
        players = defaultdict(dict)
        for (s, pid, key), r in unique.items():
            if s == side:
                if pid not in roster or r.get("team_id") != match[side + "_id"]:
                    return []
                if key in PLAYER_KEYS and (r["value"] is None or not math.isfinite(r["value"])
                                           or r["value"] < 0 or not float(r["value"]).is_integer()):
                    return []
                if key in PLAYER_KEYS:
                    players[pid][key] = r["value"]
        starters = {pid for pid, r in roster.items() if r.get("substitute") is False}
        active = {pid for pid, v in players.items() if v.get("minutesPlayed", 0) > 0}
        if len(starters) != 11 or not starters <= active:
            return []
        if sum(players[pid]["minutesPlayed"] for pid in active) != 990:
            return []
        if any(not (0 < players[pid]["minutesPlayed"] <= 90) for pid in active):
            return []
        if any("totalPass" not in players[pid] or "touches" not in players[pid] for pid in active):
            return []
        sums[side] = defaultdict(float)
        for pid in active:
            for key, value in players[pid].items():
                sums[side][key] += value
                keys.add(key)
        v = sums[side]
        if v["totalPass"] <= 0 or v["accuratePass"] > v["totalPass"]:
            return []
        if v["totalShots"] != sum(v[k] for k in ("onTargetScoringAttempt", "shotOffTarget", "blockedScoringAttempt")):
            return []
    context = {"source_key": f"derived|sofascore|/event/{eid}/lineups|player_totals",
               "raw_hash": snapshot[1], "observed_at": snapshot[0]}
    result = []
    for key, (name, group, player_key) in COUNTS.items():
        if player_key in keys:
            for side, v in sums.items():
                result.append(_row(match, side, key, name, group, v[player_key], context))
    for key, (name, group, numerator, denominator) in RATIOS.items():
        if denominator not in keys:
            continue
        for side, v in sums.items():
            value, total = v[numerator], v[denominator]
            if key == "aerialDuelsPercentage":
                total += value
            if not (0 <= value <= total) or total == 0:
                continue
            percent = math.floor(100 * value / total + 0.5)
            display = f"{percent}%" if key == "wonTacklePercent" else f"{value:g}/{total:g} ({percent}%)"
            result.append(_row(match, side, key, name, group, value, context, total=total, display=display))
    if {"duelWon", "duelLost", "aerialWon", "aerialLost"} <= keys:
        for side, v in sums.items():
            duel_total = v["duelWon"] + v["duelLost"]
            if duel_total > 0:
                percent = math.floor(100 * v["duelWon"] / duel_total + 0.5)
                result.append(_row(match, side, "duelWonPercent", "Duels", "Duels", percent,
                                   context, display=f"{percent}%"))
            value = v["duelWon"] - v["aerialWon"]
            total = v["duelWon"] + v["duelLost"] - v["aerialWon"] - v["aerialLost"]
            if 0 <= value <= total and total > 0:
                percent = math.floor(100 * value / total + 0.5)
                result.append(_row(match, side, "groundDuelsPercentage", "Ground duels", "Duels", value,
                                   context, total=total, display=f"{value:g}/{total:g} ({percent}%)"))
    return result


def besoccer_statistics(match, observation, *, team_aliases):
    """Convert verified visible team statistics; never fetch player pages.

    The caller supplies the observed page identity and explicit name aliases.
    Reversed fixtures, different dates/rounds, and inconsistent scores fail closed.
    """
    url = urlparse(observation["url"])
    if url.scheme != "https" or url.hostname not in {"www.besoccer.com", "fr.besoccer.com"}:
        raise ValueError("Expected an observed BeSoccer match URL")
    if not url.path.startswith("/match/") or url.path.endswith(("/analysis", "/analyse", "/preview", "/avant-match")):
        raise ValueError("Expected a match page, not prediction analysis")
    date = datetime.fromtimestamp(match["kickoff_utc"], timezone.utc).date().isoformat()
    if observation["date"] != date or observation["round"] != match["round"]:
        raise ValueError("Fallback fixture date or round differs")
    for side in ("home", "away"):
        if observation[side] not in team_aliases[match[side + "_id"]]:
            raise ValueError("Fallback fixture teams/orientation differ")
    scores = {tuple(match[f"{s}_score_{kind}"] for s in ("home", "away")) for kind in ("current", "display")}
    if tuple(observation["score"]) not in scores:
        raise ValueError("Fallback score differs from both recorded score variants")
    context = {"source_key": observation["url"], "observed_at": observation["observed_at"]}
    result = []
    for item in observation["rows"]:
        period = item.get("period", "ALL")
        if period not in {"ALL", "1ST", "2ND"}:
            raise ValueError("Expected an explicitly observed match-statistics period")
        label = FRENCH.get(item["label"], item["label"])
        if label not in FALLBACK:
            continue
        key, name, group = FALLBACK[label]
        context["raw_hash"] = hashlib.sha256(_json(item).encode()).hexdigest()
        values = [float(item[s].rstrip("%")) for s in ("home", "away")]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Invalid fallback value")
        if key == "ballPossession":
            if abs(sum(values) - 100) > 0.01 or any(v > 100 for v in values):
                raise ValueError("Invalid possession percentages")
        elif any(not v.is_integer() for v in values):
            raise ValueError("Count statistic is not integral")
        for side, value in zip(("home", "away"), values):
            row = _row(match, side, key, name, group, value, context,
                       display=item[side], source=item)
            row["period"] = period
            result.append(row)
    return result


def fill_missing_statistics(existing, *candidate_groups):
    """Preserve all saved values; consume candidate groups in priority order."""
    result = [dict(r) for r in existing]
    identity = lambda r: (r["event_id"], r["period"], r["key"], r["side"])
    present = {identity(r) for r in result if r.get("value") is not None or r.get("display") not in (None, "")}
    for candidates in candidate_groups:
        for row in candidates:
            ident = identity(row)
            if ident in present:
                continue
            indexes = [i for i, old in enumerate(result) if identity(old) == ident]
            if indexes:
                for i in indexes:
                    result[i] = {**row, **{k: result[i][k] for k in ("group_name", "group_order", "stat_order")}}
            else:
                added = dict(row)
                peers = [r for r in result if (r["event_id"], r["period"], r["side"], r["group_name"]) ==
                         (row["event_id"], row["period"], row["side"], row["group_name"])]
                added["stat_order"] = max((r["stat_order"] for r in peers if r.get("stat_order") is not None), default=-1) + 1
                result.append(added)
            present.add(ident)
    return result


def publish_backfill(output, stem, observations, *, team_aliases):
    """Publish a checked normal dataset version, without extra reports/columns.

    Immutable old versions remain intact. All candidate work and rollback copies
    use a temporary directory; only the format's normal bundle is installed.
    """
    from collections import Counter
    from copy import deepcopy
    from pathlib import Path
    import shutil
    from tempfile import TemporaryDirectory

    import pyarrow.parquet as pq

    from .campaign import publish_season
    from .config import Profile, Scope
    from .export import read_tables
    from .schema import to_arrow
    from .store import json_bytes
    from ..sofascore_web_v3.campaign import collector_lock
    from ..sofascore_web_v3.publish import read_json
    from ..sofascore_web_v3.reconcile_publication import _bundle, _copy_files, _verify_files, IDENTITY
    from ..sofascore_web_v3.status_write import atomic_write_status

    output = Path(output).resolve()
    if Path(stem).name != stem or not stem:
        raise ValueError("Expected a season stem, not a path")
    with collector_lock(output / "_collection/chrome_publication.lock"):
        outer = read_json(output / f"{stem}.manifest.json")
        canonical = (output / outer["manifest"]).resolve()
        if not canonical.is_relative_to(output):
            raise ValueError("Manifest leaves output directory")
        tables = read_tables(canonical, verify_hashes=True)
        tables["coverage"] = [dict(r) for r in tables.get("coverage", [])]
        expected = {r["event_id"]: tuple(r[k] for k in IDENTITY) for r in tables["matches"]}
        if set(observations) - expected.keys():
            raise ValueError("Fallback includes matches outside this season")
        prior = _bundle(output, stem, expected)
        by_event = {r["event_id"]: r for r in tables["matches"]}
        changes = {}
        for eid, observation in observations.items():
            match = by_event[eid]
            saved = [r for r in tables.get("statistics", []) if r["event_id"] == eid]
            pooled = pool_player_statistics(match, tables.get("lineup_players", []), tables.get("player_statistics", []))
            fallback = besoccer_statistics(match, observation, team_aliases=team_aliases) if observation else []
            merged = fill_missing_statistics(saved, pooled, fallback)
            if merged == saved:
                continue
            tables["statistics"] = [r for r in tables.get("statistics", []) if r["event_id"] != eid] + merged
            coverage = [r for r in tables.get("coverage", []) if r["event_id"] == eid and r["component"] == "statistics"]
            if len(coverage) != 1:
                raise ValueError("Expected one existing statistics coverage record")
            # Keep the original last provider-attempt status; row sources identify
            # each actual contribution. 'present' means some data, not all metrics.
            coverage[0].update(status="present", source_key=f"backfill|/event/{eid}/statistics",
                               observed_at=max(r["observed_at"] for r in merged if r.get("observed_at") is not None),
                               raw_hash=hashlib.sha256(_json(merged).encode()).hexdigest())
            changed_rows = [r for r in merged if r not in saved]
            changes[eid] = dict(Counter("players" if r["source_key"].startswith("derived|") else "besoccer" for r in changed_rows))
        if not changes:
            return {"state": "unchanged", "season": stem, "matches": {}}
        grouped = defaultdict(list)
        for row in tables["statistics"]:
            grouped[row["event_id"]].append(row)
        tables["statistics"] = [r for m in tables["matches"] for r in grouped[m["event_id"]]]
        # Check the prior nested data before replacing its public entry.
        prior_nested = pq.ParquetFile(output / f"{stem}.parquet").read()
        canonical_info = read_json(canonical)
        with TemporaryDirectory(prefix="xdiyo_statistics_backfill_") as temporary:
            stage = Path(temporary) / "replacement"
            backup = Path(temporary) / "prior"
            for relative in prior["mutable"]:
                path = backup / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output / relative, path)
            reconciliation = deepcopy(outer.get("reconciliation", {}))
            if "collection" in reconciliation:
                reconciliation["collection"] = "chrome_campaign_with_statistics_backfill"
            if "scope" in reconciliation:
                reconciliation["scope"] = ("Saved fixture cohort preserved. Missing team statistics filled from "
                                           "validated saved player totals, then observed BeSoccer match statistics; "
                                           "existing row source fields identify each contribution.")
            result = publish_season(tables, stage, Scope(**canonical_info["scope"]),
                                    Profile(**outer["profile"]), stem, reconciliation)
            lineage = outer.get("reconciliation", {}).get("source_lineage")
            if lineage:
                destination = (stage / lineage).resolve()
                if not destination.is_relative_to(stage.resolve()):
                    raise ValueError("Lineage leaves bundle")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output / lineage, destination)
            staged_canonical = stage / result["manifest"]
            staged_info = read_json(staged_canonical)
            for name, info in staged_info["tables"].items():
                staged_table = pq.ParquetFile(staged_canonical.parent / info["file"]).read()
                if name in {"statistics", "coverage"}:
                    expected_table = to_arrow(name, tables[name])
                else:
                    expected_table = pq.ParquetFile(canonical.parent / canonical_info["tables"][name]["file"]).read()
                if set(staged_table.column_names) != set(expected_table.column_names):
                    raise ValueError(f"Staged relational backfill differs for {name}")
                # Earlier additive backfills appended is_awarded at the end.
                # Preserve that saved layout while still requiring exact values.
                aligned = staged_table.select(expected_table.column_names)
                if not aligned.equals(expected_table, check_metadata=True):
                    raise ValueError(f"Staged relational backfill differs for {name}")
                if aligned.column_names != staged_table.column_names:
                    table_path = staged_canonical.parent / info["file"]
                    pq.write_table(aligned, table_path, compression="zstd")
                    info.update(bytes=table_path.stat().st_size,
                                sha256=hashlib.sha256(table_path.read_bytes()).hexdigest())
                del staged_table, expected_table
            staged_canonical.write_bytes(json_bytes(staged_info))
            after = pq.ParquetFile(stage / f"{stem}.parquet").read()
            if set(prior_nested.column_names) != set(after.column_names):
                raise ValueError("Backfill changed the existing columns")
            if prior_nested.column_names != after.column_names:
                after = after.select(prior_nested.column_names)
                flat_path = stage / f"{stem}.parquet"
                pq.write_table(after, flat_path, compression="zstd")
                result.update(bytes=flat_path.stat().st_size,
                              sha256=hashlib.sha256(flat_path.read_bytes()).hexdigest())
                (stage / f"{stem}.manifest.json").write_bytes(json_bytes(result))
            if not prior_nested.schema.equals(after.schema, check_metadata=False):
                raise ValueError("Backfill changed the existing schema")
            for name in prior_nested.column_names:
                if name not in {"statistics", "coverage"} and not prior_nested[name].equals(after[name]):
                    raise ValueError(f"Backfill changed nested {name}")
            for name in ("statistics", "coverage"):
                for index, match in enumerate(tables["matches"]):
                    if match["event_id"] not in changes and prior_nested[name][index].as_py() != after[name][index].as_py():
                        raise ValueError("Backfill changed an untargeted fixture")
            replacement = _bundle(stage, stem, expected)
            immutable = {k: v for k, v in replacement["files"].items() if k not in replacement["mutable"]}
            _copy_files(stage, output, immutable, reuse_identical=True)
            _verify_files(output, prior["files"])
            try:
                for relative in replacement["mutable"]:
                    atomic_write_status(output / relative, (stage / relative).read_bytes())
                if _bundle(output, stem, expected)["files"] != replacement["files"]:
                    raise ValueError("Installed backfill differs from staged bundle")
                _verify_files(output, {k: v for k, v in prior["files"].items() if k not in prior["mutable"]})
            except BaseException:
                for relative in prior["mutable"]:
                    atomic_write_status(output / relative, (backup / relative).read_bytes())
                _verify_files(output, prior["files"])
                raise
        return {"state": "published", "season": stem, "matches": changes,
                "manifest": str(output / result["manifest"])}
