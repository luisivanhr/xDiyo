"""Thin adapters for the two supported historical notebook entry points."""
from pathlib import Path

from .client import BrowserClient
from .compatibility import export_csv, merge_recent_leagues_columns
from .config import Profile, RatePolicy, Scope
from .export import normalize_store, publish_tables
from .store import Store

ROOT = Path(__file__).resolve().parents[2]


def _collect(scope, profile, path, waiting_time, store_root, output_root, request_budget, write_csv,
             random_delay_max_seconds=None):
    from .runner import Runner
    # Preserve the old timeout argument. Actual pacing is a separate policy.
    policy = RatePolicy(page_load_timeout_seconds=waiting_time, max_attempts_per_run=request_budget,
                        random_delay_max_seconds=random_delay_max_seconds)
    with Store(store_root or ROOT/"data"/"collection_state") as store:
        client = BrowserClient(policy, driver_path=path)
        report = Runner(store, client, policy, profile).run(scope)
        if report["status"] != "complete":
            return {"run": report, "manifest": None,
                    "note": "Checkpoint saved. Resume the same scope to finish before publishing."}
        tables = normalize_store(store, scope, profile)
        manifest = publish_tables(tables, output_root or ROOT/"data"/"prepared_v2", scope=scope, profile=profile,
                                  provenance={"collection_run": report})
    result = {"run": report, "manifest": str(manifest)}
    if write_csv:
        result["csv_manifest"] = str(export_csv(manifest, Path(manifest).parent/"compatibility",
            name=f"{scope.league_name}_{scope.season_start}_{scope.season_end}", include_lineups=profile.include_lineups))
    return result


def scrape_league_data(league_id, season_id, num_rounds, num_teams, league_name, season_start,
                       season_end, path=None, waiting_time=30, with_lineups=False, with_xG=False,
                       with_comments=False, with_heatmaps=False, *, with_injuries=False,
                       with_goal_actions=False, duration_mode="metadata", store_root=None,
                       output_root=None, request_budget=40, write_csv=False, random_delay_max_seconds=None):
    """Collect with shared checkpoints and return explicit completion/output paths.

    num_teams remains accepted; it no longer invents standings when data are absent.
    Original files are preserved. CSV export is explicit and versioned.
    """
    scope = Scope(league_id, season_id, tuple(range(1, num_rounds+1)), league_name, season_start, season_end)
    profile = Profile(include_lineups=with_lineups, include_shotmap=with_xG, include_comments=with_comments,
                      include_heatmaps=with_heatmaps, include_injuries=with_injuries,
                      include_goal_actions=with_goal_actions, duration_mode=duration_mode)
    return _collect(scope, profile, path, waiting_time, store_root, output_root, request_budget, write_csv,
                    random_delay_max_seconds)


def scrape_league_recent(league_id, season_id, num_rounds, num_teams, league_name, season_start,
                         season_end, path=None, waiting_time=30, with_lineups=False, with_xG=False,
                         with_heatmaps=False, *, with_injuries=False, with_goal_actions=False,
                         duration_mode="metadata", store_root=None, output_root=None,
                         request_budget=40, write_csv=False, random_delay_max_seconds=None):
    scope = Scope(league_id, season_id, tuple(range(max(1,num_rounds-1), num_rounds+1)),
                  league_name, season_start, season_end)
    profile = Profile(include_lineups=with_lineups, include_shotmap=with_xG, include_heatmaps=with_heatmaps,
                      include_injuries=with_injuries, include_goal_actions=with_goal_actions,
                      duration_mode=duration_mode)
    return _collect(scope, profile, path, waiting_time, store_root, output_root, request_budget, write_csv,
                    random_delay_max_seconds)
