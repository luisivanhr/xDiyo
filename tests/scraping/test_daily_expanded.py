"""Offline acceptance tests for all new daily league shapes and mixed years."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import socket
import time

import pytest

from scraping.sofascore_web_v3.daily import prepare_daily_run, daily_status
from scraping.sofascore_web_v3.daily_catalog import daily_league_catalog, daily_selection, NEW_LEAGUES
from scraping.sofascore_web_v3.daily_expanded import ExpandedDailyDiscovery
from scraping.sofascore_web_v3.discovery import open_discovery
from scraping.sofascore_web_v3.campaign import Campaign
from scraping.sofascore_web_v3.daily_storage import publish_daily_campaign, read_daily_tables
from xdiyo_analytics.data.prediction import load_prediction_fixtures
from test_daily_collection import event, entry, collect


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Python sent a network request"))


def prepare(tmp_path, names, **kwargs):
    return prepare_daily_run(season="26/27", calendar_season="2026", leagues=names,
        output_root=tmp_path / "output", history_root=tmp_path / "history",
        discovery_base=tmp_path / "discovery", **kwargs)


def example(spec, *, number=1, pairs=((1, 2), (3, 4)), state="finished", stage="regular", group=False, day=1):
    cid = spec["competition_id"]
    result = []
    for i, teams in enumerate(pairs):
        eid = cid * 100000 + (10000 if stage == "Clausura" else 0) + number * 100 + i
        value = event(eid=eid, number=number, cid=cid, teams=teams, status=state,
                      kickoff=int(time.time()) + (day * 86400 if state == "notstarted" else -86400))
        years = spec["years"]
        value["season"]["year"] = str(years[0]) if years[0] == years[1] else f"{years[0]}/{years[1]}"
        value["tournament"].update(name=stage, slug=stage.lower(), isGroup=group)
        result.append(value)
    return result


def feed(discovery, events_by_cid, *, tables=None, change=None):
    paths = []
    while True:
        job = discovery.next_job()
        if job.get("done"):
            return paths
        assert not job.get("paused")
        path, cid = job["path"], job["competition_id"]
        spec = discovery.leagues[cid]
        events = events_by_cid[cid]
        if path.endswith("/seasons"):
            years = spec["years"]
            payload = {"seasons": [{"id": cid * 100, "year": str(years[0]) if years[0] == years[1] else f"{years[0]}/{years[1]}"}]}
        elif path.endswith("/standings/total"):
            from scraping.sofascore_web_v3.daily_expanded import phase, regular
            groups = defaultdict(list)
            for e in events:
                if regular(e, spec):
                    groups[phase(e, spec)].append(e)
            standings = []
            for group in groups.values():
                teams = sorted({e[s]["id"] for e in group for s in ("homeTeam", "awayTeam")})
                counts = Counter(e[s]["id"] for e in group if e["status"]["type"] in {"finished", "awarded"} for s in ("homeTeam", "awayTeam"))
                standings.append({"type": "total", "tournament": deepcopy(group[0]["tournament"]),
                    "rows": [{"team": {"id": t}, "matches": counts[t]} for t in teams]})
            payload = {"standings": (tables or {}).get(cid, standings)}
        elif path.endswith("/rounds"):
            payload = {"rounds": [{"round": n} for n in sorted({e["roundInfo"]["round"] for e in events})]}
        else:
            number = int(path.rsplit("/", 1)[1])
            payload = {"events": [e for e in events if e["roundInfo"]["round"] == number]}
        if change:
            payload = change(job, deepcopy(payload))
        discovery.observe(path, job["lease"], entry(path, payload))
        paths.append(path)


def test_catalog_all_32_new_leagues_plus_original_13():
    catalog = daily_league_catalog()
    assert len(catalog) == 45 and len(NEW_LEAGUES) == 32
    assert len({s["competition_id"] for s in catalog}) == len({s["name"] for s in catalog}) == 45
    assert next(s for s in catalog if s["name"] == "Colombia_Primera_A")["competition_id"] == 11539


def test_new_year_selectors_and_explicit_overrides():
    selected = daily_selection("26/27", ["Premier_League", "Brazil_Serie_A", "Allsvenskan"],
        calendar_season="2027", season_overrides={"Allsvenskan": "2026"})
    assert [s["years"] for s in selected] == [[2026, 2027], [2027, 2027], [2026, 2026]]
    with pytest.raises(ValueError, match="overrides"):
        daily_selection("26/27", ["Brazil_Serie_A"], season_overrides={"Typo": "2026"})
    with pytest.raises(ValueError, match="calendar"):
        daily_selection("26/27", ["Brazil_Serie_A"], season_overrides={"Brazil_Serie_A": "26/27"})
    assert daily_selection("2026", ["Brazil_Serie_A"])[0]["years"] == [2026, 2026]


@pytest.mark.parametrize("name", [s[0] for s in NEW_LEAGUES])
def test_every_new_catalog_entry_discovers_and_queues_upcoming(tmp_path, name):
    prepared = prepare(tmp_path, [name])
    with open_discovery(prepared["directory"]) as discovery:
        assert isinstance(discovery, ExpandedDailyDiscovery)
        cid, spec = next(iter(discovery.leagues.items()))
        events = example(spec) + example(spec, number=2, pairs=((1, 3), (2, 4)), state="notstarted")
        feed(discovery, {cid: events})
        summary = discovery.status()
        assert summary["ready_to_collect"], summary
        assert summary["seasons"][0]["prediction_candidates"] == 2
        campaign = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert campaign["matches"] == 4
    assert daily_status(prepared["discovery_id"], discovery_base=tmp_path / "discovery")["counts"] == {"pending": 4}


def test_calendar_and_split_publish_and_load_prediction_with_null_outcomes(tmp_path):
    prepared = prepare(tmp_path, ["Brazil_Serie_A", "Premier_League"])
    with open_discovery(prepared["directory"]) as discovery:
        inventories = {cid: example(spec) + example(spec, number=2, pairs=((1, 3), (2, 4)), state="notstarted")
                       for cid, spec in discovery.leagues.items()}
        feed(discovery, inventories)
        campaign = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
    with Campaign(campaign["directory"]) as collection:
        collect(collection)
    result = publish_daily_campaign(campaign["directory"])
    assert {s["stem"] for s in result["seasons"]} == {"Brazil_Serie_A_26_26", "Premier_League_26_27"}
    views = read_daily_tables(tmp_path / "output", view="prediction")
    assert sum(len(t["matches"]) for t in views.values()) == 4
    selection = load_prediction_fixtures(tmp_path / "output", seasons=["26_26", "26_27"],
        tables=("matches",), as_of=time.time(), verify_hashes=True)
    assert len(selection.fixtures) == 4
    for tables in views.values():
        assert all(row["status"] == "notstarted" and row["home_score_current"] is None for row in tables["matches"])
    # A later daily discovery reuses completed captures by identity, but refreshes
    # unfinished fixtures; original publication remains valid and hash checked.
    resumed = prepare(tmp_path, ["Brazil_Serie_A", "Premier_League"])
    assert resumed["discovery_id"] != prepared["discovery_id"]
    with open_discovery(resumed["directory"]) as discovery:
        feed(discovery, inventories)
        second = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert second["reused_matches"] == 4 and second["matches"] == 4


def test_two_ordinary_stages_keep_repeated_rounds_and_exclude_finals(tmp_path):
    prepared = prepare(tmp_path, ["Peru_Liga_1"])
    with open_discovery(prepared["directory"]) as discovery:
        spec = discovery.leagues[406]
        events = example(spec, stage="Apertura", state="postponed") + example(spec, stage="Clausura", state="notstarted")
        final = example(spec, stage="Final", state="notstarted")[0]
        final["id"] += 9000
        feed(discovery, {406: events + [final]})
        summary, records = discovery.season_selection(406)
        assert not summary["issues"] and len(records) == 4
        assert summary["excluded_stage_matches"] == 1
        assert {json.loads(r["payload"])["tournament"]["name"] for r in records} == {"Apertura", "Clausura"}
        assert {r["round"] for r in records} == {1}  # Never invent global round numbers.
        assert summary["superseded_fixtures"] == []


def test_conferences_and_uneven_rounds_reconcile_played_counts(tmp_path):
    prepared = prepare(tmp_path, ["MLS"])
    with open_discovery(prepared["directory"]) as discovery:
        spec = discovery.leagues[242]
        events = example(spec, pairs=((1, 2),), stage="Eastern Conference", group=True)
        events += example(spec, number=2, pairs=((3, 4),), stage="Western Conference", group=True)
        events += example(spec, number=3, pairs=((1, 3), (2, 4)), state="notstarted", stage="Eastern Conference", group=True)
        feed(discovery, {242: events})
        assert discovery.status()["ready_to_collect"]
        assert len(discovery.season_selection(242)[1]) == 4


@pytest.mark.parametrize("damage", ["lost_played_event", "foreign_season", "conflicting_duplicate", "unknown_status", "paginated", "unreviewed_group", "missing_round"])
def test_bad_inventories_block_creation_and_retain_responses(tmp_path, damage):
    prepared = prepare(tmp_path, ["A_League"])
    with open_discovery(prepared["directory"]) as discovery:
        spec = discovery.leagues[136]
        events = example(spec) + example(spec, number=2, pairs=((1, 3), (2, 4)), state="notstarted")
        def change(job, payload):
            if job["path"].endswith("/round/1"):
                if damage == "lost_played_event": payload["events"].pop()
                if damage == "foreign_season": payload["events"][0]["season"]["year"] = "25/26"
                if damage == "conflicting_duplicate": payload["events"].append({**deepcopy(payload["events"][0]), "id": 999})
                if damage == "unknown_status": payload["events"][0]["status"]["type"] = "unknown"
                if damage == "paginated": payload["hasNextPage"] = True
                if damage == "unreviewed_group": payload["events"][0]["tournament"]["isGroup"] = True
            if damage == "missing_round" and job["path"].endswith("/rounds"):
                payload["rounds"] = [{"round": 1}, {"round": 3}]
            return payload
        try:
            feed(discovery, {136: events}, change=change)
        except ValueError:
            pass
        assert not discovery.status()["ready_to_collect"]
        assert discovery.db.execute("select count(*) from responses").fetchone()[0] > 0
        with pytest.raises(ValueError):
            discovery.prepare_collection(campaign_base=tmp_path / "campaigns")


def test_fixture_window_is_stable_on_resume_and_unavailable_is_explicit(tmp_path):
    prepared = prepare(tmp_path, ["Eliteserien"])
    assert prepare(tmp_path, ["Eliteserien"])["discovery_id"] == prepared["discovery_id"]
    with open_discovery(prepared["directory"]) as discovery:
        spec = discovery.leagues[20]
        events = example(spec) + example(spec, number=2, pairs=((1, 3), (2, 4)), state="notstarted", day=25)
        feed(discovery, {20: events})
        summary, rows = discovery.season_selection(20)
        assert summary["state"] == "outside_fixture_window" and len(rows) == 2
    with open_discovery(prepared["directory"], read_only=True) as discovery:
        assert discovery.season_selection(20)[0] == summary


def test_presplit_inventory_excludes_later_rounds(tmp_path):
    prepared = prepare(tmp_path, ["Austrian_Bundesliga"])
    with open_discovery(prepared["directory"]) as discovery:
        spec = discovery.leagues[45]
        events = [e for n in range(1, 9) for e in example(spec, number=n, state="notstarted")]
        paths = feed(discovery, {45: events})
        assert discovery.status()["ready_to_collect"]
        assert not any(p.endswith(("/round/7", "/round/8")) for p in paths)
        assert discovery.season_selection(45)[0]["enumerated_rounds"] == list(range(1, 7))


def test_default_preparation_does_not_touch_an_existing_run(tmp_path):
    first = prepare(tmp_path, ["Brazil_Serie_A"])
    directory = Path(first["directory"])
    before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    second = prepare_daily_run(season="26/27", output_root=tmp_path / "output", history_root=tmp_path / "history",
                               discovery_base=tmp_path / "discovery")
    assert second["discovery_id"] != first["discovery_id"] and len(second["seasons"]) == 45
    assert {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()} == before


def test_argentina_regular_zones_and_interzonal_fixture(tmp_path):
    prepared = prepare(tmp_path, ["Liga_Profesional"])
    with open_discovery(prepared["directory"]) as discovery:
        spec = discovery.leagues[155]
        events = example(spec, pairs=((1, 2),), stage="Apertura Group A", group=True)
        other = example(spec, pairs=((3, 4),), stage="Apertura Group B", group=True)[0]
        other["id"] += 1
        events += [other]
        events += example(spec, number=2, pairs=((1, 3), (2, 4)), state="notstarted", stage="Apertura", group=False)
        feed(discovery, {155: events})
        assert discovery.status()["ready_to_collect"]
        assert discovery.season_selection(155)[0]["target_rounds"] == {"apertura": 2}


def test_colombian_championship_groups_are_not_ordinary_stages(tmp_path):
    prepared = prepare(tmp_path, ["Colombia_Primera_A"])
    with open_discovery(prepared["directory"]) as discovery:
        spec = discovery.leagues[11539]
        events = example(spec, stage="Apertura") + example(spec, stage="Clausura", state="notstarted")
        playoff = example(spec, stage="Apertura Group A", group=True)[0]
        playoff["id"] += 5000
        feed(discovery, {11539: events + [playoff]})
        summary, selected = discovery.season_selection(11539)
        assert discovery.status()["ready_to_collect"] and len(selected) == 4
        assert summary["excluded_stage_matches"] == 1


@pytest.mark.parametrize("kind", ["absent", "ambiguous"])
def test_missing_or_ambiguous_selected_season_never_falls_back(tmp_path, kind):
    prepared = prepare(tmp_path, ["Allsvenskan"])
    with open_discovery(prepared["directory"]) as discovery:
        spec = discovery.leagues[40]
        def change(job, payload):
            if job["path"].endswith("/seasons"):
                payload["seasons"] = ([{"id": 1, "year": "2027"}] if kind == "absent" else
                                      [{"id": 1, "year": "2026"}, {"id": 2, "year": "2026"}])
            return payload
        try:
            feed(discovery, {40: example(spec)}, change=change)
        except ValueError:
            pass
        assert not discovery.status()["ready_to_collect"]
        assert discovery.db.execute("select count(*) from events").fetchone()[0] == 0


def test_completed_league_summaries_are_cached_but_campaign_gate_revalidates(tmp_path, monkeypatch):
    prepared = prepare(tmp_path, ["Eliteserien"])
    with open_discovery(prepared["directory"]) as discovery:
        spec = discovery.leagues[20]
        feed(discovery, {20: example(spec)})
        def fail(_):
            raise AssertionError("full validation")
        monkeypatch.setattr(discovery, "_season_selection", fail)
        assert discovery.status()["ready_to_collect"]
        with pytest.raises(AssertionError, match="full validation"):
            discovery.prepare_collection(campaign_base=tmp_path / "campaigns")


def test_calendar_only_dictionary_and_named_knockout_round(tmp_path):
    from scraping.sofascore_web_v3.daily_expanded import regular
    prepared = prepare_daily_run(season="2026", leagues=[{"name": "Brazil_Serie_A", "competition_id": 325}],
        discovery_base=tmp_path / "discovery", output_root=tmp_path / "output", history_root=tmp_path / "history")
    with open_discovery(prepared["directory"]) as discovery:
        spec = discovery.leagues[325]
        events = example(spec)
        def change(job, payload):
            if job["path"].endswith("/rounds"):
                payload["rounds"].append({"round": 100, "name": "Round of 16"})
            return payload
        paths = feed(discovery, {325: events}, change=change)
        assert not any(p.endswith("/round/100") for p in paths)
        assert discovery.status()["ready_to_collect"]
        assert not regular(example(spec, stage="Round of 16")[0], spec)
