from __future__ import annotations

from .models import Job


def verified_finished_event(event):
    """Reviewed stale postponements; retain the original response unchanged.

    Carpi-Siena was played on 24 November 2013, after the previous day's rain
    postponement, and finished 0-1. The provider retains that result and kickoff
    alongside the old postponed status. Never infer completion from a score alone.
    https://www.repubblica.it/sport/calcio/serie-b/2013/11/24/news/palermo_latina-71816673/
    """
    # Tokushima-Yamagata, J2 round 33, was played 16 September 2013 (2-2).
    # Both clubs' archives confirm the completed fixture, despite the stale flag.
    # https://www.vortis.jp/game/match/?year=2013
    # https://www.montedioyamagata.jp/match/40/
    if (event.get("id") == 3506406
            and (event.get("tournament") or {}).get("uniqueTournament", {}).get("id") == 402
            and (event.get("season") or {}).get("id") == 5918
            and (event.get("season") or {}).get("year") == "2013"
            and (event.get("roundInfo") or {}).get("round") == 33
            and (event.get("homeTeam") or {}).get("id") == 21907
            and (event.get("awayTeam") or {}).get("id") == 21902
            and event.get("startTimestamp") == 1379325600
            and (event.get("status") or {}).get("type") == "postponed"
            and (event.get("status") or {}).get("code") == 60
            and (event.get("homeScore") or {}).get("current") == 2
            and (event.get("awayScore") or {}).get("current") == 2):
        return True
    return (event.get("id") == 4167965
        and (event.get("tournament") or {}).get("uniqueTournament", {}).get("id") == 53
        and (event.get("season") or {}).get("id") == 6821
        and (event.get("season") or {}).get("year") in {"13/14", "2013/14", "2013/2014"}
        and (event.get("roundInfo") or {}).get("round") == 15
        and (event.get("homeTeam") or {}).get("id") == 43685
        and (event.get("awayTeam") or {}).get("id") == 170650
        and event.get("startTimestamp") == 1385301600
        and (event.get("status") or {}).get("type") == "postponed"
        and (event.get("status") or {}).get("code") == 60
        and (event.get("homeScore") or {}).get("current") == 0
        and (event.get("awayScore") or {}).get("current") == 1)


def is_completed_event(event):
    """Provider completion or a fully matched, independently verified correction."""
    return (event.get("status") or {}).get("type") == "finished" or verified_finished_event(event)


def verified_abandoned_event(event):
    """Standard-Zulte's 1-2 stood after abandonment; it was never replayed.

    The arbitration decision upheld the score on 10 December 2014 (pp. 18, 22).
    https://www.bas-cbas.be/cms/resources/141210-1-sentence-arbitral.pdf
    Preserve the stale postponed response and exclude it from played collection.
    """
    return (event.get("id") == 5555478
        and (event.get("tournament") or {}).get("uniqueTournament", {}).get("id") == 38
        and (event.get("season") or {}).get("id") == 8168
        and (event.get("season") or {}).get("year") in {"14/15", "2014/15", "2014/2015"}
        and (event.get("roundInfo") or {}).get("round") == 11
        and (event.get("homeTeam") or {}).get("id") == 2901
        and (event.get("awayTeam") or {}).get("id") == 2933
        and event.get("startTimestamp") == 1413721800
        and (event.get("status") or {}).get("type") == "postponed"
        and (event.get("status") or {}).get("code") == 60
        and (event.get("homeScore") or {}).get("current") == 1
        and (event.get("awayScore") or {}).get("current") == 2)


def event_status(event):
    if verified_finished_event(event):
        return "finished"
    if verified_abandoned_event(event):
        return "abandoned"
    status = event.get("status") or {}
    kind = str(status.get("type", "")).lower()
    # Awarded and abandoned are kept separate even if the provider calls them finished.
    description = str(status.get("description", "")).lower()
    for special in ("awarded", "abandoned", "cancelled", "canceled", "postponed"):
        if special in kind or special in description:
            return "cancelled" if special == "canceled" else special
    if kind in {"finished", "inprogress", "notstarted", "interrupted"}:
        return kind
    if status.get("code") in {100, 110, 120}:
        return "finished"
    return kind or "unknown"


def round_jobs(scope, locale="en"):
    return [Job(f"/unique-tournament/{scope.league_id}/season/{scope.season_id}/events/round/{r}",
                "discovery", locale=locale, reasons=(f"round:{r}",)) for r in scope.rounds]


def unresolved_jobs(store, scope, profile, now, refresh_seconds=21600):
    """Revisit tracked fixtures regardless of their original round number."""
    jobs = []
    for event in store.events(scope.league_id, scope.season_id):
        if event_status(event) in {"finished", "cancelled", "awarded", "abandoned"}:
            continue
        job = Job(f"/event/{event['id']}", "event", event["id"], locale=profile.locale,
                  reasons=("unresolved_fixture",))
        observed = store.db.execute("SELECT observed FROM events WHERE event_id=?", (event["id"],)).fetchone()[0]
        if now >= observed + refresh_seconds and store.due(job, now):
            jobs.append(job)
    return jobs
