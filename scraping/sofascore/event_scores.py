"""Exact reviewed score projection; original event responses remain unchanged."""


def reviewed_score_correction(event):
    """Cuenca-Catolica: the provider's current/display disagrees with its own FT.

    Original HTTP 200 event and incidents are retained in test fixtures. The FT
    marker, five goals, both periods and normal-time score all say 2-3, matching
    the ordinary table and contemporary reports from Primicias and El Mercurio.
    No rule applies to another match, changed identity, extra time or shootout.
    """
    if event.get("id") != 11031965:
        return False
    home, away = event.get("homeScore", {}), event.get("awayScore", {})
    identity = (event.get("tournament", {}).get("uniqueTournament", {}).get("id"),
                event.get("tournament", {}).get("id"), event.get("season", {}).get("id"),
                event.get("homeTeam", {}).get("id"), event.get("awayTeam", {}).get("id"),
                event.get("roundInfo", {}).get("round"), event.get("startTimestamp"),
                event.get("status", {}).get("type"), event.get("status", {}).get("code"))
    if identity != (240, 1387, 48720, 5251, 21942, 9, 1696636800, "finished", 100):
        raise ValueError("Reviewed Cuenca-Catolica score identity changed")
    if any(s.get(k) is not None for s in (home, away) for k in ("extra1", "extra2", "overtime", "penalties")):
        raise ValueError("Reviewed Cuenca-Catolica score has extra-time or penalty data")
    expected = ((2, 2, 1, 1, 2), (4, 4, 2, 1, 3))
    actual = tuple(tuple(s.get(k) for k in ("current", "display", "period1", "period2", "normaltime"))
                   for s in (home, away))
    if actual != expected:
        # A provider repair is already correct; other unexpected changes require review.
        if actual == ((2, 2, 1, 1, 2), (3, 3, 2, 1, 3)):
            return False
        raise ValueError("Reviewed Cuenca-Catolica score fields changed")
    return True


def match_scores(event, side):
    value = event.get(side + "Score") or {}
    if reviewed_score_correction(event) and side == "away":
        return {**value, "current": 3, "display": 3}
    return value
