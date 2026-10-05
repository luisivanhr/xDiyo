"""Audit played-fixture coverage independently of a collector's queued IDs.

The caller supplies the reviewed regular-season format and required fixtures.
Do not apply a round-robin format to playoffs or a curtailed season.
"""
from collections import Counter
from datetime import datetime, timezone


def audit_fixture_coverage(matches, *, team_ids, meetings_per_pair, required=()):
    """Return format checks and independently reviewed fixture checks.

    ``complete`` in a publication describes collection completion. This audit
    instead checks the saved played rows. A balanced format is evidence about
    coverage, not proof that every source date or score is correct.
    ``required`` uses native team IDs, a UTC calendar date, and final scores.
    """
    teams = set(team_ids)
    if len(teams) < 2 or any(type(t) is not int or t <= 0 for t in teams):
        raise ValueError('Supply at least two distinct native team IDs')
    if type(meetings_per_pair) is not int or not 1 <= meetings_per_pair <= 4:
        raise ValueError('meetings_per_pair must be an integer from 1 to 4')
    rows = list(matches)
    issues = []
    ids = [r['event_id'] for r in rows]
    if len(ids) != len(set(ids)):
        issues.append('duplicate_event_ids')
    played = [r for r in rows if r.get('status') == 'finished' and not r.get('is_awarded')]
    if len(played) != len(rows):
        issues.append('nonplayed_or_awarded_rows_require_separate_review')
    valid = [r for r in played if r['home_id'] in teams and r['away_id'] in teams
             and r['home_id'] != r['away_id']]
    if len(valid) != len(played):
        issues.append('unexpected_team_identity')
    pairs = Counter(tuple(sorted((r['home_id'], r['away_id']))) for r in valid)
    directed = Counter((r['home_id'], r['away_id']) for r in valid)
    imbalances = []
    for a in sorted(teams):
        for b in sorted(teams):
            if a >= b:
                continue
            n = pairs[a, b]
            if n != meetings_per_pair:
                imbalances.append(dict(team_ids=[a, b], expected=meetings_per_pair, actual=n))
            elif meetings_per_pair % 2 == 0 and directed[a, b] != meetings_per_pair // 2:
                imbalances.append(dict(team_ids=[a, b], reason='home_away_imbalance'))
    checks = []
    for fixture in required:
        day = datetime.strptime(fixture['date_utc'], '%Y-%m-%d').date()
        candidates = [r for r in valid if (r['home_id'], r['away_id']) ==
                      (fixture['home_id'], fixture['away_id']) and
                      r.get('kickoff_utc') is not None and
                      datetime.fromtimestamp(r['kickoff_utc'], timezone.utc).date() == day]
        status = 'missing' if not candidates else 'duplicate' if len(candidates) > 1 else 'present'
        if status == 'present' and [candidates[0].get('home_score_current'),
                                    candidates[0].get('away_score_current')] != fixture['score']:
            status = 'score_mismatch'
        checks.append({**fixture, 'status': status, 'event_ids': [r['event_id'] for r in candidates]})
    return dict(status='incomplete' if issues or imbalances or any(c['status'] != 'present' for c in checks)
                else 'balanced', scope='reviewed regular-season format and listed external fixtures',
                teams=len(teams), expected_matches=len(teams)*(len(teams)-1)*meetings_per_pair//2,
                saved_matches=len(rows), played_matches=len(played), meetings_per_pair=meetings_per_pair,
                issues=issues, pair_imbalances=imbalances, required_fixtures=checks)
