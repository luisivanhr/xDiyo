from datetime import datetime, timezone

import pytest

from xdiyo_analytics.data.fixture_coverage import audit_fixture_coverage


def row(eid, home, away):
    return dict(event_id=eid, home_id=home, away_id=away, status='finished', is_awarded=False,
                kickoff_utc=datetime(2015, 9, 15, tzinfo=timezone.utc).timestamp(),
                home_score_current=1, away_score_current=0)


def test_queue_completion_does_not_hide_missing_fixture():
    saved = [row(1, 1, 2)]
    required = [dict(home_id=2, away_id=1, date_utc='2015-09-15', score=[1, 0])]
    audit = audit_fixture_coverage(saved, team_ids=[1, 2], meetings_per_pair=2, required=required)
    assert audit['status'] == 'incomplete'
    assert audit['expected_matches'] == 2
    assert audit['required_fixtures'][0]['status'] == 'missing'
    assert saved == [row(1, 1, 2)]


def test_full_count_with_wrong_pair_direction_is_not_balanced():
    audit = audit_fixture_coverage([row(1, 1, 2), row(2, 1, 2)], team_ids=[1, 2], meetings_per_pair=2)
    assert audit['status'] == 'incomplete'
    assert audit['pair_imbalances'][0]['reason'] == 'home_away_imbalance'


def test_cancelled_stub_does_not_count_as_played():
    audit = audit_fixture_coverage([row(1, 1, 2), {**row(2, 2, 1), 'status': 'cancelled'}],
                                  team_ids=[1, 2], meetings_per_pair=2)
    assert audit['played_matches'] == 1
    assert audit['status'] == 'incomplete'


def test_required_date_and_score_are_checked():
    rows = [row(1, 1, 2), row(2, 2, 1)]
    target = dict(home_id=1, away_id=2, date_utc='2015-09-15', score=[1, 0])
    def check(t):
        return audit_fixture_coverage(rows, team_ids=[1, 2], meetings_per_pair=2, required=[t])
    assert check(target)['status'] == 'balanced'
    assert check({**target, 'score': [0, 0]})['required_fixtures'][0]['status'] == 'score_mismatch'
    assert check({**target, 'date_utc': '2015-09-16'})['required_fixtures'][0]['status'] == 'missing'


def test_duplicate_ids_fail_even_if_counts_balance():
    audit = audit_fixture_coverage([row(1, 1, 2), row(1, 2, 1)], team_ids=[1, 2], meetings_per_pair=2)
    assert 'duplicate_event_ids' in audit['issues']
    assert audit['status'] == 'incomplete'


def test_rejects_unspecified_format():
    with pytest.raises(ValueError):
        audit_fixture_coverage([], team_ids=[1, 2], meetings_per_pair=0)
