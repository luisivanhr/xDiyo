import json
from pathlib import Path

import pandas as pd
import pytest

from examples.prepare_library_20261116 import derive_annual, REVIEW
from xdiyo_analytics.data import load_season
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.data.season_calendar import POLICY, attach_calendar, publish_calendar, validate_calendar


def roster(league, year, teams, cid=1, sid=None, tier=1, new=True):
    return dict(league=league, year=year, teams={str(t): str(t) for t in teams},
                competition_id=cid, season_id=sid or year, tier=tier, country='Country',
                stem=f'{league}_{year % 100:02d}_{year % 100:02d}', new=new)


def review(league='League', year=2020):
    return dict(closed_leagues={}, boundaries=[dict(league=league, year=year,
                complete_review=True, sources=['reviewed membership'])])


def test_annual_flags_shared_but_new_closing_entrant_not_inserted_into_opening():
    rosters = [roster('League_Apertura', 2020, [1, 2], 10, 100),
               roster('League_Clausura', 2020, [1, 2], 20, 200),
               roster('League_Apertura', 2021, [1, 3], 10, 101),
               roster('League_Clausura', 2021, [1, 3, 4], 20, 201)]
    result = derive_annual(rosters, review())
    opening = result[result.season_id.eq(101)].set_index('team_id')
    closing = result[result.season_id.eq(201)].set_index('team_id')
    assert opening.movement.to_dict() == {1: 'retained', 3: 'promoted'}
    assert closing.movement.to_dict() == {1: 'retained', 3: 'promoted', 4: 'promoted'}
    assert closing.loc[1, 'previous_season_id'] == 200
    assert opening.loc[1, 'previous_season_id'] == 100
    assert pd.isna(closing.loc[3, 'previous_season_id'])


def test_cross_tier_predecessors_and_no_end_of_current_season_leakage():
    rs = [roster('Upper', 2020, [1, 2], 10), roster('Lower', 2020, [3, 4], 20, tier=2),
          roster('Upper', 2021, [1, 3], 10), roster('Lower', 2021, [2, 4], 20, tier=2)]
    config = review('Upper')
    config['boundaries'] += review('Lower')['boundaries']
    a = derive_annual(rs, config)
    assert a.loc[(a.season_id == 2021) & (a.team_id == 2), 'movement'].item() == 'relegated'
    assert a.loc[(a.season_id == 2021) & (a.team_id == 3), 'previous_competition_id'].item() == 20
    b = derive_annual(rs + [roster('Upper', 2022, [3, 4], 10)], config)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b[b.season_id.le(2021)].reset_index(drop=True))


def test_closed_league_expansion_is_not_promotion():
    cfg = review()
    cfg['closed_leagues']['League'] = 'reviewed closed competition'
    out = derive_annual([roster('League', 2020, [1]), roster('League', 2021, [1, 2])], cfg)
    assert out.loc[out.team_id.eq(2), 'movement'].item() == 'other_entry'
    assert not out.got_promoted.any()


def test_gap_requires_review_and_provider_change_does_not_invent_predecessor():
    with pytest.raises(ValueError, match='Unresolved'):
        derive_annual([roster('League', 2020, [1]), roster('League', 2022, [2])], review())
    out = derive_annual([roster('League_Apertura', 2020, [1], 10),
                         roster('League_Apertura', 2021, [1], 11)], review())
    assert out.iloc[-1].movement == 'retained'
    assert pd.isna(out.iloc[-1].previous_season_id)


def calendar():
    return dict(policy=POLICY, competition_id=17, season_id=61627,
                annual_league='League', annual_season='2024/2025', phase='Apertura', half=1)


def test_calendar_load_preserves_exact_ids_and_missing_source_flags(export, tmp_path):
    before = load_season(export.root, 'Premier_League_24_25', include_awarded=True).matches
    tables = {p: p.read_bytes() for p in export.root.rglob('*.parquet')}
    saved = tmp_path / 'old_selection.json'
    load_season(export.root, 'Premier_League_24_25', record_path=saved)
    publish_calendar(export.root, 'Premier_League_24_25', calendar())
    loaded = load_season(export.root, 'Premier_League_24_25', include_awarded=True)
    pd.testing.assert_frame_equal(before, loaded.matches[before.columns], check_flags=False)
    assert loaded.matches.annual_season.tolist() == ['2024/2025'] * 2
    assert loaded.matches.season_half.tolist() == [1, 1]
    assert loaded.provenance['season_calendar'] == calendar()
    history = build_team_history(loaded)
    assert history.annual_season.eq('2024/2025').all()
    assert history.annual_phase.eq('Apertura').all()
    assert 'season_phase' not in history
    assert all(p.read_bytes() == data for p, data in tables.items())
    with pytest.raises(ValueError, match='changed'):
        load_season(export.root, 'Premier_League_24_25', record_path=saved)
    snapshot = export.snapshot()
    publish_calendar(export.root, 'Premier_League_24_25', calendar())
    assert export.snapshot() == snapshot


def test_interrupted_calendar_publication_recovers_without_observation_rewrite(export, monkeypatch):
    replace = Path.replace
    count = 0
    def interrupt(self, target):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError('simulated interruption')
        return replace(self, target)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'replace', interrupt)
        with pytest.raises(OSError, match='interruption'):
            publish_calendar(export.root, 'Premier_League_24_25', calendar())
    with pytest.raises(ValueError, match='Pending calendar differs'):
        publish_calendar(export.root, 'Premier_League_24_25', dict(calendar(), half=2))
    publish_calendar(export.root, 'Premier_League_24_25', calendar())
    assert export.manifest()['season_calendar'] == calendar()
    assert json.loads(export.publication_path.read_bytes())['season_calendar'] == calendar()
    assert not list(export.root.rglob('*.pending.json'))


def test_calendar_rejects_concurrent_edits_and_foreign_identity(export):
    with pytest.raises(ValueError, match='identity'):
        publish_calendar(export.root, 'Premier_League_24_25', dict(calendar(), season_id=1))
    publish_calendar(export.root, 'Premier_League_24_25', calendar())
    with pytest.raises(ValueError, match='differs'):
        publish_calendar(export.root, 'Premier_League_24_25', dict(calendar(), annual_season='wrong'))


def test_interrupted_calendar_rejects_live_mutation(export, monkeypatch):
    with monkeypatch.context() as patch:
        def interrupt(*args):
            raise OSError('interrupted')
        patch.setattr(Path, 'replace', interrupt)
        with pytest.raises(OSError):
            publish_calendar(export.root, 'Premier_League_24_25', calendar())
    export.change_manifest(lambda m: m.update(unrelated_new_observation=True))
    with pytest.raises(ValueError, match='changed concurrently'):
        publish_calendar(export.root, 'Premier_League_24_25', calendar())
    assert export.manifest()['unrelated_new_observation'] is True


@pytest.mark.parametrize('field,value', [('policy', 'unsupported'), ('half', 3), ('half', True),
    ('half', 1.0), ('annual_season', ''), ('season_id', 61627.0)])
def test_invalid_calendar_evidence_fails(field, value):
    with pytest.raises(ValueError):
        validate_calendar(dict(calendar(), **{field: value}), {'league_id': 17, 'season_id': 61627})


def test_combined_stage_mapping_is_explicit_and_order_independent():
    cal = dict(calendar(), tournaments={'10': dict(phase='Apertura', half=1),
                                     '20': dict(phase='Clausura', half=2)})
    frame = pd.DataFrame(dict(tournament_id=[20, 10], event_id=[2**53 + 1, 7]))
    out = attach_calendar(frame, cal)
    assert out.season_half.tolist() == [2, 1]
    pd.testing.assert_frame_equal(out.iloc[::-1], attach_calendar(frame.iloc[::-1], cal))
    with pytest.raises(ValueError, match='Unreviewed'):
        attach_calendar(pd.DataFrame(dict(tournament_id=[30])), cal)
    with pytest.raises(ValueError, match='exact'):
        attach_calendar(pd.DataFrame(dict(tournament_id=[pd.NA])), cal)
    with pytest.raises(ValueError, match='integer'):
        attach_calendar(pd.DataFrame(dict(tournament_id=[10.5])), cal)
    with pytest.raises(ValueError, match='Reserved'):
        attach_calendar(out, cal)


def test_annual_labels_preserve_existing_numeric_season_phase():
    frame = pd.DataFrame({'season_phase': pd.array([1, 2, None], dtype='Int8')})
    out = attach_calendar(frame, calendar())
    pd.testing.assert_series_equal(out.season_phase, frame.season_phase)
    assert out.annual_phase.eq('Apertura').all()


def test_review_preserves_transition_calendar_and_source_snapshots():
    data = json.loads(REVIEW.read_bytes())
    by_stem = {r['stem']: r for r in data['seasons']}
    assert len(by_stem) == 617
    assert sum(r['matches'] for r in by_stem.values()) == 172274
    assert by_stem['Brazil_Serie_A_20_21']['calendar']['annual_season'] == '2020'
    assert by_stem['J1_Transition_26_26']['calendar']['annual_season'] == '2026_transition'
    assert by_stem['J1_26_27']['calendar']['annual_season'] == '2026/2027'
    assert by_stem['Liga_MX_Apertura_26_26']['calendar']['annual_season'] == '2026/2027'
    assert by_stem['Colombia_Primera_A_26_26']['calendar']['tournaments']['189591']['half'] == 2
    assert by_stem['Peru_Liga_1_17_17']['calendar']['tournaments']['57598'] == dict(phase='Verano', half=None)
    assert by_stem['Greek_Super_League_20_21']['matches'] == 182
    assert sum(r['snapshot'] is True for r in by_stem.values()) == 32


def test_all_reviewed_new_movements_and_shared_annual_flags():
    data = json.loads(REVIEW.read_bytes())
    rosters = [r for r in data['seasons'] if r['new'] or r['competition_id'] in (36, 37)]
    flags = derive_annual(rosters, data)
    assert len(flags) == 5402
    assert not flags[['got_promoted', 'got_demoted']].isna().any().any()
    for opening, closing in [('Liga_Profesional_Apertura_25_25', 'Liga_Profesional_Clausura_25_25'),
                              ('Colombia_Primera_A_Apertura_17_17', 'Colombia_Primera_A_Clausura_17_17')]:
        columns = ['team_id', 'movement', 'got_promoted', 'got_demoted']
        a = flags.loc[flags.season_stem.eq(opening), columns].sort_values('team_id').reset_index(drop=True)
        b = flags.loc[flags.season_stem.eq(closing), columns].sort_values('team_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
    assert flags.loc[flags.season_stem.eq('Peru_Liga_1_25_25') & flags.team_id.eq(33894), 'movement'].item() == 'other_entry'
