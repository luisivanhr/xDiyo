"""Read-only audit with frozen counts and independently specified eligibility.

Expectations are reviewed data contracts, never regenerated from feature outputs.
The original publication-audit.csv remains the historical receipt for 7156b381.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xdiyo_analytics.data import load_seasons
from xdiyo_analytics.features.countdown import SeasonCountdown, season_countdown

HERE = Path(__file__).resolve().parent
# Independent audit policy: do not import the feature's stage/format helpers.
STAGES = {'Pro_League': [38], 'Premiership': [54, 92066, 92067],
    'Liga_MX_Apertura': [28], 'Liga_MX_Clausura': [16753],
    'Bundesliga': [42], 'Bundesliga_2': [41], 'Championship': [2],
    'Eredivisie': [39], 'J1': [82], 'J2': [3034],
    'J1_Transition': [171852, 171853], 'La_Liga': [36], 'La_Liga_2': [37],
    'Ligue_1': [4], 'Ligue_2': [19], 'Premier_League': [1], 'Serie_A': [33], 'Serie_B': [34]}


def validate_publication(stem, games, values, expected, checks):
    league, a, b = stem.rsplit('_', 2)
    stages = [28] if stem == 'Liga_MX_Clausura_10_11' else STAGES[league]
    eligible = games.tournament_id.isin(stages)
    assert len(games) == int(expected['matches']), f'{stem}: fixture count drift'
    assert int(eligible.sum()) == int(expected['expected_eligible']), f'{stem}: eligible count drift'
    assert int((~eligible).sum()) == int(expected['expected_excluded']), f'{stem}: excluded count drift'
    assert eligible.any(), f'{stem}: empty eligible population'
    assert len(values) == len(games) and values.index.equals(games.index), f'{stem}: output alignment'
    assert values.notna().equals(eligible), f'{stem}: unexplained missingness or nonmissing excluded fixture'
    assert np.isfinite(values[eligible]).all() and values[eligible].between(0, 1).all(), f'{stem}: invalid finite values'
    for check in checks:
        if check['publication'] != stem:
            continue
        selected = games.event_id.eq(check['event_id'])
        assert selected.sum() == 1, f'{stem}: missing hand-verified fixture'
        want = (check['total_rounds'] - check['effective_round']) / (check['total_rounds'] - 1)
        assert values[selected].iloc[0] == want, f'{stem}: wrong effective round for {check["event_id"]}'
    return eligible


def direct_reference(games, roster):
    """Independent exhaustive interval oracle, not production sliding windows.

    Enumerate every possible contiguous cluster, choose (-size, start), then
    compare Python integer endpoint distances one target at a time.
    """
    league = games.source_league.iloc[0]
    season = games.source_season.iloc[0]
    n = roster.team_id.nunique()
    total = 38 if league == 'Premiership' else 18 if league == 'J1_Transition' else (n-1 if n%2==0 else n)*(1 if league.startswith('Liga_MX_') else 2)
    stages = [28] if league == 'Liga_MX_Clausura' and season == '10_11' else STAGES[league]
    eligible = games.tournament_id.isin(stages).to_numpy()
    rounds = games['round'].to_numpy(dtype=float,na_value=np.nan,copy=True)
    if league == 'J1' and season == '16_16':
        phases = pd.read_csv(HERE.parents[2]/'src/xdiyo_analytics/features/data/j1_2016_phases.csv').set_index('event_id').season_phase
        rounds += (games.event_id.map(phases).to_numpy()-1)*17
    times = games.kickoff_utc.to_numpy(dtype=float,na_value=np.nan)
    groups = games.tournament_id.to_numpy() if league == 'J1_Transition' else np.zeros(len(games))
    usable = eligible & np.isfinite(times) & np.isfinite(rounds)
    intervals = {}
    for group in np.unique(groups[usable]):
        calendar = {}
        for ordinal in np.unique(rounds[usable & (groups==group)]):
            stamps=sorted(int(x) for x in times[usable & (groups==group) & (rounds==ordinal)])
            candidates=[(j-i+1,stamps[i],stamps[j]) for i in range(len(stamps)) for j in range(i,len(stamps)) if stamps[j]-stamps[i]<=96*3600]
            size,start,end=min(candidates,key=lambda x:(-x[0],x[1]))
            calendar[ordinal]=(start,end)
        intervals[group]=calendar
    out = np.full(len(games),np.nan)
    for i in range(len(games)):
        if not usable[i]:
            continue
        calendar=intervals.get(groups[i],{})
        ordinal=rounds[i];instant=int(times[i]);own=calendar.get(ordinal)
        if own is not None and own[0]<=instant<=own[1]:
            effective=ordinal
        elif calendar:
            nearest=min(calendar,key=lambda r:(max(calendar[r][0]-instant,instant-calendar[r][1],0),-r))
            effective=max(ordinal,nearest)
        else:
            effective=ordinal
        out[i] = (total-effective)/(total-1)
    return pd.Series(out,index=games.index)


def main(root=Path('data/xDiyo_data')):
    expected = pd.read_csv(HERE / 'expected-publication-counts.csv').set_index('publication')
    checks = json.loads((HERE / 'scheduled-effective-round-checks.json').read_text(encoding='utf-8'))
    publications = sorted(root.glob('*.manifest.json'))
    assert {p.name.removesuffix('.manifest.json') for p in publications} == set(expected.index), 'publication inventory drift'
    records = []
    for publication in publications:
        stem = publication.name.removesuffix('.manifest.json')
        league, a, b = stem.rsplit('_', 2)
        data = load_seasons(root, f'{a}_{b}', leagues=[league], tables=['matches', 'team_seasons'], include_awarded=True)
        games = data.matches
        values = season_countdown(games, SeasonCountdown(), team_seasons=data['team_seasons'])
        eligible = validate_publication(stem, games, values, expected.loc[stem], checks)
        reference = direct_reference(games, data['team_seasons'])
        pd.testing.assert_series_equal(values, reference, check_exact=True)
        records.append(dict(publication=stem, matches=len(games), eligible=int(eligible.sum()),
            excluded=int((~eligible).sum()), finite=int(values.notna().sum()),
            minimum=values.min(), maximum=values.max(), unexplained_missing=0,
            hand_verified=sum(c['publication'] == stem for c in checks), independent_reference=True))
    output = HERE / 'scheduled-publication-audit.csv'
    pd.DataFrame(records).to_csv(output, index=False)
    print(json.dumps(dict(publications=len(records), matches=sum(r['matches'] for r in records),
        finite=sum(r['finite'] for r in records), excluded=sum(r['excluded'] for r in records),
        hand_verified=sum(r['hand_verified'] for r in records), receipt=str(output))))


if __name__ == '__main__':
    main()
