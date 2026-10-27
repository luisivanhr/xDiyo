"""Read-only publication audit. Run with PYTHONPATH=src from repository root."""
import json
from pathlib import Path
import pandas as pd
from xdiyo_analytics.data import load_seasons
from xdiyo_analytics.features.countdown import SeasonCountdown, season_countdown

root=Path('data/xDiyo_data')
records=[]
tournaments={}
for publication in sorted(root.glob('*.manifest.json')):
    stem=publication.name.removesuffix('.manifest.json')
    league, a, b=stem.rsplit('_',2)
    season=f'{a}_{b}'
    data=load_seasons(root,season,leagues=[league],tables=['matches','team_seasons'],include_awarded=True)
    games=data.matches
    tournaments.setdefault(league,set()).update(int(t) for t in games.tournament_id.dropna().unique())
    n=data['team_seasons'].team_id.nunique()
    values=season_countdown(games,SeasonCountdown(),team_seasons=data['team_seasons'])
    finite=values.dropna()
    assert finite.between(0,1).all(),stem
    total=(38 if league=='Premiership' else 18 if league=='J1_Transition'
           else (n-1 if n%2==0 else n)*(1 if league.startswith('Liga_MX_') else 2))
    records.append(dict(planned_rounds=total,publication=stem,teams=n,matches=len(games),max_assigned_round=games['round'].max(),
        double_round_robin_total=2*(n-1 if n%2==0 else n),
        countdown_min=finite.min(),countdown_max=finite.max(),missing=int(values.isna().sum())))
output=Path(__file__).with_name('publication-audit.csv')
pd.DataFrame(records).to_csv(output,index=False)
print(json.dumps(dict(publications=len(records),matches=sum(r['matches'] for r in records),
                     missing=sum(r['missing'] for r in records),receipt=str(output))))

print({k:sorted(v) for k,v in tournaments.items()})
