"""Measure scheduled-cluster changes against the rejected majority policy."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pandas as pd
from xdiyo_analytics.data import load_seasons
from xdiyo_analytics.features.countdown import SeasonCountdown, season_countdown

BASE='643c20297cf4e2233c50d91a62bf097d18d7f985'
SOURCE='src/xdiyo_analytics/features/countdown.py'
HERE=Path(__file__).resolve().parent


def main():
    raw=subprocess.check_output(['git','show',f'{BASE}:{SOURCE}'])
    name='xdiyo_analytics.features._majority_reference'
    old=types.ModuleType(name);sys.modules[name]=old
    exec(compile(raw,SOURCE,'exec'),old.__dict__)
    rows=[];summary=[]
    for manifest in sorted(Path('data/xDiyo_data').glob('*.manifest.json')):
        stem=manifest.name.removesuffix('.manifest.json');league,a,b=stem.rsplit('_',2)
        data=load_seasons('data/xDiyo_data',f'{a}_{b}',leagues=[league],tables=['matches','team_seasons'],include_awarded=True)
        games=data.matches
        previous=old.season_countdown(games,old.SeasonCountdown(),team_seasons=data['team_seasons'])
        current=season_countdown(games,SeasonCountdown(),team_seasons=data['team_seasons'])
        assert previous.isna().equals(current.isna()),stem
        changed=~(previous.eq(current)|(previous.isna()&current.isna()))
        n=data['team_seasons'].team_id.nunique()
        total=38 if league=='Premiership' else 18 if league=='J1_Transition' else (n-1 if n%2==0 else n)*(1 if league.startswith('Liga_MX_') else 2)
        for i in np.flatnonzero(changed):
            g=games.iloc[i]
            rows.append(dict(publication=stem,event_id=int(g.event_id),home=g.home_name,away=g.away_name,
                kickoff=pd.to_datetime(g.kickoff_utc,unit='s',utc=True).isoformat(),assigned_round=int(g['round']),
                total_rounds=total,old_effective_round=round(total-previous.iloc[i]*(total-1)),
                new_effective_round=round(total-current.iloc[i]*(total-1)),old_countdown=previous.iloc[i],new_countdown=current.iloc[i],
                direction='less_progress_than_majority' if current.iloc[i]>previous.iloc[i] else 'more_progress_than_majority'))
        summary.append(dict(publication=stem,fixtures=len(games),changed=int(changed.sum())))
    pd.DataFrame(rows).to_csv(HERE/'scheduled-value-changes.csv',index=False)
    pd.DataFrame(summary).to_csv(HERE/'scheduled-change-counts.csv',index=False)
    result=dict(reference_commit=BASE,reference_source_sha256=hashlib.sha256(raw).hexdigest(),
        candidate_source_sha256=hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest(),
        publications=len(summary),fixtures=sum(r['fixtures'] for r in summary),changed=len(rows),
        less_progress=sum(r['direction']=='less_progress_than_majority' for r in rows),
        more_progress=sum(r['direction']=='more_progress_than_majority' for r in rows),missing_masks_unchanged=True)
    (HERE/'scheduled-comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
