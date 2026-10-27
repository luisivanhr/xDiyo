"""Rebuild the verified phase join from the official HTML; compare without writes."""
import argparse
import hashlib
from io import StringIO
import json
from pathlib import Path

import pandas as pd
from xdiyo_analytics.data import load_seasons


def main(html, root=Path('data/xDiyo_data')):
    here=Path(__file__).resolve().parent
    evidence=json.loads((here/'j1-phase-provenance.json').read_text(encoding='utf-8'))
    raw=html.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==evidence['html_sha256'], 'Official capture changed; review before replacing reference'
    official=pd.read_html(StringIO(raw.decode('utf-8')))[0]
    assert len(official)==306
    reference=pd.DataFrame({
        'home_id':official.Home.map(evidence['team_crosswalk']),
        'away_id':official.Away.map(evidence['team_crosswalk']),
        'round':official.Sec.str.extract(r'(\d+)(?:st|nd|rd|th) Sec',expand=False).astype(int),
        'season_phase':official.Tournaments.map({'J1 1st':1,'J1 2nd':2}),
    })
    assert not reference.isna().any().any()
    matches=load_seasons(root,'16_16',leagues=['J1'],tables=['matches'],include_awarded=True).matches
    keys=['competition_id','season_id','event_id','home_id','away_id','round']
    joined=matches[keys].merge(reference,on=evidence['join_keys'],how='outer',validate='one_to_one',indicator=True)
    assert joined._merge.eq('both').all() and len(joined)==306
    result=joined[keys+['season_phase']].astype('int64').sort_values('event_id').reset_index(drop=True)
    packaged=pd.read_csv(here.parents[2]/'src/xdiyo_analytics/features/data/j1_2016_phases.csv')
    pd.testing.assert_frame_equal(result,packaged)
    print(json.dumps({'verified':len(result),'html_sha256':hashlib.sha256(raw).hexdigest(),
        'join_keys':evidence['join_keys'],'phase_counts':result.season_phase.value_counts().to_dict()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=Path('data/xDiyo_data'))
    args=parser.parse_args()
    main(args.html,args.root)
