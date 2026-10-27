"""Compare published retrospective values to the immutable reviewed implementation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import pandas as pd
from xdiyo_analytics.data import load_seasons
from xdiyo_analytics.features.countdown import SeasonCountdown, season_countdown

BASE='7156b381e3f9e352b936db2cb3273a648c3db292'
PATH='src/xdiyo_analytics/features/countdown.py'


def main():
    raw=subprocess.check_output(['git','show',f'{BASE}:{PATH}'])
    name='xdiyo_analytics.features._reviewed_countdown'
    original=types.ModuleType(name);sys.modules[name]=original
    exec(compile(raw,PATH,'exec'),original.__dict__)
    publications=0;fixtures=0
    for manifest in sorted(Path('data/xDiyo_data').glob('*.manifest.json')):
        league,a,b=manifest.name.removesuffix('.manifest.json').rsplit('_',2)
        data=load_seasons('data/xDiyo_data',f'{a}_{b}',leagues=[league],tables=['matches','team_seasons'],include_awarded=True)
        old=original.season_countdown(data.matches,original.SeasonCountdown(),team_seasons=data['team_seasons'])
        new=season_countdown(data.matches,SeasonCountdown(schedule_mode='retrospective'),team_seasons=data['team_seasons'])
        pd.testing.assert_series_equal(new,old,check_exact=True)
        publications+=1;fixtures+=len(new)
    receipt=dict(reference_commit=BASE,reference_source_sha256=hashlib.sha256(raw).hexdigest(),
        candidate_source_sha256=hashlib.sha256(Path(PATH).read_bytes()).hexdigest(),
        publications=publications,fixtures=fixtures,comparison='exact Series equality including index, dtype and NaN positions',passed=True)
    Path(__file__).with_name('published-value-parity.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
