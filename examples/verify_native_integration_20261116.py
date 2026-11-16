"""Audit the final publication and refresh demonstration without fitting."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq

from xdiyo_analytics.data import load_season
from xdiyo_analytics.data._source import _open_source
from xdiyo_analytics.features import SeasonCountdown
from xdiyo_analytics.features.countdown import season_countdown
from xdiyo_analytics.ui.recipe import prepare_recipe

ROOT=Path('data/xDiyo_data');OUT=Path('docs/analytics/native_integration_20261116')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    baseline=json.loads((OUT/'inventory.json').read_bytes())
    awards=json.loads((OUT/'awarded_materialization.json').read_bytes())
    changed={a['stem']:set(a['changed']) for a in awards}
    rows=[];names=[];keys={};duplicates=[];flagged=[];protected=0;modified=[];all_tables=0
    for i,old in enumerate(baseline['publications']):
        stem=old['stem'];src=_open_source(ROOT,stem);metadata=src.manifest['native_integration']
        for name,info in src.manifest['tables'].items():
            path=src.table_path(name)
            if sha(path)!=info['sha256']:raise ValueError('Current table fingerprint mismatch: '+str(path))
            if info['sha256']!=old['tables'][name]['sha256']:
                if stem not in changed or name!='matches':raise ValueError('Unexpected observation change: '+str(path))
                modified.append(str(path))
                original_bytes=subprocess.run(['git','show',baseline['base_revision']+':'+path.relative_to(Path.cwd()).as_posix()],check=True,capture_output=True).stdout
                if hashlib.sha256(original_bytes).hexdigest()!=old['tables'][name]['sha256']:raise ValueError('Old revision fingerprint differs')
                original=pq.ParquetFile(io.BytesIO(original_bytes)).read();actual=pq.ParquetFile(path).read()
                columns=[c for c in original.column_names if c!='is_awarded']
                assert original.select(columns).equals(actual.select(columns),check_metadata=True)
                assert original['is_awarded'].equals(actual['provider_is_awarded'])
                actual_changes={eid for eid,a,b in zip(actual['event_id'].to_pylist(),original['is_awarded'].to_pylist(),actual['is_awarded'].to_pylist()) if a!=b}
                assert actual_changes==changed[stem]
            else:protected+=1
            all_tables+=1
        data=load_season(ROOT,stem,tables=['matches','team_seasons'],include_awarded=True)
        frame=data.matches
        for cid,sid,eid in frame[['competition_id','season_id','event_id']].itertuples(index=False,name=None):
            key=(int(cid),int(sid),int(eid))
            if key in keys:duplicates.append(dict(identity=key,stems=[keys[key],stem]))
            keys[key]=stem
        for r in frame.loc[frame.is_awarded.fillna(False)].itertuples():
            if old['new']:flagged.append(dict(stem=stem,event_id=int(r.event_id),home=r.home_name,away=r.away_name))
        if old['new'] and (set(old['source_teams'])-set(metadata['teams'])):
            raise ValueError('Unverified team display name in added publication: '+stem)
        publication=json.loads((ROOT/(stem+'.manifest.json')).read_bytes())
        assert publication['native_integration']==metadata
        assert sha(ROOT/publication['season_file'])==publication['sha256']
        for side in ('home','away'):
            assert set(frame[side+'_id']).issubset(set(data['team_seasons'].team_id))
        names.append(dict(stem=stem,competition_id=old['scope']['league_id'],season_id=old['scope']['season_id'],
            old_league_key=old['scope']['league_name'],sofascore_competition=metadata['competition']['name'],
            old_season_key=stem.rsplit('_',2)[1]+'_'+stem.rsplit('_',2)[2],sofascore_season=metadata['season']['name'],
            sofascore_year=metadata['season']['year'],annual_season=old['calendar']['annual_season'],
            old_filename=stem+'.parquet',current_filename=stem+'.parquet'))
        rules=metadata['rules']
        if old['new'] and rules['countdown']['status']=='supported':
            value=season_countdown(frame,SeasonCountdown(),team_seasons=data['team_seasons'])
            assert np.isfinite(value).all() and ((value>=0)&(value<=1)).all()
        rows.append(dict(stem=stem,new=old['new'],rows=len(frame),coverage=metadata['coverage']['kind'],
            standings=rules['standings']['status'],standings_reason=rules['standings'].get('reason',''),
            countdown=rules['countdown']['status'],countdown_reason=rules['countdown'].get('reason',''),
            manifest_sha256=sha(src.path),publication_sha256=sha(ROOT/(stem+'.manifest.json'))))
        if (i+1)%100==0:print('Verified',i+1,'publications',flush=True)
    if duplicates:raise ValueError('Duplicate native identities: '+str(duplicates[:5]))
    for name,content in [('name_mapping.csv',names),('integration_status.csv',rows),('awarded_matches.csv',flagged)]:
        with (OUT/name).open('w',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(content[0]));w.writeheader();w.writerows(content)
    old_recipe=json.loads((OUT/'demo_v1.json').read_bytes());new_recipe=json.loads((OUT/'demo_v2.json').read_bytes())
    try:prepare_recipe(old_recipe)
    except ValueError as e:
        assert 'saved season manifest has changed' in str(e)
        stale_error=str(e)
    else:raise AssertionError('Stale recipe unexpectedly replayed')
    prepared=prepare_recipe(new_recipe)
    assert len(prepared.dataset.X)==380 and np.isfinite(prepared.dataset.X.to_numpy()).all()
    protected_inputs=json.loads((OUT/'protected_active_inputs.json').read_bytes())
    assert all(sha(Path(path))==expected for path,expected in protected_inputs.items())
    samples={}
    for stem in ['Allsvenskan_19_19','Brazil_Serie_A_20_21','Liga_Portugal_24_25','Colombia_Primera_A_26_26','J1_Transition_26_26','J1_26_27','Chile_Primera_Division_19_19']:
        src=_open_source(ROOT,stem);m=src.manifest
        samples[stem]=dict(scope=m['scope'],native_season=m['native_integration']['season'],
            annual=m['season_calendar'],rules=m['native_integration']['rules'],coverage=m['native_integration']['coverage'])
    receipt=dict(publications=len(rows),fixtures=len(keys),tables_verified=all_tables,unchanged_tables=protected,
        changed_tables=modified,duplicate_identities=duplicates,new_awarded_rows=len(flagged),
        awarded_corrections={k:sorted(v) for k,v in changed.items()},
        new_format_status=dict(countdown=Counter(r['countdown'] for r in rows if r['new']),standings=Counter(r['standings'] for r in rows if r['new'])),
        demo=dict(rows=len(prepared.dataset.X),folds=len(prepared.split_plan.folds),stale_recipe_failure=stale_error,trained=False),
        samples=samples,active_input_pins_unchanged=True,
        scripts={str(p):sha(p) for p in [Path(__file__),Path('examples/integrate_native_20261116.py'),Path('examples/audit_native_identity_20261116.py')]})
    (OUT/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('samples','scripts')},ensure_ascii=True))


if __name__=='__main__':main()
