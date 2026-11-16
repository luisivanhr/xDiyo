"""Read-only native-library coverage, engine and legacy-parity audit. No fits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pandas as pd
from xdiyo_analytics.data import NativeLibrary, AmbiguousNativeName, load_seasons
from xdiyo_analytics.data.native_preflight import preflight_loaded, prepare_native_history
from xdiyo_analytics.features import SeasonCountdown
from xdiyo_analytics.features.countdown import season_countdown
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.ratings import build_standings

SOURCES=['examples/verify_native_preflight_20261116.py','examples/preflight_native_library.py',
    'src/xdiyo_analytics/data/__init__.py','src/xdiyo_analytics/data/loading.py','src/xdiyo_analytics/data/native_integration.py',
    'src/xdiyo_analytics/data/native_query.py','src/xdiyo_analytics/data/native_formats.py','src/xdiyo_analytics/data/native_preflight.py',
    'src/xdiyo_analytics/ui/recipe.py','src/xdiyo_analytics/ui/refresh.py','src/xdiyo_analytics/ui/server.py','src/xdiyo_analytics/ui/build_inventory.py',
    'src/xdiyo_analytics/features/countdown.py','src/xdiyo_analytics/histories/team.py','src/xdiyo_analytics/labels/creation.py',
    'src/xdiyo_analytics/ratings/standings.py','tests/analytics/test_native_query_preflight.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_tables(history, run):
    """Independent score arithmetic at every audited cutoff/window/team."""
    from xdiyo_analytics.features.history import availability_times
    available=availability_times(history,'kickoff_at')
    checks=0
    for (cutoff,window),snap in run.snapshots.groupby(['recorded_at','window']):
        rows=[]
        for team in snap.team_id:
            past=history.loc[history.team_id.eq(team)&history.status.eq('finished')&history.kickoff_at.lt(cutoff)&available.le(cutoff)]
            past=past.sort_values(['kickoff_at','event_id'])
            if window:past=past.tail(int(window))
            gf=past.goals_for.astype(int);ga=past.goals_against.astype(int)
            rows.append(dict(team_id=team,played=len(past),points=int((3*gf.gt(ga)+gf.eq(ga)).sum()),
                             gf=int(gf.sum()),ga=int(ga.sum()),gd=int((gf-ga).sum())))
        totals=pd.DataFrame(rows)
        ordered=sorted((-r.points,-r.gd,-r.gf,r.ga) for r in totals.itertuples())
        totals['rank']=[ordered.index((-r.points,-r.gd,-r.gf,r.ga))+1 for r in totals.itertuples()]
        for column in ('played','points','gf','ga','gd','rank'):
            np.testing.assert_array_equal(snap.set_index('team_id')[column].sort_index(),totals.set_index('team_id')[column].sort_index())
            checks+=len(snap)
    return checks


def legacy_module(path, name):
    payload=subprocess.run(['git','show','e82ffdff:'+path],check=True,capture_output=True).stdout
    module=types.ModuleType(name);module.__package__=name.rsplit('.',1)[0]
    sys.modules[name]=module
    exec(compile(payload,path+'@e82ffdff','exec'),module.__dict__)
    return module,hashlib.sha256(payload).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/xDiyo_data')
    parser.add_argument('--output',default='docs/analytics/native_preflight_20261116/verification.json')
    args=parser.parse_args();root=Path(args.root).resolve();output=Path(args.output).resolve()
    if output.is_relative_to(root):raise ValueError('Audit output must be outside the dataset')
    initial_hashes={p:sha(p) for p in SOURCES}
    library=NativeLibrary(root)
    receipt=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 command=subprocess.list2cmdline([sys.executable,*sys.argv]),publications=len(library.publications),entities={},cohorts=[],failures=[])
    for kind in ('competitions','seasons','phases','teams','annual_groups'):
        entities=library.discover(kind);ambiguous=0
        for row in entities:
            population=[p for p in library.publications if p['stem'] in row['publications']]
            resolved=library._resolve(kind,[row['id']],population)
            assert len(resolved)==1
            for alias in row['aliases']:
                try:library._resolve(kind,[alias],population)
                except AmbiguousNativeName:ambiguous+=1
        receipt['entities'][kind]=dict(count=len(entities),evidence=dict(Counter(e['evidence_status'] for e in entities)),
            ambiguous_alias_checks=ambiguous,exceptions=[{k:e[k] for k in ('id','name','aliases','evidence_status','conflicts')} for e in entities if e['evidence_status']!='captured'])
    print('Identity checks complete',flush=True)
    reports=[];expected_rows=0;observed_rows=0
    for i,pub in enumerate(library.publications):
        data=library.load(query={'publications':pub['stem']},tables=['matches','team_seasons'],include_awarded=True,verify_hashes=True)
        expected_rows+=pub['manifest']['tables']['matches']['rows'];observed_rows+=len(data.matches)
        report=preflight_loaded(data,required_features=['countdown','standings','movements']);reports.extend(report['cohorts'])
        history=prepare_native_history(data,build_team_history(data),{})
        # Audit only: avoid copying large display/source attrs on every manual
        # table slice. All engine identity/rule fields remain in owned columns.
        history.attrs={}
        supported_clock={r['phase_id'] for r in report['cohorts'] if r['countdown']['status']=='supported'}
        clock_frame=data.matches.loc[data.matches.tournament_id.isin(supported_clock)]
        if len(clock_frame):
            try:
                clock=season_countdown(clock_frame,SeasonCountdown(),team_seasons=data['team_seasons'])
                assert np.isfinite(clock).all() and clock.between(0,1).all()
            except Exception as exc:
                receipt['failures'].append(dict(publication=pub['stem'],feature='countdown',error=repr(exc)))
                print(receipt['failures'][-1],flush=True)
        for row in report['cohorts']:
            result=dict(publication=pub['stem'],phase_id=row['phase_id'],loaded=row['counts']['loaded'],
                        countdown=row['countdown']['status'],standings=row['standings']['status'],readiness=row['readiness'])
            if row['standings']['status']=='supported':
                h=history.loc[history.tournament_id.eq(row['phase_id'])].copy()
                events=h.loc[h.side.eq('home')].sort_values(['kickoff_at','event_id'])
                queries=events.iloc[sorted(set([0,len(events)//2,len(events)-1]))]
                try:
                    run=build_standings(h,team_seasons=data['team_seasons'],normalized_ranks=[1,5,20],
                                       queries=queries,cutoffs=queries.kickoff_at-pd.Timedelta(days=1),available_at='kickoff_at')
                    result['independent_scalar_checks']=independent_tables(h,run)
                    result['snapshots']=len(run.snapshots)
                    result['source_sha256']=run.metadata['source_sha256']
                except Exception as exc:
                    receipt['failures'].append(dict(publication=pub['stem'],phase_id=row['phase_id'],feature='standings',error=repr(exc)))
                    print(receipt['failures'][-1],flush=True)
            receipt['cohorts'].append(result)
        if (i+1)%50==0:print(f'Audited {i+1}/{len(library.publications)} publications; failures={len(receipt["failures"])}',flush=True)
    assert expected_rows==observed_rows==sum(r['counts']['loaded'] for r in reports)
    receipt.update(fixtures=observed_rows,phase_cohorts=len(reports),
        readiness=dict(Counter(r['readiness'] for r in reports)),
        countdown=dict(Counter(r['countdown']['status'] for r in reports)),
        standings=dict(Counter(r['standings']['status'] for r in reports)),
        independent_scalar_checks=sum(r.get('independent_scalar_checks',0) for r in receipt['cohorts']))
    # Legacy storage-key callers keep values, publications and state identities.
    legacy=load_seasons(root,'24_25',leagues=['Premier_League'],tables=['matches','team_seasons'])
    h=build_team_history(legacy);queries=h.loc[h.side.eq('home')].iloc[[0,100,200,300]]
    parent,parent_sha=legacy_module('src/xdiyo_analytics/ratings/standings.py','xdiyo_analytics.ratings._audit_parent_standings')
    kwargs=dict(team_seasons=legacy['team_seasons'],normalized_ranks=[1,5,20],queries=queries,available_at='kickoff_at')
    before=parent.build_standings(h,**kwargs);after=build_standings(h,**kwargs)
    pd.testing.assert_frame_equal(before.snapshots,after.snapshots)
    assert before.publications==after.publications and before.metadata==after.metadata
    parent_clock,clock_sha=legacy_module('src/xdiyo_analytics/features/countdown.py','xdiyo_analytics.features._audit_parent_countdown')
    pd.testing.assert_series_equal(parent_clock.season_countdown(legacy.matches,parent_clock.SeasonCountdown(),team_seasons=legacy['team_seasons']),
                                   season_countdown(legacy.matches,SeasonCountdown(),team_seasons=legacy['team_seasons']))
    receipt['legacy_parity']=dict(publication='Premier_League_24_25',standings_rows=len(after.snapshots),countdown_rows=len(legacy.matches),
        standings_parent_sha256=parent_sha,countdown_parent_sha256=clock_sha,exact=True)
    protected=json.loads(Path('docs/analytics/native_integration_20261116/protected_active_inputs.json').read_bytes())
    receipt['protected_active_inputs']={path:sha(path)==digest for path,digest in protected.items()}
    assert all(receipt['protected_active_inputs'].values())
    receipt['source_sha256']={p:sha(p) for p in SOURCES}
    assert receipt['source_sha256']==initial_hashes,'Audited source changed during verification'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('cohorts','entities','source_sha256')},indent=2),flush=True)
    if receipt['failures']:raise SystemExit(1)


if __name__=='__main__':main()
