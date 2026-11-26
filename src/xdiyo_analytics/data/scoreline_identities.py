"""Deterministic scoreline team crosswalk for a pinned build, never fuzzy."""
from collections import defaultdict
from datetime import date, timedelta

from .catalogue.registry import name_key


def identity_key(row, side):
    return (row['source'], row['league'], row['season'], row[f'source_{side}_key'])


def compile_crosswalk(native, external, aliases, lookup):
    """Use dated reviews, exact names in native seasons, and fixture evidence.

    One source key can refer to different teams in different seasons. Conflicts
    within one source/league/season are held, not decided by majority vote.
    Native exact names more than five seasons away are not identity evidence.
    Fixture propagation uses a fixed seed set, not its own suggestions as
    evidence. It requires two distinct fixtures and two already-known rivals.
    """
    comps=defaultdict(set)
    raw_names=defaultdict(lambda: defaultdict(set))
    native_by_rival=defaultdict(list)
    fixture_ids={}
    for r in native:
        league=r['league']; year=int(r['season'][:4])
        comps[league].add(r['competition_id'])
        for side in ('home','away'):
            tid=r[f'{side}_id']
            name=r[f'{side}_team']
            if tid and name:
                for label in {name,r.get(f'{side}_source_name') or name}:
                    raw_names[league,name_key(label)][tid].add(year)
        if (r['home_goals'] is None or not r['home_id'] or not r['away_id']
            or not r['source_reported_date']):continue
        ident=r['fixture_key']
        if ident in fixture_ids:continue
        fixture_ids[ident]=r
        for side,other in (('home','away'),('away','home')):
            k=(league,r['season'],r['source_reported_date'],r['home_goals'],r['away_goals'],side,r[f'{other}_id'])
            native_by_rival[k].append(r)
    COMP=defaultdict(list)
    for r in aliases:
        COMP[r['competition_id'],name_key(r['name'])].append(r)

    # Evidence keyed by (source, annual league, annual season, source-team key).
    # Keep the full support on disk for review, rather than just a score.
    support=defaultdict(dict)
    scope=set()
    for r,hold in external:
        league=r['league']
        if hold or league not in comps:continue
        yr=int(r['season'][:4]); day=str(r['source_reported_date'])
        for side in ('home','away'):
            name=r[f'{side}_team']; norm=name_key(name); k=identity_key(r,side); scope.add(k)
            exact=raw_names[league,norm]
            ids=set()
            reviewed=lookup(league,name,day)
            if reviewed:ids.add(reviewed)
            ids.update(m['team_id'] for c in comps[league] for m in COMP[c,norm] if m['start']<=day<=m['end'])
            # Any surviving competition-scoped dated alias takes precedence
            # over an exact spelling outside its effective dates. If there is
            # competing *nearby* native name evidence, hold the row.
            nearby={i for i,ys in exact.items() if min(abs(y-yr) for y in ys)<=5}
            if ids:
                for i in ids:support[k].setdefault(i,set()).add('dated_review')
            if len(exact)==1 or ids:
                for i in nearby:support[k].setdefault(i,set()).add('exact_native_name')
    seeds={k:next(iter(v)) for k,v in support.items() if len(v)==1}
    voted=defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for r,hold in external:
        if hold:continue
        day=r['source_reported_date']
        for side,other in (('home','away'),('away','home')):
            k=identity_key(r,side)
            rival=seeds.get(identity_key(r,other))
            if not rival or k not in scope:continue
            candidates={}
            for shift in (0,-1,1):
                search=(r['league'],r['season'],day+timedelta(days=shift),r['home_goals'],r['away_goals'],side,rival)
                for native_r in native_by_rival[search]:candidates[native_r['fixture_key']]=native_r
            if len(candidates)!=1:continue
            dest=next(iter(candidates.values())); tid=dest[f'{side}_id']
            voted[k][tid][dest['fixture_key']].add(rival)
    decisions={}; report=[]
    for k in sorted(scope):
        observed=support.get(k,{})
        v=voted.get(k,{})
        qualified={i for i,b in v.items() if len(b)>=2 and len(set().union(*b.values()))>=2}
        direct=set(observed)
        candidates=direct|qualified
        all_candidates=set(v)|candidates
        if len(candidates)==1:
            tid=next(iter(candidates))
            # One different anchored fixture is enough to quarantine a
            # nomination instead of guessing which club the source intended.
            if all_candidates=={tid}:
                methods=sorted(observed.get(tid,()))
                if tid in qualified:methods.append('repeated_ordered_fixture')
                proof=sorted(v.get(tid,{}))
                decisions[k]=tid
                report.append(dict(zip(('source','league','season','source_team_key'),k),sofascore_team_id=tid,
                                   decision='+'.join(methods),evidence_fixture_keys=proof,
                                   alternative_ids=[]))
                continue
        report.append(dict(zip(('source','league','season','source_team_key'),k),sofascore_team_id=None,
                           decision='conflict' if len(all_candidates)>1 else 'unresolved',
                           evidence_fixture_keys=[],alternative_ids=sorted(all_candidates)))
    # If two provider IDs ever oppose one another in this scope, they
    # cannot be aliases of one club. Quarantine both mappings everywhere in
    # that league-season, not just the one impossible fixture.
    collisions=set()
    for r,hold in external:
        if hold:continue
        home,away=identity_key(r,'home'),identity_key(r,'away')
        if home in decisions and decisions[home]==decisions.get(away):
            collisions.update((home,away))
    if collisions:
        for r in report:
            k=(r['source'],r['league'],r['season'],r['source_team_key'])
            if k in collisions:
                tid=decisions.pop(k)
                r.update(sofascore_team_id=None,decision='opponents_collapse',
                         alternative_ids=[tid])
    return decisions, report
