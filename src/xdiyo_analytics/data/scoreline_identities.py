"""Deterministic scoreline team crosswalk for a pinned build, never fuzzy."""
from collections import defaultdict
from datetime import timedelta
from urllib.parse import urlsplit
import re
import unicodedata

from .catalogue.registry import name_key


def identity_key(row, side):
    return (row['source'], row['league'], row['season'], row[f'source_{side}_key'])


def slug_key(value):
    """Normalize presentation only; never strip team or reserve qualifiers."""
    text=value.casefold().translate(str.maketrans({'ł':'l','ø':'o','đ':'d','ı':'i','æ':'ae','œ':'oe'}))
    text=''.join(c for c in unicodedata.normalize('NFKD',text) if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+','-',text).strip('-') if text.isascii() else ''


# The predecessor and successor share this name; current slug alone isn't proof.
HISTORICAL_SLUG_HOLDS={('Bulgaria','cska-sofia'):2017}


def index_reviewed_source_teams(records, native):
    """Validate bounded, source-specific human-reviewed nominations."""
    league_ids=defaultdict(set)
    for row in native:
        league_ids[row['league']].update((row['home_id'],row['away_id']))
    index={}
    for row in records:
        fields=('source','league','season','source_team_key','source_name')
        if not isinstance(row,dict) or any(not isinstance(row.get(f),str) or not row[f].strip() for f in fields):
            raise ValueError('Invalid reviewed source team scope')
        key=tuple(row[f] for f in fields[:-1])
        tid=row.get('sofascore_team_id')
        urls=row.get('evidence_urls')
        if (type(tid) is not int or tid<=0 or tid not in league_ids[row['league']] or
            not isinstance(urls,list) or len(urls)<2 or any(not isinstance(u,str) for u in urls)):
            raise ValueError(f'Invalid reviewed source team evidence or target: {key}')
        parts=[urlsplit(u) for u in urls]
        if (len({u.hostname.removeprefix('www.') for u in parts if u.hostname})<2 or
            any(u.scheme!='https' or not u.hostname or u.username or u.password or u.query for u in parts)):
            raise ValueError(f'Expected two independent public HTTPS evidence hosts: {key}')
        if key in index:
            raise ValueError(f'Duplicate reviewed source team scope: {key}')
        index[key]=(row['source_name'],tid)
    return index


def compile_crosswalk(native, external, aliases, lookup, *, countries=None, slug_index=None, reviewed_source_teams=()):
    """Use dated reviews, exact names in native seasons, and fixture evidence.

    One source key can refer to different teams in different seasons. Conflicts
    within one source/league/season are held, not decided by majority vote.
    Native exact names more than five seasons away are not identity evidence.
    Fixture propagation uses a fixed seed set, not its own suggestions as
    evidence. It requires two distinct fixtures and two already-known rivals.
    """
    countries=countries or {}
    reviewed_index=index_reviewed_source_teams(reviewed_source_teams,native)
    reviews_seen=set()
    slugs=defaultdict(lambda: defaultdict(set))
    teams_by_slug={}
    comps=defaultdict(set)
    raw_names=defaultdict(lambda: defaultdict(set))
    native_by_rival=defaultdict(list)
    fixture_ids={}
    for r in native:
        league=r['league']; year=int(r['season'][:4])
        comps[league].add(r['competition_id'])
        country=countries.get(league)
        for side in ('home','away'):
            tid=r[f'{side}_id']
            name=r[f'{side}_team']
            slug=r.get(f'{side}_source_slug')
            if country and tid and name and isinstance(slug,str) and re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',slug):
                slugs[country,slug][tid].add(year)
                key=(country,slug,tid)
                entry=teams_by_slug.setdefault(key,dict(country=country,slug=slug,team_id=tid,names=set(),seasons=set()))
                entry['names'].add(name)
                entry['seasons'].add(r['season'])
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
    if slug_index is not None:
        slug_index.extend(dict(row,names=sorted(row['names']),seasons=sorted(row['seasons']))
                          for _,row in sorted(teams_by_slug.items()))
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
            if k in reviewed_index:
                reviewed_name, reviewed_id=reviewed_index[k]
                if name!=reviewed_name:
                    raise ValueError(f'Reviewed source team name disagrees with input: {k}')
                reviews_seen.add(k)
                support[k].setdefault(reviewed_id,set()).add('reviewed_source_team')
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
            country=countries.get(league); slug=slug_key(name)
            slot=(country,slug); hits=slugs.get(slot,{})
            if len(hits)==1 and yr>=HISTORICAL_SLUG_HOLDS.get(slot,0):
                tid,years=next(iter(hits.items()))
                if min(abs(y-yr) for y in years)<=10:
                    support[k].setdefault(tid,set()).add('exact_country_slug')
    if reviews_seen!=set(reviewed_index):
        missing=sorted(set(reviewed_index)-reviews_seen)
        raise ValueError(f'Reviewed source teams not found in eligible build inputs: {missing[:5]}')
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
