"""Canonical, provider-neutral identity decisions. External keys never persist."""
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import unicodedata


def name_key(value):
    return ' '.join(unicodedata.normalize('NFC', str(value)).casefold().split())


class Registry:
    """Durable allocated IDs; reviewed aliases have scope and effective dates.

    Allocation is explicit and monotonic, never based on hashes or foreign IDs.
    Existing native IDs can be reserved by seed(). Unknown aliases do not match.
    """
    def __init__(self, payload=None):
        self.payload = deepcopy(payload or dict(schema=1, revision=0, entities={}, aliases=[], squads=[]))
        if self.payload['schema'] != 1:
            raise ValueError('Unsupported canonical registry schema')
        self._index()

    def _index(self):
        self._next={kind:max(map(int,entities),default=0)+1 for kind,entities in self.payload['entities'].items()}
        self.aliases = {}
        self._alias_records = {json.dumps(a,sort_keys=True) for a in self.payload['aliases']}
        self._squad_records = {json.dumps(a,sort_keys=True) for a in self.payload['squads']}
        self._squads = {}
        for a in self.payload['squads']: self._squads.setdefault(a['team'],[]).append(a)
        for a in self.payload['aliases']:
            self.aliases.setdefault((a['kind'], a['scope'], name_key(a['name'])), []).append(a)
        self.fixture_keys = {}
        self._pairs = {}
        for key, item in self.payload['entities'].get('fixture', {}).items():
            for slot in item['occurrences']:
                self.fixture_keys[tuple(slot)] = int(key)
                self._pairs.setdefault(tuple(slot[:4]),[]).append((slot[4],slot[5],int(key)))

    def seed(self, kind, identity, **facts):
        if isinstance(identity, bool) or not isinstance(identity, int) or identity <= 0:
            raise ValueError('Canonical IDs must be exact positive integers')
        allowed={'competition':{'name','region','entity_kind','format'},'season':{'competition','label','start_year','end_year','edition','stage'},
                 'team':{'name','entity_kind'},'fixture':{'occurrences','decisions'}}
        if kind not in allowed or set(facts)-allowed[kind]:raise ValueError('Noncanonical entity fields')
        entities = self.payload['entities'].setdefault(kind, {})
        existing = entities.get(str(identity))
        if existing is not None and existing != facts:
            raise ValueError(f'Conflicting canonical {kind} {identity}')
        if existing is None:
            entities[str(identity)] = deepcopy(facts)
            self._next[kind]=max(self._next.get(kind,1),identity+1)
            self.payload['revision'] += 1
        return identity

    def allocate(self, kind, **facts):
        identity = self._next.get(kind,1)
        return self.seed(kind, identity, **facts)

    def alias(self, kind, identity, name, *, scope, start='1800-01-01', end='2200-01-01', decision):
        if str(identity) not in self.payload['entities'].get(kind, {}):
            raise ValueError(f'Unknown canonical {kind} {identity}')
        if not decision or date.fromisoformat(start) > date.fromisoformat(end):
            raise ValueError('Alias requires a reviewed decision and valid effective dates')
        a=dict(kind=kind, id=identity, name=str(name), scope=str(scope), start=start, end=end, decision=decision)
        token=json.dumps(a,sort_keys=True)
        if token not in self._alias_records:
            self.payload['aliases'].append(a); self.payload['revision'] += 1
            self._alias_records.add(token)
            self.aliases.setdefault((kind,str(scope),name_key(name)),[]).append(a)

    def resolve(self, kind, name, scope, on):
        candidates={a['id'] for a in self.aliases.get((kind,str(scope),name_key(name)), []) if a['start'] <= str(on)[:10] <= a['end']}
        if len(candidates)>1:
            raise ValueError(f'Ambiguous canonical {kind}: {name!r} in {scope} on {on}')
        return next(iter(candidates), None)

    def classify(self, team, classification, *, decision, start='1800-01-01', end='2200-01-01'):
        if classification not in ('not_reserve','reserve_or_development_squad','needs_review'):
            raise ValueError('Unknown squad classification')
        if str(team) not in self.payload['entities'].get('team', {}) or not decision:
            raise ValueError('Classification requires canonical team and review decision')
        if date.fromisoformat(start)>date.fromisoformat(end): raise ValueError('Invalid classification dates')
        record=dict(team=team, classification=classification, start=start, end=end, decision=decision)
        token=json.dumps(record,sort_keys=True)
        if token not in self._squad_records:
            self.payload['squads'].append(record); self.payload['revision'] += 1
            self._squad_records.add(token);self._squads.setdefault(team,[]).append(record)

    def squad(self, team, on):
        states={a['classification'] for a in self._squads.get(int(team),[]) if a['start']<=str(on)[:10]<=a['end']}
        if len(states)>1: raise ValueError(f'Conflicting squad decisions: team {team} on {on}')
        return next(iter(states),'needs_review')

    def fixture(self, competition, season, home, away, day, stage='', *, create=True):
        key=(int(competition),int(season),int(home),int(away),str(day)[:10],str(stage))
        if key in self.fixture_keys: return self.fixture_keys[key]
        if not create: return None
        identity=self.allocate('fixture',occurrences=[list(key)])
        self.fixture_keys[key]=identity
        self._pairs.setdefault(key[:4],[]).append((key[4],key[5],identity))
        return identity

    def nearby_fixture(self, competition, season, home, away, day):
        target=date.fromisoformat(str(day)[:10])
        found=set()
        for pair in [(competition,season,home,away),(competition,season,away,home)]:
            for when,stage,identity in self._pairs.get(pair,[]):
                if abs((date.fromisoformat(when)-target).days)<=2:found.add(identity)
        return found

    def map_occurrence(self, fixture, competition, season, home, away, day, stage='', *, decision):
        """Explicit reviewed postponement/identity correction; score is never a key."""
        if not decision: raise ValueError('A reviewed fixture mapping decision is required')
        key=(int(competition),int(season),int(home),int(away),str(day)[:10],str(stage))
        if key in self.fixture_keys and self.fixture_keys[key]!=fixture: raise ValueError('Fixture occurrence collision')
        item=self.payload['entities']['fixture'][str(fixture)]
        if list(key) not in item['occurrences']:
            item['occurrences'].append(list(key));self.payload['revision']+=1
        self.fixture_keys[key]=fixture
        self._pairs.setdefault(key[:4],[]).append((key[4],key[5],fixture))
        record=dict(decision=decision,occurrence=list(key))
        decisions=item.setdefault('decisions',[])
        if record not in decisions:
            decisions.append(record)
            self.payload['revision']+=1

    def save(self, path):
        from .storage import atomic_json
        atomic_json(path,self.payload)

    @classmethod
    def load(cls,path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))
