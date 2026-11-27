"""One identity-safe query vocabulary for native publications.

Names are exact (Unicode/case/whitespace normalized), never fuzzy. Storage keys
remain aliases. The index owns no global cache and does not write source data.
"""
from copy import deepcopy
import hashlib
import json
from numbers import Integral
from pathlib import Path
import unicodedata

from ._source import _open_source

QUERY_POLICY = 'sofascore_native_query_v1'


class AmbiguousNativeName(ValueError):
    def __init__(self, kind, name, candidates):
        self.candidates = deepcopy(candidates)
        super().__init__(f'Ambiguous {kind} name {name!r}; use an ID or narrower scope. Candidates: {candidates}')


def _name(value):
    return ' '.join(unicodedata.normalize('NFC', str(value)).casefold().split())


def _values(value):
    if value is None:
        return None
    values = value if isinstance(value, (tuple, list)) else [value]
    if not values or any(isinstance(v, bool) or not isinstance(v, (str, Integral)) or not str(v).strip() for v in values):
        raise ValueError('Native filters require nonempty names or integer IDs')
    return [int(v) if isinstance(v,Integral) else v for v in values]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def _pin(record):
    return {k: record[k] for k in ('publication_sha256','manifest_sha256')}


def validate_selection_envelope(selection):
    """Validate saved identity without pretending stale pins are current."""
    if (not isinstance(selection, dict) or selection.get('policy') != QUERY_POLICY
            or selection.get('fingerprint') != _digest({k:v for k,v in selection.items() if k!='fingerprint'})):
        raise ValueError('Invalid or tampered native selection')


class NativeLibrary:
    """Discover all publications, independently of their feature readiness.

    ``discover(kind, **query)`` lists competitions, seasons, phases, teams or
    annual_groups. ``select(**query)`` binds the exact publication population.
    Team filters select *targets*, not the history needed to calculate features.
    ``load`` retains complete selected phase histories and marks native_target.
    """
    def __init__(self, data_root):
        import pyarrow.parquet as pq
        self.root = Path(data_root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.publications = []
        self._entities = {k: {} for k in ('competitions', 'seasons', 'teams', 'phases', 'annual_groups')}
        for path in sorted(self.root.glob('*.manifest.json')):
            stem = path.name.removesuffix('.manifest.json')
            source = _open_source(self.root, stem)
            manifest = source.manifest
            meta = manifest.get('native_integration', {})
            scope = manifest['scope']; cid, sid = int(scope['league_id']), int(scope['season_id'])
            source_key, season_key = stem.rsplit('_', 2)[0], '_'.join(stem.rsplit('_', 2)[1:])
            calendar = manifest.get('season_calendar', {})
            cols = ['event_id', 'competition_id', 'season_id', 'tournament_id', 'home_id', 'away_id', 'home_name', 'away_name']
            table_path=source.table_path('matches')
            with table_path.open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=manifest['tables']['matches']['sha256']:
                    raise ValueError(f'Native query match fingerprint mismatch: {stem}')
            parquet = pq.ParquetFile(table_path)
            frame = parquet.read(columns=[c for c in cols if c in parquet.schema_arrow.names]).to_pandas()
            if len(frame)!=manifest['tables']['matches']['rows']:
                raise ValueError(f'Native query match count mismatch: {stem}')
            if frame.event_id.duplicated().any() or not frame.competition_id.eq(cid).all() or not frame.season_id.eq(sid).all():
                raise ValueError(f'Native query identity conflict: {stem}')
            phases = sorted(int(i) for i in frame.get('tournament_id', []).dropna().unique()) if 'tournament_id' in frame else []
            teams = sorted(set(map(int, frame.home_id)) | set(map(int, frame.away_id)))
            membership = {}
            for tid in phases:
                f = frame.loc[frame.tournament_id.eq(tid)]
                membership[tid] = sorted(set(map(int, f.home_id)) | set(map(int, f.away_id)))
            item = dict(stem=stem, competition_id=cid, season_id=sid, phase_ids=phases, team_ids=teams,
                        unknown_phase_rows=int(frame.tournament_id.isna().sum()) if 'tournament_id' in frame else len(frame),
                        phase_membership=membership, source_league=source_key, source_season=season_key,
                        annual_group=(calendar.get('annual_league'), calendar.get('annual_season')),
                        manifest=manifest, record=source.record, publication=json.loads(path.read_bytes()))
            self.publications.append(item)
            evidence = meta.get('evidence', {})
            self._add('competitions', cid, meta.get('competition', {}).get('name'),
                      [source_key, scope.get('league_name')], evidence.get('competition'), stem, id=cid)
            self._add('seasons', (cid, sid), meta.get('season', {}).get('name'),
                      [season_key, meta.get('season', {}).get('year')], evidence.get('season'), stem,
                      id=sid, competition_id=cid, year=meta.get('season', {}).get('year'))
            annual = item['annual_group']
            if all(annual):
                self._add('annual_groups', annual, annual[1], [annual[0]+':'+annual[1]],
                          {'policy': calendar.get('policy')}, stem, id=annual[0]+':'+annual[1], annual_league=annual[0])
            for tid in phases:
                phase = meta.get('tournaments', {}).get(str(tid), {})
                aliases = [calendar.get('tournaments', {}).get(str(tid), {}).get('phase')]
                self._add('phases', (cid, tid), phase.get('name'), aliases,
                          evidence.get('competition') if phase.get('name') else None,
                          stem, id=tid, competition_id=cid)
            for tid in teams:
                team = meta.get('teams', {}).get(str(tid), {})
                aliases = list(team.get('aliases', []))
                for side in ('home', 'away'):
                    if side+'_name' in frame:
                        aliases.extend(frame.loc[frame[side+'_id'].eq(tid), side+'_name'].dropna().tolist())
                self._add('teams', tid, team.get('name'), aliases, team.get('evidence'), stem, id=tid)

    def _add(self, kind, key, name, aliases, evidence, stem, **identity):
        row = self._entities[kind].setdefault(key, dict(**identity, name=None, aliases=[], publications=[],
                                                       evidence=[], conflicts=[], _observed=-1))
        row['publications'].append(stem)
        for alias in [name, *aliases]:
            if alias and alias not in row['aliases']:
                row['aliases'].append(alias)
        if name and evidence:
            stamp = evidence.get('observed_at', 0)
            if stamp > row['_observed']:
                row.update(name=name, _observed=stamp, conflicts=[])
            elif stamp == row['_observed'] and row['name'] != name:
                row['conflicts'] = sorted(set([*row['conflicts'], row['name'], name]))
            record = dict(name=name, **evidence)
            if record not in row['evidence']:
                row['evidence'].append(record)

    def _public(self, row):
        out = {k: deepcopy(v) for k, v in row.items() if not k.startswith('_')}
        out['evidence_status'] = 'conflicting' if row['conflicts'] else 'captured' if row['name'] else 'missing'
        out['publications'] = sorted(set(out['publications']))
        out['aliases'] = sorted(set(out['aliases']))
        return out

    def _resolve(self, kind, values, publications):
        if values is None:
            return None
        stems = {p['stem'] for p in publications}
        allowed_teams = {t for p in publications for t in p['team_ids']}
        possible = [(key, r) for key, r in self._entities[kind].items() if stems.intersection(r['publications'])
                    and (kind!='teams' or key in allowed_teams)]
        result = set()
        for value in _values(values):
            by_id = isinstance(value, Integral)
            matches = [(k, r) for k, r in possible if (r['id'] == int(value) if by_id else
                       _name(value) in {_name(a) for a in r['aliases']} or str(value) == str(r['id']))]
            if not matches:
                raise ValueError(f'No native {kind} match for {value!r} in selected scope')
            if len(matches) != 1:
                raise AmbiguousNativeName(kind, value, [self._public(r) for _, r in matches])
            key, row = matches[0]
            if row['conflicts'] and not by_id:
                raise AmbiguousNativeName(kind, value, [self._public(row)])
            result.add(key)
        return result

    def _select(self, query):
        unknown = set(query)-{'competitions','seasons','phases','teams','annual_groups','publications'}
        if unknown:
            raise ValueError(f'Unknown native query filters: {sorted(unknown)}')
        pubs = list(self.publications); resolved = {}
        if query.get('publications') is not None:
            stems = set(_values(query['publications']))
            if stems-{p['stem'] for p in pubs}:
                raise ValueError('Unknown native publication alias')
            pubs = [p for p in pubs if p['stem'] in stems]
        for kind in ('competitions','seasons','annual_groups','phases','teams'):
            ids = self._resolve(kind, query.get(kind), pubs); resolved[kind] = ids
            if ids is None:
                continue
            def belongs(p):
                return {'competitions': p['competition_id'] in ids,
                        'seasons': (p['competition_id'],p['season_id']) in ids,
                        'annual_groups': p['annual_group'] in ids,
                        'phases': any((p['competition_id'], t) in ids for t in p['phase_ids']),
                        'teams': bool(set(p['team_ids']) & ids)}[kind]
            pubs = [p for p in pubs if belongs(p)]
            if kind=='phases':
                pubs=[{**p,'team_ids':sorted({t for phase,teams in p['phase_membership'].items()
                      if (p['competition_id'],phase) in ids for t in teams})} for p in pubs]
        if not pubs:
            raise ValueError('Native query matched no publications')
        return pubs, resolved

    def discover(self, kind='publications', **query):
        pubs, resolved = self._select(query)
        stems = {p['stem'] for p in pubs}
        if kind == 'publications':
            return [dict(stem=p['stem'], competition_id=p['competition_id'], season_id=p['season_id'],
                         competition=self._public(self._entities['competitions'][p['competition_id']]),
                         season=self._public(self._entities['seasons'][p['competition_id'],p['season_id']]),
                         unknown_phase_rows=p['unknown_phase_rows'],
                         phase_ids=p['phase_ids'], annual_group=list(p['annual_group']),
                         rules=deepcopy(p['manifest'].get('native_integration', {}).get('rules', {}))) for p in pubs]
        if kind not in self._entities:
            raise ValueError(f'Unknown entity kind: {kind}')
        allowed_teams={t for p in pubs for t in p['team_ids']}
        return [self._public(r) for k,r in self._entities[kind].items()
                if stems.intersection(r['publications']) and (resolved.get(kind) is None or k in resolved[kind])
                and (kind!='teams' or k in allowed_teams)]

    def select(self, **query):
        query={key:(None if value is None else _values(value) if isinstance(value,(tuple,list)) else _values(value)[0])
               for key,value in query.items()}
        if {p.name.removesuffix('.manifest.json') for p in self.root.glob('*.manifest.json')} != {p['stem'] for p in self.publications}:
            raise ValueError('Native catalogue population changed; rebuild NativeLibrary before selecting')
        pubs, resolved = self._select(query)
        for pub in pubs:
            if _open_source(self.root,pub['stem']).record != pub['record']:
                raise ValueError('Native catalogue changed; rebuild NativeLibrary before selecting')
        result = dict(policy=QUERY_POLICY, query=deepcopy(query),
                      publications=[dict(stem=p['stem'], record=_pin(p['record'])) for p in pubs],
                      phase_ids=None if resolved['phases'] is None else sorted([list(k) for k in resolved['phases']]),
                      team_ids=None if resolved['teams'] is None else sorted(resolved['teams']))
        stems={p['stem'] for p in pubs}
        vocabulary={kind:[self._public(r) for r in records.values() if stems.intersection(r['publications'])]
                    for kind,records in self._entities.items()}
        result['vocabulary_sha256']=_digest(vocabulary)
        result['fingerprint'] = _digest(result)
        return result

    def validate_selection(self, selection, query=None):
        validate_selection_envelope(selection)
        current = self.select(**(selection['query'] if query is None else query))
        if current != selection:
            raise ValueError('Native selection population or manifest pins changed; explicitly refresh the recipe')
        return current

    def load(self, *, query=None, selection=None, tables=('matches',), record_dir=None,
             verify_hashes=False, include_awarded=False, allow_missing_statistics=False):
        from .loading import _load_selected
        from .native_formats import apply_native_scopes
        chosen = self.validate_selection(selection, query) if selection is not None else self.select(**(query or {}))
        pubs = {p['stem']:p for p in self.publications}
        selected = [(pubs[x['stem']]['source_season'],pubs[x['stem']]['source_league'],x['stem']) for x in chosen['publications']]
        data = _load_selected(self.root, selected, tables=tables, record_dir=record_dir,
                              verify_hashes=verify_hashes, include_awarded=include_awarded,
                              allow_missing_statistics=allow_missing_statistics)
        for item in chosen['publications']:
            if _pin(_open_source(self.root,item['stem']).record) != item['record']:
                raise ValueError('Native publication changed while loading')
        data.provenance['native_selection'] = chosen
        data=apply_native_scopes(data, [pubs[x['stem']] for x in chosen['publications']], chosen)
        self._attach_vocabulary(data)
        return data

    def _attach_vocabulary(self, data):
        """Use the same captured display vocabulary in loaders and reports.

        Raw names and original per-publication metadata remain untouched. Missing
        or conflicting display evidence retains its historical alias and status.
        """
        import pandas as pd
        def display(kind,key,fallback):
            row=self._entities[kind].get(key,{})
            return row.get('name') if row.get('name') and not row.get('conflicts') else fallback
        frame=data.matches
        if frame is not None:
            for column,kind,keys in [('competition_name','competitions',['competition_id']),
                                     ('season_name','seasons',['competition_id','season_id']),
                                     ('tournament_name','phases',['competition_id','tournament_id']),
                                     ('home_display_name','teams',['home_id']),('away_display_name','teams',['away_id'])]:
                original=frame.get(column,pd.Series(None,index=frame.index,dtype='object'))
                frame[column]=pd.Series([display(kind,key[0] if len(key)==1 else key,old)
                    for key,old in zip(frame[keys].itertuples(index=False,name=None),original)],index=frame.index,dtype='string')
        for source in data.provenance['sources']:
            pub=next(p for p in self.publications if p['stem']==source['season'])
            source['query_display_names']={
                'competition':display('competitions',pub['competition_id'],None),
                'season':display('seasons',(pub['competition_id'],pub['season_id']),None)}

    def preflight(self, *, query=None, **options):
        from .native_preflight import native_preflight
        return native_preflight(self, query=query, **options)
